"""Tải trọng số model (configs/candidates.yaml) và dataset đánh giá (configs/datasets.yaml).

Mỗi mục có khoá `download`:
  {type: hf, repo_id, filename, repo_type: model|dataset, dest}   -> huggingface_hub
  {type: hf_snapshot, repo_id, repo_type, dest, allow_patterns}    -> cả repo/thư mục con
  {type: url, url, dest}                                           -> HTTP trực tiếp
  {type: gdrive, id, dest}                                         -> Google Drive công khai (gdown)
  {type: ultralytics}                                              -> tên model chuẩn, tự tải khi dùng
Tuỳ chọn: extract: true (zip/tar -> thư mục dest_dir), post: [lệnh shell]...
  python scripts/download.py --models
  python scripts/download.py --datasets --names ds1 ds2
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from _common import ROOT, setup_logging  # noqa: E402

log = logging.getLogger("download")


def http_get(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Đã có %s", dest)
        return dest
    with requests.get(url, stream=True, timeout=60, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def extract(path: Path, to: Path) -> None:
    to.mkdir(parents=True, exist_ok=True)
    marker = to / ".extracted"
    if marker.exists():
        return
    log.info("Giải nén %s -> %s", path.name, to)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(to)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(to)
    else:
        raise ValueError(f"Không biết định dạng nén: {path}")
    marker.touch()


def fetch(spec: dict, name: str) -> Path | None:
    d = spec.get("download")
    if not d:
        return None
    if isinstance(d, list):                       # nhiều file cho 1 candidate (vd. two-stage)
        last = None
        for i, sub in enumerate(d):
            last = fetch({"download": sub}, f"{name}[{i}]")
        return last
    typ = d.get("type", "url")
    if typ == "ultralytics":
        # model chuẩn của Ultralytics (yolo11s.pt, yolov8s-worldv2.pt...): tải từ GitHub assets về đúng đường dẫn `weights`
        target = ROOT / spec["weights"]
        if target.exists():
            log.info("Đã có %s", target)
            return target
        from ultralytics.utils.downloads import attempt_download_asset
        target.parent.mkdir(parents=True, exist_ok=True)
        out = attempt_download_asset(str(target))
        return Path(out) if out else None
    dest = ROOT / d["dest"] if "dest" in d else None
    if typ == "hf":
        from huggingface_hub import hf_hub_download
        out = hf_hub_download(d["repo_id"], d["filename"], repo_type=d.get("repo_type", "model"),
                              local_dir=str(dest.parent if dest and dest.suffix else dest or ROOT / "weights" / name),
                              revision=d.get("revision"))
        path = Path(out)
        if dest and dest.suffix and path.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), dest)          # đổi về tên chuẩn, không giữ bản trùng
            # dọn thư mục con rỗng + .cache của huggingface_hub
            for p in (path.parent, path.parent / ".cache"):
                if p.exists() and p != dest.parent and not any(q for q in p.rglob("*") if q.is_file() and ".cache" not in q.parts):
                    shutil.rmtree(p, ignore_errors=True)
            cache = dest.parent / ".cache"
            if cache.exists():
                shutil.rmtree(cache, ignore_errors=True)
            path = dest
    elif typ == "hf_snapshot":
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(d["repo_id"], repo_type=d.get("repo_type", "dataset"),
                                      local_dir=str(dest or ROOT / "data" / name), allow_patterns=d.get("allow_patterns"),
                                      revision=d.get("revision")))
    elif typ == "url":
        path = http_get(d["url"], dest or ROOT / "downloads" / Path(d["url"]).name.split("?")[0])
    elif typ == "gdrive":
        import gdown
        dest = dest or ROOT / "downloads" / f"{name}.bin"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            gdown.download(id=d["id"], output=str(dest), quiet=False)
        path = dest
    else:
        raise ValueError(f"type không hỗ trợ: {typ}")
    if d.get("extract"):
        extract(path, ROOT / d.get("extract_to", str(path.parent)))
    for cmd in d.get("post", []):
        log.info("post: %s", cmd)
        subprocess.run(cmd, shell=True, check=False, cwd=ROOT)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--datasets", action="store_true")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    setup_logging(a.verbose)
    if not (a.models or a.datasets):
        a.models = a.datasets = True
    jobs = []
    if a.models:
        for c in yaml.safe_load(open(ROOT / "configs" / "candidates.yaml", encoding="utf-8")).get("candidates", []):
            jobs.append((c["name"], c))
    if a.datasets:
        for c in yaml.safe_load(open(ROOT / "configs" / "datasets.yaml", encoding="utf-8")).get("datasets", []):
            jobs.append((c["name"], c))
    ok = fail = 0
    for name, spec in jobs:
        if a.names and name not in a.names:
            continue
        if spec.get("disabled"):
            continue
        try:
            p = fetch(spec, name)
            log.info("OK %s -> %s", name, p)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            log.error("LỖI %s: %s", name, exc)
            fail += 1
    log.info("Xong: %d ok, %d lỗi", ok, fail)


if __name__ == "__main__":
    main()
