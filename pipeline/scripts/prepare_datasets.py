"""Tải và chuẩn hoá dataset đánh giá về YOLO format: data/<name>/{images,labels}/... + names.txt.

Nguồn (đều tải tự do, đã xác minh 30/09/2026 — research_notes/pipeline_build/eval_datasets.md):
  kerala   : figshare 32310867 "Motorcycle Safety Violation Dataset in Heterogeneous Traffic" (CC BY 4.0) — YOLO, 1.48 GB
  hcmc_v10 : HF harijawahar/Helmet_Detection = Roboflow the-intruder/helmet-detection-dataset v10 COCO (CC BY 4.0) — 76 MB
  bikes_voc: HF cute-face/bike-helmet-dataset/helmet_voc = Kaggle andrewmvd helmet-detection (VOC, 764 ảnh) — 410 MB
  mhdd     : GitHub kunalagrawal2611 MHDD color.zip (CC BY 4.0), HCMC CCTV, chỉ lớp helmet — 655 MB (LFS)
  demo     : video Pexels/Wikimedia Commons giao thông VN
  python scripts/prepare_datasets.py --only kerala hcmc_v10 bikes_voc mhdd demo
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from _common import ROOT, setup_logging  # noqa: E402

log = logging.getLogger("prepare")
DATA = ROOT / "data"
DL = ROOT / "downloads"


def http_get(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Đã có %s", dest.name)
        return dest
    with requests.get(url, stream=True, timeout=120, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, mininterval=5) as bar:
            for chunk in r.iter_content(4 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def write_names(d: Path, names: list[str]) -> None:
    (d / "names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


# ---------------- kerala (YOLO sẵn) ----------------

def prep_kerala() -> None:
    out = DATA / "kerala"
    if (out / "names.txt").exists():
        log.info("kerala: đã chuẩn bị"); return
    z = http_get("https://ndownloader.figshare.com/files/64908912", DL / "Motorcycle_Safety_Violation_Dataset.zip")
    log.info("kerala: giải nén (1.48 GB)...")
    with zipfile.ZipFile(z) as zf:
        members = [m for m in zf.namelist() if "/images/" in m or "/labels/" in m or m.endswith(("data.yaml", "metadata.json", "README.md"))]
        for m in tqdm(members, desc="unzip kerala", mininterval=5):
            rel = Path(*Path(m).parts[1:])            # bỏ thư mục gốc
            if m.endswith("/"):
                continue
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    write_names(out, ["Helmet", "No_Helmet"])
    log.info("kerala: %d ảnh", sum(1 for _ in (out / "images").rglob("*.jpg")))


# ---------------- hcmc_v10 (COCO -> YOLO) ----------------

def coco_to_yolo(json_path: Path, img_dir: Path, out_img: Path, out_lab: Path, cat_map: dict[int, int]) -> int:
    d = json.load(open(json_path, encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    anns: dict[int, list] = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    out_img.mkdir(parents=True, exist_ok=True); out_lab.mkdir(parents=True, exist_ok=True)
    n = 0
    for iid, im in imgs.items():
        src = img_dir / im["file_name"]
        if not src.exists():
            continue
        shutil.copy2(src, out_img / src.name)
        W, H = im["width"], im["height"]
        lines = []
        for a in anns.get(iid, []):
            if a["category_id"] not in cat_map:
                continue
            x, y, w, h = a["bbox"]
            lines.append(f"{cat_map[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        (out_lab / (src.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n += 1
    return n


def prep_hcmc_v10() -> None:
    out = DATA / "hcmc_v10"
    if (out / "names.txt").exists():
        log.info("hcmc_v10: đã chuẩn bị"); return
    url = "https://huggingface.co/datasets/harijawahar/Helmet_Detection/resolve/main/helmet%20detection%20dataset.v10i.coco%20(3).zip"
    z = http_get(url, DL / "helmet_v10_coco.zip")
    tmp = DL / "helmet_v10_coco"
    if not tmp.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
    total = 0
    for split in ("train", "valid", "test"):
        sd = next((p for p in tmp.rglob(split) if (p / "_annotations.coco.json").exists()), None)
        if sd is None:
            log.warning("hcmc_v10: không thấy split %s", split); continue
        # categories: 0 supercategory (bỏ), 1 Helmet -> 0, 2 NOHelmet -> 1 (đọc theo tên cho chắc)
        d = json.load(open(sd / "_annotations.coco.json", encoding="utf-8"))
        cat_map = {}
        for c in d["categories"]:
            nm = c["name"].strip().lower()
            if nm == "helmet": cat_map[c["id"]] = 0
            elif nm in ("nohelmet", "no_helmet", "no-helmet"): cat_map[c["id"]] = 1
        total += coco_to_yolo(sd / "_annotations.coco.json", sd, out / "images" / split, out / "labels" / split, cat_map)
    write_names(out, ["Helmet", "NOHelmet"])
    log.info("hcmc_v10: %d ảnh", total)


# ---------------- bikes_voc (VOC -> YOLO) ----------------

def prep_bikes_voc() -> None:
    out = DATA / "bikes_voc"
    if (out / "names.txt").exists():
        log.info("bikes_voc: đã chuẩn bị"); return
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download("cute-face/bike-helmet-dataset", repo_type="dataset", local_dir=str(DL / "bikes_helmets"),
                                   allow_patterns=["helmet_voc/*"]))
    img_dir = local / "helmet_voc" / "images"; ann_dir = local / "helmet_voc" / "annotations"
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    names = {"with helmet": 0, "without helmet": 1}
    n = 0
    for xml in tqdm(sorted(ann_dir.glob("*.xml")), desc="voc->yolo", mininterval=5):
        root = ET.parse(xml).getroot()
        fn = root.findtext("filename") or (xml.stem + ".png")
        src = img_dir / fn
        if not src.exists():
            cands = list(img_dir.glob(xml.stem + ".*"))
            if not cands: continue
            src = cands[0]
        W = int(root.find("size/width").text); H = int(root.find("size/height").text)
        lines = []
        for obj in root.findall("object"):
            nm = (obj.findtext("name") or "").strip().lower()
            if nm not in names: continue
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{names[nm]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        shutil.copy2(src, out / "images" / src.name)
        (out / "labels" / (src.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n += 1
    write_names(out, ["With Helmet", "Without Helmet"])
    log.info("bikes_voc: %d ảnh", n)


# ---------------- mhdd (YOLO sẵn, chỉ helmet_original) ----------------

def prep_mhdd() -> None:
    out = DATA / "mhdd"
    if (out / "names.txt").exists():
        log.info("mhdd: đã chuẩn bị"); return
    z = http_get("https://media.githubusercontent.com/media/kunalagrawal2611/Motorcycle-Helmet-Detection-Dataset--MHDD-/main/color.zip",
                 DL / "mhdd_color.zip")
    with zipfile.ZipFile(z) as zf:
        members = [m for m in zf.namelist() if "helmet_original/" in m and not m.endswith("/")]
        for m in tqdm(members, desc="unzip mhdd", mininterval=5):
            p = Path(m)
            # color/helmet_original/<split>/<images|labels>/<file> -> data/mhdd/<images|labels>/<split>/<file>
            try:
                i = p.parts.index("helmet_original")
                split, kind, fname = p.parts[i + 1], p.parts[i + 2], p.parts[-1]
            except (ValueError, IndexError):
                continue
            if kind not in ("images", "labels"):
                continue
            target = out / kind / split / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    write_names(out, ["helmet", "motorcycle"])   # theo data.yaml (classes.txt trong zip đã cũ)
    log.info("mhdd: %d ảnh", sum(1 for _ in (out / "images").rglob("*") if _.suffix.lower() in (".jpg", ".png", ".jpeg")))


# ---------------- video demo ----------------

DEMO_VIDEOS = {
    "pexels_3691658_hcmc_1080p.mp4": "https://videos.pexels.com/video-files/3691658/3691658-hd_1920_1080_30fps.mp4",
    "commons_hanoi_motorbike_ride.webm": "https://upload.wikimedia.org/wikipedia/commons/3/33/Veturo_sur_motorciklo_tra_Hanojo.webm",
    "commons_saigon_traffic_govap.webm": "https://upload.wikimedia.org/wikipedia/commons/c/c6/Saigon_traffic-oVbn3HeLDA0.webm",
}


def prep_demo() -> None:
    out = DATA / "demo_video"
    for fn, url in DEMO_VIDEOS.items():
        try:
            http_get(url, out / fn)
        except Exception as exc:  # noqa: BLE001
            log.error("demo %s: %s", fn, exc)
    (out / "SOURCES.md").write_text("\n".join(f"- {k}: {v}" for k, v in DEMO_VIDEOS.items()) +
                                    "\n\nPexels License (free use) / Wikimedia Commons CC BY-SA 4.0 (ghi nguồn khi dùng lại).\n", encoding="utf-8")


PREP = {"kerala": prep_kerala, "hcmc_v10": prep_hcmc_v10, "bikes_voc": prep_bikes_voc, "mhdd": prep_mhdd, "demo": prep_demo}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=list(PREP))
    a = ap.parse_args()
    setup_logging()
    for name in a.only:
        try:
            PREP[name]()
        except Exception as exc:  # noqa: BLE001
            log.exception("%s LỖI: %s", name, exc)


if __name__ == "__main__":
    main()
