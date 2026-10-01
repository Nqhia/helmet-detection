"""Kiểm tra nhanh mọi candidate: nạp được không, nhãn thô -> canonical, số box helmet/no_helmet trên ảnh demo,
ms/ảnh. Lưu ảnh đã vẽ vào outputs/check/<model>/ để xem định tính.

  python scripts/check_models.py --images data/demo --conf 0.25 --save 3
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2

from _common import ROOT, setup_logging  # noqa: E402

from helmet_pipeline.config import load_candidates, resolve_detector  # noqa: E402
from helmet_pipeline.detectors import build_detector  # noqa: E402
from helmet_pipeline.draw import draw_frame  # noqa: E402
from helmet_pipeline.types import Det, MotoGroup, Rider  # noqa: E402

log = logging.getLogger("check")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data" / "demo"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--roles", nargs="*", default=["helmet", "openvocab", "helmet_onnx"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--save", type=int, default=3, help="số ảnh đã vẽ lưu mỗi model")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "check"))
    a = ap.parse_args()
    setup_logging()
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    imgs = [cv2.imread(str(p)) for p in files]
    cands = {k: v for k, v in load_candidates().items() if v.get("role", "helmet") in a.roles and not v.get("disabled")}
    if a.only:
        cands = {k: v for k, v in cands.items() if k in a.only}
    rows = []
    out_root = Path(a.out)
    for name, c in cands.items():
        row = {"model": name}
        try:
            cfg = resolve_detector({"candidate": name, "conf": a.conf, **({"imgsz": a.imgsz} if a.imgsz else {})}, None, None)
            t0 = time.perf_counter()
            det = build_detector(cfg)
            row["load_s"] = round(time.perf_counter() - t0, 1)
            row["raw"] = getattr(det, "raw_names", None) or getattr(getattr(det, "head_det", None), "raw_names", None)
            row["labels"] = det.labels
            det.warmup()
            counts: dict[str, int] = {}
            ts = []
            (out_root / name).mkdir(parents=True, exist_ok=True)
            for i, (p, im) in enumerate(zip(files, imgs)):
                t0 = time.perf_counter()
                dets = det.detect(im)
                ts.append((time.perf_counter() - t0) * 1000)
                for d in dets:
                    counts[d.label] = counts.get(d.label, 0) + 1
                if i < a.save:
                    # vẽ head-level/rider-level như nhóm giả để tái dùng draw_frame
                    riders = [Rider("helmet" if d.label in ("helmet", "rider_helmet") else "no_helmet" if d.label in ("no_helmet", "rider_no_helmet") else "unknown",
                                    d.score, None, d) for d in dets if d.label not in ("motorcycle", "bicycle", "person", "plate")]
                    vis = draw_frame(im, [MotoGroup(None, riders)], coco=[d for d in dets if d.label in ("motorcycle", "plate")],
                                     banner=f"{name} conf>={a.conf}")
                    cv2.imwrite(str(out_root / name / p.name), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            row.update({"ms_mean": round(sum(ts) / len(ts), 1), "counts": counts, "ok": True})
        except Exception as exc:  # noqa: BLE001
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
            log.exception("%s lỗi", name)
        rows.append(row)
        log.info("%s", json.dumps(row, ensure_ascii=False, default=str))
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_root / "check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\n| model | ok | load s | ms/img | helmet | no_helmet | khác |\n|---|---|---|---|---|---|---|")
    for r in rows:
        c = r.get("counts", {})
        other = {k: v for k, v in c.items() if k not in ("helmet", "no_helmet")}
        print(f"| {r['model']} | {'✓' if r.get('ok') else '✗ ' + r.get('error', '')[:60]} | {r.get('load_s', '')} | {r.get('ms_mean', '')} | "
              f"{c.get('helmet', 0)} | {c.get('no_helmet', 0) + c.get('rider_no_helmet', 0)} | {other} |")


if __name__ == "__main__":
    main()
