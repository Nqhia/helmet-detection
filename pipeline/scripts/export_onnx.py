"""Export model mũ (.pt) sang ONNX end-to-end (đã NMS trong graph, output (1, max_det, 6)) — đúng
định dạng YoloDetector của AI worker (detect/yolo.py) — rồi kiểm tra parity .pt vs .onnx.

  python scripts/export_onnx.py --helmet <candidate> --imgsz 640 --opset 17 --check-dir data/<ds>/images
Sinh: weights/onnx/<name>_<imgsz>.onnx, <name>.txt (nhãn thô, 1 dòng/lớp), <name>.json (hợp đồng:
input, output, nhãn thô -> canonical, ngưỡng đề xuất).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT, setup_logging  # noqa: E402

from helmet_pipeline.config import load_candidates, resolve_detector  # noqa: E402
from helmet_pipeline.detectors import OnnxDetector, UltralyticsDetector  # noqa: E402
from helmet_pipeline.types import iou  # noqa: E402

log = logging.getLogger("export")


def parity(pt: UltralyticsDetector, ox: OnnxDetector, images: list[np.ndarray]) -> dict:
    n_pt = n_ox = matched = 0
    ious = []
    for im in images:
        a = pt.detect(im)
        b = ox.detect(im)
        n_pt += len(a); n_ox += len(b)
        used = set()
        for d in a:
            best, bi = 0.0, -1
            for j, e in enumerate(b):
                if j in used or e.label != d.label:
                    continue
                v = iou(d.box, e.box)
                if v > best:
                    best, bi = v, j
            if best >= 0.5:
                matched += 1; used.add(bi); ious.append(best)
    return {"images": len(images), "boxes_pt": n_pt, "boxes_onnx": n_ox, "matched_iou>=0.5": matched,
            "mean_iou_matched": float(np.mean(ious)) if ious else None,
            "match_rate": matched / max(n_pt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--helmet", help="tên candidate")
    ap.add_argument("--weights", help="hoặc đường dẫn .pt trực tiếp")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17, help="17 an toàn cho onnxruntime 1.19 của worker")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--no-nms", action="store_true", help="export thô (1,4+nc,N) — worker cũng parse được")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "weights" / "onnx"))
    ap.add_argument("--check-dir", help="thư mục ảnh để kiểm tra parity")
    ap.add_argument("--check-n", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    setup_logging(a.verbose)

    if a.helmet:
        cfg = resolve_detector({"candidate": a.helmet}, load_candidates(), None)
    elif a.weights:
        cfg = {"name": Path(a.weights).stem, "weights": a.weights, "classes": None, "keep_unmapped": True}
    else:
        raise SystemExit("--helmet hoặc --weights")
    if cfg.get("backend") not in (None, "ultralytics"):
        raise SystemExit("Chỉ export được backend ultralytics (.pt)")
    cfg["conf"] = 0.001                         # so parity ở ngưỡng thấp rồi lọc sau
    cfg["imgsz"] = a.imgsz
    pt = UltralyticsDetector(cfg)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.get('name') or pt.name}_{a.imgsz}"

    log.info("Export %s -> ONNX (imgsz=%d, nms=%s, opset=%d)", cfg["weights"], a.imgsz, not a.no_nms, a.opset)
    kw = dict(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=True, dynamic=a.dynamic,
              nms=not a.no_nms, max_det=a.max_det, conf=0.001, iou=0.5)
    if a.half:
        kw["half"] = True
    exported = pt.model.export(**kw)
    onnx_path = out_dir / f"{name}.onnx"
    shutil.move(str(exported), onnx_path)
    (out_dir / f"{name}.txt").write_text("\n".join(pt.raw_names) + "\n", encoding="utf-8")
    contract = {
        "onnx": onnx_path.name, "labels_file": f"{name}.txt", "input": f"1x3x{a.imgsz}x{a.imgsz} RGB float32 /255, letterbox",
        "output": f"(1, {a.max_det}, 6) = [x1,y1,x2,y2,conf,cls] toạ độ letterbox" if not a.no_nms else "(1, 4+nc, N) thô, cần NMS",
        "raw_names": pt.raw_names, "canonical_map": cfg.get("classes"), "box_level": cfg.get("box_level", "head"),
        # ngưỡng phục vụ: `conf_serve` khai trong candidates.yaml (đo ở benchmark), không thì lấy conf của candidate
        "suggested_conf": cfg.get("conf_serve", cfg.get("conf", 0.3)),
        "source_weights": os.path.relpath(cfg["weights"], ROOT).replace("\\", "/") if os.path.isabs(str(cfg["weights"])) else str(cfg["weights"]),
        "source_candidate": cfg.get("name"), "opset": a.opset,
    }
    json.dump(contract, open(out_dir / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("-> %s", onnx_path)

    if a.check_dir:
        files = sorted(p for p in Path(a.check_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: a.check_n]
        imgs = [cv2.imread(str(p)) for p in files]
        imgs = [im for im in imgs if im is not None]
        ox_cfg = {**cfg, "weights": str(onnx_path), "labels_file": str(out_dir / f"{name}.txt"), "conf": 0.25, "device": "cpu"}
        pt.conf = 0.25
        ox = OnnxDetector(ox_cfg)
        rep = parity(pt, ox, imgs)
        rep["onnx_ms_cpu"] = ox.last_ms
        log.info("Parity: %s", json.dumps(rep))
        json.dump(rep, open(out_dir / f"{name}_parity.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
