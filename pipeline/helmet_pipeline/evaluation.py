"""Đánh giá detector mũ trên dataset YOLO-format với class mapping về {helmet, no_helmet}.

Hai chế độ khớp GT:
- level=head : GT là box đầu -> khớp IoU>=0.5 cùng lớp (chuẩn VOC/COCO AP50).
- level=rider: GT là box người/xe kèm trạng thái -> dự đoán đầu khớp khi TÂM đầu nằm trong GT box
               và cùng trạng thái (cho phép so model head-level với GT rider-level).
Chỉ số: AP50 từng lớp, mAP50, P/R/F1 tại ngưỡng vận hành, recall no_helmet tại precision>=0.9,
recall theo cỡ GT (nhỏ/vừa/lớn), số lần nhầm helmet<->no_helmet.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .detectors import ClassMapper
from .types import Det, iou

logger = logging.getLogger(__name__)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EvalDataset:
    name: str
    images_dir: Path
    labels_dir: Path
    names: list[str]                       # tên lớp theo id trong file label
    mapping: dict[str, str]                # raw -> helmet|no_helmet|ignore
    level: str = "head"                    # head | rider
    max_images: int | None = None
    subdirs: list[str] | None = None       # chỉ lấy các split này (vd. ["val","test"]); None = tất cả

    def items(self) -> Iterable[tuple[Path, list[Det]]]:
        mapper = ClassMapper(self.mapping)
        roots = [self.images_dir / s for s in self.subdirs] if self.subdirs else [self.images_dir]
        files = sorted(p for r in roots for p in r.rglob("*") if p.suffix.lower() in IMG_EXT)
        if self.max_images:
            # lấy đều trên toàn tập (không chỉ đầu danh sách) để không lệch theo camera/split
            step = max(1, len(files) // self.max_images)
            files = files[::step][: self.max_images]
        for img_path in files:
            rel = img_path.relative_to(self.images_dir).with_suffix(".txt")
            lab_path = self.labels_dir / rel
            gts: list[Det] = []
            if lab_path.exists():
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                with open(lab_path, encoding="utf-8") as f:
                    for ln in f:
                        parts = ln.split()
                        if len(parts) < 5:
                            continue
                        cid = int(float(parts[0]))
                        raw = self.names[cid] if cid < len(self.names) else str(cid)
                        lab = mapper(raw)
                        if lab is None:
                            continue
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                        box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                        gts.append(Det(lab, 1.0, box, raw, "gt", cid))
            yield img_path, gts


def load_dataset(spec: dict, root: Path) -> EvalDataset:
    """spec từ configs/datasets.yaml: {name, path, images, labels, names|names_file, map_to, level, max_images}."""
    base = Path(spec["path"]) if os.path.isabs(spec["path"]) else root / spec["path"]
    names = spec.get("names")
    if names is None and spec.get("names_file"):
        nf = base / spec["names_file"]
        if nf.suffix in (".yaml", ".yml"):
            import yaml
            n = yaml.safe_load(open(nf, encoding="utf-8")).get("names")
            names = [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)
        else:
            names = [ln.strip() for ln in open(nf, encoding="utf-8") if ln.strip()]
    if names is None and (base / "names.txt").exists():
        names = [ln.strip() for ln in open(base / "names.txt", encoding="utf-8") if ln.strip()]
    if names is None:
        raise FileNotFoundError(f"dataset {spec['name']}: không có `names`, `names_file` hay {base / 'names.txt'} — "
                                f"chạy scripts/prepare_datasets.py trước")
    return EvalDataset(spec["name"], base / spec.get("images", "images"), base / spec.get("labels", "labels"),
                       list(names), dict(spec["map_to"]), spec.get("level", "head"), spec.get("max_images"),
                       spec.get("subdirs"))


# ---------------- AP ----------------

def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match(pred: Det, gt: Det, level: str, iou_thr: float) -> bool:
    if level == "rider":
        cx, cy = pred.cx, pred.cy
        return gt.box[0] <= cx <= gt.box[2] and gt.box[1] <= cy <= gt.box[3]
    return iou(pred.box, gt.box) >= iou_thr


def _size_bucket(d: Det) -> str:
    s = max(d.w, d.h)
    return "small<16" if s < 16 else "16-32" if s < 32 else "32-64" if s < 64 else "large>=64"


@dataclass
class EvalResult:
    dataset: str
    model: str
    n_images: int
    n_gt: dict
    ap50: dict
    map50: float
    op_conf: float
    precision: dict
    recall: dict
    f1: dict
    recall_no_helmet_at_p90: float
    recall_by_size: dict
    status_confusions: int
    ms_per_image: float
    extra: dict = field(default_factory=dict)

    def row(self) -> str:
        r = self.recall.get("no_helmet", float("nan"))
        p = self.precision.get("no_helmet", float("nan"))
        return (f"| {self.model} | {self.dataset} | {self.n_images} | {self.map50:.3f} | {self.ap50.get('helmet', float('nan')):.3f} | "
                f"{self.ap50.get('no_helmet', float('nan')):.3f} | {p:.2f}/{r:.2f} | {self.recall_no_helmet_at_p90:.2f} | "
                f"{self.recall_by_size.get('small<16', float('nan')):.2f}/{self.recall_by_size.get('16-32', float('nan')):.2f} | "
                f"{self.status_confusions} | {self.ms_per_image:.1f} |")

    @staticmethod
    def header() -> str:
        return ("| model | dataset | imgs | mAP50 | AP50 helmet | AP50 no_helmet | P/R no_helmet@op | R no_helmet@P90 | "
                "R small<16 / 16-32 | swaps | ms/img |\n|---|---|---|---|---|---|---|---|---|---|---|")

    def to_dict(self) -> dict:
        return {k: (v if not isinstance(v, float) else round(v, 4)) for k, v in self.__dict__.items()}


def evaluate(detect_fn: Callable[[np.ndarray], list[Det]], ds: EvalDataset, model_name: str,
             classes: tuple[str, ...] = ("helmet", "no_helmet"), iou_thr: float = 0.5, op_conf: float = 0.4,
             progress: bool = True) -> EvalResult:
    preds_all: dict[str, list[tuple[float, int, bool]]] = {c: [] for c in classes}   # (score, img_id, tp?)
    n_gt = {c: 0 for c in classes}
    tp_op = {c: 0 for c in classes}; fp_op = {c: 0 for c in classes}
    size_tot: dict[str, int] = {}; size_hit: dict[str, int] = {}
    swaps = 0
    n_img = 0
    t_total = 0.0
    for img_id, (path, gts) in enumerate(ds.items()):
        img = cv2.imread(str(path))
        if img is None:
            continue
        n_img += 1
        t0 = time.perf_counter()
        preds = [d for d in detect_fn(img) if d.label in classes]
        t_total += time.perf_counter() - t0
        for g in gts:
            if g.label in n_gt:
                n_gt[g.label] += 1
                size_tot[_size_bucket(g)] = size_tot.get(_size_bucket(g), 0) + 1
        # khớp theo lớp, greedy theo score (chuẩn AP)
        for c in classes:
            gts_c = [g for g in gts if g.label == c]
            used = [False] * len(gts_c)
            for p in sorted((p for p in preds if p.label == c), key=lambda d: -d.score):
                best, best_v = -1, -1.0
                for j, g in enumerate(gts_c):
                    if used[j]:
                        continue
                    v = (1.0 if _match(p, g, ds.level, iou_thr) else 0.0) if ds.level == "rider" else iou(p.box, g.box)
                    if v > best_v:
                        best, best_v = j, v
                ok = best >= 0 and (best_v >= iou_thr if ds.level != "rider" else best_v > 0)
                if ok:
                    used[best] = True
                preds_all[c].append((p.score, img_id, ok))
                if p.score >= op_conf:
                    if ok:
                        tp_op[c] += 1
                        b = _size_bucket(gts_c[best])
                        size_hit[b] = size_hit.get(b, 0) + 1
                    else:
                        fp_op[c] += 1
        # nhầm trạng thái: dự đoán ở ngưỡng vận hành khớp vị trí GT lớp KHÁC
        for p in (p for p in preds if p.score >= op_conf):
            other = [g for g in gts if g.label in classes and g.label != p.label]
            if any(_match(p, g, ds.level, iou_thr) for g in other) and not any(_match(p, g, ds.level, iou_thr) for g in gts if g.label == p.label):
                swaps += 1
        if progress and n_img % 200 == 0:
            logger.info("[%s/%s] %d ảnh...", model_name, ds.name, n_img)

    ap50, prec, rec, f1 = {}, {}, {}, {}
    r_at_p90 = 0.0
    for c in classes:
        arr = sorted(preds_all[c], key=lambda t: -t[0])
        if n_gt[c] == 0:
            ap50[c] = float("nan"); prec[c] = rec[c] = f1[c] = float("nan")
            continue
        tp = np.cumsum([1 if t[2] else 0 for t in arr]); fp = np.cumsum([0 if t[2] else 1 for t in arr])
        recall_c = tp / n_gt[c] if len(arr) else np.array([0.0])
        prec_c = tp / np.maximum(tp + fp, 1e-9) if len(arr) else np.array([0.0])
        ap50[c] = _ap_from_pr(recall_c, prec_c) if len(arr) else 0.0
        if c == "no_helmet" and len(arr):
            mask = prec_c >= 0.9
            r_at_p90 = float(recall_c[mask].max()) if mask.any() else 0.0
        P = tp_op[c] / max(tp_op[c] + fp_op[c], 1); R = tp_op[c] / n_gt[c]
        prec[c], rec[c], f1[c] = P, R, (2 * P * R / (P + R) if P + R else 0.0)
    valid = [v for v in ap50.values() if v == v]
    return EvalResult(ds.name, model_name, n_img, n_gt, ap50, float(np.mean(valid)) if valid else float("nan"),
                      op_conf, prec, rec, f1, r_at_p90,
                      {k: size_hit.get(k, 0) / v for k, v in size_tot.items()}, swaps,
                      (t_total / max(n_img, 1)) * 1e3)


def latency(detect_fn: Callable[[np.ndarray], list[Det]], images: list[np.ndarray], warmup: int = 5, runs: int = 30) -> dict:
    if not images:
        raise ValueError("latency(): cần ít nhất 1 ảnh")
    for im in images[:warmup]:
        detect_fn(im)
    ts = []
    for i in range(runs):
        im = images[i % len(images)]
        t0 = time.perf_counter(); detect_fn(im); ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)), "fps": float(1000 / a.mean())}
