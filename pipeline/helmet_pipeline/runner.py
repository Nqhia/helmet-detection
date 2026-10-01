"""Hàm dùng chung cho scripts/eval.py và scripts/bench.py: dựng detect_fn theo chế độ, chạy eval, đo latency."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import ROOT, PipelineConfig, load_candidates, load_config, resolve_detector
from .detectors import BaseDetector, build_detector
from .evaluation import EvalDataset, EvalResult, evaluate, latency, load_dataset
from .pipeline import HelmetPipeline
from .types import Det

logger = logging.getLogger(__name__)


def helmet_candidates(candidates: dict[str, dict] | None = None, roles=("helmet", "openvocab")) -> dict[str, dict]:
    candidates = candidates or load_candidates()
    return {k: v for k, v in candidates.items() if v.get("role", "helmet") in roles and not v.get("disabled")}


def build_detect_fn(cfg: PipelineConfig, candidate: str, stage: str = "detector",
                    overrides: dict | None = None) -> tuple[Callable[[np.ndarray], list[Det]], str, object]:
    """stage='detector': chỉ model mũ full-frame. stage='pipeline': COCO -> crop/both theo cfg.helmet_mode."""
    cfg = copy.deepcopy(cfg)
    hd = {"candidate": candidate}
    hd.update(overrides or {})
    cfg.helmet_detector = resolve_detector(hd, None, cfg.device)
    cfg.output.save_evidence = False
    cfg.output.save_events = False
    if stage == "detector":
        det: BaseDetector = build_detector(cfg.helmet_detector)
        det.warmup(det.imgsz if hasattr(det, "imgsz") else 640)
        return det.detect, det.name, det
    pipe = HelmetPipeline(cfg)
    pipe.helmet_det.warmup()
    pipe.vehicle_det.warmup()

    def fn(img: np.ndarray) -> list[Det]:
        coco = pipe.vehicle_det.detect(img)
        return pipe._detect_helmet(img, coco)          # noqa: SLF001 — dùng nội bộ có chủ đích
    return fn, f"{pipe.helmet_det.name}+{cfg.helmet_mode}", pipe


def run_eval(cfg: PipelineConfig, candidate: str, datasets: list[EvalDataset], stage: str = "detector",
             op_conf: float = 0.4, overrides: dict | None = None) -> list[EvalResult]:
    fn, name, _ = build_detect_fn(cfg, candidate, stage, overrides)
    out = []
    for ds in datasets:
        if ds.level == "rider" and stage == "detector":
            pass                                        # vẫn đánh giá được nhờ khớp tâm-trong-box
        logger.info("Eval %s trên %s (%s)...", name, ds.name, ds.level)
        out.append(evaluate(fn, ds, name, op_conf=op_conf))
        logger.info(out[-1].row())
    return out


def sample_images(datasets: list[EvalDataset], n: int = 8) -> list[np.ndarray]:
    imgs: list[np.ndarray] = []
    for ds in datasets:
        for path, _ in ds.items():
            im = cv2.imread(str(path))
            if im is not None:
                imgs.append(im)
            if len(imgs) >= n:
                return imgs
    return imgs


def measure_latency(cfg: PipelineConfig, candidate: str, images: list[np.ndarray], device: str | None,
                    stage: str = "detector", runs: int = 30) -> dict:
    fn, name, _ = build_detect_fn(cfg, candidate, stage, {"device": device} if device else None)
    r = latency(fn, images, runs=runs)
    r.update({"model": name, "device": device or "auto", "stage": stage})
    return r


def load_datasets(spec_path: str | Path | None = None, only: list[str] | None = None,
                  max_images: int | None = None) -> list[EvalDataset]:
    import yaml
    spec_path = Path(spec_path or ROOT / "configs" / "datasets.yaml")
    data = yaml.safe_load(open(spec_path, encoding="utf-8")) or {}
    out = []
    for spec in data.get("datasets", []):
        if spec.get("disabled"):
            continue
        if only and spec["name"] not in only:
            continue
        if max_images:
            spec = {**spec, "max_images": max_images}
        base = Path(spec["path"]) if Path(spec["path"]).is_absolute() else ROOT / spec["path"]
        if not (base / spec.get("images", "images")).exists():
            logger.warning("Bỏ dataset %s: chưa có %s (chạy scripts/prepare_datasets.py)", spec["name"], base / spec.get("images", "images"))
            continue
        try:
            ds = load_dataset(spec, ROOT)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            logger.warning("Bỏ dataset %s: %s", spec["name"], exc)
            continue
        out.append(ds)
    return out
