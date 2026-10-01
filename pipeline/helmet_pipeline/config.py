"""Nạp cấu hình YAML -> dataclass. Một file pipeline (configs/pipeline.yaml) + một file danh mục
model (configs/candidates.yaml). `helmet_detector.candidate: <tên>` tham chiếu sang danh mục."""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .association import AssocConfig
from .temporal import TemporalConfig

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CropConfig:
    expand: float = 1.25       # nới vùng xe+người quanh tâm
    up: float = 0.6            # nới thêm lên trên theo chiều cao (đầu người ở trên box xe)
    min_px: int = 48           # bỏ vùng quá nhỏ
    max_crops: int = 8         # trần số crop mỗi khung (cảnh đông)
    merge_iou: float = 0.5     # NMS gộp kết quả giữa các crop / với full-frame


@dataclass
class TrackerConfig:
    iou_thr: float = 0.3
    max_misses: int = 15
    min_hits: int = 2


@dataclass
class OutputConfig:
    dir: str = "outputs/run"
    save_video: bool = False
    save_events: bool = True
    save_evidence: bool = True
    draw_coco: bool = False
    draw_pedestrians: bool = False
    jpeg_quality: int = 90


@dataclass
class AlprConfig:
    enabled: bool = False
    detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    ocr_model: str = "global-plates-mobile-vit-v2-model"
    min_conf: float = 0.5


@dataclass
class PipelineConfig:
    device: str | None = None                 # None = ultralytics tự chọn (cuda nếu có)
    vehicle_detector: dict = field(default_factory=dict)
    helmet_detector: dict = field(default_factory=dict)
    helmet_mode: str = "full"                 # full | crop | both
    fullframe_fallback: bool = True           # không thấy xe nào -> vẫn chạy model mũ full-frame
    use_helmet_vehicles: bool = True          # model mũ có lớp xe -> gộp vào tầng COCO
    min_vehicle_px: int = 24
    crop: CropConfig = field(default_factory=CropConfig)
    assoc: AssocConfig = field(default_factory=AssocConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    alpr: AlprConfig = field(default_factory=AlprConfig)
    camera: str = "cam0"


def _dc_from_dict(cls, d: dict | None):
    d = d or {}
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        logger.warning("Bỏ khoá không biết trong %s: %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in d.items() if k in names})


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(p: str | None, root: Path = ROOT) -> str | None:
    if p is None:
        return None
    if os.path.isabs(p) or "://" in p:
        return p
    cand = root / p
    return str(cand) if cand.exists() else p     # tên model ultralytics (yolo11n.pt) giữ nguyên để auto-download


def load_candidates(path: str | Path | None = None) -> dict[str, dict]:
    path = path or ROOT / "configs" / "candidates.yaml"
    data = load_yaml(path)
    out = {}
    for c in data.get("candidates", []):
        out[c["name"]] = c
    return out


def resolve_detector(cfg: dict, candidates: dict[str, dict] | None, device: str | None) -> dict:
    """Trộn `candidate` (từ danh mục) với ghi đè inline; chuẩn hoá đường dẫn weights."""
    cfg = copy.deepcopy(cfg or {})
    if "candidate" in cfg:
        candidates = candidates or load_candidates()
        name = cfg.pop("candidate")
        if name not in candidates:
            raise KeyError(f"Không có candidate {name!r} trong configs/candidates.yaml. Có: {sorted(candidates)}")
        base = copy.deepcopy(candidates[name])
        base.update(cfg)
        cfg = base
        cfg.setdefault("name", name)
    if "weights" in cfg:
        cfg["weights"] = resolve_path(cfg["weights"])
    if "labels_file" in cfg:
        cfg["labels_file"] = resolve_path(cfg["labels_file"])
    if device is not None and "device" not in cfg:
        cfg["device"] = device
    return cfg


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None,
                candidates_path: str | Path | None = None) -> PipelineConfig:
    raw = load_yaml(path) if path else {}
    for k, v in (overrides or {}).items():          # "a.b.c": value
        cur = raw
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    candidates = load_candidates(candidates_path)
    device = raw.get("device")
    cfg = PipelineConfig(
        device=device,
        vehicle_detector=resolve_detector(raw.get("vehicle_detector"), candidates, device),
        helmet_detector=resolve_detector(raw.get("helmet_detector"), candidates, device),
        helmet_mode=raw.get("helmet_mode", "full"),
        fullframe_fallback=raw.get("fullframe_fallback", True),
        use_helmet_vehicles=raw.get("use_helmet_vehicles", True),
        min_vehicle_px=raw.get("min_vehicle_px", 24),
        crop=_dc_from_dict(CropConfig, raw.get("crop")),
        assoc=_dc_from_dict(AssocConfig, raw.get("assoc")),
        tracker=_dc_from_dict(TrackerConfig, raw.get("tracker")),
        temporal=_dc_from_dict(TemporalConfig, raw.get("temporal")),
        output=_dc_from_dict(OutputConfig, raw.get("output")),
        alpr=_dc_from_dict(AlprConfig, raw.get("alpr")),
        camera=raw.get("camera", "cam0"),
    )
    return cfg


def dump_config(cfg: PipelineConfig) -> dict:
    return asdict(cfg)
