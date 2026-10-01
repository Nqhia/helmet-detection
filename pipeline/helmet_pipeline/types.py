"""Kiểu dữ liệu chung + hàm hình học. Toạ độ box luôn là (x1, y1, x2, y2) pixel theo ảnh gốc."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

Box = tuple[float, float, float, float]

# Nhãn chuẩn (canonical) mà mọi detector phải map về. Nhãn ngoài tập này bị bỏ.
CANONICAL = {
    "person", "motorcycle", "bicycle",          # từ COCO detector
    "helmet", "no_helmet", "head",              # head-level (head = đầu chưa rõ trạng thái)
    "rider_helmet", "rider_no_helmet", "rider", # rider-level (box người (+xe))
    "plate",
}
STATUS_LABELS = {"helmet", "no_helmet"}
VEHICLE_LABELS = {"motorcycle", "bicycle"}
RIDER_LEVEL_LABELS = {"rider_helmet", "rider_no_helmet", "rider"}


@dataclass(slots=True)
class Det:
    label: str
    score: float
    box: Box
    raw_label: str = ""
    source: str = ""
    cls_id: int = -1

    @property
    def w(self) -> float: return self.box[2] - self.box[0]
    @property
    def h(self) -> float: return self.box[3] - self.box[1]
    @property
    def area(self) -> float: return max(0.0, self.w) * max(0.0, self.h)
    @property
    def cx(self) -> float: return (self.box[0] + self.box[2]) / 2
    @property
    def cy(self) -> float: return (self.box[1] + self.box[3]) / 2


@dataclass(slots=True)
class Rider:
    status: str                      # "helmet" | "no_helmet" | "unknown"
    score: float                     # độ tin cậy của trạng thái (0 nếu unknown)
    person: Det | None = None        # box người (COCO) hoặc box rider-level
    head: Det | None = None          # box đầu (helmet detector)
    role: str = "rider"              # "driver" | "passenger" | "rider" (chưa xác định)

    @property
    def box(self) -> Box:
        if self.person is not None:
            return self.person.box
        assert self.head is not None
        return self.head.box


@dataclass(slots=True)
class MotoGroup:
    vehicle: Det | None              # box xe (motorcycle/bicycle) hoặc None nếu chỉ có rider-level box
    riders: list[Rider] = field(default_factory=list)
    track_id: int | None = None
    track_confirmed: bool = False
    direction: tuple[float, float] | None = None   # (dx, dy) px/frame từ tracker

    @property
    def n_no_helmet(self) -> int:
        return sum(1 for r in self.riders if r.status == "no_helmet")

    @property
    def n_helmet(self) -> int:
        return sum(1 for r in self.riders if r.status == "helmet")

    @property
    def max_no_helmet_score(self) -> float:
        return max((r.score for r in self.riders if r.status == "no_helmet"), default=0.0)

    def union_box(self) -> Box:
        boxes = [r.box for r in self.riders] + ([self.vehicle.box] if self.vehicle else [])
        return union(boxes)

    @property
    def violation(self) -> bool:
        return self.n_no_helmet > 0


@dataclass(slots=True)
class Event:
    """Sự kiện vi phạm đã xác nhận theo track (1 lần/track)."""
    track_id: int
    frame_idx: int
    timestamp: float
    riders_total: int
    riders_no_helmet: int
    score: float
    box: Box
    evidence_path: str = ""
    crop_path: str = ""
    plate: str | None = None
    plate_score: float | None = None
    camera: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id, "frame_idx": self.frame_idx, "timestamp": self.timestamp,
            "riders_total": self.riders_total, "riders_no_helmet": self.riders_no_helmet,
            "score": round(self.score, 4), "box": [round(v, 1) for v in self.box],
            "evidence_path": self.evidence_path, "crop_path": self.crop_path,
            "plate": self.plate, "plate_score": self.plate_score, "camera": self.camera, **self.extra,
        }


# ---------------- hình học ----------------

def clip_box(b: Box, w: int, h: int) -> Box:
    return (min(max(b[0], 0.0), w), min(max(b[1], 0.0), h), min(max(b[2], 0.0), w), min(max(b[3], 0.0), h))


def inter_area(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: Box, b: Box) -> float:
    i = inter_area(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Phần diện tích `inner` nằm trong `outer` (0..1)."""
    ai = area(inner)
    return inter_area(inner, outer) / ai if ai > 0 else 0.0


def expand(b: Box, ratio: float, w: int | None = None, h: int | None = None,
           up: float = 0.0) -> Box:
    """Nới box quanh tâm theo `ratio`; `up` nới thêm về phía trên theo tỉ lệ chiều cao
    (đầu người ngồi xe nằm TRÊN box xe)."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    bw, bh = (b[2] - b[0]) * ratio, (b[3] - b[1]) * ratio
    out = (cx - bw / 2, cy - bh / 2 - up * (b[3] - b[1]), cx + bw / 2, cy + bh / 2)
    if w is not None and h is not None:
        out = clip_box(out, w, h)
    return out


def union(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(boxes, dtype=float)
    return (float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max()))


def nms(dets: list[Det], iou_thr: float, class_aware: bool = True) -> list[Det]:
    """NMS greedy theo score; dùng để gộp kết quả full-frame + crop hoặc nhiều crop chồng nhau."""
    out: list[Det] = []
    for d in sorted(dets, key=lambda x: x.score, reverse=True):
        keep = True
        for k in out:
            if class_aware and k.label != d.label:
                continue
            if iou(k.box, d.box) > iou_thr:
                keep = False
                break
        if keep:
            out.append(d)
    return out
