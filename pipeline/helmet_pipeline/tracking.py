"""Tracker IoU đơn giản (SORT-lite, không Kalman) cho nhóm xe. Đủ cho bỏ phiếu theo track ở 5–15 fps.

- Dự đoán vị trí = box cuối + vận tốc (EMA của dịch chuyển tâm).
- Ghép Hungarian (scipy) nếu có, không thì greedy theo IoU giảm dần.
- Track "confirmed" sau min_hits lượt khớp; xoá sau max_misses lượt mất.
Trong AI worker, engine đã có tracker riêng — module này chỉ dùng cho bản standalone.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .types import Box, iou

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # noqa: BLE001
    _lsa = None


@dataclass
class Track:
    id: int
    box: Box
    score: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1
    vx: float = 0.0
    vy: float = 0.0
    min_hits: int = 2
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits

    def predicted(self) -> Box:
        return (self.box[0] + self.vx, self.box[1] + self.vy, self.box[2] + self.vx, self.box[3] + self.vy)

    def direction(self, min_speed: float = 0.5) -> tuple[float, float] | None:
        """Hướng đi trung bình (px/frame) trên lịch sử; None nếu gần như đứng yên."""
        if len(self.history) < 3:
            return None
        pts = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in self.history])
        d = pts[-1] - pts[0]
        d = d / max(len(self.history) - 1, 1)
        if float(np.hypot(*d)) < min_speed:
            return None
        return float(d[0]), float(d[1])


class IouTracker:
    def __init__(self, iou_thr: float = 0.3, max_misses: int = 15, min_hits: int = 2, ema: float = 0.6):
        self.iou_thr = iou_thr
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.ema = ema
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def _match(self, preds: list[Box], dets: list[Box]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not preds or not dets:
            return [], list(range(len(preds))), list(range(len(dets)))
        M = np.zeros((len(preds), len(dets)), dtype=float)
        for i, p in enumerate(preds):
            for j, d in enumerate(dets):
                M[i, j] = iou(p, d)
        pairs: list[tuple[int, int]] = []
        if _lsa is not None:
            ri, ci = _lsa(-M)
            for i, j in zip(ri, ci):
                if M[i, j] >= self.iou_thr:
                    pairs.append((int(i), int(j)))
        else:
            used_i, used_j = set(), set()
            for i, j in sorted(((i, j) for i in range(M.shape[0]) for j in range(M.shape[1])), key=lambda t: -M[t]):
                if M[i, j] < self.iou_thr:
                    break
                if i in used_i or j in used_j:
                    continue
                pairs.append((i, j)); used_i.add(i); used_j.add(j)
        mi = {i for i, _ in pairs}
        mj = {j for _, j in pairs}
        return pairs, [i for i in range(len(preds)) if i not in mi], [j for j in range(len(dets)) if j not in mj]

    def update(self, boxes: list[Box], scores: list[float] | None = None) -> list[Track]:
        """Cập nhật với các box của frame hiện tại. Trả list Track tương ứng từng box (cùng thứ tự)."""
        scores = scores or [1.0] * len(boxes)
        ids = list(self.tracks.keys())
        preds = [self.tracks[i].predicted() for i in ids]
        pairs, un_t, un_d = self._match(preds, boxes)
        out: list[Track | None] = [None] * len(boxes)
        for ti, dj in pairs:
            t = self.tracks[ids[ti]]
            nb = boxes[dj]
            dx = ((nb[0] + nb[2]) - (t.box[0] + t.box[2])) / 2
            dy = ((nb[1] + nb[3]) - (t.box[1] + t.box[3])) / 2
            t.vx = self.ema * t.vx + (1 - self.ema) * dx
            t.vy = self.ema * t.vy + (1 - self.ema) * dy
            t.box = nb
            t.score = scores[dj]
            t.hits += 1
            t.misses = 0
            t.age += 1
            t.history.append(nb)
            out[dj] = t
        for ti in un_t:
            t = self.tracks[ids[ti]]
            t.misses += 1
            t.age += 1
            if t.misses > self.max_misses:
                del self.tracks[ids[ti]]
        for dj in un_d:
            t = Track(self._next, boxes[dj], scores[dj], min_hits=self.min_hits)
            t.history.append(boxes[dj])
            self.tracks[t.id] = t
            self._next += 1
            out[dj] = t
        return out  # type: ignore[return-value]

    def active_ids(self) -> set[int]:
        return set(self.tracks.keys())
