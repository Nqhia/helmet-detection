"""Vẽ overlay: box xe (vàng; đỏ nếu vi phạm), đầu (xanh = helmet, đỏ = no_helmet, xám = unknown),
track id, vai trò, banner thống kê. Trả bản sao, không sửa frame gốc."""
from __future__ import annotations

import cv2
import numpy as np

from .types import Det, MotoGroup

C_HELMET = (80, 200, 60)
C_NOHELMET = (40, 40, 230)
C_UNKNOWN = (160, 160, 160)
C_VEH = (0, 210, 255)
C_VEH_BAD = (0, 0, 255)
C_COCO = (200, 120, 0)
C_PED = (120, 120, 120)


def _label(img, text, x, y, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y0 = max(int(y) - th - 4, 0)
    cv2.rectangle(img, (int(x), y0), (int(x) + tw + 4, y0 + th + 4), color, -1)
    cv2.putText(img, text, (int(x) + 2, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, groups: list[MotoGroup], pedestrians: list[Det] | None = None,
               coco: list[Det] | None = None, banner: str | None = None, fps: float | None = None) -> np.ndarray:
    img = frame.copy()
    if coco:
        for d in coco:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), C_COCO, 1)
            _label(img, f"{d.label} {d.score:.2f}", x1, y1, C_COCO, 0.4)
    for g in groups:
        if g.vehicle is not None:
            x1, y1, x2, y2 = (int(v) for v in g.vehicle.box)
            col = C_VEH_BAD if g.violation else C_VEH
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if g.violation else 1)
            tid = f"#{g.track_id}" if g.track_id is not None else ""
            _label(img, f"{g.vehicle.label} {tid} riders={len(g.riders)} noH={g.n_no_helmet}", x1, y1, col, 0.45)
        for r in g.riders:
            col = {"helmet": C_HELMET, "no_helmet": C_NOHELMET}.get(r.status, C_UNKNOWN)
            b = r.head.box if r.head is not None else r.box
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            txt = f"{r.status} {r.score:.2f}" if r.status != "unknown" else "?"
            if r.role in ("driver", "passenger"):
                txt = f"{r.role[0].upper()}:{txt}"
            _label(img, txt, x1, y1, col, 0.42)
            if r.person is not None and r.head is not None:
                px1, py1, px2, py2 = (int(v) for v in r.person.box)
                cv2.rectangle(img, (px1, py1), (px2, py2), col, 1)
    for d in pedestrians or []:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), C_PED, 1)
    if banner or fps is not None:
        text = (banner or "") + (f"  {fps:.1f} FPS" if fps is not None else "")
        cv2.rectangle(img, (0, 0), (min(img.shape[1], 12 + 9 * len(text)), 26), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img
