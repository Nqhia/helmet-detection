"""HelmetPipeline — điều phối toàn bộ: COCO -> mũ (full/crop/both) -> association -> track -> bỏ phiếu
-> sự kiện + bằng chứng (+ALPR). Không phụ thuộc nguồn video (xem scripts/infer.py)."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .association import assign_roles, build_groups
from .config import PipelineConfig
from .detectors import BaseDetector, build_detector
from .draw import draw_frame
from .temporal import ViolationMonitor
from .tracking import IouTracker
from .types import Box, Det, Event, MotoGroup, clip_box, expand, nms, union, VEHICLE_LABELS

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame_idx: int
    ts: float
    coco: list[Det]
    helmet: list[Det]
    groups: list[MotoGroup]
    pedestrians: list[Det]
    events: list[Event]
    timings_ms: dict = field(default_factory=dict)

    @property
    def n_violating_now(self) -> int:
        return sum(1 for g in self.groups if g.violation)


class HelmetPipeline:
    def __init__(self, cfg: PipelineConfig, out_dir: str | Path | None = None):
        self.cfg = cfg
        self.vehicle_det: BaseDetector = build_detector(cfg.vehicle_detector)
        self.helmet_det: BaseDetector = build_detector(cfg.helmet_detector)
        self.tracker = IouTracker(cfg.tracker.iou_thr, cfg.tracker.max_misses, cfg.tracker.min_hits)
        self.monitor = ViolationMonitor(cfg.temporal)
        self.frame_idx = 0
        self.out_dir = Path(out_dir or cfg.output.dir)
        self.events: list[Event] = []
        self._alpr = None
        if cfg.alpr.enabled:
            try:
                from fast_alpr import ALPR
                self._alpr = ALPR(detector_model=cfg.alpr.detector_model, ocr_model=cfg.alpr.ocr_model)
                logger.info("ALPR bật (%s / %s)", cfg.alpr.detector_model, cfg.alpr.ocr_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Không bật được fast-alpr: %s", exc)
        if cfg.output.save_events or cfg.output.save_evidence:
            (self.out_dir / "events").mkdir(parents=True, exist_ok=True)
        logger.info("Pipeline: vehicle=%s helmet=%s(level=%s) mode=%s",
                    self.vehicle_det.name, self.helmet_det.name, self.helmet_det.box_level, cfg.helmet_mode)

    def reset(self) -> None:
        """Xoá trạng thái tracker + bỏ phiếu (đổi nguồn, ảnh rời không liên tục). Không nạp lại model."""
        self.tracker = IouTracker(self.cfg.tracker.iou_thr, self.cfg.tracker.max_misses, self.cfg.tracker.min_hits)
        self.monitor = ViolationMonitor(self.cfg.temporal)

    # ---------------- helmet detection theo chế độ ----------------

    def _crop_regions(self, coco: list[Det], w: int, h: int) -> list[Box]:
        vehicles = [d for d in coco if d.label in VEHICLE_LABELS]
        persons = [d for d in coco if d.label == "person"]
        regions: list[Box] = []
        for v in vehicles:
            near = [p.box for p in persons
                    if abs(p.cx - v.cx) < v.w * 0.8 and p.box[3] > v.box[1] and p.box[1] < v.box[3]]
            reg = union([v.box] + near)
            reg = expand(reg, self.cfg.crop.expand, w, h, up=self.cfg.crop.up)
            if min(reg[2] - reg[0], reg[3] - reg[1]) < self.cfg.crop.min_px:
                continue
            regions.append(reg)
        # gộp vùng chồng nhau nhiều để bớt crop; ưu tiên vùng lớn (gần camera)
        regions.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
        merged: list[Box] = []
        for r in regions:
            if any(_iou(r, m) > 0.6 for m in merged):
                continue
            merged.append(r)
        return merged[: self.cfg.crop.max_crops]

    def _detect_helmet(self, frame: np.ndarray, coco: list[Det]) -> list[Det]:
        h, w = frame.shape[:2]
        mode = self.cfg.helmet_mode
        dets: list[Det] = []
        regions = self._crop_regions(coco, w, h) if mode in ("crop", "both") else []
        if mode == "full" or mode == "both" or (not regions and self.cfg.fullframe_fallback):
            dets += self.helmet_det.detect(frame)
        if regions:
            crops = [frame[int(r[1]):int(r[3]), int(r[0]):int(r[2])] for r in regions]
            batch = self.helmet_det.detect_batch(crops)
            for r, cd in zip(regions, batch):
                ox, oy = r[0], r[1]
                for d in cd:
                    b = (d.box[0] + ox, d.box[1] + oy, d.box[2] + ox, d.box[3] + oy)
                    dets.append(Det(d.label, d.score, clip_box(b, w, h), d.raw_label, d.source + "@crop", d.cls_id))
            dets = nms(dets, self.cfg.crop.merge_iou, class_aware=False)
        return dets

    # ---------------- một khung hình ----------------

    def process(self, frame: np.ndarray, ts: float | None = None) -> FrameResult:
        ts = time.monotonic() if ts is None else ts
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        coco = self.vehicle_det.detect(frame)
        coco = [d for d in coco if d.label == "person" or min(d.w, d.h) >= self.cfg.min_vehicle_px]
        t1 = time.perf_counter()

        helmet = self._detect_helmet(frame, coco)
        t2 = time.perf_counter()

        # model mũ có lớp xe (motorbike/bike) -> bổ sung vào tầng COCO (COCO hay miss xe máy nghiêng/khuất)
        if self.cfg.use_helmet_vehicles:
            extra = [Det(d.label, d.score, d.box, d.raw_label, d.source, d.cls_id)
                     for d in helmet if d.label in VEHICLE_LABELS and min(d.w, d.h) >= self.cfg.min_vehicle_px]
            if extra:
                coco = nms(coco + extra, 0.6, class_aware=True)

        groups, peds = build_groups(coco, helmet, self.cfg.assoc, self.helmet_det.box_level)

        # tracking theo box xe (ổn định hơn union khi số người đổi); không có xe -> union rider
        tboxes = [g.vehicle.box if g.vehicle else g.union_box() for g in groups]
        tscores = [g.vehicle.score if g.vehicle else max(r.score for r in g.riders) for g in groups]
        tracks = self.tracker.update(tboxes, tscores)
        events: list[Event] = []
        for g, t in zip(groups, tracks):
            g.track_id = t.id
            g.track_confirmed = t.hits >= self.cfg.tracker.min_hits
            g.direction = t.direction()
            assign_roles(g)
            ev = self.monitor.observe(g, self.frame_idx, ts, frame if self.cfg.temporal.keep_best_frame else None,
                                      track_age=t.hits)
            if ev is not None:
                self._finalize_event(ev, frame)
                events.append(ev)
        if self.frame_idx % 50 == 0:
            self.monitor.prune(self.tracker.active_ids(), ts)
        t3 = time.perf_counter()

        res = FrameResult(self.frame_idx, ts, coco, helmet, groups, peds, events,
                          {"coco": (t1 - t0) * 1e3, "helmet": (t2 - t1) * 1e3, "assoc_track": (t3 - t2) * 1e3,
                           "total": (t3 - t0) * 1e3})
        self.frame_idx += 1
        self.events += events
        return res

    # ---------------- bằng chứng ----------------

    def _finalize_event(self, ev: Event, cur_frame: np.ndarray) -> None:
        ev.camera = self.cfg.camera
        best_frame, best_group = self.monitor.best_evidence(ev.track_id)
        frame = best_frame if best_frame is not None else cur_frame
        h, w = frame.shape[:2]
        crop_box = clip_box(expand(ev.box, 1.3, w, h, up=0.2), w, h)
        x1, y1, x2, y2 = (int(v) for v in crop_box)
        crop = frame[y1:y2, x1:x2]
        if self._alpr is not None and crop.size:
            try:
                res = self._alpr.predict(crop)
                best = max(res, key=lambda r: (r.ocr.confidence if r.ocr else 0.0), default=None)
                if best is not None and best.ocr and best.ocr.confidence >= self.cfg.alpr.min_conf:
                    ev.plate, ev.plate_score = best.ocr.text, float(best.ocr.confidence)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ALPR lỗi: %s", exc)
        if self.cfg.output.save_evidence:
            stem = f"trk{ev.track_id:05d}_f{ev.frame_idx:07d}"
            vis = draw_frame(frame, [best_group] if best_group else [], banner=f"VI PHAM track {ev.track_id}: "
                             f"{ev.riders_no_helmet}/{ev.riders_total} khong mu" + (f" | {ev.plate}" if ev.plate else ""))
            ev.evidence_path = str(self.out_dir / "events" / f"{stem}.jpg")
            cv2.imwrite(ev.evidence_path, vis, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.output.jpeg_quality])
            if crop.size:
                ev.crop_path = str(self.out_dir / "events" / f"{stem}_crop.jpg")
                cv2.imwrite(ev.crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.output.jpeg_quality])
        logger.info("VI PHẠM track=%d riders=%d no_helmet=%d score=%.2f plate=%s -> %s",
                    ev.track_id, ev.riders_total, ev.riders_no_helmet, ev.score, ev.plate, ev.evidence_path)

    def save_events(self, path: str | Path | None = None) -> str:
        path = Path(path or self.out_dir / "events.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.events], f, ensure_ascii=False, indent=2)
        return str(path)

    def summary(self) -> dict:
        s = self.monitor.summary()
        s.update({"frames": self.frame_idx, "events": len(self.events)})
        return s


def _iou(a: Box, b: Box) -> float:
    from .types import iou
    return iou(a, b)
