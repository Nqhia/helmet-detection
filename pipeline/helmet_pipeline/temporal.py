"""Bỏ phiếu theo track (N trong M lượt) và phát sự kiện vi phạm 1 lần/track kèm khung bằng chứng tốt nhất.

Lý do: model mũ trên 1 khung hình nhấp nháy (đầu nhỏ, mờ, che). Chỉ báo vi phạm khi một xe (track)
bị bắt "có người không mũ" đủ N lượt trong M lượt quan sát gần nhất — cùng cơ chế detect_confirm_n/m
của AI worker nhưng gom theo xe thay vì theo nhãn toàn khung.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .types import Event, MotoGroup


@dataclass
class TemporalConfig:
    window: int = 8            # M lượt quan sát gần nhất của track
    min_hits: int = 3          # N lượt có no_helmet (score >= min_score)
    min_score: float = 0.35    # ngưỡng score no_helmet để tính là 1 lượt "có"
    min_track_age: int = 2     # bỏ track quá non (nhiễu 1 khung)
    cooldown_s: float = 0.0    # 0 = mỗi track chỉ phát 1 sự kiện; > 0 = cho phép phát lại sau chừng đó giây (track sống lâu)
    keep_best_frame: bool = True
    min_track_speed_px: float = 0.0   # > 0: chỉ tính lượt "có no_helmet" khi xe đang di chuyển >= ngưỡng (px/frame)
                                      # — cho camera CỐ ĐỊNH, loại người đứng cạnh xe đỗ. Dashcam/camera động: để 0.


@dataclass
class TrackState:
    obs: deque = field(default_factory=deque)          # (frame_idx, ts, n_riders, n_no_helmet, max_score)
    emitted_at: float | None = None
    best_score: float = 0.0
    best_frame: np.ndarray | None = None
    best_group: MotoGroup | None = None
    best_frame_idx: int = -1
    best_ts: float = 0.0
    riders_hist: Counter = field(default_factory=Counter)
    noh_hist: Counter = field(default_factory=Counter)
    first_seen: float = 0.0
    last_seen: float = 0.0


class ViolationMonitor:
    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self.states: dict[int, TrackState] = {}
        self.n_events = 0

    def observe(self, group: MotoGroup, frame_idx: int, ts: float, frame: np.ndarray | None,
                track_age: int) -> Event | None:
        tid = group.track_id
        if tid is None:
            return None
        st = self.states.get(tid)
        if st is None:
            st = self.states[tid] = TrackState(first_seen=ts)
            st.obs = deque(maxlen=self.cfg.window)
        st.last_seen = ts
        score = group.max_no_helmet_score
        moving = True
        if self.cfg.min_track_speed_px > 0:
            d = group.direction
            moving = d is not None and float(np.hypot(*d)) >= self.cfg.min_track_speed_px
        n_noh = group.n_no_helmet if (score >= self.cfg.min_score and moving) else 0
        st.obs.append((frame_idx, ts, len(group.riders), n_noh, score))
        st.riders_hist[len(group.riders)] += 1
        if n_noh:
            st.noh_hist[n_noh] += 1
        # chỉ giữ khung khi thật sự có lượt no_helmet đủ ngưỡng (tránh giữ 1 khung 1080p cho mọi track có score 0.05)
        if n_noh and score > st.best_score and self.cfg.keep_best_frame and frame is not None:
            st.best_score = score
            st.best_frame = frame.copy()
            st.best_group = group
            st.best_frame_idx = frame_idx
            st.best_ts = ts

        if track_age < self.cfg.min_track_age:
            return None
        hits = sum(1 for o in st.obs if o[3] > 0)
        if hits < self.cfg.min_hits:
            return None
        if st.emitted_at is not None and (self.cfg.cooldown_s <= 0 or (ts - st.emitted_at) < self.cfg.cooldown_s):
            return None                                   # 1 sự kiện / track (hoặc chưa hết cooldown)
        st.emitted_at = ts
        self.n_events += 1
        riders_total = st.riders_hist.most_common(1)[0][0] if st.riders_hist else len(group.riders)
        riders_noh = st.noh_hist.most_common(1)[0][0] if st.noh_hist else group.n_no_helmet
        return Event(
            track_id=tid, frame_idx=st.best_frame_idx if st.best_frame is not None else frame_idx,
            timestamp=st.best_ts if st.best_frame is not None else ts,
            riders_total=max(riders_total, riders_noh), riders_no_helmet=riders_noh,
            score=st.best_score if st.best_frame is not None else score,
            box=(st.best_group or group).union_box(),
            extra={"hits": hits, "window": len(st.obs), "first_seen": st.first_seen},
        )

    def best_evidence(self, track_id: int) -> tuple[np.ndarray | None, MotoGroup | None]:
        st = self.states.get(track_id)
        if st is None:
            return None, None
        return st.best_frame, st.best_group

    def prune(self, active_ids: set[int], now: float | None = None, ttl_s: float = 120.0, max_frames_kept: int = 64) -> None:
        """Bỏ state của track đã mất quá lâu; giải phóng khung bằng chứng của track không còn active;
        trần số khung đang giữ (RTSP 24/7 không được phình RAM)."""
        now = now if now is not None else time.monotonic()
        for tid in list(self.states):
            st = self.states[tid]
            if tid not in active_ids:
                if (now - st.last_seen) > ttl_s:
                    del self.states[tid]
                elif st.best_frame is not None and st.emitted_at is not None:
                    st.best_frame = None                  # đã phát sự kiện, không cần giữ khung nữa
        holders = [st for st in self.states.values() if st.best_frame is not None]
        if len(holders) > max_frames_kept:
            for st in sorted(holders, key=lambda s: s.last_seen)[: len(holders) - max_frames_kept]:
                st.best_frame = None

    def summary(self) -> dict:
        tracks = len(self.states)
        return {"tracks_seen": tracks, "events": self.n_events}
