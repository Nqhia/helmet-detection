"""Gán đầu -> người -> xe để tạo MotoGroup (nhóm xe + các người ngồi trên xe).

Đầu vào: list Det từ COCO detector (person/motorcycle/bicycle) và từ helmet detector
(head-level: helmet/no_helmet/head; hoặc rider-level: rider_helmet/rider_no_helmet/rider).

Nguyên tắc:
- Người gắn với xe khi phần lớn thân người chồng lên box xe (overlap = inter/area(person)) và tâm
  người nằm trong khoảng ngang của xe (đã nới). COCO hay gộp 2 người ngồi thành 1 box hoặc bỏ sót
  người ngồi -> KHÔNG dựa vào số box person để đếm người; đếm theo ĐẦU.
- Đầu gắn với xe nếu nằm trong "vùng rider" của xe (phía trên box xe) hoặc nằm trong 1 person đã gắn xe.
- Đầu không gần xe nào = người đi bộ -> bỏ (trừ khi require_vehicle=False).
- Rider-level box (đã kèm trạng thái) được coi như person có status; nếu không có xe nào chồng thì
  tự lập nhóm (box rider-level thường đã bao gồm xe).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import (RIDER_LEVEL_LABELS, STATUS_LABELS, VEHICLE_LABELS, Box, Det, MotoGroup, Rider,
                    containment, inter_area, iou, nms)


@dataclass
class AssocConfig:
    person_vehicle_min_overlap: float = 0.15   # inter(person, vehicle)/area(person)
    person_vehicle_x_expand: float = 1.15      # nới ngang box xe khi xét tâm người (1.4 cũ gom cả người đứng cạnh xe)
    head_person_min_containment: float = 0.5   # phần đầu nằm trong box người
    head_person_max_rel_y: float = 0.6         # tâm đầu phải nằm ở 60% trên của box người
    head_vehicle_up: float = 1.3               # vùng đầu trên xe: từ veh.y1 - up*veh.h tới veh.y2
    head_vehicle_x_expand: float = 1.15        # nới ngang box xe khi xét tâm đầu
    head_max_rel_size: float = 0.9             # đầu không được rộng hơn 0.9*bề rộng xe (lọc đầu người khác)
    head_min_rel_size: float = 0.08            # đầu phải rộng >= 8% bề rộng xe (đầu bé hơn = xe khác ở xa phía sau)
    head_rel_size_typical: float = 0.25        # tỉ lệ đầu/xe điển hình (xe máy nhìn từ sau ~0.3, ngang ~0.12) để chấm điểm
    include_bicycle: bool = False              # luật VN: xe đạp thường không bắt buộc mũ; xe đạp điện COCO thường ra "motorcycle"
    require_vehicle: bool = True               # chỉ tính người gắn với xe (bỏ người đi bộ)
    max_riders: int = 4
    min_head_px: int = 6
    head_dedup_iou: float = 0.5                # gộp helmet/no_helmet trùng trên cùng 1 đầu
    vehicle_dedup_iou: float = 0.7             # gộp motorcycle/bicycle trùng


def _x_in(cx: float, box: Box, expand_ratio: float) -> bool:
    bcx = (box[0] + box[2]) / 2
    half = (box[2] - box[0]) * expand_ratio / 2
    return bcx - half <= cx <= bcx + half


def _in_rider_zone(head: Det, veh: Det, cfg: AssocConfig) -> bool:
    if not _x_in(head.cx, veh.box, cfg.head_vehicle_x_expand):
        return False
    top = veh.box[1] - cfg.head_vehicle_up * veh.h
    if not (top <= head.cy <= veh.box[3]):
        return False
    rel = head.w / max(veh.w, 1.0)
    return cfg.head_min_rel_size <= rel <= cfg.head_max_rel_size


def _zone_score(head: Det, veh: Det, cfg: AssocConfig) -> float:
    """Điểm gán đầu->xe khi không có box người: gần đỉnh xe hơn và tỉ lệ đầu/xe hợp lý hơn thì cao hơn (< 1)."""
    dy = max(0.0, (veh.box[1] - head.box[3]) / max(head.h, 1.0))       # số "đầu" phía trên đỉnh xe
    rel = head.w / max(veh.w, 1.0)
    return 1.0 - 0.1 * dy - 0.2 * abs(np.log(max(rel, 1e-3) / cfg.head_rel_size_typical))


def assign_persons_to_vehicles(persons: list[Det], vehicles: list[Det], cfg: AssocConfig) -> dict[int, list[int]]:
    """Trả {vehicle_idx: [person_idx...]}; người không gắn được thì không xuất hiện."""
    out: dict[int, list[int]] = {i: [] for i in range(len(vehicles))}
    for pi, p in enumerate(persons):
        best, best_score = None, 0.0
        for vi, v in enumerate(vehicles):
            ov = inter_area(p.box, v.box) / max(p.area, 1.0)
            if ov < cfg.person_vehicle_min_overlap or not _x_in(p.cx, v.box, cfg.person_vehicle_x_expand):
                continue
            # người ngồi xe: chân/thân ở khoảng box xe; phạt độ lệch tâm ngang
            score = ov - 0.2 * abs(p.cx - v.cx) / max(v.w, 1.0)
            if score > best_score:
                best, best_score = vi, score
        if best is not None:
            out[best].append(pi)
    return out


def _best_person_for_head(head: Det, persons: list[Det], cfg: AssocConfig) -> int | None:
    best, best_key = None, None
    for i, p in enumerate(persons):
        c = containment(head.box, p.box)
        if c < cfg.head_person_min_containment:
            continue
        rel_y = (head.cy - p.box[1]) / max(p.h, 1.0)
        if rel_y > cfg.head_person_max_rel_y:
            continue
        key = (c, -rel_y)
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def _status_from_label(label: str) -> tuple[str, bool]:
    if label in ("helmet", "rider_helmet"):
        return "helmet", True
    if label in ("no_helmet", "rider_no_helmet"):
        return "no_helmet", True
    return "unknown", False


def build_groups(coco: list[Det], helmet: list[Det], cfg: AssocConfig,
                 helmet_box_level: str = "head") -> tuple[list[MotoGroup], list[Det]]:
    """Trả (groups, pedestrians). `pedestrians` = đầu/rider không gắn với xe nào (để vẽ/debug)."""
    vehicles = [d for d in coco if d.label == "motorcycle" or (cfg.include_bicycle and d.label == "bicycle")]
    vehicles = nms(vehicles, cfg.vehicle_dedup_iou, class_aware=False)
    persons = [d for d in coco if d.label == "person"]

    if helmet_box_level == "rider":
        return _build_groups_rider_level(vehicles, helmet, cfg)

    heads = [d for d in helmet if d.label in STATUS_LABELS or d.label == "head"]
    heads = [d for d in heads if min(d.w, d.h) >= cfg.min_head_px]
    heads = nms(heads, cfg.head_dedup_iou, class_aware=False)      # helmet vs no_helmet cùng đầu -> giữ score cao

    pv = assign_persons_to_vehicles(persons, vehicles, cfg)
    v_persons_all = [[persons[i] for i in pv[vi]] for vi in range(len(vehicles))]

    # Mỗi đầu chọn XE PHÙ HỢP NHẤT (không phải xe đầu tiên thoả điều kiện): ưu tiên đầu nằm trong box người
    # đã gắn xe (điểm >= 2), sau đó mới tới vùng phía trên xe (điểm < 1, gần đỉnh xe và đúng tỉ lệ thì cao hơn).
    assigned: dict[int, list[tuple[float, int, int | None]]] = {vi: [] for vi in range(len(vehicles))}
    used_heads: set[int] = set()
    for hi, hd in enumerate(heads):
        best: tuple[float, int, int | None] | None = None
        for vi, v in enumerate(vehicles):
            pidx = _best_person_for_head(hd, v_persons_all[vi], cfg)
            if pidx is not None:
                score = 2.0 + containment(hd.box, v_persons_all[vi][pidx].box)
            elif _in_rider_zone(hd, v, cfg):
                score = _zone_score(hd, v, cfg)
            else:
                continue
            if best is None or score > best[0]:
                best = (score, vi, pidx)
        if best is not None:
            assigned[best[1]].append((best[0], hi, best[2]))
            used_heads.add(hi)

    groups: list[MotoGroup] = []
    for vi, v in enumerate(vehicles):
        v_persons = v_persons_all[vi]
        riders: list[Rider] = []
        used_persons: set[int] = set()
        for _, hi, pidx in sorted(assigned[vi], key=lambda t: -t[0]):
            hd = heads[hi]
            status, _ = _status_from_label(hd.label)
            person = v_persons[pidx] if pidx is not None and pidx not in used_persons else None
            if pidx is not None:
                used_persons.add(pidx)
            riders.append(Rider(status=status, score=hd.score if status != "unknown" else 0.0,
                                person=person, head=hd))
        # người gắn xe nhưng không có đầu nào -> rider trạng thái unknown (model mũ không bắt được)
        for pi, p in enumerate(v_persons):
            if pi not in used_persons:
                riders.append(Rider(status="unknown", score=0.0, person=p, head=None))
        if not riders:
            continue                                   # xe đỗ / không có người
        riders.sort(key=lambda r: (r.head.score if r.head else 0.0), reverse=True)
        riders = riders[: cfg.max_riders]
        groups.append(MotoGroup(vehicle=v, riders=riders))

    pedestrians = [hd for hi, hd in enumerate(heads) if hi not in used_heads]
    if not cfg.require_vehicle:
        # chế độ không cần xe (vd. công trường): mỗi đầu lẻ là 1 nhóm riêng
        for hd in pedestrians:
            status, _ = _status_from_label(hd.label)
            groups.append(MotoGroup(vehicle=None, riders=[Rider(status, hd.score, None, hd)]))
        pedestrians = []
    return groups, pedestrians


def _build_groups_rider_level(vehicles: list[Det], helmet: list[Det], cfg: AssocConfig) -> tuple[list[MotoGroup], list[Det]]:
    riders_d = [d for d in helmet if d.label in RIDER_LEVEL_LABELS]
    riders_d = nms(riders_d, 0.6, class_aware=False)
    groups: list[MotoGroup] = []
    used: set[int] = set()
    for v in vehicles:
        rs: list[Rider] = []
        for i, rd in enumerate(riders_d):
            if i in used:
                continue
            if inter_area(rd.box, v.box) / max(min(rd.area, v.area), 1.0) >= 0.3:
                status, _ = _status_from_label(rd.label)
                rs.append(Rider(status, rd.score if status != "unknown" else 0.0, person=rd, head=None))
                used.add(i)
        if rs:
            groups.append(MotoGroup(vehicle=v, riders=rs[: cfg.max_riders]))
    # rider-level box không chồng xe nào: box thường đã bao cả xe -> tự lập nhóm
    for i, rd in enumerate(riders_d):
        if i in used:
            continue
        status, _ = _status_from_label(rd.label)
        groups.append(MotoGroup(vehicle=None, riders=[Rider(status, rd.score, person=rd, head=None)]))
    return groups, []


def assign_roles(group: MotoGroup) -> None:
    """Gán driver/passenger theo hướng di chuyển (từ tracker). Không có hướng -> giữ 'rider'
    và sắp theo x tăng để chỉ số ổn định."""
    d = group.direction
    if len(group.riders) <= 1:
        group.riders.sort(key=lambda r: r.box[0])
        if len(group.riders) == 1:
            group.riders[0].role = "driver"      # 1 người trên xe = người lái
        return
    if d is None or (abs(d[0]) < 0.5 and abs(d[1]) < 0.5):
        group.riders.sort(key=lambda r: r.box[0])
        for r in group.riders:
            r.role = "rider"
        return
    dx, dy = d
    n = float(np.hypot(dx, dy)) or 1.0
    ux, uy = dx / n, dy / n
    # chiếu tâm rider lên hướng đi: người "phía trước" (chiếu lớn nhất) là lái
    group.riders.sort(key=lambda r: -(((r.box[0] + r.box[2]) / 2) * ux + ((r.box[1] + r.box[3]) / 2) * uy))
    for i, r in enumerate(group.riders):
        r.role = "driver" if i == 0 else "passenger"
