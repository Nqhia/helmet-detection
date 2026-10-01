"""Test đơn vị không cần model: hình học, association, tracker, bỏ phiếu.  python -m unittest tests.test_core"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helmet_pipeline.association import AssocConfig, assign_roles, build_groups  # noqa: E402
from helmet_pipeline.temporal import TemporalConfig, ViolationMonitor  # noqa: E402
from helmet_pipeline.tracking import IouTracker  # noqa: E402
from helmet_pipeline.types import Det, MotoGroup, Rider, containment, iou, nms  # noqa: E402


def D(label, box, score=0.9):
    return Det(label, score, tuple(float(v) for v in box))


class TestGeometry(unittest.TestCase):
    def test_iou_containment(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertAlmostEqual(containment((2, 2, 4, 4), (0, 0, 10, 10)), 1.0)

    def test_nms_class_agnostic(self):
        dets = [D("helmet", (0, 0, 10, 10), 0.9), D("no_helmet", (1, 1, 10, 10), 0.8)]
        self.assertEqual(len(nms(dets, 0.5, class_aware=False)), 1)
        self.assertEqual(len(nms(dets, 0.5, class_aware=True)), 2)


class TestAssociation(unittest.TestCase):
    def setUp(self):
        self.cfg = AssocConfig()

    def test_two_riders_one_person_box(self):
        # xe 100x120 tại (100,200); COCO gộp 2 người thành 1 box; 2 đầu ở trên xe
        coco = [D("motorcycle", (100, 200, 200, 320)), D("person", (110, 100, 190, 300))]
        helmet = [D("helmet", (120, 100, 145, 125)), D("no_helmet", (160, 105, 185, 130), 0.7)]
        groups, peds = build_groups(coco, helmet, self.cfg, "head")
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].riders), 2)
        self.assertEqual(groups[0].n_no_helmet, 1)
        self.assertEqual(peds, [])

    def test_head_above_vehicle_without_person(self):
        coco = [D("motorcycle", (100, 200, 200, 320))]
        helmet = [D("no_helmet", (135, 120, 165, 150))]        # COCO không thấy người
        groups, _ = build_groups(coco, helmet, self.cfg, "head")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].riders[0].status, "no_helmet")
        self.assertIsNone(groups[0].riders[0].person)

    def test_pedestrian_ignored(self):
        coco = [D("motorcycle", (100, 200, 200, 320)), D("person", (500, 100, 560, 320))]
        helmet = [D("no_helmet", (515, 100, 545, 130))]
        groups, peds = build_groups(coco, helmet, self.cfg, "head")
        self.assertEqual(groups, [])                             # xe không người -> bỏ
        self.assertEqual(len(peds), 1)

    def test_head_prefers_own_vehicle_over_big_nearby_box(self):
        # xe A rất to gần camera (không có người), xe B nhỏ hơn có box người chứa đầu -> đầu phải về B
        coco = [D("motorcycle", (0, 300, 450, 900)), D("motorcycle", (380, 450, 520, 620)), D("person", (400, 330, 500, 600))]
        helmet = [D("helmet", (430, 340, 470, 380))]
        groups, _ = build_groups(coco, helmet, self.cfg, "head")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].vehicle.box[0], 380.0)

    def test_bicycle_excluded_by_default(self):
        coco = [D("bicycle", (100, 200, 200, 320))]
        helmet = [D("no_helmet", (135, 120, 165, 150))]
        groups, peds = build_groups(coco, helmet, self.cfg, "head")
        self.assertEqual(groups, [])
        self.assertEqual(len(peds), 1)

    def test_parked_vehicle_skipped(self):
        groups, _ = build_groups([D("motorcycle", (0, 0, 50, 60))], [], self.cfg, "head")
        self.assertEqual(groups, [])

    def test_rider_level(self):
        coco = [D("motorcycle", (100, 200, 200, 320))]
        helmet = [D("rider_no_helmet", (105, 90, 195, 320))]
        groups, _ = build_groups(coco, helmet, self.cfg, "rider")
        self.assertEqual(groups[0].n_no_helmet, 1)
        groups, _ = build_groups([], helmet, self.cfg, "rider")   # không có xe -> tự lập nhóm
        self.assertEqual(len(groups), 1)

    def test_roles_by_direction(self):
        g = MotoGroup(D("motorcycle", (0, 0, 100, 50)),
                      [Rider("helmet", 0.9, None, D("helmet", (10, 0, 20, 10))),
                       Rider("no_helmet", 0.9, None, D("no_helmet", (60, 0, 70, 10)))])
        g.direction = (5.0, 0.0)          # đi sang phải -> người bên phải là lái
        assign_roles(g)
        self.assertEqual([r.role for r in g.riders], ["driver", "passenger"])
        self.assertEqual(g.riders[0].status, "no_helmet")


class TestTrackingTemporal(unittest.TestCase):
    def test_tracker_keeps_id(self):
        tr = IouTracker(iou_thr=0.3, max_misses=3)
        ids = []
        for i in range(5):
            t = tr.update([(100 + i * 5, 100, 200 + i * 5, 200)])
            ids.append(t[0].id)
        self.assertEqual(len(set(ids)), 1)
        self.assertIsNotNone(tr.tracks[ids[0]].direction())

    def test_monitor_n_of_m(self):
        mon = ViolationMonitor(TemporalConfig(window=5, min_hits=3, min_score=0.3, min_track_age=1))
        g = MotoGroup(D("motorcycle", (0, 0, 100, 100)), [Rider("no_helmet", 0.8, None, D("no_helmet", (10, 0, 30, 20)))])
        g.track_id = 7
        evs = [mon.observe(g, i, float(i), None, track_age=i + 1) for i in range(4)]
        self.assertTrue(evs[0] is None and evs[1] is None)
        self.assertIsNotNone(evs[2])                              # lượt thứ 3 -> đủ N=3
        self.assertIsNone(evs[3])                                 # cooldown, không phát lại
        self.assertEqual(evs[2].riders_no_helmet, 1)


if __name__ == "__main__":
    unittest.main()
