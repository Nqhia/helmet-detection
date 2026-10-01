"""Chạy pipeline trên ảnh / thư mục ảnh / video / RTSP / webcam.

Ví dụ:
  python scripts/infer.py --source data/demo/clip.mp4 --out outputs/demo --save-video
  python scripts/infer.py --source rtsp://user:pass@ip:554/... --every 3 --show
  python scripts/infer.py --source frames/ --helmet <candidate> --mode crop
Kết quả video/RTSP: <out>/events.json + <out>/events/*.jpg (sự kiện đã xác nhận N/M theo track), <out>/annotated.mp4
(nếu --save-video), <out>/summary.json. Ảnh/thư mục ảnh: <out>/frames.json (kết quả từng ảnh: xe, người, trạng thái mũ)
và ảnh đã vẽ — KHÔNG có sự kiện vì xác nhận theo track cần video liên tục.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2

from _common import ROOT, setup_logging  # noqa: E402

from helmet_pipeline.config import load_config  # noqa: E402
from helmet_pipeline.draw import draw_frame  # noqa: E402
from helmet_pipeline.pipeline import HelmetPipeline  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
log = logging.getLogger("infer")


def iter_source(src: str, every: int, max_frames: int | None, meta: dict | None = None):
    """Sinh (frame, ts, is_video, tên). ts theo giây video (file) hoặc monotonic (stream). meta['fps'] = fps nguồn."""
    p = Path(src)
    meta = meta if meta is not None else {}
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        for i, f in enumerate(files[:max_frames]):
            im = cv2.imread(str(f))
            if im is None:
                log.warning("Bỏ qua, không đọc được ảnh: %s", f)
                continue
            yield im, float(i), False, f.stem
        return
    if p.is_file() and p.suffix.lower() in IMG_EXT:
        im = cv2.imread(str(p))
        if im is None:
            raise SystemExit(f"Không đọc được ảnh: {src}")
        yield im, 0.0, False, p.stem
        return
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    meta["fps"] = fps
    is_stream = not p.is_file()
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            ts = time.monotonic() if is_stream else i / fps
            yield frame, ts, True, f"f{i:07d}"
            n += 1
            if max_frames and n >= max_frames:
                break
        i += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pipeline.yaml"))
    ap.add_argument("--helmet", help="tên candidate trong configs/candidates.yaml (ghi đè helmet_detector)")
    ap.add_argument("--vehicle", help="tên candidate cho vehicle_detector")
    ap.add_argument("--mode", choices=["full", "crop", "both"])
    ap.add_argument("--conf", type=float, help="ngưỡng model mũ")
    ap.add_argument("--imgsz", type=int, help="kích thước input model mũ (1280 cho camera xa/đầu nhỏ)")
    ap.add_argument("--device", help="cuda:0 | cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1, help="xử lý mỗi N khung")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--save-frames", action="store_true", help="lưu từng khung đã vẽ (ảnh/thư mục ảnh)")
    ap.add_argument("--draw-coco", action="store_true")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--alpr", action="store_true", help="bật fast-alpr trên ảnh bằng chứng")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    setup_logging(a.verbose)

    ov: dict = {}
    # ghi đè theo khoá con để GIỮ conf/imgsz của profile (ghi đè cả dict sẽ rơi về mặc định của candidate)
    if a.helmet: ov["helmet_detector.candidate"] = a.helmet
    if a.vehicle: ov["vehicle_detector.candidate"] = a.vehicle
    if a.mode: ov["helmet_mode"] = a.mode
    if a.device: ov["device"] = a.device
    if a.out: ov["output.dir"] = a.out
    if a.no_evidence: ov["output.save_evidence"] = False
    if a.alpr: ov["alpr.enabled"] = True
    cfg = load_config(a.config, ov)
    if a.conf is not None:
        cfg.helmet_detector["conf"] = a.conf
    if a.imgsz:
        cfg.helmet_detector["imgsz"] = a.imgsz
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = HelmetPipeline(cfg, out_dir)

    writer = None
    frames_dir = out_dir / "frames"
    if a.save_frames:
        frames_dir.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    stage_ms: dict[str, float] = {}
    n = 0
    meta: dict = {}
    frame_records: list[dict] = []          # ảnh/thư mục ảnh: kết quả từng ảnh (không có sự kiện theo track)
    for frame, ts, is_video, stem in iter_source(a.source, a.every, a.max_frames, meta):
        if not is_video:
            pipe.reset()                    # ảnh rời không liên tục: không để tracker nối box giữa các ảnh
        res = pipe.process(frame, ts)
        n += 1
        if not is_video:
            frame_records.append({
                "file": stem, "vehicles": len(res.groups), "violating_now": res.n_violating_now,
                "groups": [{"vehicle": ([round(v, 1) for v in g.vehicle.box] if g.vehicle else None),
                            "riders": [{"status": r.status, "score": round(r.score, 3), "role": r.role,
                                        "box": [round(v, 1) for v in r.box]} for r in g.riders]} for g in res.groups],
            })
        for k, v in res.timings_ms.items():
            stage_ms[k] = stage_ms.get(k, 0.0) + v
        # ảnh/thư mục ảnh: luôn lưu bản đã vẽ (rẻ); video/stream: chỉ khi --save-video/--save-frames/--show
        need_draw = a.show or a.save_video or a.save_frames or not is_video
        if need_draw:
            fps_now = n / max(time.perf_counter() - t_start, 1e-6)
            vis = draw_frame(frame, res.groups, res.pedestrians if cfg.output.draw_pedestrians else None,
                             res.coco if (a.draw_coco or cfg.output.draw_coco) else None,
                             banner=f"xe={len(res.groups)} vi pham={res.n_violating_now} su kien={len(pipe.events)}",
                             fps=fps_now)
            if a.save_video and is_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(1.0, float(meta.get("fps", 25.0)) / a.every), (w, h))
                writer.write(vis)
            if a.save_frames or (not is_video and not a.save_video):
                cv2.imwrite(str(frames_dir / f"{stem}.jpg") if a.save_frames else str(out_dir / f"{stem}_annotated.jpg"), vis)
            if a.show:
                cv2.imshow("helmet", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        if n % 100 == 0:
            log.info("%d khung, %.1f FPS, %d sự kiện", n, n / (time.perf_counter() - t_start), len(pipe.events))
    if writer is not None:
        writer.release()
    if a.show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    summary = {
        "source": a.source, "frames": n, "elapsed_s": round(elapsed, 2), "fps": round(n / max(elapsed, 1e-6), 2),
        "avg_ms": {k: round(v / max(n, 1), 2) for k, v in stage_ms.items()},
        "events": len(pipe.events), "helmet_model": pipe.helmet_det.name, "vehicle_model": pipe.vehicle_det.name,
        "mode": cfg.helmet_mode, **pipe.summary(),
    }
    if frame_records:                       # nguồn là ảnh: ghi frames.json thay vì events.json
        summary["frames_json"] = str(out_dir / "frames.json")
        summary["violating_images"] = sum(1 for r in frame_records if r["violating_now"])
        with open(out_dir / "frames.json", "w", encoding="utf-8") as f:
            json.dump(frame_records, f, ensure_ascii=False, indent=2)
    else:
        pipe.save_events()
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("XONG: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
