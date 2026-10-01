# helmet_pipeline — phát hiện người đi xe máy không đội mũ bảo hiểm (model sẵn có, không train)

Pipeline hoàn chỉnh từ ảnh/video/RTSP đến sự kiện vi phạm kèm ảnh bằng chứng, dùng **model pretrained
công khai** (không huấn luyện) và được thiết kế để leader chuyển thẳng vào AI worker theo khuôn
`detect/weapon/` (xem [../integration/INTEGRATION.md](../integration/INTEGRATION.md)). Kết quả benchmark, model được chọn và lý do
nằm ở [../reports/benchmark_report.md](../reports/benchmark_report.md) và [../reports/model_test_report.md](../reports/model_test_report.md).

**Kết quả chính (30/09/2026, 19 model × 4 dataset):** model được chọn là `smolry_yolo11s` (YOLO11s, AGPL-3.0),
điểm tổng hợp 0.575, 18.6 ms/ảnh trên RTX 2050; dự phòng nhẹ `jingjing_yolov8n_plate` (YOLOv8n 6 MB, tốt nhất trên
CCTV góc cao). Ba profile sẵn: `configs/pipeline.yaml` (mặc định, full 640), `configs/pipeline_far.yaml` (camera xa,
full 1280), `configs/pipeline_lite.yaml` (nhẹ, jingjing + both). ONNX bàn giao trong `../models/` (repo) hoặc `weights/onnx/` (khi tự export) đã kiểm chứng
bằng chính backend onnxruntime.

```
frame ──> COCO detector (yolo11s: person / motorcycle / bicycle)
      ──> helmet detector (pretrained; full-frame và/hoặc crop vùng xe+người rồi upscale)
      ──> association: đầu -> người -> xe  (đếm người theo ĐẦU, không tin số box person của COCO)
      ──> tracker IoU theo xe  -> vai trò lái/ngồi sau theo hướng đi
      ──> bỏ phiếu N/M theo track -> sự kiện vi phạm 1 lần/xe + ảnh bằng chứng (+ biển số nếu bật ALPR)
```

## Cài đặt (Windows/Linux, Python 3.10+)

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA; CPU: bỏ --index-url
pip install -r requirements.txt
python scripts/download.py --models        # tải trọng số trong configs/candidates.yaml
python scripts/prepare_datasets.py         # tải + chuẩn hoá 4 dataset đánh giá (~2.6 GB, tuỳ chọn)
```
`setup_env.sh` là script cho Windows + Git Bash (đã dùng để dựng môi trường đo: torch 2.11+cu128, ultralytics 8.4.166); Linux dùng các lệnh trên.

## Chạy

```bash
# ảnh / thư mục ảnh -> ảnh đã vẽ + frames.json (kết quả từng ảnh; sự kiện chỉ có với video)
python scripts/infer.py --source data/demo --out outputs/demo
# video -> annotated.mp4 + events/ (ảnh bằng chứng) + events.json + summary.json
python scripts/infer.py --source clip.mp4 --save-video --out outputs/clip
# RTSP, xử lý mỗi 3 khung, hiện cửa sổ
python scripts/infer.py --source "rtsp://user:pass@ip:554/stream" --every 3 --show
# đổi model mũ / chế độ / ngưỡng
python scripts/infer.py --source clip.mp4 --helmet <candidate> --mode crop --conf 0.35
```
Tuỳ chọn: `--alpr` (cần `pip install fast-alpr`) đọc biển số trên ảnh bằng chứng; `--draw-coco` vẽ thêm box COCO;
`--device cpu`.

Cấu hình mặc định: [configs/pipeline.yaml](configs/pipeline.yaml). Danh mục model: [configs/candidates.yaml](configs/candidates.yaml).
Dataset đánh giá: [configs/datasets.yaml](configs/datasets.yaml).

## Đánh giá & benchmark

```bash
python scripts/eval.py  --helmet modelA modelB --stage detector           # model mũ full-frame, AP50 head-level
python scripts/eval.py  --helmet modelA --stage pipeline --mode crop      # qua COCO -> crop
python scripts/bench.py --cpu                                             # mọi candidate x mọi dataset + latency GPU/CPU
```
Chỉ số: AP50 từng lớp, mAP50, P/R tại ngưỡng vận hành, **recall `no_helmet` tại precision >= 0.9** (chỉ số
quan trọng nhất cho cảnh báo), recall theo cỡ đầu (<16 px, 16–32 px...), số lần nhầm helmet<->no_helmet, ms/ảnh.
Dataset rider-level (box người+xe kèm trạng thái) được khớp theo "tâm đầu nằm trong box" nên vẫn so được với model head-level.

## Export cho AI worker

```bash
python scripts/export_onnx.py --helmet <candidate> --imgsz 640 --opset 17 --check-dir data/<dataset>/images
```
Ra `weights/onnx/<name>_640.onnx` (end-to-end, output `(1, max_det, 6)` như YOLO26 của worker), `.txt` nhãn,
`.json` hợp đồng (map nhãn, ngưỡng) và `_parity.json` (so .pt vs .onnx). Chi tiết: [../integration/INTEGRATION.md](../integration/INTEGRATION.md).

## Cấu trúc

```
helmet_pipeline/
  types.py         Det / Rider / MotoGroup / Event + hình học (iou, containment, expand, nms)
  detectors.py     UltralyticsDetector (.pt) | OnnxDetector (port YoloDetector worker) | OpenVocabDetector (YOLO-World/YOLOE)
  association.py   build_groups(): đầu -> người -> xe; assign_roles() theo hướng đi
  tracking.py      IouTracker (SORT-lite, Hungarian/greedy)
  temporal.py      ViolationMonitor: N/M theo track, giữ khung bằng chứng tốt nhất, cooldown
  pipeline.py      HelmetPipeline.process(frame) -> FrameResult; full/crop/both; lưu bằng chứng; ALPR tuỳ chọn
  evaluation.py    dataset YOLO-format + class map; AP50, R@P90, recall theo cỡ, nhầm trạng thái
  runner.py        dựng detect_fn theo stage, run_eval, measure_latency
  config.py, draw.py
scripts/           infer.py | eval.py | bench.py | download.py | export_onnx.py
configs/           pipeline.yaml | candidates.yaml | datasets.yaml
tests/test_core.py python -m unittest tests.test_core   (12 test, không cần model)
data/demo/         12 ảnh giao thông VN (Wikimedia Commons, CC BY / CC BY-SA) — tác giả và license từng ảnh trong data/demo/SOURCES.json
```

## Giới hạn đã biết

- Không có model nào được train trên dữ liệu Việt Nam; xem ../reports/benchmark_report.md §11 về lỗi đặc thù (áo mưa, mũ lưỡi trai, nón lá).
- COCO detector gộp/bỏ sót người ngồi sau ở cảnh đông — pipeline đếm người theo đầu để giảm ảnh hưởng, nhưng
  rider-role (lái/ngồi sau) chỉ tin được khi có hướng đi từ tracker.
- Bản standalone dùng tracker + N/M riêng; khi vào worker thì dùng tracker/rule của engine (INTEGRATION.md §3).
