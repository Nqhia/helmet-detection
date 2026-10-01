# helmet-detection

Phát hiện người đi xe máy **không đội mũ bảo hiểm** từ camera, dùng model pretrained công khai (không huấn luyện), kèm benchmark 19 model trên 4 dataset, pipeline hoàn chỉnh (COCO → model mũ → gán đầu–người–xe → tracker → bỏ phiếu theo track → sự kiện + ảnh bằng chứng), ONNX bàn giao và hướng dẫn tích hợp vào AI worker.

| Thư mục | Nội dung |
|---|---|
| [pipeline/](pipeline/) | package `helmet_pipeline`, scripts (infer / eval / bench / download / prepare_datasets / export_onnx), configs (3 profile), tests, 12 ảnh demo |
| [models/](models/) | `smolry_yolo11s_640.onnx` + nhãn + hợp đồng I/O + parity; [MODEL_CARD.md](models/MODEL_CARD.md) |
| [reports/](reports/) | [benchmark_report.md](reports/benchmark_report.md) (19 model × 4 dataset, 2 vòng, latency, license), [model_test_report.md](reports/model_test_report.md) (model cuối trên các tập test) |
| [integration/](integration/) | [INTEGRATION.md](integration/INTEGRATION.md): đưa vào AI worker theo khuôn Consumer `requires=("object",)` |

## Kết quả chính (30/09/2026)

- Model chính **smolry_yolo11s** (YOLO11s, AGPL-3.0): điểm tổng hợp 0.575 trên 4 dataset và 0.518 khi bỏ bộ Kaggle có rò rỉ (đứng đầu 19 model ở cả hai cách tính), tập test TP.HCM mAP50 0.739 (P/R no_helmet 0.76/0.79), Kerala CCTV góc cao 0.483; 18.6 ms/ảnh RTX 2050, 82 ms CPU onnxruntime.
- Dự phòng nhẹ **jingjing_yolov8n_plate** (YOLOv8n 6 MB): Kerala 0.486 với AP50 no_helmet 0.419 cao nhất; 12 ms GPU / 37 ms CPU. Không phân phối ONNX vì repo gốc không có LICENSE.
- Với camera xa (đầu < 16 px): chạy smolry ở 1280 (`pipeline/configs/pipeline_far.yaml`) tăng recall đầu nhỏ 2.5 lần.
- Không model công khai nào đủ cho phạt nguội tự động trên CCTV góc cao (AP50 no_helmet ≤ 0.44); cần fine-tune trên dữ liệu camera thật — pipeline đã sẵn để thay model.

## Chạy nhanh

```bash
cd pipeline
python -m venv .venv && .venv/Scripts/activate        # Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU; CPU: bỏ --index-url
pip install -r requirements.txt
python scripts/download.py --models --names coco_yolo11s smolry_yolo11s
python scripts/infer.py --source data/demo --out outputs/demo                     # ảnh
python scripts/infer.py --source clip.mp4 --save-video --out outputs/clip         # video / RTSP
python scripts/infer.py --source data/demo --helmet smolry_yolo11s_onnx           # chạy bản ONNX trong ../models
```

## Giấy phép và nguồn

- Mã nguồn trong repo: chưa gắn license (mặc định giữ bản quyền). Xem `reports/benchmark_report.md` §10 cho từng model.
- `models/smolry_yolo11s_640.onnx`: dẫn xuất từ trọng số AGPL-3.0 (Ultralytics YOLO11, tác giả Smolry) — phân phối theo AGPL-3.0, văn bản license tại `models/LICENSE-AGPL-3.0.txt`.
- Ảnh demo `pipeline/data/demo/`: Wikimedia Commons, CC BY / CC BY-SA (nguồn từng ảnh trong `SOURCES.json`). Ảnh sự kiện từ video demo không đưa vào repo (có biển số xe thật).
- Dataset đánh giá không nằm trong repo; `scripts/prepare_datasets.py` tải lại (Kerala figshare CC BY 4.0, Roboflow v10 CC BY 4.0, Kaggle bikes CC0, MHDD CC BY 4.0).
