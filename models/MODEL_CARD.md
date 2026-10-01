# Model card — `smolry_yolo11s_640.onnx`

| | |
|---|---|
| Tác vụ | phát hiện đầu người đội mũ / không đội mũ bảo hiểm (2 lớp, head-level) trong ảnh giao thông |
| Kiến trúc | YOLO11s (9.4 M tham số, 21.6 GFLOPs @640), export ONNX end-to-end (NMS trong graph), opset 17 |
| Nguồn | https://huggingface.co/Smolry/Helmet-classifer — file `helmet.pt`, tác giả Smolry, 08/2026 |
| Dữ liệu train (theo tác giả) | Roboflow fork "Helmet-and-Non-Helmet-Detection" v2 (traffic/CCTV), 100 epoch, imgsz 640 |
| License | **AGPL-3.0** (trọng số Ultralytics YOLO). File ONNX này là dẫn xuất, phân phối theo AGPL-3.0 — văn bản đầy đủ: `LICENSE-AGPL-3.0.txt` cùng thư mục. Dùng trong sản phẩm đóng cần Ultralytics Enterprise License. |
| Kích thước | 38 MB (ONNX FP32) |

## Hợp đồng I/O (khớp `detect/yolo.py` của AI worker)

- Input `images`: `1×3×640×640`, RGB, float32 chia 255, letterbox (giữ tỉ lệ, pad 0 giữa khung).
- Output: `(1, 100, 6)` = `[x1, y1, x2, y2, conf, cls]` theo toạ độ khung letterbox 640; hàng không dùng có conf 0.
- Nhãn thô (`smolry_yolo11s_640.txt`, id = số dòng): `0 helmet`, `1 no-helmet`.
- Map canonical: `helmet → helmet`, `no-helmet → no_helmet`.
- Ngưỡng đề xuất: 0.30 (kết hợp xác nhận 3/8 lượt theo track); 0.50 nếu chạy không có temporal voting.
- Chi tiết: `smolry_yolo11s_640.json`; parity .pt ↔ ONNX: `smolry_yolo11s_640_parity.json` (90% box trùng IoU ≥ 0.5, IoU trung bình 0.94 trên 30 ảnh Kerala).

## Kết quả trên tập test (bản ONNX, onnxruntime CPU; chi tiết `../reports/model_test_report.md`)

| Tập test | mAP50 | AP50 helmet | AP50 no_helmet | P / R no_helmet @0.30 |
|---|---|---|---|---|
| Kerala CCTV góc cao (394 ảnh, có IR đêm) | 0.475 | 0.634 | 0.317 | 0.73 / 0.29 |
| TP.HCM khung video Pexels (200 ảnh) | 0.739 | 0.754 | 0.724 | 0.76 / 0.79 |
| Kaggle bikes web (764 ảnh) | 0.759 | 0.861 | 0.657 | 0.60 / 0.83 |
| MHDD camera công cộng TP.HCM (100 ảnh, chỉ helmet, đầu ~10 px) | 0.391 | 0.391 | – | – |

Latency: 18.6 ms/ảnh RTX 2050 (torch), 82 ms/ảnh Core i5-12450H (onnxruntime CPU), 45.6 ms ở 1280 (GPU).

## Giới hạn

- Không train trên dữ liệu Việt Nam có nhãn đầy đủ; CCTV góc cao chỉ đạt AP50 no_helmet ≈ 0.32.
- Nhầm mũ bảo hiểm màu đen ở xa/ngược sáng thành không mũ; chưa đo với áo mưa, hood, mũ lưỡi trai nhựa, khẩu trang.
- Đầu < 16 px: recall thấp ở 640 (0.11); dùng input 1280 (0.27) hoặc crop vùng xe.

## `jingjing_yolov8n_plate_640` (dự phòng, chỉ có hợp đồng)

`jingjing_yolov8n_plate_640.txt/.json/_parity.json` mô tả bản ONNX của YOLOv8n từ https://github.com/Jingjing468/motorcycle-helmet-violation-detection (`model/helmet_best.pt`, lớp `bike, helmet, no-helmet, number-plate`). Repo gốc **không có LICENSE** nên file ONNX không được đưa vào đây; tự tạo bằng:

```bash
cd ../pipeline && python scripts/download.py --models --names jingjing_yolov8n_plate && python scripts/export_onnx.py --helmet jingjing_yolov8n_plate --imgsz 640 --opset 17
```
