# Báo cáo model cuối trên các tập test: smolry_yolo11s (chính) và jingjing_yolov8n_plate (dự phòng)

*Ngày 30/09/2026. Đánh giá bằng `pipeline/scripts/eval.py --datasets-file configs/datasets_test.yaml --stage detector` (model mũ chạy full-frame, không qua COCO/crop), ngưỡng AP 0.05, ngưỡng vận hành 0.30 (mặc định pipeline) và 0.50. Máy: RTX 2050 4 GB, Core i5-12450H. Bản `.pt` chạy torch GPU; bản `_onnx` chạy onnxruntime CPU qua backend giống `detect/yolo.py` của AI worker.*

## 1. Tập test

| Tập | Ảnh | Nguồn, license | Đặc điểm | GT helmet / no_helmet |
|---|---|---|---|---|
| kerala_cctv_test | 394 | figshare 32310867 (NIT Calicut, 05/2026), CC BY 4.0 | CCTV góc cao Ấn Độ, 1920×1080 + crop, có khung IR đêm | ~350 / ~270 |
| hcmc_v10_test | 200 | Roboflow the-intruder v10 (HF harijawahar), CC BY 4.0 | khung video Pexels TP.HCM + ảnh stock, kéo giãn 640×640 | 212 / 48 (**chỉ 48 no_helmet: R@P90 có bước 1/48 ≈ 0.02, đọc với sai số lớn**) |
| bikes_voc_all | 764 | Kaggle andrewmvd (HF cute-face), CC0 / CC BY 4.0 | ảnh web ~400 px, xe máy + xe đạp; **nhiều model công khai đã train trên ảnh này** (không phải smolry/jingjing theo tài liệu tác giả, nhưng không loại trừ được) | ~1,050 / ~380 |
| mhdd_hcmc_test | 100 | MHDD SoICT 2024, CC BY 4.0 | camera công cộng TP.HCM 1280×720 ngày+đêm, đầu ~10 px, **chỉ có lớp helmet** | ~290 / 0 |

Cách khớp: IoU ≥ 0.5 cùng lớp; AP50 nội suy toàn điểm; `R@P90` = recall no_helmet lớn nhất khi precision ≥ 0.9; `swaps` = dự đoán trúng vị trí GT nhưng sai trạng thái (helmet ↔ no_helmet).

## 2. smolry_yolo11s — kết quả

### 2.1 Độ chính xác (ngưỡng vận hành 0.30)

| Tập | mAP50 | AP50 helmet | AP50 no_helmet | P / R no_helmet | R@P90 | R đầu <16 px / 16–32 px | swaps |
|---|---|---|---|---|---|---|---|
| kerala_cctv_test | **0.483** | 0.633 | 0.332 | 0.70 / 0.29 | 0.14 | – | 55 |
| hcmc_v10_test | **0.739** | 0.754 | 0.724 | 0.76 / 0.79 | 0.44 | – / 0.33 | 3 |
| bikes_voc_all | 0.759 | 0.851 | 0.667 | 0.63 / 0.84 | 0.06 | 0.70 / 0.90 | 35 |
| mhdd_hcmc_test | 0.382 (helmet) | 0.382 | – | – | – | 0.11 / 0.42 | 4 |

### 2.2 Bản ONNX bàn giao (`models/smolry_yolo11s_640.onnx`, onnxruntime CPU)

| Tập | mAP50 .pt → ONNX | AP50 no_helmet .pt → ONNX | P / R no_helmet ONNX | ms/ảnh CPU |
|---|---|---|---|---|
| kerala_cctv_test | 0.483 → 0.475 | 0.332 → 0.317 | 0.73 / 0.29 | 83 |
| hcmc_v10_test | 0.739 → 0.739 | 0.724 → 0.724 | 0.76 / 0.79 | 95 |
| bikes_voc_all | 0.759 → 0.759 | 0.667 → 0.657 | 0.60 / 0.83 | 91 |
| mhdd_hcmc_test | 0.382 → 0.391 | – | – | 93 |

Chênh lệch ≤ 0.015 mAP50 do letterbox chữ nhật (Ultralytics) so với letterbox vuông (worker). Parity box trên 30 ảnh Kerala: 90% box trùng IoU ≥ 0.5, IoU trung bình 0.94. Latency CPU đo sạch (không tiến trình khác): 82 ms/ảnh (12 FPS).

### 2.3 Ngưỡng vận hành: 0.30 so với 0.50

| Tập | P / R no_helmet @0.30 | P / R no_helmet @0.50 | swaps @0.30 → @0.50 |
|---|---|---|---|
| kerala_cctv_test | 0.70 / 0.29 | 0.75 / 0.21 | 55 → 37 |
| hcmc_v10_test | 0.76 / 0.79 | 0.80 / 0.77 | 3 → 2 |
| bikes_voc_all | 0.63 / 0.84 | 0.65 / 0.80 | 35 → 31 |

Tăng ngưỡng lên 0.50 đổi ~2–8 điểm recall lấy ~2–5 điểm precision; với xác nhận N/M theo track (3/8) phía sau, giữ 0.30 hợp lý hơn vì temporal voting đã lọc nhiễu nhấp nháy. Ngưỡng 0.35 cho `min_score` của bỏ phiếu.

### 2.4 Full-frame 1280 (profile `pipeline_far.yaml`)

| Tập | mAP50 @640 → @1280 | AP50 no_helmet | R@P90 | R đầu <16 / 16–32 px | ms/ảnh GPU |
|---|---|---|---|---|---|
| kerala_cctv_test | 0.483 → 0.453 | 0.332 → 0.269 | 0.14 → 0.20 | – | 39 |
| mhdd_hcmc_test | 0.382 → **0.507** | – | – | 0.11 / 0.42 → **0.27 / 0.62** | 36 |

1280 chỉ dành cho camera xa (đầu < 16 px): tăng recall đầu nhỏ gấp 2.5 lần trên MHDD, nhưng giảm nhẹ trên Kerala (nhiều crop đầu đã to sẵn).

### 2.5 Latency

| Cấu hình | Thiết bị | ms/ảnh | FPS |
|---|---|---|---|
| .pt, 640 | RTX 2050 (torch FP32) | 18.6 | 54 |
| .pt, 1280 | RTX 2050 | 36–46 (tuỳ lần đo) | 22–28 |
| ONNX end-to-end, 640 | i5-12450H onnxruntime CPU | 82 | 12 |
| Cả pipeline (COCO yolo11s + smolry 640 + gán) | RTX 2050 | 40–46 | 22–25 |

## 3. jingjing_yolov8n_plate — kết quả (dự phòng nhẹ)

| Tập | mAP50 (.pt / ONNX) | AP50 helmet | AP50 no_helmet | P / R no_helmet @0.30 | P / R @0.50 | swaps |
|---|---|---|---|---|---|---|
| kerala_cctv_test | **0.486 / 0.496** | 0.553 | **0.419 / 0.441** | 0.60 / 0.47 | 0.69 / 0.38 | 59 |
| hcmc_v10_test | 0.624 / 0.624 | 0.672 | 0.575 | 0.63 / 0.69 | 0.69 / 0.65 | 9 |
| bikes_voc_all | 0.630 / 0.647 | 0.653 | 0.608 | 0.57 / 0.75 | 0.64 / 0.62 | 47 |
| mhdd_hcmc_test | 0.024 / 0.032 | 0.024 | – | – | – | 0 |

Latency: 12 ms GPU; ONNX CPU 37–41 ms (27 FPS). Trên CCTV góc cao (Kerala) jingjing có recall no_helmet cao hơn smolry (0.47 so với 0.29 ở cùng ngưỡng) nhưng precision thấp hơn (0.60 so với 0.70); trên cảnh TP.HCM và đầu nhỏ thì kém rõ rệt (MHDD 0.024 full-frame; cần chế độ `both` để lên 0.18). Bản ONNX của jingjing **không** nằm trong repo vì repo gốc không có LICENSE; tự export bằng `scripts/export_onnx.py --helmet jingjing_yolov8n_plate`.

## 4. Đọc kết quả cho triển khai

- **Cảnh Việt Nam tầm gần/trung (cổng, bãi xe, giao lộ camera thấp)**: smolry đạt P/R no_helmet 0.76/0.79 trên khung TP.HCM; đủ cho cảnh báo có người duyệt.
- **CCTV góc cao / xa**: mọi model công khai đều yếu (AP50 no_helmet ≤ 0.44, R@P90 ≤ 0.20). Không đủ cho phạt nguội tự động; cần fine-tune trên dữ liệu camera thật.
- **Lỗi thấy trên demo** (ảnh sự kiện giữ trong thư mục làm việc, không đưa vào repo vì có biển số xe thật): mũ đen ở xa ngược sáng bị đọc là no_helmet; xe đạp bị COCO nhận thành motorcycle nên người đi xe đạp bị báo; người đứng cạnh xe đỗ (chỉ lọc được bằng `min_track_speed_px` ở camera cố định).
- **Chưa đo được** vì không có dataset: áo mưa/hood, mũ lưỡi trai nhựa, nón lá trên xe máy (demo cho thấy nón lá → no_helmet đúng), khẩu trang che nửa đầu, đêm mưa.

## 5. License và nguồn

- smolry_yolo11s: https://huggingface.co/Smolry/Helmet-classifer (file `helmet.pt`), YOLO11s train trên Roboflow "Helmet-and-Non-Helmet-Detection" v2; **AGPL-3.0** (trọng số Ultralytics). Bản ONNX trong `models/` là dẫn xuất và được phân phối theo AGPL-3.0; sản phẩm đóng cần Ultralytics Enterprise License.
- jingjing_yolov8n_plate: https://github.com/Jingjing468/motorcycle-helmet-violation-detection (`model/helmet_best.pt`), YOLOv8n; repo không có LICENSE → không phân phối lại.
- Dataset test: Kerala (CC BY 4.0), hcmc_v10 (CC BY 4.0), MHDD (CC BY 4.0), Kaggle bikes (CC0 / CC BY 4.0).

## 6. Tái lập

```bash
cd pipeline
python scripts/download.py --models --names smolry_yolo11s jingjing_yolov8n_plate coco_yolo11s
python scripts/prepare_datasets.py --only kerala hcmc_v10 bikes_voc mhdd
python scripts/eval.py --datasets-file configs/datasets_test.yaml --helmet smolry_yolo11s smolry_yolo11s_onnx jingjing_yolov8n_plate --op-conf 0.30
python scripts/eval.py --datasets-file configs/datasets_test.yaml --datasets kerala_cctv_test mhdd_hcmc_test --helmet smolry_yolo11s --imgsz 1280
python scripts/export_onnx.py --helmet smolry_yolo11s --imgsz 640 --opset 17 --check-dir data/kerala/images/test
```
