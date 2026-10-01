# Báo cáo: pipeline nhận diện không đội mũ bảo hiểm xe máy bằng model sẵn có

> Bản sao của `REPORT.md` trong dự án gốc. Đường dẫn `outputs/bench/*`, `weights/onnx/*`, `data/*` trong bài là của thư mục làm việc gốc; trong repo này ONNX nằm ở `../models/`, số liệu tập test ở `model_test_report.md`. Ảnh sự kiện demo không đưa vào repo vì chứa biển số xe thật.

*Ngày 30/09/2026. Máy đo: Windows 11, RTX 2050 4 GB, Core i5-12450H; torch 2.11+cu128, ultralytics 8.4.166, onnxruntime 1.30. Ràng buộc của đợt này: không huấn luyện, chỉ dùng trọng số công khai tải được không cần đăng nhập, dự án đứng riêng trong `Documents/Helmet/helmet_pipeline`, không sửa `ecovision-plat`. Toàn bộ số liệu trong báo cáo lấy từ `outputs/bench/*/results.md` và `results.json`.*

## 1. Kết luận và khuyến nghị

- **Model chính: `smolry_yolo11s`** (YOLO11s, 19 MB, lớp `helmet`/`no-helmet` ở mức đầu người). Ổn định nhất trên cả 4 dataset thuộc 4 domain khác nhau, điểm tổng hợp 0.575 (full-frame) và 0.555 (qua pipeline both), cao hơn model thứ hai 0.05 điểm. Latency 18.6 ms/ảnh 640 trên RTX 2050, 82 ms CPU onnxruntime. **License AGPL-3.0** (trọng số Ultralytics): dùng trong sản phẩm đóng cần Ultralytics Enterprise License hoặc tuân thủ AGPL.
- **Phương án nhẹ / dự phòng: `jingjing_yolov8n_plate`** (YOLOv8n, 6 MB, thêm lớp `bike` và `number-plate`). Tốt nhất trên bộ CCTV góc cao Kerala (mAP50 0.496, AP50 no_helmet 0.431, cao nhất trong 19 model), 12 ms GPU và 37 ms CPU onnxruntime. Repo gốc không có file LICENSE nên không phân phối lại trọng số.
- **Bản ONNX bàn giao** đã export end-to-end (output `(1,100,6)`, opset 17) và **đánh giá lại trực tiếp bằng backend onnxruntime giống `detect/yolo.py` của worker**: mAP50 trên Kerala bằng hoặc nhỉnh hơn bản .pt (smolry 0.446 vs 0.435; jingjing 0.481 vs 0.473), tức không mất độ chính xác khi export.
- **Chế độ chạy**: `full` (một lượt toàn khung) là mặc định hợp lý cho camera tầm gần/trung; `both` (full + crop vùng xe) chỉ đáng dùng ở camera cao/xa (đầu < 16 px), nơi nó tăng recall trên đầu nhỏ (jingjing trên MHDD: 0.027 → 0.183) nhưng tốn 3–4 lần thời gian. Chi tiết mục 8.
- **Giới hạn quan trọng**: không model nào được huấn luyện trên dữ liệu Việt Nam có nhãn đầy đủ; trên CCTV góc cao (Kerala) model tốt nhất chỉ đạt AP50 no_helmet ≈ 0.43 và recall no_helmet ≈ 0.40 ở precision 0.6. Muốn đạt mức triển khai phạt nguội cần fine-tune trên dữ liệu camera của mình (mục 11). Pipeline đã sẵn sàng để thay model mà không đổi mã.

## 2. Phạm vi và cách làm

1. Rà soát và tải 19 model mũ bảo hiểm pretrained (16 detector một tầng, 2 two-stage đầu + classifier, 1 open-vocabulary) và 4 vehicle detector; tên lớp đọc trực tiếp từ checkpoint (`research_notes/pipeline_build/pretrained_models.md`).
2. Xây pipeline hoàn chỉnh, độc lập, cấu hình bằng YAML, hỗ trợ 4 backend (Ultralytics .pt, ONNX, open-vocab, two-stage) — mục 3.
3. Tải 4 dataset đánh giá công khai (không cần đăng nhập), chuẩn hoá về YOLO format — mục 5.
4. Benchmark vòng 1 (full-frame, 600 ảnh/bộ lấy mẫu đều) trên tất cả model; vòng 2 trên các model dẫn đầu với toàn bộ ảnh qua pipeline (COCO → crop), thử imgsz 1280, đo latency GPU/CPU; kiểm chứng ONNX.
5. Chọn model, export ONNX kèm hợp đồng cho worker, chạy demo video, viết tài liệu tích hợp (`INTEGRATION.md`).

## 3. Pipeline đã xây

```
frame ──> COCO detector (yolo11s: person / motorcycle / bicycle)
      ──> helmet detector (full-frame, hoặc crop vùng xe+người nới 1.25x + 0.6h lên trên, hoặc cả hai + NMS)
      ──> association: đầu -> người -> xe (đếm người theo ĐẦU; đầu không gần xe nào = người đi bộ, bỏ)
      ──> IouTracker theo box xe -> hướng đi -> vai trò lái / ngồi sau
      ──> ViolationMonitor: >= N lượt có no_helmet trong M lượt gần nhất -> 1 sự kiện / xe, giữ khung tốt nhất
      ──> ảnh bằng chứng (xe + người + trạng thái), crop, events.json (+ biển số nếu bật fast-alpr)
```

| Thành phần | File | Ghi chú |
|---|---|---|
| Kiểu dữ liệu, hình học, NMS | `helmet_pipeline/types.py` | nhãn canonical: helmet, no_helmet, head, rider_*, person, motorcycle, bicycle, plate |
| Backend detector | `helmet_pipeline/detectors.py` | `OnnxDetector` là port của `detect/yolo.py` (letterbox vuông, parse end-to-end/thô) |
| Gán đầu–người–xe, vai trò | `helmet_pipeline/association.py` | xử lý COCO gộp 2 người thành 1 box hoặc bỏ sót người ngồi sau |
| Tracker, bỏ phiếu | `tracking.py`, `temporal.py` | chỉ cho bản standalone; trong worker dùng tracker + rule N/M của engine |
| Điều phối, bằng chứng, ALPR | `helmet_pipeline/pipeline.py` | `HelmetPipeline.process(frame)`; `use_helmet_vehicles` gộp lớp xe của model mũ vào tầng COCO |
| Đánh giá | `helmet_pipeline/evaluation.py` | AP50 từng lớp, R no_helmet @ P≥0.9, recall theo cỡ đầu, nhầm helmet↔no_helmet |
| CLI | `scripts/infer.py, eval.py, bench.py, check_models.py, download.py, prepare_datasets.py, export_onnx.py` | |
| Test | `tests/test_core.py` | 12 test, không cần model |

Thiết kế bám theo `detect/weapon` của worker (Consumer nhận `ctx.prior["object"]`), nên khi tích hợp chỉ cần thay `OnnxDetector` bằng `YoloDetector` và bỏ tracker/temporal (xem `INTEGRATION.md`).

## 4. Model đã thử

| Model (candidate) | Kiến trúc | Lớp thô | Dữ liệu train (theo tác giả) | Kích thước | License |
|---|---|---|---|---|---|
| smolry_yolo11s | YOLO11s | helmet, no-helmet | Roboflow "Helmet-and-Non-Helmet-Detection" v2 (traffic/CCTV) | 19 MB | AGPL-3.0 |
| jingjing_yolov8n_plate | YOLOv8n | bike, helmet, no-helmet, number-plate | Roboflow "Helmet and Number Plate Detection for Motorbike Safety" v4 (~8.5k ảnh) | 6 MB | không có LICENSE |
| rafay_yolo11s | YOLO11s | helmet, no_helmet | Kaggle "unified-dataset" | 19 MB | không ghi |
| sharath_yolov8n | YOLOv8n | With helmet, Without helmet | rider CCTV, 28 epoch | 6 MB | Apache-2.0 |
| justlikethat_yolov8m | YOLOv8m | helmet, nohelmet | Kaggle | 52 MB | không ghi |
| nnsohamnn_yolo11m | YOLO11m | With Helmet, Without Helmet | Roboflow Bike Helmet Detection (1,376 ảnh) | 40 MB | MIT |
| iamtsr_yolov8n | YOLOv8n | With Helmet, Without Helmet | Roboflow ~1,737 ảnh | 6 MB | MIT |
| ganesh_yolov8n | YOLOv8n | With Helmet, Without Helmet | Roboflow Bike-Helmet-Detection v2 | 6 MB | không ghi |
| dtdat_yolo26m | YOLO26m | motorbike, helmet, non-helmet | 2,309/330/659 ảnh đa nguồn + pseudo-label (tác giả VN, 06/2026) | 44 MB | không ghi |
| dtdat_rtdetr_l | RT-DETR-L | như trên | như trên | 66 MB | không ghi |
| vhoc_stage2_yolo26l | YOLO26l | helmet, nohelmet, licenseplate | crop người đi xe, giao thông VN (tác giả VN, 06/2026) | 53 MB | không ghi |
| anubhav_yolov8s_rider_plate | YOLOv8s | with/without helmet, rider, number plate | Kaggle (Ấn Độ) | 23 MB | không ghi |
| lephuocthai_yolo11l | YOLO11l | LP, helmet, no helmet, Triple riding, Using mobile, Wheeling | gộp Kaggle (tác giả VN) | 51 MB | không ghi |
| csay_cambodia_yolov8s | YOLOv8s | helmet, no_helmet | video giao thông Campuchia (ngày/đêm/mưa) | 22 MB | không ghi |
| punmyidol_yolo11m | YOLO11m | With-Helmet, Without Helmet | Roboflow | 40 MB | không ghi |
| bilalgondal_yolov8n_riderlevel | YOLOv8n | driver/passenger with/without helmet, bike | không rõ | 6 MB | không ghi |
| twostage_abel_head_effnet | YOLOv8n head + EfficientNet-B0 | head → helmet/no_helmet | head: không rõ (MIT); classifier: CCTV Ấn Độ (Apache-2.0) | 6 + 16 MB | MIT / Apache-2.0 |
| twostage_hollywood_head_effnet | YOLOv8n head (HollywoodHeads) + EfficientNet-B0 | như trên | CC-BY-NC-4.0 (head) | 6 + 16 MB | **không dùng thương mại** |
| yoloworld_s | YOLO-World v2 S (zero-shot) | prompt "helmet", "human head" | Objects365 + GoldG + CC3M | 26 MB | AGPL-3.0 |

Vehicle detector: `coco_yolo11s` (mặc định), `coco_yolo11n`, `coco_yolo26n` (cùng họ với object detector của worker), `vhoc_stage1_motorcyclist_yolo26l` (box người+xe, VN). Bị loại trước benchmark: các checkpoint định dạng yolov5-repo (không nạp được bằng ultralytics), model AI City Challenge (không tải được / định dạng Co-DETR), Roboflow hosted (cần API key).

## 5. Dataset đánh giá

| Tên | Nguồn, license | Nội dung | Ảnh dùng | Vai trò |
|---|---|---|---|---|
| kerala_cctv | figshare 32310867, NIT Calicut, công bố 05/2026, CC BY 4.0 | CCTV góc cao Ấn Độ 1920×1080 + crop, 928 khung IR đêm, nhãn đầu Helmet/No_Helmet | val+test 1,179 (vòng 1: 600) | **thước đo chính**: mới nhất, khó nằm trong tập train của model nào, có đêm |
| hcmc_v10 | Roboflow the-intruder v10 (HF harijawahar), CC BY 4.0 | khung video Pexels TP.HCM + ảnh stock, kéo giãn 640×640, nhãn Helmet/NOHelmet | 1,518 (vòng 1: 600) | cảnh Việt Nam tầm gần; có thể trùng nguồn với dữ liệu train của vài model |
| bikes_voc | Kaggle andrewmvd (HF mirror), CC0/CC BY 4.0 | 764 ảnh web ~400 px, xe máy + xe đạp, nhãn With/Without Helmet | 764 (vòng 1: 600) | **rò rỉ**: cùng ảnh với Roboflow "Bike Helmet Detection" mà nnsohamnn/ganesh/iamtsr/sharath/rafay đã train — chỉ dùng tham khảo |
| mhdd_hcmc | MHDD SoICT 2024 (GitHub), CC BY 4.0 | camera công cộng TP.HCM 1280×720 ngày+đêm, đầu ~10 px, **chỉ có lớp helmet** | 1,000 (vòng 1: 600) | recall đầu rất nhỏ ở cảnh VN thật; không đo được no_helmet |

Bộ HELMET (Myanmar, OSF) và RideSafe-400 (dashcam Ấn Độ) đã xác minh tải được nhưng chỉ có nhãn mức xe/track, không dùng trong đợt này (xem `eval_datasets.md`).

## 6. Phương pháp đánh giá

- Khớp dự đoán–GT theo IoU ≥ 0.5 cùng lớp; AP50 nội suy toàn điểm (VOC 2010+), mAP50 = trung bình các lớp có GT.
- P/R tại ngưỡng vận hành 0.4; **R no_helmet @ P≥0.9** = recall cao nhất trên đường PR của lớp no_helmet mà precision còn ≥ 0.9 (chỉ số phù hợp nhất cho cảnh báo phạt nguội).
- `swaps` = số dự đoán (≥ 0.4) trúng vị trí một GT nhưng sai trạng thái (helmet ↔ no_helmet).
- Recall theo cỡ đầu (cạnh lớn nhất < 16 px, 16–32 px) để thấy giới hạn với camera xa.
- Vòng 1 lấy mẫu đều 600 ảnh mỗi bộ (không lấy 600 ảnh đầu để tránh lệch camera); vòng 2 dùng toàn bộ.
- Điểm tổng hợp = trung bình(mean mAP50 trên 4 bộ, mean AP50 no_helmet trên 3 bộ có lớp này).

## 7. Vòng 1: model mũ chạy full-frame 640 (19 model × 4 bộ × 600 ảnh)

| # | Model | mean mAP50 | mean AP50 no_helmet | mean R no_helmet@P90 | Điểm | ms/ảnh GPU |
|---|---|---|---|---|---|---|
| 1 | smolry_yolo11s | 0.587 | 0.564 | 0.235 | **0.575** | 21.4* |
| 2 | jingjing_yolov8n_plate | 0.463 | 0.589 | 0.224 | **0.526** | 17.4 |
| 3 | rafay_yolo11s | 0.369 | 0.473 | 0.410 | 0.421 | 18.7 |
| 4 | sharath_yolov8n | 0.384 | 0.438 | 0.333 | 0.411 | 15.2 |
| 5 | justlikethat_yolov8m | 0.346 | 0.400 | 0.126 | 0.373 | 29.2 |
| 6 | nnsohamnn_yolo11m | 0.293 | 0.315 | 0.081 | 0.304 | 26.6 |
| 7 | iamtsr_yolov8n | 0.320 | 0.284 | 0.012 | 0.302 | 15.6 |
| 8 | dtdat_rtdetr_l | 0.321 | 0.266 | 0.042 | 0.293 | 93.1 |
| 9 | ganesh_yolov8n | 0.289 | 0.283 | 0.038 | 0.286 | 15.9 |
| 10 | dtdat_yolo26m | 0.335 | 0.194 | 0.029 | 0.264 | 25.4 |
| 11 | vhoc_stage2_yolo26l | 0.257 | 0.242 | 0.095 | 0.249 | 56.6 |
| 12 | anubhav_yolov8s_rider_plate | 0.244 | 0.225 | 0.003 | 0.234 | 15.7 |
| 13 | lephuocthai_yolo11l | 0.124 | 0.326 | 0.000 | 0.225 | 42.9 |
| 14 | twostage_hollywood_head_effnet | 0.126 | 0.232 | 0.003 | 0.179 | 35.3 |
| 15 | twostage_abel_head_effnet | 0.097 | 0.146 | 0.004 | 0.122 | 35.1 |
| 16 | yoloworld_s | 0.234 | 0.001 | 0.000 | 0.118 | 22.1 |
| 17 | csay_cambodia_yolov8s | 0.134 | 0.074 | 0.001 | 0.104 | 17.0 |
| 18 | punmyidol_yolo11m | 0.068 | 0.021 | 0.001 | 0.045 | 46.9 |
| 19 | bilalgondal_yolov8n_riderlevel | 0.000 | 0.000 | 0.000 | 0.000 | 15.4 |

\* ms/ảnh ở bảng này đo trong lúc benchmark chạy liên tục (có thể lệch); số latency sạch ở §8d.

**Xếp hạng bỏ bộ Kaggle bikes (rò rỉ)** — chỉ Kerala + TP.HCM v10 + MHDD, để kiểm tra thứ hạng không bị ảnh rò rỉ kéo lên:

| # | Model | mean mAP50 (3 bộ) | mean AP50 no_helmet (2 bộ) | mean R no_helmet@P90 | Điểm |
|---|---|---|---|---|---|
| 1 | smolry_yolo11s | 0.528 | 0.508 | 0.322 | **0.518** |
| 2 | jingjing_yolov8n_plate | 0.404 | 0.568 | 0.304 | **0.486** |
| 3 | justlikethat_yolov8m | 0.252 | 0.346 | 0.189 | 0.299 |
| 4 | dtdat_rtdetr_l | 0.308 | 0.276 | 0.062 | 0.292 |
| 5 | dtdat_yolo26m | 0.319 | 0.196 | 0.031 | 0.258 |
| 6 | rafay_yolo11s | 0.208 | 0.297 | 0.253 | 0.253 |
| 7 | anubhav_yolov8s_rider_plate | 0.221 | 0.274 | 0.005 | 0.247 |
| 8 | vhoc_stage2_yolo26l | 0.210 | 0.239 | 0.142 | 0.225 |
| 9 | lephuocthai_yolo11l | 0.111 | 0.327 | 0.000 | 0.219 |
| 10 | sharath_yolov8n | 0.210 | 0.220 | 0.090 | 0.215 |

Hai vị trí đầu không đổi; rafay và sharath rớt từ #3–4 xuống #6 và #10 khi bỏ bộ rò rỉ. Với pipeline both (§8a) cũng vậy: smolry 0.488, jingjing 0.453, justlikethat 0.301.

Theo từng bộ cho các model dẫn đầu (mAP50 / AP50 no_helmet):

| Model | Kerala CCTV | TP.HCM v10 | Kaggle bikes (rò rỉ) | MHDD TP.HCM (helmet) |
|---|---|---|---|---|
| smolry_yolo11s | 0.465 / 0.295 | 0.707 / 0.721 | 0.764 / 0.675 | 0.411 |
| jingjing_yolov8n_plate | **0.496 / 0.431** | 0.690 / 0.705 | 0.639 / 0.631 | 0.027 |
| dtdat_rtdetr_l | 0.438 / 0.298 | 0.267 / 0.254 | 0.362 / 0.245 | 0.218 |
| dtdat_yolo26m | 0.398 / 0.247 | 0.175 / 0.145 | 0.382 / 0.189 | 0.385 |
| rafay_yolo11s | 0.047 / 0.027 | 0.576 / 0.568 | 0.852 / 0.824 | 0.001 |
| sharath_yolov8n | 0.196 / 0.064 | 0.411 / 0.377 | 0.906 / 0.872 | 0.024 |

Nhận xét:
- Chỉ smolry và jingjing giữ được kết quả trên **cả** CCTV góc cao lẫn cảnh Việt Nam tầm gần. Các model lineage "Bike Helmet Detection" (nnsohamnn, ganesh, iamtsr, sharath, rafay) đạt 0.67–0.91 trên Kaggle bikes nhưng sụp trên Kerala (≤ 0.28): điểm Kaggle của nhóm này là rò rỉ tập train, không có giá trị dự báo.
- Hai model tác giả Việt Nam (dtdat, vhoc) hợp lý trên Kerala nhưng kém trên khung TP.HCM 640×640 kéo giãn; vhoc_stage2 được train trên crop nên full-frame là bất lợi (mục 8).
- Two-stage (head detector generic + classifier CCTV Ấn Độ) và zero-shot YOLO-World không dùng được: head detector bỏ sót đầu có mũ, classifier gán no_helmet tràn lan (hàng trăm swaps); YOLO-World gần như không sinh no_helmet.
- bilalgondal (rider-level) không so được với GT mức đầu bằng IoU nên 0; trên ảnh demo nó vẫn ra box rider_helmet hợp lý — chỉ hữu ích nếu có dataset rider-level.
- Kerala vẫn khó với mọi model: AP50 no_helmet tốt nhất 0.43, recall no_helmet ở precision 0.9 gần 0 — không model công khai nào đủ cho phạt nguội tự động trên CCTV góc cao mà không fine-tune.

## 8. Vòng 2: qua pipeline đầy đủ, chế độ crop, imgsz 1280, latency

### 8a. Pipeline `both` (COCO yolo11s → full + crop, toàn bộ 4,461 ảnh)

| # | Model | mean mAP50 | mean AP50 no_helmet | Điểm | Kerala mAP50 / AP no_helmet | TP.HCM v10 | MHDD |
|---|---|---|---|---|---|---|---|
| 1 | smolry_yolo11s | 0.573 | 0.538 | **0.555** | 0.440 / 0.252 | 0.660 | 0.429 |
| 2 | jingjing_yolov8n_plate | 0.472 | 0.520 | **0.496** | 0.453 / 0.339 | 0.620 | 0.183 |
| 3 | rafay_yolo11s | 0.381 | 0.446 | 0.413 | | | |
| 4 | sharath_yolov8n | 0.400 | 0.416 | 0.408 | | | |
| 5 | justlikethat_yolov8m | 0.367 | 0.389 | 0.378 | | | |
| 6 | dtdat_rtdetr_l | 0.335 | 0.270 | 0.302 | | | |
| 7 | vhoc_stage2_yolo26l | 0.321 | 0.259 | 0.290 | 0.439 / 0.222 | 0.254 | 0.135 |
| 8 | dtdat_yolo26m | 0.338 | 0.191 | 0.264 | 0.354 / 0.177 | 0.161 | 0.382 |

- Thứ hạng hai model đầu không đổi so với vòng 1; khoảng cách 0.06 điểm.
- Crop giúp đúng chỗ dự đoán: MHDD (đầu ~10 px) jingjing 0.027 → 0.183, smolry 0.411 → 0.429; vhoc_stage2 (train trên crop) Kerala 0.359 → 0.439. Ngược lại trên hcmc_v10 (ảnh đã là crop 640) both làm smolry giảm 0.707 → 0.660 vì crop thêm sinh false positive.
- Chi phí: cả pipeline both 62–77 ms/ảnh (yolo11s COCO ~20 ms + full + tới 8 crop) so với 17–21 ms full-frame.

### 8b. Chế độ `crop` đơn thuần (COCO → chỉ crop vùng xe, toàn bộ ảnh)

| Model | Điểm | Kerala mAP50 | TP.HCM v10 | MHDD | ms/ảnh |
|---|---|---|---|---|---|
| smolry_yolo11s + crop | 0.502 | 0.396 | 0.635 | 0.261 | 70–99 |
| jingjing_yolov8n_plate + crop | 0.454 | 0.379 | 0.600 | 0.194 | 41–49 |
| vhoc_stage2_yolo26l + crop | 0.285 | 0.425 | 0.236 | 0.149 | 76–211 |
| dtdat_yolo26m + crop | 0.216 | 0.238 | 0.154 | 0.179 | 72–137 |

Crop đơn thuần kém hơn cả `full` lẫn `both` với smolry (Kerala 0.396 so với 0.435 full / 0.440 both): recall bị chặn bởi COCO detector tìm xe (ảnh Kerala nhiều crop đơn xe góc cao mà COCO bỏ sót). Với model nano jingjing thì crop vẫn có ích ở đầu nhỏ (MHDD 0.027 → 0.194). Kết luận: `crop` không nên là mặc định; `both` chỉ cho model nhẹ ở camera xa.

### 8c. Full-frame ở imgsz 1280 (Kerala 1,179 ảnh + MHDD 1,000 ảnh)

| Model | Kerala mAP50 @640 → @1280 | MHDD mAP50 @640 → @1280 | R đầu <16 px / 16–32 px @1280 | ms/ảnh GPU |
|---|---|---|---|---|
| smolry_yolo11s | 0.435 → 0.432 | 0.411 → **0.516** | 0.30 / 0.56 (từ 0.09 / 0.44) | 45.6 |
| jingjing_yolov8n_plate | 0.473 → 0.407 | 0.027 → 0.165 | 0.03 / 0.16 | 26.0 |
| dtdat_yolo26m | 0.398 → 0.244 | 0.385 → 0.347 | 0.17 / 0.23 | 88.3 |

Với smolry, **1280 full-frame là cách rẻ nhất để xử lý đầu nhỏ**: tốt hơn `both` trên MHDD (0.516 so với 0.429) mà chỉ 45 ms (so với 77 ms), và không làm giảm Kerala. Với jingjing và dtdat, 1280 làm giảm kết quả (model học ở 640, đầu bị phóng quá lớn so với phân bố train).

### 8d. Latency (không có tiến trình khác trên GPU/CPU, batch 1, ảnh Kerala thật)

| Model | GPU RTX 2050 torch FP32 | CPU i5-12450H torch (mean / p50) | CPU onnxruntime end-to-end |
|---|---|---|---|
| smolry_yolo11s @640 | 18.6 ms (54 FPS) | 225 / 140 ms | **82 ms (12 FPS)** |
| jingjing_yolov8n_plate @640 | 12.0 ms (83 FPS) | 110 / 57 ms | **37 ms (27 FPS)** |
| sharath_yolov8n @640 | 12.3 ms | 61 / 57 ms | – |
| dtdat_yolo26m @640 | 38.6 ms | 643 / 597 ms | – |
| smolry_yolo11s @1280 | 45.6 ms | – | – |

Cả pipeline (COCO yolo11s + smolry full 640 + association) trên RTX 2050 khoảng 40–46 ms/khung 1080p; chế độ both 62–77 ms. Latency ở 1280 dao động 36–46 ms giữa các lần đo. Số CPU torch dao động mạnh (p50 thấp hơn mean nhiều) do Windows scheduling; với worker dùng onnxruntime CPU, nên lấy số onnxruntime làm chuẩn và giữ `interval_s` 0.5–1.0 như weapon.

### 8e. Kiểm chứng bản ONNX (backend onnxruntime CPU, giống worker) trên Kerala 1,179 ảnh

| Model | mAP50 | AP50 helmet | AP50 no_helmet | P/R no_helmet@0.4 | ms/ảnh CPU* |
|---|---|---|---|---|---|
| smolry_yolo11s (.pt, torch CPU) | 0.435 | 0.608 | 0.263 | 0.69 / 0.23 | 172 |
| **smolry_yolo11s_onnx** | 0.446 | 0.614 | 0.279 | 0.70 / 0.24 | 196 |
| jingjing_yolov8n_plate (.pt, torch CPU) | 0.473 | 0.544 | 0.401 | 0.61 / 0.38 | 62 |
| **jingjing_yolov8n_plate_onnx** | 0.481 | 0.544 | 0.419 | 0.63 / 0.40 | 72 |

\* đo khi GPU đang chạy benchmark song song nên ms/ảnh CPU chỉ mang tính tương đối. Parity box (.pt vs .onnx, 30 ảnh, IoU ≥ 0.5): smolry 90% / IoU 0.94, jingjing 98.5% / IoU 0.97; chênh lệch do Ultralytics dùng letterbox chữ nhật còn worker dùng letterbox vuông — bản ONNX đo trực tiếp không kém bản .pt.

## 9. Model được chọn và cấu hình mặc định

| Profile | File | Model mũ | Chế độ | Dùng khi | Số đo tham chiếu |
|---|---|---|---|---|---|
| **Mặc định** | `configs/pipeline.yaml` | smolry_yolo11s, conf 0.30, imgsz 640 | `full` | camera tầm gần/trung (cổng, bãi xe, giao lộ gần), đầu ≥ 16 px | mAP50 Kerala 0.435 (val+test 1,179 ảnh) / TP.HCM 0.707 (600 ảnh mẫu) / MHDD 0.411 (600 ảnh mẫu); trên split test: 0.483 / 0.739 / 0.382 (xem `model_test_report.md`); 18.6 ms GPU |
| Camera xa | `configs/pipeline_far.yaml` | smolry_yolo11s, imgsz 1280 | `full` | camera cao/xa, đầu < 16 px | MHDD mAP50 0.516 (1,000 ảnh) / 0.507 (split test); 45.6 ms GPU |
| Nhẹ | `configs/pipeline_lite.yaml` | jingjing_yolov8n_plate, imgsz 640 | `both` | CPU / nhiều luồng; cần thêm biển số | Kerala mAP50 0.453 / AP50 no_helmet 0.339 (pipeline both, val+test); 12 ms GPU |

Lý do chọn smolry_yolo11s thay vì jingjing dù jingjing nhỉnh hơn trên Kerala: smolry hơn 0.05–0.08 điểm tổng hợp ở cả ba cách chạy, hơn hẳn trên cảnh Việt Nam tầm gần (TP.HCM 0.707 so với 0.690; MHDD 0.411 so với 0.027 full-frame), và là model duy nhất còn tăng được bằng imgsz 1280. jingjing giữ vai trò dự phòng nhẹ và là lựa chọn nếu camera chủ yếu góc cao và cần lớp biển số từ cùng model.

Tham số khác (đo trên demo và dataset): ngưỡng vận hành 0.30–0.40 (ở 0.4 smolry đạt P 0.66–0.87 cho no_helmet tuỳ bộ); bỏ phiếu 3/8 lượt ở 2–5 Hz; `fullframe_fallback: true`; `use_helmet_vehicles: true` (jingjing/dtdat có lớp xe); `require_vehicle: true` để không báo người đi bộ.

### Demo video (cấu hình mặc định, xử lý mỗi 2 khung, RTX 2050, gồm cả decode + vẽ + ghi video)

| Video | Khung xử lý | FPS đầu-cuối | ms/khung (COCO + mũ + gán) | Nhóm xe theo dõi | Sự kiện vi phạm |
|---|---|---|---|---|---|
| Pexels 3691658, vòng xoay TP.HCM 1080p (Pexels License) | 665 | 13.0 | 46 (24 + 21 + 1) | 220 | 8 |
| Wikimedia "Veturo sur motorciklo tra Hanojo", dashcam xe máy Hà Nội (CC BY-SA 4.0) | 627 | 13.5 | 43 | 128 | 2 |
| Wikimedia "Saigon traffic" Gò Vấp, dashcam giờ cao điểm (CC BY-SA 4.0) | 702 | 13.7 | 43 | 214 | 3 |

Kiểm tra bằng mắt 13 ảnh bằng chứng (`outputs/demo/*/events/`): 9 đúng, 4 sai.
- Đúng (9): người đội nón lá, người đầu trần trên xe máy đang chạy (TP.HCM: 6/8; Gò Vấp: 2/3; Hà Nội: 1/2).
- Sai (4): (a) mũ bảo hiểm **màu đen, ở xa, ngược sáng** bị đọc là no_helmet 0.75 (Gò Vấp) — lỗi của model, cũng là lỗi Thanh Hóa từng gặp với áo mưa; (b) người đi **xe đạp** bị COCO nhận là motorcycle nên vẫn bị báo (TP.HCM); (c) hai người **đứng cạnh xe máy đang đỗ** trên vỉa hè trong video dashcam (Hà Nội) — với camera cố định, đặt `temporal.min_track_speed_px: 1.5` sẽ loại trường hợp này, video dashcam thì không phân biệt được.
- Vòng chạy đầu (trước khi siết association) còn lỗi box xe đạp to gần camera "nuốt" đầu người ở xe khác; đã sửa bằng cách cho mỗi đầu chọn xe phù hợp nhất (ưu tiên có box người, rồi tỉ lệ đầu/xe và khoảng cách tới đỉnh xe), thu hẹp vùng ngang xe (1.4 → 1.15) và loại COCO `bicycle` khỏi vi phạm (luật VN không bắt buộc mũ với xe đạp thường). Test đơn vị: 12/12.

## 10. Giấy phép

- Ultralytics công bố trọng số YOLO (v5/v8/11/26, YOLO-World) theo **AGPL-3.0** và coi model fine-tune từ đó cũng AGPL; sản phẩm đóng chạy qua mạng cần Enterprise License. Việc chọn AGPL hay Enterprise là quyết định ở cấp sản phẩm, áp dụng chung cho mọi model họ YOLO của Ultralytics.
- smolry: AGPL-3.0 (tác giả ghi rõ). jingjing: repo GitHub không có LICENSE, tức mặc định "all rights reserved" của tác giả — nên xin phép hoặc chỉ dùng nội bộ để đánh giá. sharath (Apache-2.0), nnsohamnn/iamtsr (MIT) sạch hơn nhưng yếu trên CCTV.
- Dataset đánh giá: Kerala, hcmc_v10, MHDD CC BY 4.0; Kaggle bikes CC0/CC BY 4.0. Ảnh demo Wikimedia Commons CC BY / CC BY-SA; video Pexels License.
- twostage_hollywood dùng head detector CC-BY-NC-4.0 — không được dùng thương mại (đã loại).

## 11. Hạn chế, rủi ro và việc tiếp theo

1. **Không có dữ liệu Việt Nam có nhãn no_helmet ở góc camera giám sát**: hcmc_v10 là khung video tầm gần kéo giãn, MHDD không có lớp no_helmet. Kết quả trên Kerala (CCTV góc cao, đêm IR) là ước lượng gần nhất cho camera giao lộ và nó chỉ ở mức AP50 no_helmet ≈ 0.43.
2. **Lỗi đặc thù VN chưa đo được**: áo mưa/hood bị đọc là không mũ, mũ lưỡi trai/mũ thời trang bị đọc là mũ, nón lá, khẩu trang. Không dataset công khai nào gắn nhãn các trường hợp này.
3. **Rò rỉ dữ liệu**: nhiều model train trên Roboflow/Kaggle trùng với bộ đánh giá. Điểm tổng hợp chính (§7, §8a) vẫn gồm Kaggle bikes; bảng xếp hạng bỏ Kaggle ở §7 cho cùng hai vị trí đầu, còn rafay/sharath tụt hạng. Không loại trừ hcmc_v10 trùng nguồn với smolry/jingjing; Kerala là bộ ít rủi ro nhất.
4. **COCO detector là trần recall** cho chế độ crop và cho association: xe máy nghiêng/khuất, người ngồi sau bị gộp box. Pipeline đếm theo đầu và có fallback full-frame để giảm ảnh hưởng.
5. **Tracker/N-M của bản standalone** chỉ để demo; trong worker dùng engine.
6. Việc tiếp theo có giá trị nhất: thu 5–20 nghìn đầu người từ camera thật (split theo camera), fine-tune smolry/jingjing với imgsz 1280 hoặc P2 head, đo lại bằng chính `scripts/eval.py`; bổ sung lớp `cap_or_fashion` và mẫu áo mưa/nón lá làm hard negative (xem báo cáo nghiên cứu `reports/Mô hình nhận diện mũ bảo hiểm.md`, mục 10).

## 12. Bàn giao

| Đường dẫn | Nội dung |
|---|---|
| `helmet_pipeline/` | mã pipeline + scripts + tests + configs |
| `weights/onnx/smolry_yolo11s_640.{onnx,txt,json}`, `jingjing_yolov8n_plate_640.{onnx,txt,json}` | model ONNX end-to-end + nhãn + hợp đồng + parity |
| `weights/helmet/*.pt`, `weights/coco/*.pt` | 19 model .pt gốc (không commit; tải lại bằng `scripts/download.py --models`) |
| `data/{kerala,hcmc_v10,bikes_voc,mhdd}` | dataset đánh giá đã chuẩn hoá (tải lại bằng `scripts/prepare_datasets.py`) |
| `data/demo`, `data/demo_video` | 12 ảnh + 3 video giao thông VN có giấy phép mở |
| `outputs/bench/round*` | results.md / results.json / latency.json của từng vòng |
| `INTEGRATION.md` | hướng dẫn tích hợp vào AI worker |
| `../research_notes/pipeline_build/*.md` | ghi chú rà soát model và dataset (URL, kích thước, license đã xác minh) |
| `../reports/Mô hình nhận diện mũ bảo hiểm.md` | báo cáo deep research nền (văn liệu, dataset, họ model, triển khai, VN) |
