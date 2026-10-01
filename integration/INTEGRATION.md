# Hướng dẫn tích hợp `helmet` vào AI worker (dành cho leader)

Tài liệu này mô tả cách đưa pipeline này vào `ecovision-platform-ai-worker` **mà không sửa gì trong repo đó từ phía dự án này**. Toàn bộ mã trong `helmet_pipeline/` được viết để có thể copy từng phần sang `detect/helmet/` theo đúng khuôn của `detect/weapon/`.

## 1. Vị trí trong kiến trúc worker

```
engine (core/engine.py)
  stage 0: ObjectDetector (COCO detector của worker)  ──> ctx.prior["object"] = [person, motorcycle, bicycle, ...]
  stage 1: HelmetConsumer  requires=("object",)  ──> list[Detection(label="no_helmet"|"helmet", ...)]
  engine: tracker theo họ nhãn, rule N/M (detect_confirm_n/m hoặc khoá riêng helmet_confirm_n/m), sinks (MQTT, snapshot)
```

- Helmet là **Consumer** (giống `WeaponConsumer`): nhận box `motorcycle`/`bicycle`/`person` từ `ctx.prior["object"]`, không tự detect lại xe.
- Trả `Detection` theo đúng luật vàng §2 của `detect/base.py`: chỉ frame vào, list `Detection` ra; không publish, không lưu, không giữ state dedup (engine lo tracker + N/M).
- Nhãn sinh ra: `no_helmet` (chính, để cảnh báo) và `helmet` (để hiển thị/thống kê). Hai nhãn này cần được thêm vào `ALLOWED_LABELS` trong `detect/base.py` để hiện trên UI.

## 2. Hợp đồng model ONNX (khớp `detect/yolo.py`)

`scripts/export_onnx.py` xuất model mũ ở dạng **end-to-end đã NMS**, output `(1, max_det, 6) = [x1, y1, x2, y2, conf, cls]` theo toạ độ letterbox — chính là nhánh `_parse_end2end` mà `YoloDetector` của worker đã xử lý cho YOLO26. Kèm theo:

| File | Nội dung |
|---|---|
| `weights/onnx/<name>_<imgsz>.onnx` | model, opset 17 (chạy được trên onnxruntime 1.19.2 của worker) |
| `weights/onnx/<name>_<imgsz>.txt` | nhãn thô của model, 1 dòng/lớp, đúng thứ tự class id (giống file nhãn của object detector) |
| `weights/onnx/<name>_<imgsz>.json` | hợp đồng: input, output, map nhãn thô -> canonical, ngưỡng đề xuất, `box_level` |
| `weights/onnx/<name>_<imgsz>_parity.json` | kết quả so .pt vs .onnx trên ảnh thật (tỉ lệ box khớp IoU>=0.5) |

Copy `.onnx` + `.txt` vào `models/helmet/` của worker. `YoloDetector(model_file, input_size, labels_file=...)` dùng được ngay; phần **map nhãn thô -> canonical** (ví dụ `"With Helmet" -> helmet`, `"Without Helmet" -> no_helmet`) nằm trong `.json`, cần đưa vào `detect/helmet/config.py` (`HELMET_CLASS_MAP`).

Nếu model bàn giao là **rider-level** (box người+xe đã kèm trạng thái, `box_level: rider`), bỏ qua bước gán đầu->xe ở §4 và coi mỗi box là một người.

## 3. Cấu trúc module đề xuất `detect/helmet/`

```
detect/helmet/
  README.md      # mô tả cascade + hợp đồng train<->serve (input size, class map, ngưỡng)
  config.py      # HelmetConfig(BaseSettings): HELMET_MODEL, HELMET_INPUT_SIZE, HELMET_LABELS_FILE,
                 #   HELMET_CLASS_MAP, HELMET_THRESHOLD, HELMET_MODE(full|crop|both), HELMET_CROP_EXPAND,
                 #   HELMET_CROP_UP, HELMET_MAX_CROPS, HELMET_REQUIRE_VEHICLE, HELMET_CONFIRM_N/M
  detector.py    # HelmetDetector: dùng detect.yolo.YoloDetector; crop theo vùng xe+người; map box về khung gốc
  pipeline.py    # HelmetConsumer: name="helmet", requires=("object",), labels=["no_helmet","helmet"],
                 #   cost_class="medium", interval_s=0.5..1.0, max_objects=8, runtime_key="model_helmet_enabled"
```

Mapping mã từ dự án này:

| Trong `helmet_pipeline/` | Sang worker |
|---|---|
| `detectors.OnnxDetector` | thay bằng `detect.yolo.YoloDetector` (cùng thuật toán letterbox/parse) |
| `pipeline.HelmetPipeline._crop_regions` + `_detect_helmet` | `detect/helmet/detector.py` (chế độ full/crop/both, NMS gộp `types.nms`) |
| `association.build_groups` (+ `AssocConfig`) | giữ nguyên, nhận `ctx.prior["object"]` làm `coco`, trả `MotoGroup` |
| `types.Det` | chuyển sang `detect.base.Detection` (bbox theo toạ độ `ctx.image`, engine cộng `ctx.offset`) |
| `tracking.IouTracker`, `temporal.ViolationMonitor` | **KHÔNG** đem sang — engine đã có tracker + rule N/M |
| `draw.py`, `scripts/*` | không cần |

## 4. Trả `Detection` như thế nào

Từ mỗi `MotoGroup` sau `build_groups`:

```python
for g in groups:                       # mỗi xe có người
    for r in g.riders:
        if r.status == "unknown":
            continue                   # model mũ không bắt được đầu -> không kết luận
        box = r.head.box if r.head else r.box
        out.append(Detection(label=r.status, score=r.score, bbox=tuple(box), source="helmet",
                             extra={"vehicle_bbox": g.vehicle.box if g.vehicle else None,
                                    "riders_total": len(g.riders), "riders_no_helmet": g.n_no_helmet,
                                    "role": r.role}))
```

- Đầu người **không gắn được với xe nào** (người đi bộ) bị bỏ khi `HELMET_REQUIRE_VEHICLE=True` (mặc định; tránh phạt người đi bộ/đứng chờ). Đặt `False` cho camera cổng/công trường.
- Mỗi đầu được gán cho **xe phù hợp nhất** (ưu tiên đầu nằm trong box person đã gắn xe; nếu không thì theo tỉ lệ đầu/xe 8–90% bề rộng xe và khoảng cách tới đỉnh xe), không phải xe đầu tiên thoả điều kiện — tránh box xe to gần camera nuốt đầu người ở xe khác. Vùng ngang xe chỉ nới 1.15 lần để không gom người đứng cạnh xe.
- COCO `bicycle` **không** tính vi phạm (`include_bicycle=False`): luật VN không bắt buộc mũ với xe đạp thường; xe đạp điện thường được COCO nhận là motorcycle. Lưu ý COCO vẫn có thể nhận nhầm xe đạp thành motorcycle.
- Camera cố định: chỉ tính lượt vi phạm khi track xe đang di chuyển (`min_track_speed_px` ≈ 1.5 px/khung ở bản standalone; trong worker lấy vận tốc từ tracker của engine) để loại người đứng cạnh xe đỗ.
- Box `no_helmet` nên là **box đầu** (nhỏ) để tracker của engine bám ổn định; `extra.vehicle_bbox` dùng cho snapshot bằng chứng (cắt cả xe + người, xem §6).

## 5. Ngưỡng và xác nhận đa khung

- Ngưỡng model: lấy `suggested_conf` trong `.json` (đã đo ở benchmark theo recall `no_helmet` tại precision >= 0.9).
- Xác nhận N/M: khuyến nghị **3/8** ở nhịp 2–5 Hz (bản standalone dùng `temporal.window=8, min_hits=3`), cùng ý với `WEAPON_CONFIRM_N/M`. Dùng khoá riêng `helmet_confirm_n/m` để UI chỉnh được mà không ảnh hưởng fire/smoke.
- Dedup theo xe: engine track theo họ nhãn `no_helmet`; một xe có 2 người không mũ sẽ ra 2 track — chấp nhận được (đúng luật: mỗi người là một lỗi). Nếu muốn 1 sự kiện/xe, gom theo `extra.vehicle_bbox` ở tầng rule.

## 6. Bằng chứng (theo dự thảo QCVN 05:2026/BCA và Thông tư 73/2024/TT-BCA)

Ảnh bằng chứng cần thấy **xe + người + trạng thái mũ**, ưu tiên kèm biển số: snapshot nên cắt `expand(union(vehicle_bbox, head_bbox), 1.3, up=0.2)` thay vì chỉ box đầu. ALPR của worker (`detect/alpr`) chạy được trên cùng vùng đó. Bản standalone minh hoạ ở `pipeline.HelmetPipeline._finalize_event`.

## 7. Chi phí & lập lịch

Đo trên RTX 2050 (xem `REPORT.md` phần latency) để chọn `cost_class`/`interval_s`. Chế độ `crop` chạy thêm 1 lượt cho tối đa `HELMET_MAX_CROPS` vùng xe mỗi khung; ở CPU onnxruntime nên dùng `full` hoặc giới hạn `max_crops=4` và `interval_s=1.0` như weapon.

## 8. Checklist tích hợp

1. Copy `models/helmet/<name>.onnx`, `<name>.txt`; thêm `HELMET_*` vào `.env.example`.
2. Tạo `detect/helmet/{config,detector,pipeline}.py` theo §3; `register(HelmetConsumer)` trong `detect/registry.py` sau `AlprConsumer`.
3. Thêm `"helmet", "no_helmet"` vào `ALLOWED_LABELS`; thêm khoá `model_helmet_enabled`, `helmet_confirm_n`, `helmet_confirm_m` vào `runtime._SPEC`.
4. Chạy `scripts/export_onnx.py --check-dir <ảnh thật>` và đối chiếu `parity.json` trước khi thay model.
5. Kiểm tra trên 1 camera thật 1 ngày: đếm cảnh báo/giờ, tỉ lệ đúng khi người duyệt xem lại; chỉnh `HELMET_THRESHOLD` và N/M.
