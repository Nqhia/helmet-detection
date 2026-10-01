"""Backend detector. Mọi backend trả list[Det] với nhãn CANONICAL đã map, toạ độ ảnh gốc.

- UltralyticsDetector : .pt (YOLOv8/v9/v10/11/12/26, RT-DETR của ultralytics) qua package ultralytics.
- OnnxDetector        : .onnx end-to-end (1,max_det,6) hoặc thô (1,4+nc,N) — port từ YoloDetector của
                         AI worker để kiểm tra parity trước khi bàn giao.
- OpenVocabDetector   : YOLO-World / YOLOE zero-shot với text prompt; suy ra no_helmet = head không
                         chồng helmet.
Cấu hình mỗi detector là dict (xem configs/candidates.yaml):
  backend, weights, imgsz, conf, iou, device, half, classes {raw->canonical|ignore}, box_level,
  keep_unmapped, prompts (openvocab), derive_no_helmet (openvocab).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Sequence

import cv2
import numpy as np

from .types import CANONICAL, Det, iou

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum())


_CANON_BY_NORM = {_norm(c): c for c in CANONICAL}


class ClassMapper:
    """Map nhãn thô của model -> nhãn canonical. So khớp không phân biệt hoa/thường, khoảng trắng, gạch."""

    def __init__(self, mapping: dict[str, str] | None, keep_unmapped: bool = False):
        self._map = {_norm(k): v for k, v in (mapping or {}).items()}
        self.keep_unmapped = keep_unmapped

    def __call__(self, raw: str) -> str | None:
        v = self._map.get(_norm(raw))
        if v is None:
            if self.keep_unmapped:
                return _CANON_BY_NORM.get(_norm(raw))
            return None
        if v == "ignore":
            return None
        if v not in CANONICAL:
            raise ValueError(f"Nhãn canonical không hợp lệ: {v!r} (từ {raw!r})")
        return v


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Ảnh xám (H,W) / BGRA (H,W,4) -> BGR 3 kênh; giữ nguyên nếu đã đúng."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image


class BaseDetector:
    name: str = "base"
    box_level: str = "head"          # "head" | "rider" | "coco"
    labels: list[str] = []
    last_ms: float = 0.0

    def detect(self, image: np.ndarray) -> list[Det]:
        raise NotImplementedError

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Det]]:
        return [self.detect(im) for im in images]

    def warmup(self, size: int = 640) -> None:
        self.detect(np.zeros((size, size, 3), dtype=np.uint8))


class UltralyticsDetector(BaseDetector):
    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        self.cfg = cfg
        self.name = cfg.get("name") or os.path.splitext(os.path.basename(cfg["weights"]))[0]
        self.box_level = cfg.get("box_level", "head")
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.25))
        self.iou_thr = float(cfg.get("iou", 0.5))
        self.device = cfg.get("device", None)
        self.half = bool(cfg.get("half", False))
        self.max_det = int(cfg.get("max_det", 300))
        if str(cfg.get("arch", "")).lower() == "rtdetr":
            from ultralytics import RTDETR
            self.model = RTDETR(cfg["weights"])          # RT-DETR cần predictor riêng (resize vuông, không letterbox)
        else:
            self.model = YOLO(cfg["weights"], task=cfg.get("task", "detect"))
        names = self.model.names
        self.raw_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        self.mapper = ClassMapper(cfg.get("classes"), cfg.get("keep_unmapped", False))
        self.labels = sorted({m for m in (self.mapper(n) for n in self.raw_names) if m})
        unmapped = [n for n in self.raw_names if self.mapper(n) is None]
        if len(unmapped) <= 12:
            logger.info("[%s] raw=%s -> canonical=%s (bỏ: %s)", self.name, self.raw_names, self.labels, unmapped)
        else:
            logger.info("[%s] %d raw classes -> canonical=%s", self.name, len(self.raw_names), self.labels)

    def _to_dets(self, res) -> list[Det]:
        out: list[Det] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for b, s, c in zip(xyxy, conf, cls):
            raw = self.raw_names[c] if c < len(self.raw_names) else str(c)
            lab = self.mapper(raw)
            if lab is None:
                continue
            out.append(Det(lab, float(s), tuple(float(v) for v in b), raw, self.name, int(c)))
        return out

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Det]]:
        if not images:
            return []
        t0 = time.perf_counter()
        kw = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou_thr, max_det=self.max_det, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        if self.half:
            kw["half"] = True
        results = self.model.predict([ensure_bgr(im) for im in images], **kw)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return [self._to_dets(r) for r in results]

    def detect(self, image: np.ndarray) -> list[Det]:
        return self.detect_batch([image])[0]


class OpenVocabDetector(UltralyticsDetector):
    """YOLO-World / YOLOE với prompt. cfg['prompts'] = {canonical: [text, ...]}.
    Nếu derive_no_helmet=True: box 'head' không chồng (IoU<0.3) với box 'helmet' -> no_helmet;
    box 'head' chồng helmet bị bỏ (helmet giữ)."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        self.cfg = cfg
        self.name = cfg.get("name") or os.path.splitext(os.path.basename(cfg["weights"]))[0]
        self.box_level = cfg.get("box_level", "head")
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.1))
        self.iou_thr = float(cfg.get("iou", 0.5))
        self.device = cfg.get("device", None)
        self.half = bool(cfg.get("half", False))
        self.max_det = int(cfg.get("max_det", 300))
        self.derive = bool(cfg.get("derive_no_helmet", True))
        self.model = YOLO(cfg["weights"])
        prompts: dict[str, list[str]] = cfg["prompts"]
        self.raw_names, mapping = [], {}
        for canon, texts in prompts.items():
            for t in texts:
                self.raw_names.append(t)
                mapping[t] = canon
        # YOLOE cần set_classes(names, embeddings); YOLO-World chỉ cần names.
        try:
            self.model.set_classes(self.raw_names)
        except TypeError:
            self.model.set_classes(self.raw_names, self.model.get_text_pe(self.raw_names))
        self.mapper = ClassMapper(mapping)
        self.labels = sorted(set(mapping.values()) | ({"no_helmet"} if self.derive and "head" in mapping.values() else set()))
        logger.info("[%s] prompts=%s -> %s", self.name, self.raw_names, self.labels)

    def _to_dets(self, res) -> list[Det]:
        dets = super()._to_dets(res)
        if not self.derive:
            return dets
        helmets = [d for d in dets if d.label == "helmet"]
        out = [d for d in dets if d.label != "head"]
        for d in dets:
            if d.label != "head":
                continue
            if any(iou(d.box, h.box) >= 0.3 for h in helmets):
                continue
            out.append(Det("no_helmet", d.score, d.box, d.raw_label, d.source, d.cls_id))
        return out


class OnnxDetector(BaseDetector):
    """Port của detect/yolo.py (AI worker): letterbox -> ONNX -> parse end2end (1,max_det,6) hoặc thô."""

    def __init__(self, cfg: dict):
        import onnxruntime as ort
        self.cfg = cfg
        self.name = cfg.get("name") or os.path.splitext(os.path.basename(cfg["weights"]))[0]
        self.box_level = cfg.get("box_level", "head")
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.25))
        self.iou_thr = float(cfg.get("iou", 0.5))
        want_gpu = cfg.get("device") not in (None, "cpu")
        providers = cfg.get("providers") or (["CUDAExecutionProvider", "CPUExecutionProvider"] if want_gpu
                                             else ["CPUExecutionProvider"])
        avail = ort.get_available_providers()
        providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        if cfg.get("threads"):
            so.intra_op_num_threads = int(cfg["threads"])
        self.session = ort.InferenceSession(cfg["weights"], sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        labels_file = cfg.get("labels_file")
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, encoding="utf-8") as f:
                self.raw_names = [ln.strip() for ln in f if ln.strip()]
        else:
            self.raw_names = list(cfg.get("raw_names") or [])
        self.mapper = ClassMapper(cfg.get("classes"), cfg.get("keep_unmapped", False))
        self.labels = sorted({m for m in (self.mapper(n) for n in self.raw_names) if m})
        logger.info("[%s] onnx providers=%s raw=%s -> %s", self.name, providers, self.raw_names, self.labels)

    def _letterbox(self, image):
        h, w = image.shape[:2]
        gain = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(w * gain), int(h * gain)
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        left, top = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
        return canvas, gain, left, top

    def _mk(self, cid, score, x1, y1, x2, y2, w, h) -> Det | None:
        raw = self.raw_names[cid] if cid < len(self.raw_names) else str(cid)
        lab = self.mapper(raw)
        if lab is None:
            return None
        box = (float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)), float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)))
        return Det(lab, float(score), box, raw, self.name, int(cid))

    def detect(self, image: np.ndarray) -> list[Det]:
        t0 = time.perf_counter()
        image = ensure_bgr(image)
        h, w = image.shape[:2]
        canvas, gain, px, py = self._letterbox(image)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]
        out = self.session.run(None, {self.input_name: blob})[0]
        dets: list[Det] = []
        if out.ndim == 3 and out.shape[2] == 6 and out.shape[1] > out.shape[2]:
            for r in out[0]:
                if r[4] < self.conf:
                    continue
                d = self._mk(int(r[5]), r[4], (r[0] - px) / gain, (r[1] - py) / gain,
                             (r[2] - px) / gain, (r[3] - py) / gain, w, h)
                if d:
                    dets.append(d)
        else:
            preds = out[0].T
            scores_all = preds[:, 4:]
            cls_ids = scores_all.argmax(1)
            scores = scores_all.max(1)
            keep = scores >= self.conf
            preds, scores, cls_ids = preds[keep], scores[keep], cls_ids[keep]
            if len(preds):
                cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
                x1 = (cx - bw / 2 - px) / gain
                y1 = (cy - bh / 2 - py) / gain
                rects = np.stack([x1, y1, bw / gain, bh / gain], 1)
                for c in np.unique(cls_ids):
                    idx = np.where(cls_ids == c)[0]
                    keep_i = cv2.dnn.NMSBoxes([rects[i].tolist() for i in idx],
                                              [float(scores[i]) for i in idx], self.conf, self.iou_thr)
                    for k in np.array(keep_i).flatten():
                        i = idx[int(k)]
                        bx, by, bwd, bhd = rects[i]
                        d = self._mk(int(c), scores[i], bx, by, bx + bwd, by + bhd, w, h)
                        if d:
                            dets.append(d)
        self.last_ms = (time.perf_counter() - t0) * 1000
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets


class TwoStageDetector(BaseDetector):
    """Head detector (bất kỳ backend, nhãn 'head') -> crop đầu -> classifier torchvision (helmet / no_helmet).
    cfg = {head_detector: {...detector cfg...}, classifier: {weights, arch: efficientnet_b0|resnet18, input: 224,
           labels_map: {raw->canonical}, crop_expand: 1.2, min_prob: 0.0}}
    Checkpoint classifier: dict {'model': state_dict, 'labels': [...]} (định dạng vivekvar/helmet-v5) hoặc
    state_dict thuần + cfg['classifier']['labels']."""

    def __init__(self, cfg: dict):
        import torch
        from torchvision import models
        self.cfg = cfg
        self.name = cfg.get("name", "twostage")
        self.box_level = "head"
        hd = dict(cfg["head_detector"])
        hd.setdefault("name", self.name + ".head")
        if cfg.get("device") is not None:
            hd.setdefault("device", cfg["device"])
        self.head_det = build_detector(hd)
        c = cfg["classifier"]
        self.input = int(c.get("input", 224))
        self.crop_expand = float(c.get("crop_expand", 1.2))
        self.min_prob = float(c.get("min_prob", 0.0))
        dev = cfg.get("device")
        self.device = torch.device("cuda" if (dev not in ("cpu",) and torch.cuda.is_available()) else "cpu")
        ckpt = torch.load(c["weights"], map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state, labels = ckpt["model"], list(ckpt.get("labels") or c.get("labels") or [])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state, labels = ckpt["state_dict"], list(c.get("labels") or [])
        else:                                            # state_dict thuần (OrderedDict)
            state, labels = ckpt, list(c.get("labels") or [])
        if not labels:
            # suy số lớp từ lớp cuối để báo lỗi rõ ràng
            last = [v for k, v in state.items() if k.endswith(("classifier.1.weight", "fc.weight"))]
            n_out = int(last[0].shape[0]) if last else "?"
            raise ValueError(f"Classifier cần cfg.classifier.labels (checkpoint không có nhãn; lớp cuối có {n_out} đầu ra)")
        arch = c.get("arch", "efficientnet_b0")
        m = getattr(models, arch)(weights=None)
        if arch.startswith("efficientnet"):
            m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(labels))
        elif arch.startswith("resnet"):
            m.fc = torch.nn.Linear(m.fc.in_features, len(labels))
        else:
            raise ValueError(f"arch classifier chưa hỗ trợ: {arch}")
        m.load_state_dict(state)
        self.cls = m.eval().to(self.device)
        self.cls_labels = labels
        self.mapper = ClassMapper(c.get("labels_map") or {l: l for l in labels}, keep_unmapped=True)
        self.labels = sorted({m_ for m_ in (self.mapper(l) for l in labels) if m_})
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        logger.info("[%s] head=%s + %s(%s) -> %s", self.name, self.head_det.name, arch, labels, self.labels)

    def _prep(self, crop: np.ndarray):
        import torch
        x = cv2.resize(crop, (self.input, self.input)).astype(np.float32)[:, :, ::-1] / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def detect(self, image: np.ndarray) -> list[Det]:
        import torch
        t0 = time.perf_counter()
        image = ensure_bgr(image)
        h, w = image.shape[:2]
        heads = [d for d in self.head_det.detect(image) if d.label in ("head", "helmet", "no_helmet")]
        if not heads:
            self.last_ms = (time.perf_counter() - t0) * 1000
            return []
        crops, keep = [], []
        for d in heads:
            cx, cy = d.cx, d.cy
            bw, bh = d.w * self.crop_expand, d.h * self.crop_expand
            x1, y1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            x2, y2 = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crops.append(self._prep(image[y1:y2, x1:x2]))
            keep.append(d)
        if not crops:
            return []
        with torch.no_grad():
            probs = torch.softmax(self.cls(torch.stack(crops).to(self.device)).float(), dim=1).cpu().numpy()
        out: list[Det] = []
        for d, p in zip(keep, probs):
            i = int(p.argmax())
            lab = self.mapper(self.cls_labels[i])
            if lab is None or float(p[i]) < self.min_prob:
                continue
            out.append(Det(lab, float(p[i]) * d.score if self.cfg.get("multiply_head_score", False) else float(p[i]),
                           d.box, self.cls_labels[i], self.name, i))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out


def build_detector(cfg: dict) -> BaseDetector:
    backend = (cfg.get("backend") or ("onnx" if str(cfg.get("weights", "")).endswith(".onnx") else "ultralytics")).lower()
    if backend == "ultralytics":
        return UltralyticsDetector(cfg)
    if backend == "onnx":
        return OnnxDetector(cfg)
    if backend in ("openvocab", "yoloworld", "yoloe"):
        return OpenVocabDetector(cfg)
    if backend == "twostage":
        return TwoStageDetector(cfg)
    raise ValueError(f"backend không hỗ trợ: {backend}")
