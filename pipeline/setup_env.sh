#!/usr/bin/env bash
# Tạo venv Windows (Python 3.12) và cài torch CUDA + ultralytics. Log: setup_env.log
set -u
cd "$(dirname "$0")"
PY="${PY:-$LOCALAPPDATA/Programs/Python/Python312/python.exe}"
echo "[setup] python: $PY"; "$PY" -V || exit 1
[ -d .venv ] || "$PY" -m venv .venv || exit 1
VPY=".venv/Scripts/python.exe"
"$VPY" -m pip install --upgrade pip wheel setuptools -q
ok=0
for cu in cu128 cu126 cu124; do
  echo "[setup] thử torch index $cu ..."
  if "$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu" -q 2>>setup_env.err; then
    if "$VPY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
      echo "[setup] torch $cu OK, CUDA khả dụng"; ok=1; break
    else
      echo "[setup] torch $cu cài được nhưng CUDA không khả dụng, thử index khác"
    fi
  else
    echo "[setup] index $cu thất bại"
  fi
done
[ $ok -eq 1 ] || echo "[setup] CẢNH BÁO: không có CUDA, sẽ chạy CPU"
"$VPY" -m pip install -q "ultralytics>=8.3" onnx onnxruntime onnxslim huggingface_hub requests tqdm pyyaml lap pycocotools-windows 2>>setup_env.err || \
"$VPY" -m pip install -q "ultralytics>=8.3" onnx onnxruntime onnxslim huggingface_hub requests tqdm pyyaml lap 2>>setup_env.err
"$VPY" - <<'PYEOF'
import torch, ultralytics, cv2, numpy, onnxruntime
print("[setup] torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("[setup] ultralytics", ultralytics.__version__, "| opencv", cv2.__version__, "| numpy", numpy.__version__, "| onnxruntime", onnxruntime.__version__)
PYEOF
echo "[setup] DONE"
