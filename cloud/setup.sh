#!/bin/bash
# cloud/setup.sh
# Run once on a fresh RunPod pod (or Lambda Labs instance) to install all dependencies.
# RunPod runs as root — sudo is not needed but harmless.
#
# Usage:
#   bash cloud/setup.sh

set -e

echo "============================================"
echo "  H-ATGNN environment setup (RunPod / Lambda)"
echo "============================================"

# ── Detect torch + CUDA versions ──────────────────────────────────────────────
TORCH_VER=$(python -c "import torch; v=torch.__version__; print(v.split('+')[0])")
CUDA_TAG=$(python -c "
import torch
cuda = torch.version.cuda          # e.g. '12.1'
tag  = 'cu' + cuda.replace('.','') # e.g. 'cu121'
print(tag)
")

echo ""
echo "  Detected: torch=${TORCH_VER}  cuda_tag=${CUDA_TAG}"
echo ""

# ── PyG extras (scatter / sparse / cluster) ───────────────────────────────────
# These require --no-build-isolation so they can see the installed torch.
PYG_WHL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "[1/4] Installing torch-geometric + extras from ${PYG_WHL}"
pip install torch-geometric
pip install --no-build-isolation torch-scatter torch-sparse torch-cluster \
    -f "${PYG_WHL}"

# ── torchaudio codec backend ──────────────────────────────────────────────────
echo "[2/4] Installing torchaudio backends"
pip install soundfile
# torchcodec — skip on older CUDA if wheel unavailable (torchaudio falls back to soundfile)
pip install torchcodec 2>/dev/null || echo "  torchcodec not available for this env, skipping (soundfile will be used)"

# ── NLP / embedding deps ──────────────────────────────────────────────────────
echo "[3/4] Installing NLP deps (sentence-transformers, transformers)"
pip install sentence-transformers transformers

# ── General deps ──────────────────────────────────────────────────────────────
echo "[4/4] Installing general deps"
pip install scikit-learn numpy tqdm wandb muq

echo ""
echo "  Setup complete. Run: python scripts/verify_setup.py"
echo ""
