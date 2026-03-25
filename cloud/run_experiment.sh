#!/bin/bash
# cloud/run_experiment.sh
# Full experiment launcher for Lambda Labs.
# Runs setup checks, generates embeddings (if needed), then trains.
#
# Usage:
#   # All 6 CNN ablations (Table 1)
#   bash cloud/run_experiment.sh --run_all_ablations --epochs 50
#
#   # Full H-ATGNN with CNN backbone
#   bash cloud/run_experiment.sh --text_init --hierarchy --clap --run_name full_hatgnn
#
#   # Full H-ATGNN with MERT backbone (Table 2)
#   bash cloud/run_experiment.sh --backbone mert --text_init --hierarchy --run_name hatgnn_mert
#
#   # Lambda sweep
#   for lam in 0.1 0.3 0.5 1.0; do
#       bash cloud/run_experiment.sh --text_init --hierarchy --clap --lam $lam --run_name lam_$lam
#   done

set -e

LAMBDA_STORAGE=${LAMBDA_STORAGE:-/home/ubuntu/storage}
DATASET_DIR="${LAMBDA_STORAGE}/mtg_jamendo"
EMB_DIR="${LAMBDA_STORAGE}/embeddings"
OUTPUTS_DIR="${LAMBDA_STORAGE}/outputs"

echo "============================================"
echo "  H-ATGNN experiment launcher"
echo "  Storage  : ${LAMBDA_STORAGE}"
echo "  Args     : $@"
echo "============================================"

# ── Symlink storage paths into repo ──────────────────────────────────────────
mkdir -p data
[ ! -L "data/mtg_jamendo" ] && ln -s "${DATASET_DIR}" data/mtg_jamendo
mkdir -p "${EMB_DIR}"
[ ! -L "embeddings" ]       && ln -s "${EMB_DIR}" embeddings
mkdir -p "${OUTPUTS_DIR}"
[ ! -L "outputs" ]          && ln -s "${OUTPUTS_DIR}" outputs

# ── Verify data ───────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Checking dataset ..."
python scripts/download_mtg_jamendo.py --output "${DATASET_DIR}" --verify
echo ""

# ── Generate label embeddings (once, cached in Lambda storage) ────────────────
EMB_PATH="embeddings/label_embs.pt"
if [ ! -f "${EMB_PATH}" ]; then
    echo "[2/3] Generating text label embeddings ..."
    python -m utils.text_embeddings \
        --model sentence-transformers/all-mpnet-base-v2 \
        --output "${EMB_PATH}" \
        --analyse
else
    echo "[2/3] Label embeddings already exist at ${EMB_PATH} — skipping"
fi
echo ""

# ── Verify full environment ───────────────────────────────────────────────────
echo "[3/3] Environment check ..."
python scripts/verify_setup.py --data_root "${DATASET_DIR}"
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Starting training ..."
echo ""
python train.py \
    --data_root "${DATASET_DIR}" \
    --output_dir "${OUTPUTS_DIR}" \
    --emb_path "${EMB_PATH}" \
    "$@"
