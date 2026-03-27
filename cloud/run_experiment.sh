#!/bin/bash
# cloud/run_experiment.sh
# Full experiment launcher for RunPod (and compatible with Lambda Labs).
# Runs setup checks, generates embeddings (if needed), then trains.
#
# Usage:
#   # All 6 CNN ablations (Table 1)
#   bash cloud/run_experiment.sh --run_all_ablations --epochs 50
#
#   # SSL experiments (Table 2: MERT + MuQ)
#   bash cloud/run_experiment.sh --run_ssl_experiments --epochs 50
#
#   # Auto-shutdown when done (pod terminates, billing stops)
#   AUTO_SHUTDOWN=1 bash cloud/run_experiment.sh --run_all_ablations --epochs 50
#
#   # Lambda sweep
#   for lam in 0.1 0.3 0.5 1.0; do
#       bash cloud/run_experiment.sh --text_init --hierarchy --cross_modal clap \
#           --lam $lam --run_name lam_$lam
#   done

set -e

# Load .env if present (API keys — not committed to git)
if [ -f ".env" ]; then
    set -a && source .env && set +a
fi

# RunPod: persistent Network Volume mounts at /runpod-volume by default
# Lambda Labs: use /home/ubuntu/storage
# Override by setting STORAGE_ROOT in .env or environment
STORAGE_ROOT=${STORAGE_ROOT:-/workspace}
DATASET_DIR="${STORAGE_ROOT}/mtg_jamendo"
EMB_DIR="${STORAGE_ROOT}/embeddings"
OUTPUTS_DIR="${STORAGE_ROOT}/outputs"
HF_CACHE_DIR="${STORAGE_ROOT}/hf_cache"

# Store HuggingFace model weights on persistent storage so they survive
# pod termination and don't need to be re-downloaded next time
mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

# ── W&B config ────────────────────────────────────────────────────────────────
# Set WANDB_API_KEY before running, e.g.:
#   export WANDB_API_KEY=your_key_here
#   bash cloud/run_experiment.sh --run_all_ablations ...
#
# Or pass project/entity directly:
#   WANDB_PROJECT=hatgnn WANDB_ENTITY=your_entity bash cloud/run_experiment.sh ...
WANDB_PROJECT=${WANDB_PROJECT:-""}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_TAGS=${WANDB_TAGS:-"cloud,lambda"}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}

# Build W&B flags — only added to train.py if WANDB_PROJECT is set
WANDB_FLAGS=""
if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_FLAGS="--wandb_project ${WANDB_PROJECT}"
    [ -n "${WANDB_ENTITY}" ] && WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    [ -n "${WANDB_TAGS}" ]   && WANDB_FLAGS="${WANDB_FLAGS} --wandb_tags ${WANDB_TAGS}"
fi

echo "============================================"
echo "  H-ATGNN experiment launcher"
echo "  Storage  : ${STORAGE_ROOT}"
echo "  W&B      : ${WANDB_PROJECT:-disabled}"
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

# ── W&B auth check ────────────────────────────────────────────────────────────
if [ -n "${WANDB_PROJECT}" ]; then
    if [ -z "${WANDB_API_KEY}" ]; then
        echo "WARNING: WANDB_PROJECT is set but WANDB_API_KEY is not."
        echo "         Set it with: export WANDB_API_KEY=your_key_here"
        echo "         Continuing without W&B logging."
        WANDB_FLAGS=""
    else
        echo "W&B auth  : key found, logging to project '${WANDB_PROJECT}'"
        python -c "import wandb; wandb.login()" 2>/dev/null && echo "W&B login : OK" || echo "W&B login : FAILED (continuing anyway)"
    fi
fi

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Starting training ..."
echo ""
python train.py \
    --data_root "${DATASET_DIR}" \
    --output_dir "${OUTPUTS_DIR}" \
    --emb_path "${EMB_PATH}" \
    ${WANDB_FLAGS} \
    "$@"
TRAIN_EXIT=$?

# ── Auto-shutdown ─────────────────────────────────────────────────────────────
# Runs only if AUTO_SHUTDOWN=1 is set.
# On Lambda Labs, powering off terminates the instance and stops billing.
# Checkpoints are already uploaded to W&B artifacts before this point.
if [ "${AUTO_SHUTDOWN}" = "1" ]; then
    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo ""
        echo "WARNING: training exited with code ${TRAIN_EXIT}."
        echo "Skipping auto-shutdown — review logs before closing the instance."
    else
        echo ""
        echo "Training complete. Shutting down in 60 seconds..."
        echo "Press Ctrl+C to cancel."
        sleep 60
        sudo shutdown -h now
    fi
fi
