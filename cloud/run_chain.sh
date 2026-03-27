#!/bin/bash
# cloud/run_chain.sh
# Full automated pipeline: smoketest → wait for download → experiments → shutdown.
#
# Usage:
#   # Table 1 (CNN ablations) with auto-shutdown
#   AUTO_SHUTDOWN=1 bash cloud/run_chain.sh --table 1
#
#   # Table 2 (SSL experiments) with auto-shutdown
#   AUTO_SHUTDOWN=1 bash cloud/run_chain.sh --table 2

set -a; [ -f ".env" ] && source .env; set +a

STORAGE_ROOT=${STORAGE_ROOT:-/workspace}
DATASET_DIR="${STORAGE_ROOT}/mtg_jamendo"
EMB_DIR="${STORAGE_ROOT}/embeddings"
OUTPUTS_DIR="${STORAGE_ROOT}/outputs"
HF_CACHE_DIR="${STORAGE_ROOT}/hf_cache"
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
TABLE=${2:-1}   # default Table 1

# Parse --table argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --table) TABLE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
mkdir -p "${EMB_DIR}" "${OUTPUTS_DIR}" "${HF_CACHE_DIR}"

cd "$(dirname "$0")/.."   # repo root

WANDB_FLAGS=""
if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_FLAGS="--wandb_project ${WANDB_PROJECT}"
    [ -n "${WANDB_ENTITY}" ] && WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
fi

LOG="${STORAGE_ROOT}/chain_table${TABLE}.log"
echo "================================================" | tee -a $LOG
echo "  H-ATGNN chain: Table ${TABLE}  $(date)" | tee -a $LOG
echo "================================================" | tee -a $LOG

# ── Symlinks ──────────────────────────────────────────────────────────────────
mkdir -p data
[ ! -L "data/mtg_jamendo" ] && ln -sf "${DATASET_DIR}" data/mtg_jamendo
[ ! -L "embeddings" ]       && ln -sf "${EMB_DIR}" embeddings
[ ! -L "outputs" ]          && ln -sf "${OUTPUTS_DIR}" outputs

# ── Step 1: Smoke test (2 epochs, CNN) ───────────────────────────────────────
echo "[1/4] Running smoke test..." | tee -a $LOG

SMOKE_LOG="${STORAGE_ROOT}/smoketest.log"
# Generate label embeddings if needed
EMB_PATH="embeddings/label_embs.pt"
[ ! -f "${EMB_PATH}" ] && python -m utils.text_embeddings --output "${EMB_PATH}"

python train.py \
    --text_init --hierarchy \
    --epochs 2 --batch_size 24 \
    --run_name smoketest_cnn \
    --data_root "${DATASET_DIR}" \
    --output_dir "${OUTPUTS_DIR}" \
    --emb_path "${EMB_PATH}" \
    ${WANDB_FLAGS} --wandb_tags smoketest \
    2>&1 | tee "${SMOKE_LOG}"

if ! grep -q "TEST" "${SMOKE_LOG}"; then
    echo "[chain] ABORT: smoke test failed. Check ${SMOKE_LOG}" | tee -a $LOG
    exit 1
fi
echo "[1/4] Smoke test passed." | tee -a $LOG

# ── Step 2: Wait for data download to finish ─────────────────────────────────
echo "[2/4] Waiting for download to finish..." | tee -a $LOG
while pgrep -f download_mtg_jamendo > /dev/null; do
    CHUNKS=$(ls "${DATASET_DIR}/audio/" 2>/dev/null | wc -l)
    echo "  ... ${CHUNKS}/20 chunks ready  $(date +%H:%M)" | tee -a $LOG
    sleep 120
done
echo "[2/4] Download complete. $(ls ${DATASET_DIR}/audio/ | wc -l)/20 chunks." | tee -a $LOG

# ── Step 3: Run experiments ───────────────────────────────────────────────────
echo "[3/4] Starting Table ${TABLE} experiments..." | tee -a $LOG

if [ "$TABLE" = "1" ]; then
    python train.py \
        --run_all_ablations \
        --epochs 50 --batch_size 24 \
        --data_root "${DATASET_DIR}" \
        --output_dir "${OUTPUTS_DIR}" \
        --emb_path "${EMB_PATH}" \
        ${WANDB_FLAGS} --wandb_tags "table1,cnn,cloud" \
        2>&1 | tee -a $LOG
elif [ "$TABLE" = "2" ]; then
    python train.py \
        --run_ssl_experiments \
        --epochs 50 --batch_size 32 \
        --data_root "${DATASET_DIR}" \
        --output_dir "${OUTPUTS_DIR}" \
        --emb_path "${EMB_PATH}" \
        ${WANDB_FLAGS} --wandb_tags "table2,ssl,cloud" \
        2>&1 | tee -a $LOG
fi

TRAIN_EXIT=${PIPESTATUS[0]}
echo "[3/4] Experiments finished with exit code ${TRAIN_EXIT} at $(date)" | tee -a $LOG

# ── Step 4: Shutdown ──────────────────────────────────────────────────────────
if [ "${AUTO_SHUTDOWN}" = "1" ]; then
    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "[4/4] Training failed — NOT shutting down so you can inspect." | tee -a $LOG
    else
        echo "[4/4] All done. Shutting down in 60s... (Ctrl+C to cancel)" | tee -a $LOG
        sleep 60
        sudo shutdown -h now
    fi
else
    echo "[4/4] AUTO_SHUTDOWN=0 — pod staying alive." | tee -a $LOG
fi
