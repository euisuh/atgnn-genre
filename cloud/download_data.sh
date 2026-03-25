#!/bin/bash
# cloud/download_data.sh
# Downloads MTG-Jamendo to Lambda persistent storage and symlinks it into the repo.
#
# Lambda Storage mounts at a path you choose when creating it (e.g. /home/ubuntu/storage).
# Set LAMBDA_STORAGE to that mount point before running.
#
# Usage:
#   export LAMBDA_STORAGE=/home/ubuntu/storage   # set to your mount path
#   bash cloud/download_data.sh                  # metadata + splits + chunks 0-9 (~32GB)
#   bash cloud/download_data.sh --chunks 0 1 2   # specific chunks only
#   bash cloud/download_data.sh --meta_only      # metadata + splits only

set -e

# ── Config ────────────────────────────────────────────────────────────────────
LAMBDA_STORAGE=${LAMBDA_STORAGE:-/home/ubuntu/storage}
DATASET_DIR="${LAMBDA_STORAGE}/mtg_jamendo"
REPO_DATA_LINK="data/mtg_jamendo"

# ── Validate storage mount ────────────────────────────────────────────────────
if [ ! -d "${LAMBDA_STORAGE}" ]; then
    echo "ERROR: Lambda storage not found at ${LAMBDA_STORAGE}"
    echo "  - Make sure you attached a persistent storage volume to this instance."
    echo "  - Set LAMBDA_STORAGE to the correct mount path, e.g.:"
    echo "      export LAMBDA_STORAGE=/home/ubuntu/storage"
    exit 1
fi

echo "============================================"
echo "  MTG-Jamendo dataset download"
echo "  Storage : ${LAMBDA_STORAGE}"
echo "  Dataset : ${DATASET_DIR}"
echo "============================================"

mkdir -p "${DATASET_DIR}"

# ── Symlink data/ -> Lambda storage ──────────────────────────────────────────
# This lets all existing code use data/mtg_jamendo without changes.
mkdir -p data
if [ ! -L "${REPO_DATA_LINK}" ]; then
    ln -s "${DATASET_DIR}" "${REPO_DATA_LINK}"
    echo "  Symlinked ${REPO_DATA_LINK} -> ${DATASET_DIR}"
else
    echo "  Symlink ${REPO_DATA_LINK} already exists"
fi

# ── Download ──────────────────────────────────────────────────────────────────
# Parse args: forward everything to the Python download script.
# Default: download chunks 0-9 (~32GB, ~50k tracks — enough for solid results).
if [ "$#" -eq 0 ]; then
    echo "  Downloading metadata + splits + chunks 0-9 (~32GB)"
    python scripts/download_mtg_jamendo.py \
        --output "${DATASET_DIR}" \
        --chunks 0 1 2 3 4 5 6 7 8 9
else
    python scripts/download_mtg_jamendo.py \
        --output "${DATASET_DIR}" \
        "$@"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
python scripts/download_mtg_jamendo.py --output "${DATASET_DIR}" --verify
