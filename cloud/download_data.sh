#!/bin/bash
# cloud/download_data.sh
# Downloads MTG-Jamendo to persistent storage and symlinks it into the repo.
#
# MTG-Jamendo has 100 chunks (0-99), each ~3.2GB. Tracks are ordered by ID so
# sequential chunks (e.g. 0-9) skew toward a narrow slice of the catalogue.
# The default here downloads 20 evenly-spaced chunks (~64GB, ~100k tracks) to
# get a representative genre/mood distribution across the full dataset.
#
# RunPod: Network Volume mounts at /runpod-volume by default
# Lambda Labs: use /home/ubuntu/storage
# Override by setting STORAGE_ROOT in .env or environment.
#
# Usage:
#   bash cloud/download_data.sh                  # stratified 20 chunks (~64GB) [default]
#   bash cloud/download_data.sh --small          # stratified 10 chunks (~32GB, quick test)
#   bash cloud/download_data.sh --full           # all 100 chunks (~320GB, full dataset)
#   bash cloud/download_data.sh --chunks 0 5 10  # specific chunks
#   bash cloud/download_data.sh --meta_only      # metadata + splits only

set -e

# Load .env if present
if [ -f ".env" ]; then
    set -a && source .env && set +a
fi

# ── Config ────────────────────────────────────────────────────────────────────
STORAGE_ROOT=${STORAGE_ROOT:-/runpod-volume}
DATASET_DIR="${STORAGE_ROOT}/mtg_jamendo"
REPO_DATA_LINK="data/mtg_jamendo"

# ── Validate storage mount ────────────────────────────────────────────────────
if [ ! -d "${STORAGE_ROOT}" ]; then
    echo "ERROR: Persistent storage not found at ${STORAGE_ROOT}"
    echo "  - RunPod: make sure you attached a Network Volume to this pod."
    echo "  - Set STORAGE_ROOT in .env to the correct mount path, e.g.:"
    echo "      STORAGE_ROOT=/runpod-volume"
    exit 1
fi

echo "============================================"
echo "  MTG-Jamendo dataset download"
echo "  Storage : ${STORAGE_ROOT}"
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

# ── Chunk selection ───────────────────────────────────────────────────────────
# Evenly-spaced chunks give a better genre/mood distribution than sequential ones.
# 100 chunks total → step=5 gives 20 chunks, step=10 gives 10 chunks.
STRATIFIED_20="0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95"
STRATIFIED_10="0 10 20 30 40 50 60 70 80 90"
ALL_CHUNKS=$(seq 0 99 | tr '\n' ' ')

# ── Download ──────────────────────────────────────────────────────────────────
if [ "$1" = "--small" ]; then
    echo "  Downloading stratified 10 chunks (~32GB)"
    python scripts/download_mtg_jamendo.py \
        --output "${DATASET_DIR}" \
        --chunks ${STRATIFIED_10}
elif [ "$1" = "--full" ]; then
    echo "  Downloading all 100 chunks (~320GB)"
    python scripts/download_mtg_jamendo.py \
        --output "${DATASET_DIR}" \
        --chunks ${ALL_CHUNKS}
elif [ "$#" -eq 0 ]; then
    echo "  Downloading stratified 20 chunks (~64GB, recommended)"
    python scripts/download_mtg_jamendo.py \
        --output "${DATASET_DIR}" \
        --chunks ${STRATIFIED_20}
else
    # Pass args directly (e.g. --chunks 3 7 42, or --meta_only)
    python scripts/download_mtg_jamendo.py \
        --output "${DATASET_DIR}" \
        "$@"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
python scripts/download_mtg_jamendo.py --output "${DATASET_DIR}" --verify
