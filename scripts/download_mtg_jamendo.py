"""
scripts/download_mtg_jamendo.py

Downloads the MTG-Jamendo dataset (audio + metadata + splits).

MTG-Jamendo is hosted on Zenodo and requires accepting a license.
This script handles:
  1. Downloading metadata TSVs and split files (small, no license needed)
  2. Downloading audio in chunks from Zenodo (requires free Zenodo account
     OR direct access via the MTG GitHub instructions)

Official repo: https://github.com/MTG/mtg-jamendo-dataset

Usage:
    # Download metadata + splits only (fast, no audio)
    python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --meta_only

    # Download everything (audio is ~320GB total, use --chunks to get a subset)
    python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --chunks 0 1 2

    # Download just chunk 0 (~3.2GB, ~5000 tracks, enough for a quick test)
    python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --chunks 0
"""

import os
import argparse
import subprocess
import hashlib
import urllib.request
from pathlib import Path


# ── MTG-Jamendo Zenodo URLs ───────────────────────────────────────────────────
# These are the official public URLs from the MTG GitHub repo.
# Full list: https://github.com/MTG/mtg-jamendo-dataset/blob/master/scripts/download/zenodo.py

GITHUB_DATA = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data"

METADATA_FILES = {
    "autotagging.tsv":            f"{GITHUB_DATA}/autotagging.tsv",
    "autotagging_genre.tsv":      f"{GITHUB_DATA}/autotagging_genre.tsv",
    "autotagging_moodtheme.tsv":  f"{GITHUB_DATA}/autotagging_moodtheme.tsv",
    "autotagging_instrument.tsv": f"{GITHUB_DATA}/autotagging_instrument.tsv",
    "tracks.tsv":                 f"{GITHUB_DATA}/tracks.tsv",
    "tags.tsv":                   f"{GITHUB_DATA}/tags.tsv",
}

# Split files hosted on GitHub (no login needed)
GITHUB_RAW = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master"
SPLIT_FILES = {
    "split-0/train.tsv":      f"{GITHUB_RAW}/data/splits/split-0/autotagging_moodtheme-train.tsv",
    "split-0/validation.tsv": f"{GITHUB_RAW}/data/splits/split-0/autotagging_moodtheme-validation.tsv",
    "split-0/test.tsv":       f"{GITHUB_RAW}/data/splits/split-0/autotagging_moodtheme-test.tsv",
}

# Also grab the genre splits
SPLIT_FILES_GENRE = {
    "split-0/train_genre.tsv":      f"{GITHUB_RAW}/data/splits/split-0/autotagging_genre-train.tsv",
    "split-0/validation_genre.tsv": f"{GITHUB_RAW}/data/splits/split-0/autotagging_genre-validation.tsv",
    "split-0/test_genre.tsv":       f"{GITHUB_RAW}/data/splits/split-0/autotagging_genre-test.tsv",
}

# Audio chunks — each is a tar.gz of ~3.2GB containing ~5000 tracks
# 100 chunks total = ~320GB. For experiments, chunks 0-9 (~32GB) is enough.
MTG_CDN_BASE = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio"
AUDIO_CHUNK_URL = f"{MTG_CDN_BASE}/raw_30s_audio-{{chunk:02d}}.tar"
N_AUDIO_CHUNKS  = 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str, desc: str = ""):
    """Download with progress display."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  [skip] {desc or os.path.basename(dest)} already exists")
        return

    print(f"  Downloading {desc or os.path.basename(dest)} ...")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r    {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest + ".tmp", reporthook)
        os.rename(dest + ".tmp", dest)
        print(f"\r    done      ")
    except Exception as e:
        print(f"\n  ERROR downloading {url}: {e}")
        if os.path.exists(dest + ".tmp"):
            os.remove(dest + ".tmp")
        raise


def extract_tar(tar_path: str, dest_dir: str):
    """Extract tar (or tar.gz), placing audio files in dest_dir/audio/."""
    print(f"  Extracting {os.path.basename(tar_path)} ...")
    os.makedirs(dest_dir, exist_ok=True)
    result = subprocess.run(
        ["tar", "-xf", tar_path, "-C", dest_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: tar exited {result.returncode}: {result.stderr}")
    else:
        print(f"  Extracted to {dest_dir}")
    os.remove(tar_path)   # always remove tar to save disk space


# ── Main ──────────────────────────────────────────────────────────────────────

def download_metadata(output_dir: str):
    print("\n[1/3] Downloading metadata TSVs ...")
    for name, url in METADATA_FILES.items():
        dest = os.path.join(output_dir, name)
        try:
            download_file(url, dest, name)
        except Exception:
            print(f"  Could not download {name} — check your internet connection")
            print(f"  Manual download: {url}")


def download_splits(output_dir: str):
    print("\n[2/3] Downloading split files ...")
    split_dir = os.path.join(output_dir, "splits")

    all_splits = {**SPLIT_FILES, **SPLIT_FILES_GENRE}
    for rel_path, url in all_splits.items():
        dest = os.path.join(split_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            download_file(url, dest, rel_path)
        except Exception:
            print(f"  Could not download {rel_path}")
            print(f"  Manual download: {url}")

    # Create unified train/val/test TSVs that merge mood+genre labels
    _merge_splits(split_dir)


def _merge_splits(split_dir: str):
    """
    The dataset has separate TSVs for mood and genre tasks.
    Merge them so each track has all its labels in one row.
    """
    import csv
    from collections import defaultdict

    for subset in ["train", "validation", "test"]:
        mood_path  = os.path.join(split_dir, "split-0", f"{subset}.tsv")
        genre_path = os.path.join(split_dir, "split-0", f"{subset}_genre.tsv")
        out_path   = os.path.join(split_dir, "split-0", f"{subset}_merged.tsv")

        if not os.path.exists(mood_path) or not os.path.exists(genre_path):
            continue
        if os.path.exists(out_path):
            continue

        # Read both, merge tags by track_id
        tracks = defaultdict(set)

        def _read_tsv(path, tracks):
            with open(path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    tid = row.get("TRACK_ID") or row.get("track_id", "")
                    tags = row.get("TAGS") or row.get("tags", "")
                    tracks[tid].update(t.strip() for t in tags.split(",") if t.strip())

        _read_tsv(mood_path,  tracks)
        _read_tsv(genre_path, tracks)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["track_id", "tags"])
            for tid, tags in sorted(tracks.items()):
                writer.writerow([tid, ",".join(sorted(tags))])

        print(f"  Merged {subset}: {len(tracks)} tracks -> {out_path}")


def download_audio(output_dir: str, chunks: list):
    print(f"\n[3/3] Downloading audio chunks: {chunks}")
    print(f"  Each chunk ~3.2GB. Total: ~{len(chunks)*3.2:.0f}GB")
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for chunk in chunks:
        url      = AUDIO_CHUNK_URL.format(chunk=chunk)
        tar_path = os.path.join(output_dir, f"audio_chunk_{chunk:02d}.tar")
        try:
            download_file(url, tar_path, f"audio chunk {chunk:02d}")
            extract_tar(tar_path, audio_dir)
        except Exception as e:
            print(f"\n  Failed on chunk {chunk}: {e}")
            print(f"  You can retry with: python scripts/download_mtg_jamendo.py "
                  f"--output {output_dir} --chunks {chunk}")


def verify_structure(output_dir: str):
    print("\n[Verify] Checking directory structure ...")
    required = [
        "autotagging_genre.tsv",
        "autotagging_moodtheme.tsv",
        "splits/split-0/train.tsv",
        "splits/split-0/validation.tsv",
        "splits/split-0/test.tsv",
    ]
    ok = True
    for rel in required:
        path = os.path.join(output_dir, rel)
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {rel}")
        if not exists:
            ok = False

    audio_dir = os.path.join(output_dir, "audio")
    if os.path.exists(audio_dir):
        n_files = sum(1 for _ in Path(audio_dir).rglob("*.mp3"))
        print(f"  [OK] audio/  ({n_files} mp3 files found)")
    else:
        print(f"  [MISSING] audio/  (run with --chunks 0 to download)")
        ok = False

    if ok:
        print("\n  Dataset ready.")
    else:
        print("\n  Some files missing. See above.")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",    type=str, default="data/mtg_jamendo",
                        help="Root directory for the dataset")
    parser.add_argument("--meta_only", action="store_true",
                        help="Download metadata and splits only (no audio)")
    parser.add_argument("--chunks",    type=int, nargs="+", default=None,
                        help="Which audio chunks to download (0-99). "
                             "Omit to skip audio. Use 0 for a small test set.")
    parser.add_argument("--verify",    action="store_true",
                        help="Only verify existing download, don't download")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.verify:
        verify_structure(args.output)
    else:
        download_metadata(args.output)
        download_splits(args.output)
        if not args.meta_only and args.chunks:
            download_audio(args.output, args.chunks)
        verify_structure(args.output)
