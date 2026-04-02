"""
scripts/precompute_spectrograms.py

Pre-computes log-mel spectrograms for all MTG-Jamendo audio files and saves
them as float16 .pt tensors next to the source mp3s.

Why:
  Real-time mp3 decode + resample + mel at training time is ~0.5–2s/file on CPU.
  At batch_size=16 with 4 workers that makes each batch 4–8s of loading, leaving
  the GPU idle ~57% of the time. Loading a pre-saved tensor is ~2–10ms — 100–200x
  faster. After precompute the GPU stays at >90% utilisation during training.

Output:
  For each audio file at:
    data/mtg_jamendo/audio/00/track_0000000.mp3
  This script writes:
    data/mtg_jamendo/audio/00/track_0000000.spec.pt   <- tensor (1, 128, 1024) float16

Usage:
    python scripts/precompute_spectrograms.py --data_root /workspace/mtg_jamendo
    python scripts/precompute_spectrograms.py --data_root /workspace/mtg_jamendo --workers 8
    python scripts/precompute_spectrograms.py --data_root /workspace/mtg_jamendo --overwrite

Time estimate (RTX 4090 / NVMe):
    ~10k tracks: ~5–10 min
    ~50k tracks: ~25–50 min

Disk cost:
    Each spec is (1, 128, 1024) float16 = 262 KB
    ~50k tracks → ~13 GB extra on the volume
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from tqdm import tqdm


# ── Parameters — must match HATGNNConfig defaults in configs/default.py ──────
SAMPLE_RATE = 16000
N_MELS      = 128
HOP_MS      = 10
WIN_MS      = 25
MAX_FRAMES  = 1024
F_MIN       = 20
F_MAX       = 8000


def _make_transform():
    hop_len = int(HOP_MS * SAMPLE_RATE / 1000)   # 160
    win_len = int(WIN_MS * SAMPLE_RATE / 1000)   # 400
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=win_len,
        hop_length=hop_len,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
    )


def process_file(mp3_path: str, overwrite: bool) -> tuple[str, bool, str]:
    """
    Load one mp3, compute log-mel spectrogram, save as float16 .pt.
    Returns (path, success, error_message).
    """
    spec_path = mp3_path.replace(".mp3", ".spec.pt")
    if not overwrite and os.path.exists(spec_path):
        return mp3_path, True, "skipped"

    try:
        waveform, sr = torchaudio.load(mp3_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_transform = _make_transform()
        mel = mel_transform(waveform)                    # (1, n_mels, T)
        log_mel = torch.log(mel + 1e-8)

        # Pad / crop to MAX_FRAMES
        T_frames = log_mel.shape[-1]
        if T_frames < MAX_FRAMES:
            log_mel = F.pad(log_mel, (0, MAX_FRAMES - T_frames))
        else:
            log_mel = log_mel[..., :MAX_FRAMES]

        torch.save(log_mel.half(), spec_path)            # save as float16
        return mp3_path, True, ""
    except Exception as e:
        return mp3_path, False, str(e)


def collect_mp3s(audio_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.endswith(".mp3"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute log-mel spectrograms")
    parser.add_argument("--data_root", default="data/mtg_jamendo",
                        help="Root of MTG-Jamendo dataset (contains audio/)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if .spec.pt already exists")
    args = parser.parse_args()

    audio_dir = os.path.join(args.data_root, "audio")
    if not os.path.isdir(audio_dir):
        print(f"ERROR: audio dir not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    mp3s = collect_mp3s(audio_dir)
    if not mp3s:
        print("No .mp3 files found — is the dataset downloaded?", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(mp3s)} mp3 files in {audio_dir}")
    print(f"Workers: {args.workers}  Overwrite: {args.overwrite}")
    print(f"Output: <track>.spec.pt  shape=(1,{N_MELS},{MAX_FRAMES}) float16 ≈ 262KB each")
    est_gb = len(mp3s) * 262 / 1024 / 1024
    print(f"Estimated disk usage: {est_gb:.1f} GB")
    print()

    ok = skipped = failed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_file, p, args.overwrite): p for p in mp3s}
        with tqdm(total=len(mp3s), unit="file") as bar:
            for fut in as_completed(futures):
                path, success, msg = fut.result()
                if msg == "skipped":
                    skipped += 1
                elif success:
                    ok += 1
                else:
                    failed += 1
                    errors.append((path, msg))
                bar.update(1)
                bar.set_postfix(ok=ok, skip=skipped, fail=failed)

    print(f"\nDone. processed={ok}  skipped={skipped}  failed={failed}")
    if errors:
        print(f"\nFailed files ({len(errors)}):")
        for p, e in errors[:20]:
            print(f"  {p}: {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")


if __name__ == "__main__":
    main()
