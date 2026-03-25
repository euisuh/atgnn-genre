"""
scripts/precompute_clap.py

Pre-computes CLAP audio embeddings for all tracks in MTG-Jamendo
and saves them as individual .pt files alongside the audio.

Why pre-compute instead of computing during training?
  CLAP inference is ~3x slower than the CNN backbone. Running it
  in the training loop would bottleneck the GPU. Pre-computing once
  means training uses cached 512-dim vectors with near-zero overhead.

Output:
  For each audio file at:
    data/mtg_jamendo/audio/00/track_0000000.mp3
  This script writes:
    data/mtg_jamendo/clap_embs/00/track_0000000.pt   <- tensor shape (512,)

Usage:
    python scripts/precompute_clap.py \
        --data_root data/mtg_jamendo \
        --model laion/clap-htsat-fused \
        --batch_size 32 \
        --device cuda

    # CPU fallback (slow but works):
    python scripts/precompute_clap.py --device cpu

Time estimate:
    ~18k tracks on a single A100: ~25 minutes
    ~18k tracks on CPU: ~4-6 hours
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm


def load_clap_model(model_id: str, device: str):
    """
    Loads LAION-CLAP from HuggingFace.
    pip install laion-clap  (or transformers >= 4.30 has it too)
    """
    try:
        # Try laion_clap package first (official)
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
        model.load_ckpt()  # downloads weights automatically
        print(f"Loaded CLAP via laion_clap package")
        return model, "laion_clap"
    except ImportError:
        pass

    try:
        # Fallback: HuggingFace transformers ClapModel
        from transformers import ClapModel, ClapProcessor
        model = ClapModel.from_pretrained(model_id).to(device)
        processor = ClapProcessor.from_pretrained(model_id)
        model.eval()
        print(f"Loaded CLAP via transformers ({model_id})")
        return (model, processor), "transformers"
    except Exception as e:
        print(f"ERROR: Could not load CLAP model.")
        print(f"Install with: pip install laion-clap")
        print(f"Or: pip install transformers>=4.30")
        print(f"Detail: {e}")
        sys.exit(1)


@torch.no_grad()
def embed_batch_laion(model, audio_paths, sr=48000):
    """Embed a batch using the laion_clap package."""
    embeddings = model.get_audio_embedding_from_filelist(
        x=audio_paths, use_tensor=True
    )
    return embeddings.cpu()   # (B, 512)


@torch.no_grad()
def embed_batch_transformers(model_tuple, audio_paths, device, sr=48000):
    """Embed a batch using HuggingFace transformers ClapModel."""
    model, processor = model_tuple
    waveforms = []
    for path in audio_paths:
        wav, file_sr = torchaudio.load(path)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        # CLAP expects up to 10s; truncate/pad to 480000 samples @ 48kHz
        target_len = sr * 10
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        waveforms.append(wav.squeeze(0).numpy())

    inputs = processor(
        audios=waveforms,
        return_tensors="pt",
        sampling_rate=sr,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.get_audio_features(**inputs)
    return out.cpu()   # (B, 512)


def find_audio_files(data_root: str):
    audio_dir = os.path.join(data_root, "audio")
    if not os.path.isdir(audio_dir):
        print(f"ERROR: audio directory not found at {audio_dir}")
        print("Run scripts/setup_data.py first.")
        sys.exit(1)
    files = sorted(Path(audio_dir).rglob("*.mp3"))
    files += sorted(Path(audio_dir).rglob("*.wav"))
    return [str(f) for f in files]


def get_output_path(audio_path: str, data_root: str) -> str:
    """Mirror the audio path structure under clap_embs/"""
    rel = os.path.relpath(audio_path,
                          os.path.join(data_root, "audio"))
    stem = os.path.splitext(rel)[0]
    return os.path.join(data_root, "clap_embs", stem + ".pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="data/mtg_jamendo")
    p.add_argument("--model",      default="laion/clap-htsat-fused",
                   help="HuggingFace model id (used only for transformers fallback)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite",  action="store_true",
                   help="Recompute even if .pt file already exists")
    p.add_argument("--sample_rate",type=int, default=48000,
                   help="CLAP expects 48kHz audio")
    args = p.parse_args()

    print(f"\nCLAP Embedding Pre-computation")
    print(f"{'='*50}")
    print(f"Data root  : {args.data_root}")
    print(f"Device     : {args.device}")
    print(f"Batch size : {args.batch_size}")
    print(f"{'='*50}\n")

    # Load model
    model_obj, backend = load_clap_model(args.model, args.device)

    # Find all audio files
    audio_files = find_audio_files(args.data_root)
    print(f"Found {len(audio_files):,} audio files\n")

    # Filter already-done
    if not args.overwrite:
        todo = [f for f in audio_files
                if not os.path.exists(get_output_path(f, args.data_root))]
        print(f"  {len(audio_files) - len(todo):,} already done, "
              f"{len(todo):,} remaining\n")
    else:
        todo = audio_files

    if not todo:
        print("All embeddings already computed. Done.")
        return

    # Process in batches
    errors = []
    for i in tqdm(range(0, len(todo), args.batch_size),
                  desc="Computing CLAP embeddings"):
        batch_paths = todo[i : i + args.batch_size]

        # Skip files that can't be loaded
        valid_paths = []
        for p_ in batch_paths:
            try:
                torchaudio.info(p_)
                valid_paths.append(p_)
            except Exception:
                errors.append(p_)
                continue

        if not valid_paths:
            continue

        try:
            if backend == "laion_clap":
                embs = embed_batch_laion(model_obj, valid_paths,
                                          sr=args.sample_rate)
            else:
                embs = embed_batch_transformers(model_obj, valid_paths,
                                                args.device,
                                                sr=args.sample_rate)

            # Save individual .pt files
            for j, path in enumerate(valid_paths):
                out_path = get_output_path(path, args.data_root)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(embs[j], out_path)

        except Exception as e:
            errors.extend(valid_paths)
            tqdm.write(f"Batch error at index {i}: {e}")
            continue

    print(f"\nDone. Embeddings saved to {args.data_root}/clap_embs/")
    if errors:
        print(f"Failed on {len(errors)} files:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more")

        # Save error list
        err_path = os.path.join(args.data_root, "clap_errors.txt")
        with open(err_path, "w") as f:
            f.write("\n".join(errors))
        print(f"Full error list saved to {err_path}")


if __name__ == "__main__":
    main()
