"""
scripts/precompute_muqmulan.py

Pre-computes MuQ-MuLan audio embeddings for all tracks in MTG-Jamendo
and saves them as individual .pt files — identical layout to precompute_clap.py.

Why pre-compute?
  MuQ-MuLan inference is slower than training a linear layer on top. Running it
  every forward pass would bottleneck the GPU. Pre-computing once means training
  uses cached embeddings with near-zero overhead.

Output:
  For each audio file at:
    data/mtg_jamendo/audio/00/track_0000000.mp3
  This script writes:
    data/mtg_jamendo/muqmulan_embs/00/track_0000000.pt   <- tensor shape (512,)

Usage:
    python scripts/precompute_muqmulan.py \
        --data_root data/mtg_jamendo \
        --model tencent-ailab/MuQ-MuLan \
        --batch_size 16 \
        --device cuda

    # CPU fallback (slow but works for smoke testing):
    python scripts/precompute_muqmulan.py --device cpu --batch_size 4

Time estimate:
    ~18k tracks on a single A100: ~35 minutes
    ~18k tracks on CPU: ~6-10 hours
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


_MODEL_SR = {
    "tencent-ailab/MuQ-MuLan": 24000,
    "tencent-ailab/MuQ-MuLan-large": 24000,
}


def load_muqmulan(model_id: str, device: str):
    from transformers import AutoModel
    print(f"Loading MuQ-MuLan: {model_id}")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def embed_batch(model, waveforms: torch.Tensor, device: str):
    """
    waveforms : (B, T) float32 at model_sr
    returns   : (B, D) cpu tensor
    """
    waveforms = waveforms.to(device)
    out = model(input_values=waveforms)

    if hasattr(out, "audio_embeds"):
        emb = out.audio_embeds
    elif hasattr(out, "pooler_output"):
        emb = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        emb = out.last_hidden_state.mean(dim=1)
    else:
        raise ValueError(
            f"Cannot extract audio embedding from model output: {list(vars(out).keys())}"
        )
    return emb.cpu()


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load audio, mono, resample to target_sr -> (T,) float32"""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    # Truncate/pad to 30s
    max_len = target_sr * 30
    if wav.shape[1] > max_len:
        wav = wav[:, :max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]))
    return wav.squeeze(0)   # (T,)


def find_audio_files(data_root: str):
    audio_dir = os.path.join(data_root, "audio")
    if not os.path.isdir(audio_dir):
        print(f"ERROR: audio directory not found at {audio_dir}")
        sys.exit(1)
    files = sorted(Path(audio_dir).rglob("*.mp3"))
    files += sorted(Path(audio_dir).rglob("*.wav"))
    return [str(f) for f in files]


def get_output_path(audio_path: str, data_root: str) -> str:
    rel  = os.path.relpath(audio_path, os.path.join(data_root, "audio"))
    stem = os.path.splitext(rel)[0]
    return os.path.join(data_root, "muqmulan_embs", stem + ".pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="data/mtg_jamendo")
    p.add_argument("--model",      default="tencent-ailab/MuQ-MuLan")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite",  action="store_true")
    args = p.parse_args()

    model_sr = _MODEL_SR.get(args.model, 24000)

    print(f"\nMuQ-MuLan Embedding Pre-computation")
    print(f"{'='*50}")
    print(f"Data root  : {args.data_root}")
    print(f"Model      : {args.model}  (SR={model_sr}Hz)")
    print(f"Device     : {args.device}")
    print(f"Batch size : {args.batch_size}")
    print(f"{'='*50}\n")

    model = load_muqmulan(args.model, args.device)

    audio_files = find_audio_files(args.data_root)
    print(f"Found {len(audio_files):,} audio files")

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

    errors = []
    for i in tqdm(range(0, len(todo), args.batch_size),
                  desc="Computing MuQ-MuLan embeddings"):
        batch_paths = todo[i : i + args.batch_size]

        valid_paths, wavs = [], []
        for path in batch_paths:
            try:
                wav = load_audio(path, model_sr)
                valid_paths.append(path)
                wavs.append(wav)
            except Exception:
                errors.append(path)

        if not valid_paths:
            continue

        try:
            wav_batch = torch.stack(wavs)              # (B, T)
            embs = embed_batch(model, wav_batch, args.device)   # (B, D)

            for j, path in enumerate(valid_paths):
                out_path = get_output_path(path, args.data_root)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(embs[j], out_path)

        except Exception as e:
            errors.extend(valid_paths)
            tqdm.write(f"Batch error at index {i}: {e}")

    print(f"\nDone. Embeddings saved to {args.data_root}/muqmulan_embs/")
    if errors:
        print(f"Failed on {len(errors)} files.")
        err_path = os.path.join(args.data_root, "muqmulan_errors.txt")
        with open(err_path, "w") as f:
            f.write("\n".join(errors))
        print(f"Error list saved to {err_path}")


if __name__ == "__main__":
    main()
