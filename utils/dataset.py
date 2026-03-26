"""
utils/dataset.py

MTG-Jamendo dataset loader.
Download instructions:
  https://github.com/MTG/mtg-jamendo-dataset

Expected directory structure:
  data/mtg_jamendo/
    autotagging_moodtheme.tsv      <- mood labels
    autotagging_genre.tsv          <- genre labels
    autotagging_subgenre.tsv       <- sub-genre labels (if available, else derived)
    splits/
      split-0/
        train.tsv
        validation.tsv
        test.tsv
    audio/                         <- raw mp3/wav files
      00/track_000000.mp3
      ...
"""

import os
import csv
import torch
import numpy as np
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from utils.hierarchy import (
    MOODS, GENRES, SUBGENRES,
    SUBGENRE_GENRE_MAP, GENRE_TO_MOOD_PRIMARY,
)


class MTGJamendoDataset(Dataset):
    """
    Multi-label dataset returning log-mel spectrograms and
    hierarchical label vectors [mood, genre, subgenre].
    """

    def __init__(
        self,
        root: str,
        split: str = "train",          # train / validation / test
        sample_rate: int = 16000,
        n_mels: int = 128,
        hop_ms: int = 10,
        win_ms: int = 25,
        max_frames: int = 1024,
        mixup_alpha: float = 0.5,      # 0 = disabled
        time_mask: int = 192,
        freq_mask: int = 48,
        augment: bool = True,
        cross_modal_emb_dir: Optional[str] = None,  # dir of precomputed .pt embeddings
    ):
        self.root       = root
        self.split      = split
        self.sr         = sample_rate
        self.n_mels     = n_mels
        self.hop_len    = int(hop_ms * sample_rate / 1000)
        self.win_len    = int(win_ms * sample_rate / 1000)
        self.max_frames = max_frames
        self.mixup_alpha= mixup_alpha
        self.time_mask  = time_mask
        self.freq_mask  = freq_mask
        self.augment    = augment and (split == "train")
        self.cross_modal_emb_dir = cross_modal_emb_dir

        # Build label vocabs
        self.mood_to_idx    = {m: i for i, m in enumerate(MOODS)}
        self.genre_to_idx   = {g: i for i, g in enumerate(GENRES)}
        self.sub_to_idx     = {s: i for i, s in enumerate(SUBGENRES)}

        # Load split TSV
        self.items = self._load_split()

        # Mel filterbank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.win_len,
            hop_length=self.hop_len,
            n_mels=n_mels,
            f_min=20,
            f_max=8000,
        )

    def _load_split(self) -> List[Dict]:
        split_file = os.path.join(
            self.root, "splits", "split-0", f"{self.split}.tsv"
        )
        items = []
        with open(split_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # Normalise column names to lowercase (skip None overflow keys)
                row = {k.lower(): v for k, v in row.items() if k is not None}
                # Only keep tracks whose audio file is present
                audio_path = os.path.join(self.root, "audio", row["path"])
                if os.path.exists(audio_path):
                    items.append(row)
        print(f"  [{self.split}] {len(items)} tracks with audio found")
        return items

    def _parse_labels(self, row: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_mood  = len(MOODS)
        n_genre = len(GENRES)
        n_sub   = len(SUBGENRES)

        mood_vec  = np.zeros(n_mood,  dtype=np.float32)
        genre_vec = np.zeros(n_genre, dtype=np.float32)
        sub_vec   = np.zeros(n_sub,   dtype=np.float32)

        # Tags are comma-separated in the TSV, prefixed with category
        tags = row.get("tags", "").split(",")
        for tag in tags:
            tag = tag.strip().lower()
            if tag.startswith("mood/theme---"):
                t = tag.replace("mood/theme---", "").replace("-", " ")
                if t in self.mood_to_idx:
                    mood_vec[self.mood_to_idx[t]] = 1.0
            elif tag.startswith("genre---"):
                t = tag.replace("genre---", "").replace("-", " ")
                if t in self.genre_to_idx:
                    genre_vec[self.genre_to_idx[t]] = 1.0
            # Sub-genre derived from genre if not in dataset
            # (In real experiment, use MusicBrainz linked sub-genre annotations)

        # Propagate genre labels downward to sub-genres based on hierarchy
        # (weak supervision: if genre is positive, all its sub-genres get 0.5 weight)
        # In a real setup: use actual sub-genre annotations when available
        for si, gi in enumerate(SUBGENRE_GENRE_MAP):
            if genre_vec[gi] > 0:
                sub_vec[si] = 0.5   # weak label

        # Propagate genre labels upward to moods (if mood annotation missing)
        for gi in range(n_genre):
            if genre_vec[gi] > 0:
                mi = GENRE_TO_MOOD_PRIMARY[gi]
                mood_vec[mi] = max(mood_vec[mi], 0.5)

        return mood_vec, genre_vec, sub_vec

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform   # (1, T)

    def _to_logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)           # (1, n_mels, T)
        log_mel = torch.log(mel + 1e-8)
        # Pad / crop to max_frames
        T = log_mel.shape[-1]
        if T < self.max_frames:
            log_mel = F.pad(log_mel, (0, self.max_frames - T))
        else:
            log_mel = log_mel[..., :self.max_frames]
        return log_mel   # (1, n_mels, max_frames)

    def _freq_time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        # SpecAugment: time + frequency masking
        C, F, T = spec.shape
        if self.time_mask > 0:
            t0 = np.random.randint(0, max(1, T - self.time_mask))
            tw = np.random.randint(0, self.time_mask)
            spec[:, :, t0:t0+tw] = 0
        if self.freq_mask > 0:
            f0 = np.random.randint(0, max(1, F - self.freq_mask))
            fw = np.random.randint(0, self.freq_mask)
            spec[:, f0:f0+fw, :] = 0
        return spec

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]

        track_id   = row["track_id"]
        audio_path = os.path.join(self.root, "audio", row["path"])

        waveform = self._load_audio(audio_path)
        spec     = self._to_logmel(waveform)          # (1, F, T)

        if self.augment:
            spec = self._freq_time_mask(spec)

        mood_vec, genre_vec, sub_vec = self._parse_labels(row)
        labels = np.concatenate([mood_vec, genre_vec, sub_vec])

        # Pad/crop waveform to fixed length (max_frames * hop_len samples)
        max_samples = self.max_frames * self.hop_len
        T = waveform.shape[-1]
        if T < max_samples:
            waveform = F.pad(waveform, (0, max_samples - T))
        else:
            waveform = waveform[..., :max_samples]

        item = {
            "spec":     spec,
            "waveform": waveform,
            "labels":   torch.tensor(labels, dtype=torch.float),
            "track_id": track_id,
        }

        # Load precomputed cross-modal embedding (CLAP or MuQ-MuLan) if available
        if self.cross_modal_emb_dir:
            rel_stem = os.path.splitext(row["path"])[0]   # e.g. "00/track_0000000"
            emb_path = os.path.join(self.cross_modal_emb_dir, rel_stem + ".pt")
            if os.path.exists(emb_path):
                item["cross_modal_emb"] = torch.load(emb_path, map_location="cpu",
                                                      weights_only=True)

        return item


def collate_mixup(batch, alpha=0.5):
    """
    Mixup augmentation at the batch level.
    Mixes pairs of (spec, waveform, label) within the batch.
    """
    specs     = torch.stack([b["spec"]     for b in batch])
    waveforms = torch.stack([b["waveform"] for b in batch])
    labels    = torch.stack([b["labels"]   for b in batch])
    ids       = [b["track_id"] for b in batch]

    # Cross-modal embeddings are optional (only present if precomputed emb dir set)
    cm_embs = None
    if "cross_modal_emb" in batch[0]:
        cm_list = [b.get("cross_modal_emb") for b in batch]
        if all(e is not None for e in cm_list):
            cm_embs = torch.stack(cm_list)

    if alpha > 0 and np.random.random() > 0.5:
        lam  = np.random.beta(alpha, alpha)
        perm = torch.randperm(specs.size(0))
        specs     = lam * specs     + (1 - lam) * specs[perm]
        waveforms = lam * waveforms + (1 - lam) * waveforms[perm]
        labels    = lam * labels    + (1 - lam) * labels[perm]
        if cm_embs is not None:
            cm_embs = lam * cm_embs + (1 - lam) * cm_embs[perm]

    out = {"spec": specs, "waveform": waveforms, "labels": labels, "track_id": ids}
    if cm_embs is not None:
        out["cross_modal_emb"] = cm_embs
    return out


def get_dataloaders(cfg):
    from functools import partial

    # Resolve cross-modal embedding dir for precomputed embeddings
    cm_emb_dir = cfg.cross_modal_emb_dir or None
    if cm_emb_dir and not os.path.isabs(cm_emb_dir):
        cm_emb_dir = os.path.join(cfg.data_root, cm_emb_dir)

    train_ds = MTGJamendoDataset(
        root=cfg.data_root, split="train",
        sample_rate=cfg.sample_rate, n_mels=cfg.n_mels,
        hop_ms=cfg.hop_ms, win_ms=cfg.win_ms,
        max_frames=cfg.max_frames,
        mixup_alpha=cfg.mixup_alpha,
        time_mask=cfg.time_mask, freq_mask=cfg.freq_mask,
        augment=True,
        cross_modal_emb_dir=cm_emb_dir,
    )
    val_ds = MTGJamendoDataset(
        root=cfg.data_root, split="validation",
        sample_rate=cfg.sample_rate, n_mels=cfg.n_mels,
        hop_ms=cfg.hop_ms, win_ms=cfg.win_ms,
        max_frames=cfg.max_frames, augment=False,
        cross_modal_emb_dir=cm_emb_dir,
    )
    test_ds = MTGJamendoDataset(
        root=cfg.data_root, split="test",
        sample_rate=cfg.sample_rate, n_mels=cfg.n_mels,
        hop_ms=cfg.hop_ms, win_ms=cfg.win_ms,
        max_frames=cfg.max_frames, augment=False,
        cross_modal_emb_dir=cm_emb_dir,
    )

    collate = partial(collate_mixup, alpha=cfg.mixup_alpha)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=4,
                              collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True)

    return train_loader, val_loader, test_loader
