"""
Experiment configuration for H-ATGNN.
Edit configs/default.py or pass a YAML override to train.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class HATGNNConfig:
    # ── Model dims ──────────────────────────────────────────────────────────
    patch_dim:  int = 128      # CNN backbone output channels
    label_dim:  int = 128      # label embedding dimension
    clap_dim:   int = 512      # CLAP audio embedding dim (LAION-CLAP default)
    max_nodes:  int = 256      # max patch nodes (F/8 * T/8)

    # ── Graph ───────────────────────────────────────────────────────────────
    k:      int = 9            # k-NN for PGN
    k_plg:  int = 9            # k-NN for patch->label edges
    n_pgn:  int = 4            # number of PGN blocks

    # ── Label hierarchy ─────────────────────────────────────────────────────
    n_mood:     int = 8        # e.g. Joyful, Tense, Melancholic, Energetic,
                               #      Calm, Dark, Romantic, Aggressive
    n_genre:    int = 15       # e.g. Jazz, Rock, Classical, Electronic ...
    n_subgenre: int = 50       # e.g. Bebop, Post-bop, Indie Rock ...

    # These are filled by build_hierarchy() in utils/hierarchy.py
    hierarchy_mask:   Optional[torch.Tensor] = None
    genre_to_mood:    Optional[List[int]]    = None
    subgenre_to_genre:Optional[List[int]]    = None

    # ── Training ────────────────────────────────────────────────────────────
    lr:           float = 5e-4
    lr_warmup:    int   = 1000       # steps
    batch_size:   int   = 24
    epochs:       int   = 50
    lam:          float = 0.5        # consistency loss weight
    mixup_alpha:  float = 0.5
    time_mask:    int   = 192        # frames
    freq_mask:    int   = 48         # bins

    # ── Data ────────────────────────────────────────────────────────────────
    sample_rate:  int   = 16000
    n_mels:       int   = 128
    hop_ms:       int   = 10
    win_ms:       int   = 25
    max_frames:   int   = 1024

    # ── Ablation switches ────────────────────────────────────────────────────
    use_text_init:  bool = True   # initialise label embs from text LM
    use_hierarchy:  bool = True   # use DAG-masked LLG
    use_clap:       bool = True   # use CLAP cross-modal fusion

    # ── Paths ────────────────────────────────────────────────────────────────
    data_root:   str = "data/mtg_jamendo"
    output_dir:  str = "outputs"
    clap_ckpt:   str = "laion/clap-htsat-fused"   # HuggingFace model id
    text_lm_id:  str = "sentence-transformers/all-mpnet-base-v2"
