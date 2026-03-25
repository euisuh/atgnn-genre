"""
train.py — main training script for H-ATGNN ablation experiments.

Usage:
    # Full model
    python train.py --text_init --hierarchy --clap

    # Ablation: no text init
    python train.py --hierarchy --clap --run_name no_text_init

    # Ablation: no hierarchy
    python train.py --text_init --clap --run_name no_hierarchy

    # Ablation: baseline (ATGNN replication)
    python train.py --run_name baseline

    # Run all 6 ablations sequentially
    python train.py --run_all_ablations
"""

import os
import argparse
import json
import time
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from configs.default import HATGNNConfig
from models.hatgnn import HATGNN, HierarchicalLoss
from utils.hierarchy import get_hierarchy_config
from utils.metrics import evaluate, print_results


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text_init",  action="store_true",
                   help="Initialise label embeddings from text LM")
    p.add_argument("--hierarchy",  action="store_true",
                   help="Use hierarchical DAG-masked LLG")
    p.add_argument("--clap",       action="store_true",
                   help="Use CLAP cross-modal fusion")
    p.add_argument("--lam",        type=float, default=0.5,
                   help="Consistency loss weight")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=24)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--run_name",   type=str,   default=None)
    p.add_argument("--data_root",  type=str,   default="data/mtg_jamendo")
    p.add_argument("--output_dir", type=str,   default="outputs")
    p.add_argument("--emb_path",   type=str,   default="embeddings/label_embs.pt",
                   help="Path to precomputed text LM embeddings")
    p.add_argument("--backbone",   type=str,   default="cnn",
                   choices=["cnn", "mert"],
                   help="Backbone: 'cnn' (ATGNN-style) or 'mert' (MERT-v1-95M)")
    p.add_argument("--run_all_ablations", action="store_true",
                   help="Run all 6 ablation configs sequentially")
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Config builder ────────────────────────────────────────────────────────────

def build_config(args) -> HATGNNConfig:
    h = get_hierarchy_config()
    backbone = getattr(args, "backbone", "cnn")
    # max_nodes differs by backbone: MERT uses pooled frames, CNN uses spatial patches
    if backbone == "mert":
        max_nodes = 512
    else:
        # CNN: 3x stride-2 → F/8 * T/8 nodes
        n_mels = 128
        max_frames = 1024
        max_nodes = (n_mels // 8) * (max_frames // 8)   # 2048
    cfg = HATGNNConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lam=args.lam,
        data_root=args.data_root,
        output_dir=args.output_dir,
        use_text_init=args.text_init,
        use_hierarchy=args.hierarchy,
        use_clap=args.clap,
        backbone=backbone,
        max_nodes=max_nodes,
        **h,
    )
    return cfg


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int,
                     halve_every: int = 5, halve_after: int = 10):
    """
    Linear warmup then halve every N epochs after epoch M.
    Matches ATGNN training protocol.
    """
    def lr_lambda(step):
        epoch = step // max(1, total_steps // 50)   # approx epoch
        if step < warmup_steps:
            return step / warmup_steps
        halves = max(0, epoch - halve_after) // halve_every
        return 0.5 ** halves

    return LambdaLR(optimizer, lr_lambda)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, use_clap=False):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        spec     = batch["spec"].to(device)
        waveform = batch["waveform"].to(device)
        labels   = batch["labels"].to(device)
        clap_e   = batch.get("clap_emb")
        if clap_e is not None:
            clap_e = clap_e.to(device)
        elif use_clap:
            clap_e = None

        preds = model(spec, clap_e, waveform=waveform)
        loss  = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_epoch(model, loader, device, cfg):
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        spec     = batch["spec"].to(device)
        waveform = batch["waveform"].to(device)
        labels   = batch["labels"].numpy()
        clap_e   = batch.get("clap_emb")
        if clap_e is not None:
            clap_e = clap_e.to(device)

        preds = model(spec, clap_e, waveform=waveform).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels)

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return evaluate(
        preds, targets,
        cfg.n_mood, cfg.n_genre, cfg.n_subgenre,
        cfg.genre_to_mood, cfg.subgenre_to_genre,
    )


# ── Single experiment run ─────────────────────────────────────────────────────

def run_experiment(cfg, run_name: str, device: str,
                   emb_path: str = None) -> Dict:
    from utils.dataset import get_dataloaders

    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  text_init={cfg.use_text_init}  hierarchy={cfg.use_hierarchy}  clap={cfg.use_clap}")
    print(f"{'='*60}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ── Data ──
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # ── Model ──
    model = HATGNN(cfg).to(device)

    # Text LM initialisation
    if cfg.use_text_init and emb_path and os.path.exists(emb_path):
        from utils.text_embeddings import load_label_embeddings
        mood_v, genre_v, sub_v = load_label_embeddings(emb_path)
        model.initialise_from_text_embeddings(mood_v, genre_v, sub_v)
        print(f"  Loaded text embeddings from {emb_path}")
    elif cfg.use_text_init:
        print("  WARNING: --text_init set but emb_path not found. "
              "Run utils/text_embeddings.py first.")

    # ── Loss ──
    criterion = HierarchicalLoss(
        n_mood=cfg.n_mood,
        n_genre=cfg.n_genre,
        n_subgenre=cfg.n_subgenre,
        genre_to_mood=cfg.genre_to_mood,
        subgenre_to_genre=cfg.subgenre_to_genre,
        lam=cfg.lam if cfg.use_hierarchy else 0.0,
    ).to(device)

    # ── Optimiser ──
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler   = get_lr_scheduler(optimizer, cfg.lr_warmup, total_steps)

    # ── Training ──
    best_map  = 0.0
    best_ckpt = os.path.join(run_dir, "best.pt")
    history   = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                      optimizer, scheduler, device,
                                      use_clap=cfg.use_clap)
        val_metrics = evaluate_epoch(model, val_loader, device, cfg)
        elapsed     = time.time() - t0

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)

        print(f"  Epoch {epoch:3d}/{cfg.epochs}  "
              f"loss={train_loss:.4f}  "
              f"mAP={val_metrics['mAP_all']:.4f}  "
              f"genre={val_metrics['mAP_genre']:.4f}  "
              f"mood={val_metrics['mAP_mood']:.4f}  "
              f"({elapsed:.1f}s)")

        if val_metrics["mAP_all"] > best_map:
            best_map = val_metrics["mAP_all"]
            torch.save(model.state_dict(), best_ckpt)

    # ── Test evaluation ──
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = evaluate_epoch(model, test_loader, device, cfg)
    print_results(test_metrics, prefix=f"TEST — {run_name}")

    # ── Save results ──
    results = {
        "run_name":    run_name,
        "config": {
            "text_init": cfg.use_text_init,
            "hierarchy": cfg.use_hierarchy,
            "clap":      cfg.use_clap,
            "lam":       cfg.lam,
        },
        "best_val_mAP":  best_map,
        "test_metrics":  test_metrics,
        "history":       history,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Ablation suite ────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    # (name,            text_init, hierarchy, clap)
    ("baseline",        False,     False,     False),
    ("text_init_only",  True,      False,     False),
    ("hierarchy_only",  False,     True,      False),
    ("clap_only",       False,     False,     True),
    ("text_hierarchy",  True,      True,      False),
    ("full_hatgnn",     True,      True,      True),
]


def run_all_ablations(base_args):
    all_results = []
    for name, ti, hi, cl in ABLATION_CONFIGS:
        args = deepcopy(base_args)
        args.text_init = ti
        args.hierarchy = hi
        args.clap      = cl
        cfg = build_config(args)
        results = run_experiment(cfg, name, base_args.device, base_args.emb_path)
        all_results.append(results)

    # Print ablation summary table
    print("\n\nABLATION SUMMARY")
    print(f"{'Run':<22} {'Text':>6} {'Hier':>6} {'CLAP':>6} "
          f"{'mAP':>8} {'Genre':>8} {'Mood':>8} {'Cons':>8}")
    print("─" * 78)
    for r in all_results:
        c = r["config"]
        m = r["test_metrics"]
        print(f"{r['run_name']:<22} {str(c['text_init']):>6} {str(c['hierarchy']):>6} "
              f"{str(c['clap']):>6} "
              f"{m['mAP_all']:>8.4f} {m['mAP_genre']:>8.4f} "
              f"{m['mAP_mood']:>8.4f} {m['consistency_overall']:>8.4f}")

    # Save summary
    with open("outputs/ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to outputs/ablation_summary.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from typing import Dict
    args = parse_args()

    if args.run_all_ablations:
        run_all_ablations(args)
    else:
        cfg = build_config(args)
        run_name = args.run_name or (
            f"ti{int(args.text_init)}_hi{int(args.hierarchy)}_cl{int(args.clap)}"
        )
        run_experiment(cfg, run_name, args.device, args.emb_path)
