"""
train.py — main training script for H-ATGNN ablation experiments.

Usage:
    # Full model with W&B logging
    python train.py --text_init --hierarchy --cross_modal clap \
        --wandb_project hatgnn --wandb_entity YOUR_ENTITY

    # Ablation: no text init
    python train.py --hierarchy --cross_modal clap --run_name no_text_init

    # Ablation: baseline (ATGNN replication)
    python train.py --run_name baseline

    # Run all 6 CNN ablations sequentially (Table 1)
    python train.py --run_all_ablations --wandb_project hatgnn

    # Run all SSL backbone experiments (Table 2)
    python train.py --run_ssl_experiments --wandb_project hatgnn

    # MuQ backbone
    python train.py --backbone muq --text_init --hierarchy \
        --wandb_project hatgnn

    # MuQ + MuQ-MuLan (Exp 3)
    python train.py --backbone muq --cross_modal muqmulan \
        --text_init --hierarchy --wandb_project hatgnn
"""

import os
import argparse
import json
import time
import dataclasses
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from configs.default import HATGNNConfig
from models.hatgnn import HATGNN, HierarchicalLoss
from utils.hierarchy import get_hierarchy_config
from utils.metrics import evaluate, print_results


# ── W&B helper ────────────────────────────────────────────────────────────────

def _wandb_init(project: Optional[str], entity: Optional[str],
                run_name: str, cfg: HATGNNConfig, tags=None):
    """
    Initialise a W&B run. Returns the run object (or None if wandb is disabled).
    Logs the full HATGNNConfig as hyperparameters.
    """
    if not project:
        return None

    import wandb

    # Flatten HATGNNConfig to a plain dict — skip non-serialisable fields
    skip = {"hierarchy_mask", "genre_to_mood", "subgenre_to_genre"}
    cfg_dict = {
        k: v for k, v in dataclasses.asdict(cfg).items()
        if k not in skip
    }

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags or [],
        config=cfg_dict,
        reinit=True,
    )
    return run


def _wandb_log_epoch(run, epoch: int, train_loss: float, val_metrics: dict,
                     lr: float, grad_norm: float, elapsed: float):
    """Log all per-epoch metrics to W&B."""
    if run is None:
        return

    import wandb
    wandb.log({
        "epoch":               epoch,
        "train/loss":          train_loss,
        "train/lr":            lr,
        "train/grad_norm":     grad_norm,
        "val/mAP_all":         val_metrics["mAP_all"],
        "val/mAP_mood":        val_metrics["mAP_mood"],
        "val/mAP_genre":       val_metrics["mAP_genre"],
        "val/mAP_subgenre":    val_metrics["mAP_subgenre"],
        "val/consistency_genre_mood": val_metrics["consistency_genre_mood"],
        "val/consistency_sub_genre":  val_metrics["consistency_sub_genre"],
        "val/consistency_overall":    val_metrics["consistency_overall"],
        "val/rare_subgenre_gap":      val_metrics.get("mAP_subgenre_common", 0)
                                    - val_metrics.get("mAP_subgenre_rare", 0),
        "epoch_time_s":        elapsed,
    }, step=epoch)


def _wandb_log_test(run, test_metrics: dict, best_val_map: float):
    """Log final test metrics as W&B summary values."""
    if run is None:
        return

    import wandb
    wandb.summary["test/mAP_all"]              = test_metrics["mAP_all"]
    wandb.summary["test/mAP_mood"]             = test_metrics["mAP_mood"]
    wandb.summary["test/mAP_genre"]            = test_metrics["mAP_genre"]
    wandb.summary["test/mAP_subgenre"]         = test_metrics["mAP_subgenre"]
    wandb.summary["test/consistency_overall"]  = test_metrics["consistency_overall"]
    wandb.summary["test/rare_subgenre_gap"]    = (
        test_metrics.get("mAP_subgenre_common", 0)
        - test_metrics.get("mAP_subgenre_rare", 0)
    )
    wandb.summary["best_val_mAP"]              = best_val_map


def _wandb_log_model_info(run, model: torch.nn.Module):
    """Log parameter counts broken down by component."""
    if run is None:
        return

    import wandb

    def count_params(module):
        total     = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    total, trainable = count_params(model)
    info = {
        "model/total_params":     total,
        "model/trainable_params": trainable,
        "model/frozen_params":    total - trainable,
    }

    # Per-component breakdown
    for name, mod in model.named_children():
        t, tr = count_params(mod)
        info[f"model/params_{name}"] = t
        info[f"model/trainable_{name}"] = tr

    wandb.log(info, step=0)


def _wandb_summary_table(run, all_results: list, table_name: str):
    """Log a summary comparison table as a W&B artifact."""
    if run is None:
        return

    import wandb

    columns = ["run", "backbone", "text_init", "hierarchy", "cross_modal",
               "mAP_all", "mAP_genre", "mAP_mood", "mAP_subgenre",
               "consistency", "rare_gap"]
    rows = []
    for r in all_results:
        c = r["config"]
        m = r["test_metrics"]
        rows.append([
            r["run_name"],
            c.get("backbone", "cnn"),
            c["text_init"],
            c["hierarchy"],
            c["cross_modal"],
            round(m["mAP_all"],     4),
            round(m["mAP_genre"],   4),
            round(m["mAP_mood"],    4),
            round(m["mAP_subgenre"], 4),
            round(m["consistency_overall"], 4),
            round(m.get("mAP_subgenre_common", 0)
                  - m.get("mAP_subgenre_rare", 0), 4),
        ])

    table = wandb.Table(columns=columns, data=rows)
    wandb.log({table_name: table})


def _wandb_upload_checkpoint(run, ckpt_path: str, run_name: str):
    """Upload best.pt as a W&B artifact so it survives instance shutdown."""
    if run is None or not os.path.exists(ckpt_path):
        return
    import wandb
    artifact = wandb.Artifact(
        name=f"{run_name}-checkpoint",
        type="model",
        description=f"Best validation checkpoint for run {run_name}",
    )
    artifact.add_file(ckpt_path, name="best.pt")
    run.log_artifact(artifact)
    print(f"  Checkpoint uploaded to W&B artifacts: {run_name}-checkpoint")


def _wandb_finish(run):
    if run is not None:
        import wandb
        wandb.finish()


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text_init",  action="store_true",
                   help="Initialise label embeddings from text LM")
    p.add_argument("--hierarchy",  action="store_true",
                   help="Use hierarchical DAG-masked LLG")
    p.add_argument("--clap",       action="store_true",
                   help="(Deprecated) alias for --cross_modal clap")
    p.add_argument("--lam",        type=float, default=0.5,
                   help="Consistency loss weight")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=24)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--run_name",   type=str,   default=None)
    p.add_argument("--data_root",  type=str,   default="data/mtg_jamendo")
    p.add_argument("--output_dir", type=str,   default="outputs")
    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to last.pt to resume training from")
    p.add_argument("--emb_path",   type=str,   default="embeddings/label_embs.pt",
                   help="Path to precomputed text LM embeddings")
    p.add_argument("--backbone",   type=str,   default="cnn",
                   choices=["cnn", "mert", "muq"],
                   help="Audio backbone: cnn | mert (MERT-v1-95M) | muq (MuQ)")
    p.add_argument("--cross_modal", type=str,  default="none",
                   choices=["none", "clap", "muqmulan"],
                   help="Cross-modal fusion: none | clap | muqmulan (MuQ-MuLan)")
    p.add_argument("--run_all_ablations", action="store_true",
                   help="Run all 6 CNN ablation configs sequentially (Table 1)")
    p.add_argument("--run_ssl_experiments", action="store_true",
                   help="Run all SSL backbone experiments (Table 2: MERT + MuQ)")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # ── W&B ──
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project name. If not set, W&B logging is disabled.")
    p.add_argument("--wandb_entity",  type=str, default=None,
                   help="W&B entity (team or username). Defaults to your default entity.")
    p.add_argument("--wandb_tags",    type=str, default="",
                   help="Comma-separated W&B tags, e.g. 'table1,cnn,ablation'")

    return p.parse_args()


# ── Config builder ────────────────────────────────────────────────────────────

def build_config(args) -> HATGNNConfig:
    h = get_hierarchy_config()
    backbone    = getattr(args, "backbone", "cnn")
    cross_modal = getattr(args, "cross_modal", "none")

    # Backward compat: --clap flag maps to --cross_modal clap
    if getattr(args, "clap", False) and cross_modal == "none":
        cross_modal = "clap"

    # max_nodes: SSL backbones pool to 512 frames; CNN uses spatial patches
    if backbone in ("mert", "muq"):
        max_nodes = 512
    else:
        n_mels, max_frames = 128, 1024
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
        cross_modal=cross_modal,
        backbone=backbone,
        max_nodes=max_nodes,
        **h,
    )
    # Attach resume path (not a dataclass field, just a dynamic attr)
    cfg.resume = getattr(args, "resume", None)
    return cfg


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int,
                     halve_every: int = 5, halve_after: int = 10):
    """
    Linear warmup then halve every N epochs after epoch M.
    Matches ATGNN training protocol.
    """
    def lr_lambda(step):
        epoch  = step // max(1, total_steps // 50)
        if step < warmup_steps:
            return step / warmup_steps
        halves = max(0, epoch - halve_after) // halve_every
        return 0.5 ** halves

    return LambdaLR(optimizer, lr_lambda)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss  = 0.0
    total_gnorm = 0.0
    n_batches   = 0

    for batch in loader:
        spec     = batch["spec"].to(device)
        waveform = batch["waveform"].to(device)
        labels   = batch["labels"].to(device)
        cm_emb   = batch.get("cross_modal_emb")
        if cm_emb is not None:
            cm_emb = cm_emb.to(device)

        preds = model(spec, cm_emb, waveform=waveform)
        loss  = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
        total_gnorm += gnorm
        n_batches   += 1

    n = max(1, n_batches)
    return total_loss / n, total_gnorm / n


@torch.no_grad()
def evaluate_epoch(model, loader, device, cfg):
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        spec     = batch["spec"].to(device)
        waveform = batch["waveform"].to(device)
        labels   = batch["labels"].numpy()
        cm_emb   = batch.get("cross_modal_emb")
        if cm_emb is not None:
            cm_emb = cm_emb.to(device)

        preds = model(spec, cm_emb, waveform=waveform).cpu().numpy()
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
                   emb_path: str = None,
                   wandb_project: str = None,
                   wandb_entity: str = None,
                   wandb_tags: list = None) -> Dict:
    from utils.dataset import get_dataloaders

    print(f"\n{'='*60}")
    print(f"  Run: {run_name}")
    print(f"  backbone={cfg.backbone}  text_init={cfg.use_text_init}  "
          f"hierarchy={cfg.use_hierarchy}  cross_modal={cfg.cross_modal}")
    print(f"{'='*60}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    run_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ── W&B init ──
    wb_run = _wandb_init(wandb_project, wandb_entity, run_name, cfg,
                         tags=wandb_tags)

    # ── Data ──
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # ── Model ──
    model = HATGNN(cfg).to(device)

    # Log model parameter breakdown
    _wandb_log_model_info(wb_run, model)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_total:,} total  {n_trainable:,} trainable")

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
    optimizer   = optim.Adam(model.parameters(), lr=cfg.lr)
    total_steps = len(train_loader) * cfg.epochs
    scheduler   = get_lr_scheduler(optimizer, cfg.lr_warmup, total_steps)

    # ── Checkpoint paths ──
    best_ckpt = os.path.join(run_dir, "best.pt")
    last_ckpt = os.path.join(run_dir, "last.pt")

    # ── Resume from checkpoint ──
    start_epoch = 1
    best_map    = 0.0
    history     = []

    resume_path = getattr(cfg, "resume", None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_map    = ckpt.get("best_map", 0.0)
        history     = ckpt.get("history", [])
        print(f"  Resumed from {resume_path} (epoch {ckpt['epoch']}, best_mAP={best_map:.4f})")
    elif resume_path:
        print(f"  WARNING: --resume path not found: {resume_path}. Starting fresh.")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        train_loss, grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_metrics = evaluate_epoch(model, val_loader, device, cfg)
        elapsed     = time.time() - t0

        # Current LR (from last scheduler step)
        current_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch, "train_loss": train_loss,
            "grad_norm": grad_norm, "lr": current_lr,
            **val_metrics,
        }
        history.append(row)

        # ── W&B per-epoch log ──
        _wandb_log_epoch(wb_run, epoch, train_loss, val_metrics,
                         current_lr, grad_norm, elapsed)

        print(f"  Epoch {epoch:3d}/{cfg.epochs}  "
              f"loss={train_loss:.4f}  "
              f"mAP={val_metrics['mAP_all']:.4f}  "
              f"genre={val_metrics['mAP_genre']:.4f}  "
              f"mood={val_metrics['mAP_mood']:.4f}  "
              f"gnorm={grad_norm:.3f}  lr={current_lr:.2e}  "
              f"({elapsed:.1f}s)")

        if val_metrics["mAP_all"] > best_map:
            best_map = val_metrics["mAP_all"]
            torch.save(model.state_dict(), best_ckpt)

        # Save last checkpoint every epoch (enables resume)
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map":  best_map,
            "history":   history,
        }, last_ckpt)

    # ── Test evaluation ──
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = evaluate_epoch(model, test_loader, device, cfg)
    print_results(test_metrics, prefix=f"TEST — {run_name}")

    # ── W&B test summary + checkpoint upload ──
    _wandb_log_test(wb_run, test_metrics, best_map)
    _wandb_upload_checkpoint(wb_run, best_ckpt, run_name)

    # ── Save results ──
    results = {
        "run_name":    run_name,
        "config": {
            "backbone":    cfg.backbone,
            "text_init":   cfg.use_text_init,
            "hierarchy":   cfg.use_hierarchy,
            "cross_modal": cfg.cross_modal,
            "lam":         cfg.lam,
        },
        "best_val_mAP": best_map,
        "test_metrics": test_metrics,
        "history":      history,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _wandb_finish(wb_run)
    return results


# ── Experiment suites ─────────────────────────────────────────────────────────

# Table 1: CNN ablation (6 runs)
CNN_ABLATION_CONFIGS = [
    # (name,            text_init, hierarchy, cross_modal)
    ("baseline",        False, False, "none"),
    ("text_init_only",  True,  False, "none"),
    ("hierarchy_only",  False, True,  "none"),
    ("clap_only",       False, False, "clap"),
    ("text_hierarchy",  True,  True,  "none"),
    ("full_hatgnn",     True,  True,  "clap"),
]

# Table 2: SSL backbone experiments (5 runs)
SSL_CONFIGS = [
    # (name,                 backbone, text_init, hierarchy, cross_modal)
    ("mert_baseline",        "mert",  False, False, "none"),
    ("hatgnn_mert",          "mert",  True,  True,  "clap"),
    ("muq_baseline",         "muq",   False, False, "none"),
    ("hatgnn_muq",           "muq",   True,  True,  "clap"),
    ("hatgnn_muq_muqmulan",  "muq",   True,  True,  "muqmulan"),
]


def run_all_ablations(base_args):
    all_results = []
    tags = ([t.strip() for t in base_args.wandb_tags.split(",") if t.strip()]
            + ["table1", "cnn"])

    for name, ti, hi, cm in CNN_ABLATION_CONFIGS:
        args = deepcopy(base_args)
        args.text_init    = ti
        args.hierarchy    = hi
        args.cross_modal  = cm
        args.backbone     = "cnn"
        cfg = build_config(args)
        results = run_experiment(
            cfg, name, base_args.device, base_args.emb_path,
            wandb_project=base_args.wandb_project,
            wandb_entity=base_args.wandb_entity,
            wandb_tags=tags + [name],
        )
        all_results.append(results)

    # Console summary
    print("\n\nABLATION SUMMARY (Table 1 — CNN)")
    print(f"{'Run':<22} {'Text':>6} {'Hier':>6} {'CrossMod':>10} "
          f"{'mAP':>8} {'Genre':>8} {'Mood':>8} {'Cons':>8}")
    print("─" * 82)
    for r in all_results:
        c = r["config"]
        m = r["test_metrics"]
        print(f"{r['run_name']:<22} {str(c['text_init']):>6} {str(c['hierarchy']):>6} "
              f"{c['cross_modal']:>10} "
              f"{m['mAP_all']:>8.4f} {m['mAP_genre']:>8.4f} "
              f"{m['mAP_mood']:>8.4f} {m['consistency_overall']:>8.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to outputs/ablation_summary.json")

    # W&B summary table (logged on last run's entity for visibility)
    if base_args.wandb_project:
        import wandb
        run = wandb.init(
            project=base_args.wandb_project,
            entity=base_args.wandb_entity,
            name="ablation_summary_table1",
            tags=tags,
            reinit=True,
        )
        _wandb_summary_table(run, all_results, "table1_ablation")
        wandb.finish()


def run_ssl_experiments(base_args):
    all_results = []
    tags = ([t.strip() for t in base_args.wandb_tags.split(",") if t.strip()]
            + ["table2", "ssl"])

    for name, bb, ti, hi, cm in SSL_CONFIGS:
        args = deepcopy(base_args)
        args.backbone     = bb
        args.text_init    = ti
        args.hierarchy    = hi
        args.cross_modal  = cm
        cfg = build_config(args)
        results = run_experiment(
            cfg, name, base_args.device, base_args.emb_path,
            wandb_project=base_args.wandb_project,
            wandb_entity=base_args.wandb_entity,
            wandb_tags=tags + [bb, name],
        )
        all_results.append(results)

    # Console summary
    print("\n\nSSL BACKBONE SUMMARY (Table 2)")
    print(f"{'Run':<26} {'BB':>6} {'Text':>6} {'Hier':>6} {'CrossMod':>10} "
          f"{'mAP':>8} {'Genre':>8} {'Mood':>8}")
    print("─" * 88)
    for r in all_results:
        c = r["config"]
        m = r["test_metrics"]
        print(f"{r['run_name']:<26} {c['backbone']:>6} {str(c['text_init']):>6} "
              f"{str(c['hierarchy']):>6} {c['cross_modal']:>10} "
              f"{m['mAP_all']:>8.4f} {m['mAP_genre']:>8.4f} "
              f"{m['mAP_mood']:>8.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ssl_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to outputs/ssl_summary.json")

    # W&B summary table
    if base_args.wandb_project:
        import wandb
        run = wandb.init(
            project=base_args.wandb_project,
            entity=base_args.wandb_entity,
            name="ssl_summary_table2",
            tags=tags,
            reinit=True,
        )
        _wandb_summary_table(run, all_results, "table2_ssl")
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    if args.run_all_ablations:
        run_all_ablations(args)
    elif args.run_ssl_experiments:
        run_ssl_experiments(args)
    else:
        cfg = build_config(args)
        run_name = args.run_name or (
            f"{cfg.backbone}_ti{int(cfg.use_text_init)}"
            f"_hi{int(cfg.use_hierarchy)}_cm{cfg.cross_modal}"
        )
        run_experiment(
            cfg, run_name, args.device, args.emb_path,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_tags=wandb_tags,
        )
