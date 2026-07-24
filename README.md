# H-ATGNN — Hierarchical Audio Tagging Graph Neural Network

Research code for hierarchical music auto-tagging on
[MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/): a graph neural network
that jointly predicts **mood → genre → sub-genre** labels, with the label
hierarchy encoded directly in the message-passing graph.

The model extends **ATGNN** (Singh et al., *Audio Tagging Graph Neural Network*,
IEEE Signal Processing Letters, 2024) along three axes — text-initialised label
embeddings, a DAG-masked label hierarchy, and cross-modal (CLAP / MuQ-MuLan)
fusion — each of which is independently ablatable from the CLI.

> **Status:** active research code. The full training pipeline (data → precompute
> → train → evaluate) runs end-to-end on GPU; final experiment results are not yet
> committed to this repo, so no metrics are reported below. Session-by-session
> engineering notes live in [`reports/`](reports/).

---

## Research context

| | |
|---|---|
| **Base method** | ATGNN (Singh et al., IEEE SPL 2024) — patch-graph + patch-label-graph audio tagging |
| **This work** | H-ATGNN: adds a label ontology (8 moods → 15 genres → 48 sub-genres) as a DAG mask over the label-label graph, plus text-LM label initialisation and cross-modal fusion |
| **Dataset** | MTG-Jamendo (autotagging genre + mood/theme subsets, official split-0) |
| **Target claim** | Hierarchical message passing propagates signal from well-represented parent genres to rare sub-genres, narrowing the common-vs-rare mAP gap. Measured by `compute_rare_class_metrics` and `compute_consistency` in `utils/metrics.py` |
| **Write-up** | Manuscript prepared for KSMI / OpenReview submission (not in this repo) |

### Architecture

```
Audio ─► CNN | MERT | MuQ backbone ─► PGN blocks (k-NN patch graph) ─┐
Audio ─► CLAP / MuQ-MuLan encoder ──────────────────────────────────┤─► Gated fusion
Labels ─► Sentence-Transformer (frozen) ────────────────────────────┘        │
                                                                             ▼
                                              H-PLG (level-wise patch→label) ─► H-LLG
                                              (DAG-masked label→label)        ─► heads
```

Ablation axes (`train.py` flags):

- `--text_init` — initialise label embeddings from `all-mpnet-base-v2` instead of randomly
- `--hierarchy` — DAG-masked H-LLG + level-wise H-PLG message passing
- `--clap` / `--cross_modal {clap,muqmulan}` — gated fusion of patch features with a
  pretrained audio-text embedding
- `--backbone {cnn,mert,muq}` — CNN (Table 1) vs. self-supervised backbones (Table 2)

Loss: BCE over all labels + `--lam`-weighted hierarchical consistency term
(`HierarchicalLoss`, `models/hatgnn.py`) that penalises child predictions
exceeding their parent's.

---

## Repository structure

```
train.py                        Training loop, ablation runner, CLI
models/
  hatgnn.py                     HATGNN, PGN / H-PLG / H-LLG blocks, backbones,
                                GatedFusion, MuQLanEmbedder, HierarchicalLoss
utils/
  hierarchy.py                  Mood→genre→sub-genre ontology + DAG mask builder
  dataset.py                    MTG-Jamendo loader (cached specs, mixup, SpecAugment)
  text_embeddings.py            Sentence-Transformer label embedding generator
  metrics.py                    mAP, hierarchy consistency, rare-class metrics
configs/default.py              HATGNNConfig dataclass (all hyperparameters)
scripts/
  download_mtg_jamendo.py       Dataset downloader / verifier
  precompute_spectrograms.py    Offline log-mel cache (.spec.pt)
  precompute_clap.py            Offline CLAP audio embeddings
  precompute_muqmulan.py        Offline MuQ-MuLan audio embeddings
  verify_setup.py               Environment checker
cloud/
  setup.sh                      Dependency installer (auto-detects CUDA version)
  download_data.sh              Dataset download to persistent volume + symlinks
  run_experiment.sh             Launcher: data check → embeddings → train
  run_chain.sh                  Smoke test → wait for data → experiments → shutdown
reports/                        Dated engineering session reports
```

---

## Setup

Python 3.10+, PyTorch 2.x. PyTorch Geometric extensions must match your CUDA build.

```bash
pip install torch torchaudio
pip install -r requirements.txt

# PyG extensions — pick the index matching your torch/CUDA build, e.g. torch 2.4 + cu124:
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
# CPU-only: -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

`torch_cluster` is **required** (not optional): `PGNBlock` calls `knn_graph` on
every forward pass.

Verify:

```bash
python scripts/verify_setup.py
```

Optional: `pip install laion-clap` for the CLAP backend, `wandb` for logging
(`--wandb_project`).

---

## Reproduction

### 1. Data

```bash
# Metadata + official splits only (~50 MB)
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --meta_only

# Audio, one chunk (~3.2 GB, enough for a smoke test)
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --chunks 0

# Verify
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --verify
```

Expected layout:

```
data/mtg_jamendo/
  autotagging_moodtheme.tsv
  autotagging_genre.tsv
  splits/split-0/{train,validation,test}.tsv
  audio/00/track_0000000.mp3 ...
```

### 2. Precompute (strongly recommended)

Decoding mp3 + computing mel spectrograms at training time is the dominant
bottleneck — the GPU sits idle waiting on the dataloader. Cache them once:

```bash
python scripts/precompute_spectrograms.py --data_root data/mtg_jamendo
```

`dataset.py` picks up `<track>.spec.pt` automatically and falls back to on-the-fly
decoding when a cache file is missing.

Cross-modal embeddings (only needed for `--cross_modal`):

```bash
python scripts/precompute_clap.py      --data_root data/mtg_jamendo
python scripts/precompute_muqmulan.py  --data_root data/mtg_jamendo
```

### 3. Label text embeddings (run once)

```bash
python -m utils.text_embeddings \
    --model sentence-transformers/all-mpnet-base-v2 \
    --output embeddings/label_embs.pt \
    --analyse          # nearest-neighbour sanity check
```

### 4. Train

```bash
# Smoke test — a few steps, verifies the whole path
python train.py --epochs 1 --max_steps 20 --batch_size 8

# Full model
python train.py --text_init --hierarchy --clap --epochs 50 --run_name full_hatgnn

# Table 1 — all six CNN ablations, sequentially
python train.py --run_all_ablations --epochs 50

# Table 2 — self-supervised backbones
python train.py --run_ssl_experiments --epochs 50

# Consistency-loss weight sweep
for lam in 0.1 0.3 0.5 1.0; do
    python train.py --text_init --hierarchy --clap --lam $lam --run_name "lam_${lam}"
done
```

Individual ablations:

| Run | Command |
|---|---|
| Baseline (ATGNN replication) | `python train.py --run_name baseline` |
| + text init | `python train.py --text_init --run_name text_init_only` |
| + hierarchy | `python train.py --hierarchy --run_name hierarchy_only` |
| + CLAP | `python train.py --clap --run_name clap_only` |
| text + hierarchy | `python train.py --text_init --hierarchy --run_name text_hierarchy` |
| full | `python train.py --text_init --hierarchy --clap --run_name full_hatgnn` |

Training checkpoints every epoch to `outputs/{run_name}/last.pt`; resume with
`--resume outputs/{run_name}/last.pt`.

### 5. Evaluate

Metrics are computed at the end of each run and written to
`outputs/{run_name}/results.json` (overall / genre / mood mAP, hierarchy
consistency, common-vs-rare sub-genre gap). `--run_all_ablations` additionally
prints a summary table and writes `outputs/ablation_summary.json`.

---

## Running on a cloud GPU

Local CPU training is smoke-test only. The `cloud/` scripts target RunPod (and
work on any box with a persistent volume). Set `STORAGE_ROOT` to the volume mount
path — `/workspace` on RunPod pods, `/runpod-volume` for a Network Volume,
`/home/ubuntu/storage` on Lambda Labs.

```bash
git clone https://github.com/euisuh/atgnn-genre.git && cd atgnn-genre
bash cloud/setup.sh                       # deps, auto-detects CUDA

export STORAGE_ROOT=/workspace
bash cloud/download_data.sh               # 20 stratified chunks (~64 GB), default
# --small (10 chunks) | --full (all 100) | --chunks 0 5 10 | --meta_only

bash cloud/run_experiment.sh --run_all_ablations --epochs 50
```

Chunks are downloaded evenly spaced rather than sequentially: MTG-Jamendo orders
tracks by ID, so consecutive chunks skew toward a narrow slice of the catalogue.

Fully automated pipeline (smoke test → wait for download → experiments → shutdown):

```bash
AUTO_SHUTDOWN=1 bash cloud/run_chain.sh --table 1
```

Data, embeddings and outputs live under `${STORAGE_ROOT}` and are symlinked into
the repo, so they survive pod termination. Pull results back with:

```bash
rsync -avz <user>@<host>:${STORAGE_ROOT}/outputs/ outputs/
```

See [`cloud/README.md`](cloud/README.md) for instance-type recommendations.

---

## Contributions

- **H-LLG** — label-label graph attention masked by a mood→genre→sub-genre DAG, so
  a label only exchanges messages with its ontological neighbours instead of all
  labels (`HierarchicalLLGBlock`, `utils/hierarchy.build_hierarchy`).
- **H-PLG** — level-wise patch→label message passing, with separate k-NN
  neighbourhoods per hierarchy level.
- **Text-initialised label nodes** — label embeddings seeded from a frozen
  sentence-transformer, giving the label graph semantic structure before training.
- **Cross-modal gated fusion** — CLAP or MuQ-MuLan clip embeddings fused into
  patch features through a learned gate, with an offline precompute path so the
  frozen encoder never runs in the training loop.
- **Hierarchical consistency loss** — penalises a child label scoring above its
  parent; reported as an explicit consistency metric alongside mAP.
- **Reproduction harness** — single-flag ablations, one-command cloud runs, offline
  spectrogram/embedding caches, resumable checkpoints.

Engineering decisions and their rationale are logged per session in
[`reports/`](reports/).

---

## Citation

This repository builds on:

- Singh et al., *ATGNN: Audio Tagging Graph Neural Network*, IEEE Signal Processing Letters, 2024
- Bogdanov et al., *The MTG-Jamendo Dataset for Automatic Music Tagging*, ICML ML4MD Workshop, 2019
- Wu et al., *Large-Scale Contrastive Language-Audio Pretraining (CLAP)*, ICASSP 2023
- Zhu et al., *MuQ / MuQ-MuLan: Self-Supervised Music Representation Learning*, 2025
