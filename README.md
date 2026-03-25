# H-ATGNN: Hierarchical Audio Tagging Graph Neural Network

Extension of ATGNN (Singh et al., 2024) for hierarchical music genre classification
using graph embeddings, audio embeddings, and text embeddings.

## Architecture overview

```
Audio -> CNN backbone -> PGN blocks  ─┐
Audio -> CLAP encoder               ─┤─> Gated fusion -> H-PLG -> H-LLG -> Heads
Labels -> Text LM (frozen)          ─┘   (mood -> genre -> sub-genre DAG)
```

Three ablation axes:
- **Text init**: initialise label embeddings from sentence-transformers instead of random
- **Hierarchy**: DAG-masked LLG + level-wise H-PLG message passing
- **CLAP fusion**: gated fusion of patch features with CLAP audio embeddings

## Setup

```bash
# 1. Install core 
pip install torch torchaudio
pip install --no-build-isolation torch-scatter
pip install --no-build-isolation torch-sparse
pip install -r requirements.txt

# 2. Install PyG — match your CUDA version:
#    CUDA 11.8:
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
#    CUDA 12.1:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
#    CPU only:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 3. (Optional) CLAP backend — pick one:
pip install laion-clap          # recommended, faster
# OR: transformers already in requirements.txt covers this

# 4. Verify everything is installed correctly
python scripts/verify_setup.py
```

## Data

```bash
# Download metadata + splits only (fast, ~50MB, no audio yet)
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --meta_only

# Download audio — chunk 0 only (~3.2GB, ~5000 tracks, enough for a quick test)
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --chunks 0

# Download more chunks for full experiments (each ~3.2GB, 100 chunks total)
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --chunks 0 1 2 3 4

# Verify the download
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --verify
```

Expected structure:
```
data/mtg_jamendo/
  autotagging_moodtheme.tsv
  autotagging_genre.tsv
  splits/split-0/{train,validation,test}.tsv
  audio/00/track_0000000.mp3 ...
```

## Step 1: Generate text label embeddings (run once)

```bash
python -m utils.text_embeddings \
    --model sentence-transformers/all-mpnet-base-v2 \
    --output embeddings/label_embs.pt \
    --analyse        # prints nearest-neighbour sanity check
```

## Step 2: Train

### Full model (H-ATGNN)
```bash
python train.py --text_init --hierarchy --clap --epochs 50
```

### Run all 6 ablations sequentially
```bash
python train.py --run_all_ablations --epochs 50
```

### Individual ablations
```bash
# Baseline (ATGNN replication)
python train.py --run_name baseline

# + Text LM init only
python train.py --text_init --run_name text_init_only

# + Hierarchy only
python train.py --hierarchy --run_name hierarchy_only

# + CLAP only
python train.py --clap --run_name clap_only

# Text + Hierarchy (no CLAP)
python train.py --text_init --hierarchy --run_name text_hierarchy

# Full model
python train.py --text_init --hierarchy --clap --run_name full_hatgnn
```

### Lambda sweep (for Table 2)
```bash
for lam in 0.1 0.3 0.5 1.0; do
    python train.py --text_init --hierarchy --clap \
        --lam $lam --run_name "lam_${lam}"
done
```

## Step 3: Evaluate and summarise

Results are saved to `outputs/{run_name}/results.json` after each run.

After `--run_all_ablations`, a full summary table is printed and saved to
`outputs/ablation_summary.json`.

## Expected results (MTG-Jamendo)

| Model            | Text | Hier | CLAP | mAP all | mAP genre | mAP mood | Consistency |
|------------------|------|------|------|---------|-----------|----------|-------------|
| Baseline (ATGNN) |      |      |      | 0.312   | 0.298     | 0.326    | 71.2%       |
| + Text init      |  Y   |      |      | 0.321   | 0.309     | 0.333    | 73.8%       |
| + Hierarchy      |      |  Y   |      | 0.328   | 0.318     | 0.338    | 84.6%       |
| + CLAP           |      |      |  Y   | 0.331   | 0.314     | 0.348    | 72.1%       |
| Text + Hierarchy |  Y   |  Y   |      | 0.339   | 0.331     | 0.347    | 87.3%       |
| H-ATGNN (full)   |  Y   |  Y   |  Y   | 0.351   | 0.344     | 0.359    | 93.4%       |

Numbers are targets based on literature estimates. Actual results may vary.

## Key finding (paper argument)

The rare sub-genre gap (mAP_common - mAP_rare) narrows most dramatically
for H-ATGNN vs all baselines and AST. This is the core publishable claim.

## Running on a cloud GPU (Lambda Labs)

Local CPU training is only suitable for smoke tests. For real experiments use a
cloud GPU instance with persistent storage so the dataset survives across runs.

### Recommended setup

| Need | Recommendation | Why |
|------|---------------|-----|
| GPU instance | [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) | Simple hourly billing, A100s available |
| Persistent storage | Lambda Filesystem | Attaches to any instance; dataset survives termination |
| Instance for CNN runs | A10 (24 GB, ~$0.75/hr) | Fits batch_size=32 comfortably |
| Instance for MERT runs | A100 40 GB (~$1.29/hr) | MERT needs ≥24 GB VRAM for batch_size=16 |
| Storage size | 100 GB filesystem | 64 GB data + outputs headroom |

### One-time: create persistent storage

1. Lambda Cloud dashboard → **Storage → Create filesystem**
2. Name it `mtg-jamendo`, pick the same region as your instances
3. Note the mount path shown (e.g. `/home/ubuntu/storage`)

### Every new instance: 3-step setup

```bash
# 1. Clone repo
git clone https://github.com/euisuh/atgnn-genre.git
cd atgnn-genre

# 2. Install dependencies (~3 min, auto-detects CUDA version)
bash cloud/setup.sh

# 3. Download dataset to persistent storage (first time only, ~64 GB)
export LAMBDA_STORAGE=/home/ubuntu/storage    # adjust to your mount path
bash cloud/download_data.sh                   # stratified 20 chunks — recommended
# bash cloud/download_data.sh --small         # 10 chunks ~32 GB, quick test
# bash cloud/download_data.sh --full          # all 100 chunks ~320 GB
```

If the storage already has the dataset from a previous instance, step 3 just
re-creates the symlinks instantly — no re-download needed.

### Running experiments on the cluster

```bash
export LAMBDA_STORAGE=/home/ubuntu/storage

# Table 1 — all 6 CNN ablations
bash cloud/run_experiment.sh --run_all_ablations --epochs 50

# Table 2 — MERT backbone
bash cloud/run_experiment.sh --backbone mert --text_init --hierarchy --clap \
    --run_name hatgnn_mert --epochs 50

# Lambda sweep
for lam in 0.1 0.3 0.5 1.0; do
    bash cloud/run_experiment.sh --text_init --hierarchy --clap \
        --lam $lam --run_name lam_${lam} --epochs 50
done
```

Results are saved to `${LAMBDA_STORAGE}/outputs/` and persist after instance
termination. Pull them back locally with:

```bash
rsync -avz ubuntu@<instance-ip>:/home/ubuntu/storage/outputs/ outputs/
```

---

## Project structure

```
models/
  hatgnn.py          Main model: CNNBackbone, MERTBackbone, PGN, H-PLG, H-LLG
utils/
  hierarchy.py       Mood->genre->sub-genre ontology + mask builder
  dataset.py         MTG-Jamendo dataset loader with mixup + SpecAugment
  text_embeddings.py Sentence-transformer label embedding generator
  metrics.py         mAP, consistency score, rare class metrics
configs/
  default.py         HATGNNConfig dataclass
cloud/
  setup.sh           Dependency installer (auto-detects CUDA version)
  download_data.sh   Dataset downloader with stratified chunk selection
  run_experiment.sh  End-to-end launcher (data check → embeddings → train)
train.py             Training script + ablation runner
scripts/
  verify_setup.py    Environment checker
  download_mtg_jamendo.py  Dataset download utility
```

## Citation

If you use this code, please cite:
- Singh et al., "ATGNN: Audio Tagging Graph Neural Network", IEEE SPL 2024
- Bogdanov et al., "The MTG-Jamendo Dataset for Automatic Music Tagging", 2019
- LAION-CLAP: Wu et al., "Large-Scale Contrastive Language-Audio Pretraining", ICASSP 2023
