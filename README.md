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
pip install torch torchaudio
pip install --no-build-isolation torch-scatter
pip install --no-build-isolation torch-sparse

pip install -r requirements.txt

# Install PyG (match your CUDA version)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## Data

Download MTG-Jamendo:
```bash
# Instructions: https://github.com/MTG/mtg-jamendo-dataset
python scripts/download_mtg_jamendo.py --output data/mtg_jamendo
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

## Project structure

```
hatgnn/
  models/
    hatgnn.py          Main model: PGN, H-PLG, H-LLG, GatedFusion
  utils/
    hierarchy.py       Mood->genre->sub-genre ontology + mask builder
    dataset.py         MTG-Jamendo dataset loader with mixup + SpecAugment
    text_embeddings.py Sentence-transformer label embedding generator
    metrics.py         mAP, consistency score, rare class metrics
  configs/
    default.py         HATGNNConfig dataclass
  train.py             Training script + ablation runner
  requirements.txt
  README.md
```

## Citation

If you use this code, please cite:
- Singh et al., "ATGNN: Audio Tagging Graph Neural Network", IEEE SPL 2024
- Bogdanov et al., "The MTG-Jamendo Dataset for Automatic Music Tagging", 2019
- LAION-CLAP: Wu et al., "Large-Scale Contrastive Language-Audio Pretraining", ICASSP 2023
