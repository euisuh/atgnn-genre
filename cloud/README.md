# Cloud setup — Lambda Labs

## One-time: create persistent storage

1. Go to **Lambda Cloud → Storage → Create filesystem**
2. Name it `mtg-jamendo`, pick a region
3. Note the mount path (e.g. `/home/ubuntu/storage`) — set this as `LAMBDA_STORAGE`

The same storage volume attaches to any instance you spin up, so you only
download the dataset once.

---

## Every new instance: 3-step setup

```bash
# 1. Clone repo
git clone https://github.com/euisuh/atgnn-genre.git
cd atgnn-genre

# 2. Install dependencies (~3 min)
bash cloud/setup.sh

# 3. Download dataset to persistent storage (first time only, ~32GB for 10 chunks)
export LAMBDA_STORAGE=/home/ubuntu/storage    # adjust if your mount path differs
bash cloud/download_data.sh
```

If the dataset is already on storage from a previous instance, step 3 is instant
(just re-creates the symlinks).

---

## Running experiments

```bash
export LAMBDA_STORAGE=/home/ubuntu/storage

# All 6 CNN ablations — Table 1
bash cloud/run_experiment.sh --run_all_ablations --epochs 50

# Full model with MERT backbone — Table 2
bash cloud/run_experiment.sh --backbone mert --text_init --hierarchy --clap \
    --run_name hatgnn_mert --epochs 50

# Lambda sweep
for lam in 0.1 0.3 0.5 1.0; do
    bash cloud/run_experiment.sh --text_init --hierarchy --clap \
        --lam $lam --run_name lam_${lam} --epochs 50
done
```

Results are saved to `${LAMBDA_STORAGE}/outputs/` and persist across instances.

---

## Recommended instance types

| Task | Instance | VRAM | Est. time (50 epochs, 10 chunks) |
|------|----------|------|----------------------------------|
| CNN ablations | A10 (24GB) | 24GB | ~4h |
| MERT full model | A100 (40GB) | 40GB | ~6h |
| All ablations + MERT | A100 (80GB) | 80GB | ~10h |

MERT needs at least 24GB VRAM for batch_size=16. Use A100 for full runs.

---

## Pulling results back locally

```bash
# From your Mac
rsync -avz ubuntu@<instance-ip>:/home/ubuntu/storage/outputs/ outputs/
```

Or download `outputs/ablation_summary.json` directly from the Lambda web console.
