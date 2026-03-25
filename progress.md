# Experiment Progress

## Status: Pipeline verified, ablation run in progress

---

## Environment
- Python 3.14, torch 2.11, torch-geometric 2.7, torchaudio 2.11
- CPU only (Mac mini) — GPU needed for full experiments
- Venv: `source genre/bin/activate`

## Data
- MTG-Jamendo chunk 0 downloaded: 586 mp3 files
- Splits in `data/mtg_jamendo/splits/split-0/`
- Available for training: 112 train / 41 val / 46 test tracks (chunk 0 only)
- For full experiments: download more chunks (each ~3.2GB)
  ```bash
  python scripts/download_mtg_jamendo.py --output data/mtg_jamendo --chunks 1 2 3 4
  ```

## Smoke test results (2 epochs, chunk 0 only)

| Backbone | mAP_all | consistency | batch_size | epoch time (CPU) |
|----------|---------|-------------|------------|-----------------|
| CNN      | 0.0849  | 0.8502      | 24         | ~365s           |
| MERT     | 0.1242  | 0.9586      | 2          | ~294s           |

MERT shows stronger early signal even with a much smaller batch size.

## In progress
- Full 6-ablation CNN run: `python train.py --run_all_ablations --epochs 50`
  - Started on chunk 0 data (112 train tracks)
  - Results will appear in `outputs/ablation_summary.json`

## Next steps
1. Download more audio chunks for meaningful training data
2. Run full ablations on GPU with larger data
3. Implement MERT ablation table (Table 2 in paper):
   ```bash
   python train.py --backbone mert --text_init --hierarchy --run_name hatgnn_mert
   ```
4. Add MERT to `--run_all_ablations` sweep (currently CNN only)

## Paper recommendation (from Claude)
Run two backbone variants for two strong tables:
- **Table 1**: CNN backbone — direct ATGNN comparison
- **Table 2**: MERT backbone — vs. strong self-supervised baseline

Key claim to demonstrate: rare sub-genre gap (mAP_common - mAP_rare) narrows
most dramatically for H-ATGNN vs all baselines.
