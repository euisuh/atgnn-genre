# Project: H-ATGNN

## Setup
- Venv: `genre/` — always activate with `source genre/bin/activate`
- Python 3.14, torch 2.11, torch-geometric 2.7

## File structure
- models/hatgnn.py, utils/{dataset,hierarchy,metrics,text_embeddings}.py, configs/default.py
- Each subdir has __init__.py — imports match what train.py expects

## Data
- MTG-Jamendo audio: 9/20 chunks already on RunPod Network Volume at `/workspace/mtg_jamendo/audio/` (chunks 00,05,10,15,20,25,30,35,40)
- Remaining 11 chunks will download fast on next pod (skip logic in place)
- Splits at `/workspace/mtg_jamendo/splits/split-0/`
- Label embeddings at `/workspace/embeddings/label_embs.pt`

## Cloud (RunPod) — next session TODO
1. Create A40 pod, **attach existing Network Volume** (data already there)
2. `git clone` + `scp .env` + `git pull`
3. Run: `AUTO_SHUTDOWN=1 bash cloud/run_chain.sh --table 1`
   - Runs smoke test, waits for remaining 11 chunks, runs Table 1 (6 CNN runs × 50 epochs), shuts down
4. After pod terminates: create H100 SXM pod, attach same volume, run `--table 2`

## RunPod specifics
- Network Volume mounts at `/workspace` (NOT `/runpod-volume`)
- STORAGE_ROOT=/workspace already set in .env
- Do NOT delete the Network Volume between pods — audio data (46GB) lives there

## Paper recommendations (from Claude)
- Use MTG-Jamendo as primary dataset — right fit for the hierarchy task
- Run two backbone variants:
  1. **Lightweight CNN** — for direct ATGNN comparison (Table 1)
  2. **MERT** — for "vs. strong baseline" result (Table 2)
- Goal: two strong tables instead of one
- Remind user to implement MERT backbone after pipeline is set up

## Known issues (all fixed)
- Original Zenodo URLs were wrong (404) — fixed to use GitHub raw + CDN
- Audio files are .tar not .tar.gz
- RunPod tar extraction fails ownership check — fixed with --no-same-owner
- Corrupt audio files crash training — fixed with try/except in dataset.__getitem__