# Project: H-ATGNN

## Setup
- Venv: `genre/` — always activate with `source genre/bin/activate`
- Python 3.14, torch 2.11, torch-geometric 2.7

## File structure
- models/hatgnn.py, utils/{dataset,hierarchy,metrics,text_embeddings}.py, configs/default.py
- Each subdir has __init__.py — imports match what train.py expects

## Data
- data/mtg_jamendo/ — metadata TSVs downloaded, audio chunk 0 in progress
- Splits already at data/mtg_jamendo/splits/split-0/

## Paper recommendations (from Claude)
- Use MTG-Jamendo as primary dataset — right fit for the hierarchy task
- Run two backbone variants:
  1. **Lightweight CNN** — for direct ATGNN comparison (Table 1)
  2. **MERT** — for "vs. strong baseline" result (Table 2)
- Goal: two strong tables instead of one
- Remind user to implement MERT backbone after pipeline is set up

## Known issues
- Original Zenodo URLs were wrong (404) — fixed to use GitHub raw + CDN
- Audio files are .tar not .tar.gz