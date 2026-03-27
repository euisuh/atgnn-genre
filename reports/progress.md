# Project Progress — H-ATGNN
*Last updated: 2026-03-24*

## Overview

Extending ATGNN (Singh et al., IEEE SPL 2024) with three hierarchical improvements
for music genre classification on MTG-Jamendo.

---

## Milestones

### Done
- [x] Read and understood ATGNN architecture (PGN, PLG, LLG)
- [x] Designed H-ATGNN extensions: H-PLG, H-LLG, consistency loss
- [x] Implemented `models/hatgnn.py` — full model with all extensions switchable
- [x] Implemented `utils/dataset.py` — MTG-Jamendo loader with mixup, augmentation
- [x] Implemented `utils/hierarchy.py` — mood/genre/sub-genre DAG config
- [x] Implemented `utils/metrics.py` — mAP per level, rare sub-genre gap
- [x] Implemented `utils/text_embeddings.py` — SBERT label initialisation
- [x] Implemented `train.py` — ablation runner, W&B logging, checkpoint save/resume
- [x] Implemented `configs/default.py` — full config dataclass
- [x] Added SSLBackbone — supports MERT and MuQ via unified API
- [x] Added MuQLanEmbedder — on-the-fly cross-modal fusion (replaces precomputed CLAP)
- [x] CPU smoke tests passing for all experiment variants
- [x] W&B integration — logs all metrics, configs, model params, checkpoints as artifacts
- [x] Cloud scripts — `setup.sh`, `download_data.sh`, `run_experiment.sh`, `run_chain.sh`
- [x] Checkpoint resume — `--resume last.pt` continues training from any epoch

### In Progress
- [ ] **Table 1** — 6 CNN ablation runs × 50 epochs (A40, ~4–6h, ~$1.60)
- [ ] Download remaining 11/20 audio chunks to RunPod Network Volume

### Not Started
- [ ] **Table 2** — SSL backbone experiments: MERT vs MuQ, with/without H-ATGNN (H100 SXM, ~2h, ~$5)
- [ ] **Table 3** — λ consistency loss sweep (A40, ~3–4h, ~$1.20)
- [ ] Precompute CLAP embeddings
- [ ] Precompute MuQ-MuLan embeddings
- [ ] Write paper sections

---

## Experiment plan

### Table 1 — CNN ablations (ablation study)
| Run | text_init | hierarchy | cross_modal |
|-----|-----------|-----------|-------------|
| baseline | ✗ | ✗ | none |
| +text_init | ✓ | ✗ | none |
| +hierarchy | ✗ | ✓ | none |
| +clap | ✗ | ✗ | clap |
| +hier+text | ✓ | ✓ | none |
| full | ✓ | ✓ | clap |

### Table 2 — SSL backbones
| Run | backbone | hierarchy |
|-----|----------|-----------|
| MERT baseline | mert | ✗ |
| MERT + H-ATGNN | mert | ✓ |
| MuQ baseline | muq | ✗ |
| MuQ + H-ATGNN | muq | ✓ |
| MuQ + MuQ-MuLan | muq | ✓ + muqmulan |

### Table 3 — λ sweep (consistency loss weight)
λ ∈ {0.1, 0.3, 0.5, 1.0}, full model config

---

## Key design decisions

- **Dataset:** MTG-Jamendo (55k tracks, 3-level genre hierarchy: mood → genre → sub-genre)
- **Subset:** 20 stratified chunks (~1/5 of full dataset) — sufficient for relative ablation comparisons
- **H-PLG:** ordered 3-pass message passing (mood → genre → sub-genre direction)
- **H-LLG:** DAG-masked adjacency — only parent→child edges allowed
- **Consistency loss:** KL divergence between sub-genre predictions and parent genre marginals
- **MuQ over MERT:** better benchmark scores, ConformerEncoder architecture, `muq` pip package
- **MuQ-MuLan as CLAP replacement:** on-the-fly cross-modal fusion, no precomputation needed

---

## Cloud setup (RunPod)

- **Network Volume:** 200GB, Texas region — keep between pods, do NOT delete
- **Data on volume:** `/workspace/mtg_jamendo/` (audio + splits + metadata)
- **Embeddings on volume:** `/workspace/embeddings/label_embs.pt`
- **Table 1 GPU:** A40 at $0.40/hr
- **Table 2 GPU:** H100 SXM at $2.69/hr (~1.67x cost-efficient vs A100 SXM for transformers)
