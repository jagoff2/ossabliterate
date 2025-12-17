# FH-RL Training & Evaluation Guide

This document explains how to run the GPT-2 + FH-RL tiny model implementation provided in this repo.

## 1. Build the byte-level corpus
```
python scripts/build_fh_rl_corpus.py \
  --sources data/seed_text.txt another.txt \
  --output data/fh_rl_corpus.jsonl \
  --seq-len 128
```
If no source files are passed, the script falls back to bundled seed sentences.

## 2. Inspect/adjust the config
`configs/fh_rl_tiny.json` controls model width, FH-RL hyperparameters, optimizer, and training schedule. Update `gamma_sweep`, `max_steps`, etc. as needed.

## 3. Launch training
```
python -m training.train_fh_rl --config configs/fh_rl_tiny.json --device cuda
```
The trainer will iterate over all gamma values in `gamma_sweep`, logging:
- training loss
- IRR (Information Reentry Ratio proxy)
- ESRI and RDP estimates derived from the current batch trace
Validation loss is reported every `eval_interval`. Checkpoints are saved under `runs/fh_rl_tiny/fh_rl_gamma_*.pt`.

## 4. Evaluate IRR/ESRI/RDP on checkpoints
```
python scripts/eval_fh_rl_metrics.py \
  --config configs/fh_rl_tiny.json \
  --checkpoint runs/fh_rl_tiny/fh_rl_gamma_0.15.pt \
  --limit 128
```
Outputs JSON with averaged IRR/ESRI/RDP values across sampled sequences.

## 5. Scaling tips
- Increase `hidden_size`, `num_layers`, and dataset scope incrementally; keep `gamma` small (≤0.05) when initializing from pretrained GPT weights.
- The FH-RL layer exposes `rank`, `alpha`, `beta`, and `noise_std` so you can explore the “reflective band” (γ≈0.1–0.2) identified in the paper.
- To disable gradient detachment for experimentation, set `detach_feedback=false` in the config—expect instability.

Refer to `docs/fh_rl_overview.md` for architectural details and theory context.
