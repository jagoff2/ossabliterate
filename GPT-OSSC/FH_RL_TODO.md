# FH-RL GPT-2 Integration TODO

> Status legend: [ ] pending, [~] in progress, [x] done. Update immediately after each task.

1. [x] Formalize FH-RL-on-GPT-2 design spec (layer placement, state handling, training params) in `docs/fh_rl_overview.md`.
2. [x] Implement reusable FH-RL module (`meta_transformer/fh_rl_layer.py`) with fast-weight updates, homeostasis, and reentry projection.
3. [x] Fork GPT-2 block (`meta_transformer/gpt2_fh_rl_block.py`) embedding FH-RL between attention and MLP, wired for detaching feedback gradients.
4. [x] Build dataset & config for byte-level synthetic corpus (`scripts/build_fh_rl_corpus.py`, `configs/fh_rl_tiny.json`).
5. [x] Create end-to-end training script (`training/train_fh_rl.py`) matching paper hyperparameters and logging IRR/ESRI/RDP per step.
6. [x] Implement evaluation utilities (`scripts/eval_fh_rl_metrics.py`) computing IRR/ESRI/RDP on saved checkpoints.
7. [x] Document training/eval workflow in README section (`docs/fh_rl_training.md`) and link from root README.
8. [x] Provide example launch commands + smoke-test notebook/log to verify sanity (e.g., `runs/fh_rl_smoke`), capturing sample metrics.
9. [x] Prepare paper-faithful experiment config (`configs/fh_rl_paper.json`) and logging plan for baseline (γ=0) vs reflective (γ≈0.15).
10. [x] Extend `training/train_fh_rl.py` to emit per-step JSON logs for IRR/ESRI/RDP and eval summaries.
11. [x] Execute GPU training run with the paper config, covering both γ settings, and store checkpoints under `runs/fh_rl_paper`.
12. [x] Implement & run comparison script (`scripts/summarize_fh_rl_runs.py`) to report baseline vs FH-RL metrics from recorded checkpoints.
