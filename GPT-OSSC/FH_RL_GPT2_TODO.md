# GPT-2 FH-RL Integration TODO
(Status legend: [ ] pending, [~] in progress, [x] done)

1. [x] Define integration plan (baseline checkpoints, corpus choice, eval targets) in `docs/fh_rl_gpt2_plan.md`.
2. [x] Implement HuggingFace GPT-2 block wrapper inserting FH-RL (new module `meta_transformer/models/hf_gpt2_fh_rl.py`).
   2.1. [x] Create FH-RL aware block class mirroring `GPT2Block` but with post-attention FH-RL stage.
   2.2. [x] Provide factory that loads pretrained GPT-2 weights, initializes FH-RL params (γ≈0), and exposes fine-tune interface.
3. [x] Prepare training corpus + dataloader for real text (use WikiText-103 subset) via `scripts/build_wikitext_subset.py` + config `configs/fh_rl_gpt2_finetune.json`.
4. [x] Extend training script (`training/train_fh_rl_gpt2.py`) to fine-tune the HuggingFace model with FH-RL on that corpus (mixed precision, γ warmup from 0 to target 0.1).
5. [x] Add OOD evaluation script (`scripts/eval_fh_rl_gpt2.py`) that compares baseline GPT-2 vs FH-RL checkpoint on a held-out corpus (e.g., PTB).
6. [x] Run fine-tuning (baseline frozen vs FH-RL model) and record losses/perplexities.
7. [x] Run OOD evaluation to produce benchmark table; store summary in `runs/fh_rl_gpt2/ood_summary.json` and report results.
