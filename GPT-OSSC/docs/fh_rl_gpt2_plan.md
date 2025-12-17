# GPT-2 + FH-RL Fine-Tuning Plan

Goal: retrofit HuggingFace `gpt2` (124M) with FH-RL layers between self-attention and MLP, fine-tune on a real corpus, and benchmark OOD perplexity versus the baseline checkpoint.

## Models
- **Baseline**: vanilla `gpt2` from HuggingFace (no FH-RL). Fine-tune on target corpus using the same learning schedule for fair comparison.
- **FH-RL**: clone of GPT-2 where each transformer block gains the FH-RL module (rank=32, α=0.2, β=0.1). Initialize `gamma=0.0` and linearly ramp to 0.1 over the first 2k steps.

## Data
- **Training corpus**: WikiText-103 (word-level). Use HuggingFace datasets to fetch, then tokenize with GPT-2 BPE to produce LM training samples (block size 512). Script: `scripts/build_wikitext_subset.py`.
- **OOD evaluation**: Penn Treebank validation split, tokenized identically. This stresses generalization beyond WikiText domain.

## Training Schedule
- Batch size: 8 (global), seq len 512, gradient accumulation 4 → effective batch 32.
- Optimizer: AdamW lr 5e-5, β=(0.9,0.95), weight decay 0.01.
- Steps: 20k (baseline) + 20k (FH-RL). Mixed precision (torch.cuda.amp) to keep memory manageable.
- FH-RL gamma schedule: 0 → 0.1 over first 2k steps, then hold.

## Metrics
- Track train/val cross-entropy + perplexity.
- Log IRR/ESRI/RDP for FH-RL model per eval interval.
- OOD eval: perplexity on PTB for both checkpoints.

## Outputs
- Checkpoints in `runs/fh_rl_gpt2/baseline.pt` and `runs/fh_rl_gpt2/fh_rl.pt`.
- Logs (`train_baseline.jsonl`, `train_fh_rl.jsonl`).
- OOD summary JSON comparing perplexities.
