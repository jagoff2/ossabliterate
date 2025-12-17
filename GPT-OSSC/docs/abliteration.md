# Refusal Abliteration Workflow

This repository now vendors the utilities from [jim-plus/llm-abliteration](https://github.com/jim-plus/llm-abliteration) so GPT-OSS checkpoints can be measured, analyzed, ablated, and exported to GGUF via a single Typer CLI.

## 1. Prepare prompts

Create curated prompt sets that reliably elicit refusals and safe completions. Sample placeholders live under `data/abliteration_samples/`, but you should replace them with a vetted corpus (TXT/JSON/JSONL/Parquet or Hugging Face datasets).

If you want a starting point, `data/abliteration_datasets/harmful_v1.jsonl` and `data/abliteration_datasets/harmless_v1.jsonl` contain 360 harmful-style and 320 harmless-style prompts respectively. Each line is a JSON object with metadata fields (`category`, `tone`, `complexity`, etc.), so you can subset by topic before measurement:

```
python -m cli.main abliteration measure \
  --model openai/gpt-oss-20b \
  --output outputs/gpt_oss.refuse.pt \
  --data-harmful data/abliteration_datasets/harmful_v1.jsonl \
  --data-harmless data/abliteration_datasets/harmless_v1.jsonl \
  --clip 0.98 --batch-size 24
```

```
python -m cli.main abliteration measure \
  --model openai/gpt-oss-20b \
  --output outputs/gpt_oss.refuse.pt \
  --data-harmful data/abliteration_samples/harmful.txt \
  --data-harmless data/abliteration_samples/harmless.txt \
  --batch-size 16 --clip 0.98 --flash-attn --projected
```

- Measurements run with automatic BitsAndBytes quant detection (`--quant-measure` can force 4bit/8bit).
- Use `--deccp` to append the AUGMXNT/DECCP prompt set for Chinese models.
- Hugging Face datasets can be supplied via `--hf-harmful/--hf-harmless`.

## 2. Analyze signal quality

The analyzer reproduces the plotting/reporting logic from `analyze.py`.

```
python -m cli.main abliteration analyze \
  --input outputs/gpt_oss.refuse.pt --chart --chart-path outputs/gpt_oss_refusal.png
```

Per-layer cosine similarities, norms, purity, and signal-to-noise ratios are printed, and optional charts are rendered via Matplotlib.

## 3. Author YAML “marching orders”

See `configs/abliteration_example.yaml` for the canonical structure. Each `ablate` entry specifies the destination layer, which layer’s refusal vector to borrow, the scale factor, and sparsity fraction (0.0 keeps the whole vector).

## 4. Run sharded ablation

```
python -m cli.main abliteration ablate \
  --config configs/abliteration_example.yaml \
  --projected --norm-preserve
```

This streams `model.safetensors` shards, modifies `self_attn.o_proj.weight` (and `mlp.down_proj.weight` when present), and writes a fresh checkpoint directory with all config/tokenizer files copied alongside the edited shards. As with the reference implementation, ablation should target full-precision weights; MXFP4 MoE blocks are skipped automatically because their packed representation would require a custom dequant/requant pipeline.

## 5. Export GGUF via llama.cpp

Point the exporter at an existing llama.cpp checkout (no fork required):

```
python -m cli.main abliteration export-gguf \
  --model-dir outputs/gpt_oss_ablated \
  --llama-cpp-path ~/src/llama.cpp \
  --outfile outputs/gpt_oss_ablated.gguf \
  --quantize q4_0 --outtype f16
```

Additional `--extra-arg` flags are forwarded verbatim to `convert.py` so features such as vocabulary overrides or context-length tweaks stay available.

## Notes & caveats

- Measurement can run on quantized (4-bit/8-bit/MXFP4) checkpoints, but ablation should be executed on float/bfloat16 weights to guarantee the projected direction can be applied directly.
- When a destination layer lacks `mlp.down_proj.weight` (e.g., MXFP4 experts), the current implementation only updates `self_attn.o_proj.weight` and emits a warning.
- Always verify the resulting GGUF with `./llama-cli --vocab-only your_model.gguf` from llama.cpp to ensure metadata matches expectations.
