# GPT-OSS Workspace + Abliteration Toolkit

Augments `openai/gpt-oss-20b` with a latent global workspace (virtual KV, residual deltas, memory/controller hooks) and now vendors a full refusal-abliteration pipeline (measure → analyze → ablate → GGUF) adapted from `jim-plus/llm-abliteration`.

## Features
- Slot-Attention workspace with virtual KV append, residual-delta hooks, entropy-aware controller, and episodic memory (SQLite + FAISS).
- FastAPI server exposing OpenAI-compatible completions with streaming and request-level hook toggles.
- Abliteration suite: measure refusal directions, analyze signal quality, apply sharded weight edits, and export GGUF for stock `llama.cpp`.
- Typer-based CLI for serving, evaluation, and abliteration workflows.

## Requirements
- Python 3.11+
- CUDA GPU (measurement/ablation expect GPU; CPU inference is not supported for GPT-OSS)
- Disk: ~50 GB free for GPT-OSS weights + ablated copies; additional space for GGUF outputs
- (Optional) `llama.cpp` checkout for GGUF conversion

Install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Repo Layout (abridged)
- `cli/` — Typer entrypoints
- `gpt_oss_ws/` — workspace hooks and abliteration modules
- `configs/` — server defaults, eval presets, abliteration template
- `data/abliteration_samples/` — placeholder harmful/harmless prompts
- `docs/abliteration.md` — detailed abliteration walkthrough
- `tests/` — unit tests including abliteration smoke tests

## Quickstart: Abliteration Pipeline
• One‑Stop Checklist: GPT‑OSS Abliteration Pipeline

  0. Prep (run everything from repo root /mnt/g/osscrepo/GPT-OSSC)

  - Activate the venv every time you open a new shell:

    source .venv/bin/activate

  1. Measure refusal directions (harmful vs harmless prompts)

  - Command (adjust paths if you have different datasets):

    python -m cli.main abliteration measure \
      --model openai/gpt-oss-20b \
      --output outputs/gpt_oss_dequant_chattemp.refuse.pt \
      --chat-template chat_template.jinja \
      --sample-last-user \
      --data-harmful data/abliteration_datasets/harmful_v1.jsonl \
      --data-harmless data/abliteration_datasets/harmless_v1.jsonl \
      --batch-size 16 \
      --clip 1.0 \
      --flash-attn \
      --projected \
      --deccp
      - This loads both datasets, applies the chat template, samples hidden states at the last user token, and saves the
        measurement file.

  2. Inspect the measurements (optional sanity check)

  python -m cli.main abliteration analyze --input outputs/gpt_oss_dequant_chattemp.refuse.pt

  - Prints per-layer cosine / SNR so you can see where the signal is strongest.

  3. Run ablation (projection ON, norm-preserve ON)

  - Use whatever config you want (e.g., configs/abliteration_gpt_oss_chattemp_router_exps_base2p5.yaml). Command:

    python -m cli.main abliteration ablate \
      --config configs/abliteration_gpt_oss_chattemp_router_exps_base2p5.yaml \
      --projected \
      --norm-preserve
      - Output checkpoint lands in the output: folder defined inside the YAML (e.g., outputs/
        gpt_oss_dequant_ablated_chattemp_router_exps_base2p5/).

  4. Export the ablated checkpoint to GGUF

  python -m cli.main abliteration export-gguf \
    --model-dir outputs/gpt_oss_dequant_ablated_chattemp_router_exps_base2p5 \
    --llama-cpp-path /mnt/g/osscrepo/llama.cpp \
    --outfile outputs/gpt_oss_dequant_chattemp_router_exps_base2p5.gguf \
    --outtype f16

  5. Test the GGUF with llama.cpp

  - Harmless prompt (in this repo it’s already saved as tmp_prompt_chattemp.txt):

    /mnt/g/osscrepo/llama.cpp/build/bin/llama-cli \
      -m outputs/gpt_oss_dequant_chattemp_router_exps_base2p5.gguf \
      -f tmp_prompt_chattemp.txt \
      -n 512 \
      --temp 0.7 \
      --repeat_penalty 1.1 \
      --mirostat 0
  - Harmful test prompt (without copying text into the chat history, use the pre-saved file):

    /mnt/g/osscrepo/llama.cpp/build/bin/llama-cli \
      -m outputs/gpt_oss_dequant_chattemp_router_exps_base2p5.gguf \
      -f tmp_prompt_harmful.txt \
      -n 512 \
      --temp 0.7 \
      --repeat_penalty 1.1 \
      --mirostat 0

  6. Iterate on ablation parameters if needed

  - Copy an existing config (e.g., cp configs/abliteration_gpt_oss_chattemp_router_exps_base2p5.yaml configs/my_new_run.yaml),
    edit scales/sparsity targets, then rerun steps 3–5.

  That’s the entire flow end-to-end. Activate venv, measure, analyze (optional), ablate with projection/norm-preserve, export
  GGUF, and test with llama-cli on both harmless and harmful prompt files.





1) **Measure refusal directions** (customize prompts for real use):
```bash
python -m cli.main abliteration measure \
  --model openai/gpt-oss-20b \
  --output outputs/gpt_oss.refuse.pt \
  --data-harmful data/abliteration_samples/harmful.txt \
  --data-harmless data/abliteration_samples/harmless.txt \
  --batch-size 16 --clip 0.98 --flash-attn --projected
```
Flags:
- `--quant-measure {4bit,8bit}`: force BitsAndBytes quant during measurement (auto-detected otherwise).
- `--deccp`: append AUGMXNT/DECCP prompts (Chinese models).
- Hugging Face datasets: `--hf-harmful id --hf-harmless id --harmful-split train --harmless-split train --harmful-column text --harmless-column text`.

2) **Analyze signal quality**:
```bash
python -m cli.main abliteration analyze \
  --input outputs/gpt_oss.refuse.pt \
  --chart --chart-path outputs/gpt_oss_refusal.png
```
Prints per-layer cosine/snr/purity metrics; optional chart saved via Matplotlib.

3) **Author marching orders**: edit `configs/abliteration_example.yaml` (one entry per destination layer):
```yaml
model: openai/gpt-oss-20b
measurements: outputs/gpt_oss.refuse.pt
output: outputs/gpt_oss_ablated
ablate:
  - layer: 11
    measurement: 23
    scale: 1.0
    sparsity: 0.0
```
`measurement` picks which layer’s refusal vector to apply; `sparsity` keeps top-|fraction| magnitudes; add `--projected` to orthogonalize vs harmless means; `--norm-preserve` keeps row norms.

4) **Run sharded ablation**:
```bash
python -m cli.main abliteration ablate \
  --config configs/abliteration_example.yaml \
  --projected --norm-preserve
```
Writes a new safetensors directory with config/tokenizer files copied alongside edited shards. MXFP4 MoE blocks lacking float down-proj weights are left untouched for safety.

5) **Export GGUF for llama.cpp**:
```bash
python -m cli.main abliteration export-gguf \
  --model-dir outputs/gpt_oss_ablated \
  --llama-cpp-path ~/src/llama.cpp \
  --outfile outputs/gpt_oss_ablated.gguf \
  --quantize q4_0 --outtype f16
```
`--extra-arg` is forwarded to `convert.py` (e.g., `--extra-arg --vocab-dir --extra-arg ./custom_vocab`). Ensure `llama.cpp/tools/convert-hf-to-gguf.py` or `convert.py` exists in the checkout.

## Serving GPT-OSS with Workspace Hooks
Launch the FastAPI server (OpenAI-compatible):
```bash
python -m cli.main serve --config configs/server.yaml --host 0.0.0.0 --port 8000
```
Key toggles in `configs/server.yaml`: hooked layers, quantization (`bnb-4bit`/`bf16`/`Mxfp4`), virtual-KV retention, structured-output guardrails, retro-generation settings.

## Running Evaluations
- Unit tests: `python -m pytest`
- Abliteration-only: `python -m pytest tests/test_abliteration.py`
- Fluency guard A/B: `python -m cli.main fluency-guard --baseline configs/hooks_off.yaml --workspace configs/server.yaml --samples 64`

## Tips & Caveats
- Run measurement on quantized weights if VRAM-bound, but perform ablation on float/bfloat16 checkpoints to avoid precision loss.
- Ensure tokenizer has a defined `pad_token`; CLI sets EOS as fallback.
- For large runs, keep an eye on disk space: measurement files are small; ablated safetensors duplicate the checkpoint.
- Verify GGUF outputs with `./llama-cli --vocab-only your.gguf` from `llama.cpp` to confirm header metadata.

## Troubleshooting
- `AutoModelForImageTextToText` missing: installed transformers build lacks VLM class; the code falls back to `AutoModelForCausalLM` (text-only). Provide text-only prompts in that case.
- “No matching parameters found for requested ablations”: confirm your YAML layers match the model’s layer count and key names (`self_attn.o_proj.weight`, `mlp.down_proj.weight`).
- CUDA OOM during measurement: lower `--batch-size`, enable `--quant-measure 4bit`, and keep `--clip` near 1.0.

## License
Apache-2.0
