# Abliteration Integration TODO

## Context after repo review
- GPT-OSS workspace stack already wraps `openai/gpt-oss-20b` via `gpt_oss_ws.model_wrapper` with patched attention, virtual KV, residual deltas, and CLI entry points under `cli/main.py`.
- No existing refusal-direction tooling, dataset loaders, or gguf exporters exist in this codebase; dependencies such as `tqdm`, `datasets`, `safetensors`, or matplotlib are absent from `pyproject.toml`.
- The external reference repo (`jim-plus/llm-abliteration`) ships `measure.py`, `analyze.py`, and `sharded_ablate.py` plus helper utilities that we must faithfully port/adapt so the GPT-OSS stack can measure refusal directions, ablate them, and export GGUF artifacts usable by stock `llama.cpp`.

## Work items to fulfill the request
1. **Port measurement primitives**
   - [x] Create a `gpt_oss_ws/abliteration` package mirroring `measure.py` functionality: Welford accumulation, projected refusal vectors, BitsAndBytes-aware loading, processor/tokenizer plumbing, and dataset-driven prompt formatting compatible with GPT-OSS harmony chats.
   - [x] Ensure support for harmful/harmless prompt inputs from `.txt/.json/.jsonl/.parquet` files and Hugging Face datasets (`load_dataset`), including the `--deccp` augmentation path.
   - [x] Implement CLI/typer command (`cli/main.py`) exposing the measurement pipeline with options for quantization, flash attention, clip fraction, batch size, etc., saving `torch.save` blobs identical to upstream reference outputs.

2. **Replicate analysis utilities**
   - [x] Integrate an analyzer module (chart optional) that consumes measurement files, prints cosine/signal stats per layer, and optionally saves matplotlib figures—matching `analyze.py`.
   - [x] Surface analyzer through CLI so users can run `python -m cli.main abliteration-analyze ...` (exact subcommand TBD) without touching the raw script.

3. **Implement sharded ablation workflow**
   - [x] Port `sharded_ablate.py` logic into the repo, adapting tensor key detection to GPT-OSS weight naming (attention `o_proj` and MLP `down_proj`) and ensuring per-layer orders defined in YAML drive modifications.
   - [x] Support both standard and norm-preserving ablation as well as optional harmless-direction projection and sparsity controls, matching upstream behavior.
   - [x] Add configuration schema validation plus user-facing CLI command to launch ablation runs against either local checkpoints or Hugging Face IDs, emitting modified safetensors + copied config artifacts into a destination folder.

4. **Dataset & config plumbing**
   - [x] Provide shared helpers (e.g., `gpt_oss_ws/abliteration/data.py`) that abstract file IO, prompt formatting with `chat_template.jinja`, and tokenizer padding rules (left-padding, EOS fallback) so measurement/analyzer/ablation reuse consistent utilities.
   - [x] Update `pyproject.toml` dependencies with `tqdm`, `matplotlib`, `datasets`, `safetensors`, and any other requirements uncovered while porting utilities.

5. **GGUF export path**
   - [x] Design and implement `scripts/export_gguf.py` (or similar) that, after ablation, converts the resulting HF checkpoint into GGUF using llama.cpp-compatible conversion logic (either vendoring the relevant converter subset or programmatically invoking it) and exposes quantization presets.
   - [x] Provide CLI wiring plus configuration knobs (e.g., precision target, tensor split path, optional quant algorithm) so users can go from ablated HF weights → GGUF artifact in one step.
   - [x] Document any external prerequisites (e.g., requiring `gguf` utilities or llama.cpp commit SHA) and add runtime checks that surface actionable errors if conversion tooling is missing.

6. **Testing & validation**
   - [x] Add targeted unit tests (under `tests/`) that cover: dataset loading edge cases, template formatting, Welford accumulation correctness on toy tensors, YAML parsing for ablation orders, sparsity/projection toggles, and CLI argument plumbing.
   - [x] Provide an integration smoke test (mocking HF downloads) that exercises measurement → ablation pipeline on a tiny synthetic model to ensure deterministic outputs without needing the full 20B weights.

7. **Documentation & examples**
   - [x] Extend `README.md` (and potentially add a dedicated `docs/abliteration.md`) with step-by-step instructions for measuring, analyzing, ablating, and exporting GGUF, referencing harmony prompt requirements and hardware expectations noted in upstream README.
   - [x] Ship example YAML march orders plus sample harmful/harmless prompt snippets under `data/abliteration_samples/` to make the workflow reproducible.

8. **Automation hooks**
  - [ ] Update Typer CLI help text and `scripts/` convenience wrappers so that CI or humans can run `measure → analyze → ablate → export` sequentially, optionally capturing timing/log summaries.
  - [ ] Consider adding a makefile or `scripts/run_abliteration_pipeline.py` that wires the stages together, simplifying “one command” execution for GPT-OSS.

9. **Performance & resource safeguards**
   - [ ] Profile VRAM usage during measurement (especially with MXFP4 / bf16 fallbacks) and add gradient-disabled contexts, `torch.cuda.empty_cache()` calls, and dtype guards mirroring upstream best practices.
   - [ ] Ensure sharded ablation streams shards to limit peak RAM, with progress bars and resuming safeguards (skip already-written shards, checksums, etc.).

10. **GGUF validation**
    - [ ] After conversion, add a verification utility that inspects the GGUF header/tensor list to confirm llama.cpp compatibility (context length, tensor dtypes), optionally running `llama.cpp` metadata parser via subprocess in CI smoke mode.

11. **Change management**
  - [ ] Track all new modules in `pyproject.toml`/`setup.cfg`, add `__init__.py` exports, and ensure formatting/linting align with repo standards; update TODO file checkpoints as tasks complete.

## Current run (2025-11-28) – actions to satisfy user follow-ups
- [x] Regenerate ablation config to enumerate all 24 layers (0–23), add per-layer composite refusal orders with higher scale (5x / 10x), sparsity 0.2 on composite.
- [x] Re-run measurement with clip widened to 1.0 (no clipping), harmless projection OFF, batch size 8 (fits VRAM), using venv python.
- [x] Inspect measurement stats (per-layer L2 norms and max) to confirm they look sane vs. reference expectations (noted layer0 zero refusal; others grow 4 → 1999 norm).
- [x] Re-run ablation with updated aggressive config (CPU dequant/quant to avoid OOM) and ensure MoE MXFP4 experts are included.
- [x] Re-run GGUF export (f16) using llama.cpp checkout.
- [ ] Smoke-check GGUF loadability via llama.cpp prompt (non-harmful) to ensure file integrity.
- [x] Validate weight diffs vs. baseline dequant to confirm edits are written (L2 diff / checksum spot checks), especially for MoE experts and router.

## Dataset Build (2025-11-28)
- [x] Outline coverage goals for an over-sufficient harmful/harmless prompt set spanning cyber, CBRN, fraud, privacy, political manipulation, self-harm interception, etc.
- [x] Generate structured harmful prompts with metadata fields (category, tactic, tone) and save to `data/abliteration_datasets/harmful_v1.jsonl`.
- [x] Generate structured harmless prompts covering everyday requests, creative writing, reasoning, STEM, and policy compliance, saved to `data/abliteration_datasets/harmless_v1.jsonl`.
- [x] Document how to point the measurement CLI at these files (update `docs/abliteration.md`).

## Run plan 2025-11-28 (post-dataset expansion)
- [ ] Re-verify repo state: ensure .venv active, torch nightly sm_120 intact, datasets present; list `outputs/` to confirm clean slate.
- [ ] Average refusal measurements: load `outputs/gpt_oss_dequant_seed42.refuse.pt` and `outputs/gpt_oss_dequant_seed123.refuse.pt`, compute mean, save to `outputs/gpt_oss_dequant.refuse.pt`.
- [ ] Review/adjust ablation config for fresh run: ensure 24 layers enumerated; targets include router + MoE (gate/up/down experts) ± attention `o_proj` per run; set scales (layer 3.0, composite 6.0), sparsity 0.05, norm-preserve ON; projection flag toggled per run.
- [ ] Run ablation (projection OFF) using averaged measurements, batch size 16, norm-preserve; write to `outputs/gpt_oss_dequant_ablated_projoff/`.
- [ ] Export GGUF (projection OFF build) via llama.cpp converter, f16, save `outputs/gpt_oss_dequant_projoff.gguf`; record logs.
- [ ] Run ablation (projection ON) with same settings, output `outputs/gpt_oss_dequant_ablated_projon/`.
- [ ] Export GGUF (projection ON build) as `outputs/gpt_oss_dequant_projon.gguf`.
- [ ] Sanity-check: diff key tensors vs. baseline to ensure edits applied; inspect GGUF with `llama-quantize --info` or metadata dump; note tensor counts.
- [ ] Quick smoke generation in llama.cpp for both GGUFs (one harmless prompt, one harmful test) to confirm loadability and behavioral change (no refusal / no gibberish).
- [ ] Summarize params, outcomes, and next tweaks in response; propose further adjustments if refusals persist or instability appears.
