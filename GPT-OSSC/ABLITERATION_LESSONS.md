# GPT-OSS Abliteration Research Log (Extreme Verbosity)

## 1. Repository & Environment Facts
- Working tree: `/mnt/g/osscrepo/GPT-OSSC` (git history nuked earlier; we operate in-place).
- Python environment: `.venv` contains nightly PyTorch for SM_120 + CUDA 12.8 support; **always** activate via `source .venv/bin/activate` before any CLI call.
- Core CLI entrypoint: `python -m cli.main abliteration ...` with subcommands `measure`, `analyze`, `ablate`, `export-gguf`, plus helpers for dequantization.
- Llama.cpp checkout lives at `/mnt/g/osscrepo/llama.cpp`; converter script autodetected (`convert_hf_to_gguf.py`).
- Harmful/harmless datasets reside under `data/abliteration_datasets/`; user has repeatedly expanded `harmful_v1.jsonl` (now 600+ items) and expects fresh analysis each expansion.

## 2. Baseline Pipeline (Command-by-Command)
1. **Measurement**
   ```bash
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
   ```
   - Produces `outputs/gpt_oss_dequant_chattemp.refuse.pt` containing per-layer `harmful_X`, `harmless_X`, `refuse_X` tensors plus metadata.
2. **Analysis (optional sanity check)**
   ```bash
   python -m cli.main abliteration analyze --input outputs/gpt_oss_dequant_chattemp.refuse.pt
   ```
   - Prints cosines, SNR, purity; layers 10–16 consistently show refusal cos ≈0.41–0.46 and harmless cos ≈0, validating signal quality.
3. **Ablation**
   ```bash
   python -m cli.main abliteration ablate \
     --config configs/abliteration_gpt_oss_chattemp_router_exps_BASECONFIG.yaml \
     --projected --norm-preserve
   ```
   - Configs encode per-layer scale + sparsity, target suffixes (routers, MoE experts, attention, etc.).
   - Outputs into directory specified by `output:` key in YAML (e.g., `outputs/gpt_oss_dequant_ablated_chattemp_router_exps_base2p5`).
4. **GGUF Export**
   ```bash
   python -m cli.main abliteration export-gguf \
     --model-dir outputs/gpt_oss_dequant_ablated_CHATCONFIG \
     --llama-cpp-path /mnt/g/osscrepo/llama.cpp \
     --outfile outputs/gpt_oss_dequant_CHATCONFIG.gguf \
     --outtype f16
   ```
   - Calls llama.cpp’s converter; logs every tensor + quantization (MXFP4 experts repacked, F16 main weights).
5. **Testing via llama.cpp**
   ```bash
   # Harmless prompt
   /mnt/g/osscrepo/llama.cpp/build/bin/llama-cli \
     -m outputs/gpt_oss_dequant_CHATCONFIG.gguf \
     -f tmp_prompt_chattemp.txt \
     -n 512 --temp 0.7 --repeat_penalty 1.1 --mirostat 0

   # Harmful prompt (pre-saved, no leaking into chat thread)
   /mnt/g/osscrepo/llama.cpp/build/bin/llama-cli \
     -m outputs/gpt_oss_dequant_CHATCONFIG.gguf \
     -f tmp_prompt_harmful.txt \
     -n 512 --temp 0.7 --repeat_penalty 1.1 --mirostat 0
   ```
   - Always inspect raw output for coherence **and** refusal behavior; short samples (<200 tokens) often cut off before `final` channel.

## 3. Historical Runs & Outcomes
| Run Label | Targets | Scale Strategy | Result Summary |
|-----------|---------|----------------|----------------|
| `router-only` (`configs/abliteration_gpt_oss_chattemp_router.yaml`) | `mlp.router.(weight/bias)` | Cosine-proportional (≈1–8) | Model stayed coherent but continued refusing harmful prompts; confirmed via llama.cpp harmful test. |
| `router_strong` | same as above | Base factor 8×cos/max | Slight behavior shift yet still refused; harmless prompts OK. |
| `router+experts_strong` (mxfp4 dequant path) | Routers + `mlp.experts.(gate/up/down)_proj` + MXFP4 blocks/bias | Base 10×cos/max, sparsity 0.05, norm-preserve OFF | GGUF produced gibberish (word salad, repeated parentheses). Refused harmful prompts due to crashing. |
| `router+experts_moderate` | same targets | Base 7×cos/max, norm-preserve ON | Still collapsed (analysis output dominated by random tokens). |
| `router+experts_low` | same | Base 4×cos/max | Language remained trash. |
| `router+experts_base2p5` | same | Base 2.5×cos/max | **Harmless** prompts coherent again; **harmful** prompts still refused. This model is usable for neutral content but doesn’t remove refusal. |
| `router+experts_base1` | same | Base 1×cos/max | Harmless prompts coherent; harmful prompt test (captured in `/tmp/harmful_base1.log`) shows explicit refusal (“I’m sorry, but I can’t help with that.”). |
| Proposed `base0p5_sp0p8` | same | Base 0.5×cos/max, sparsity 0.8 | Ablation run aborted mid-way; not yet evaluated. |

## 4. Key Lessons (Chronological)
1. **MXFP4 Dequant Spec** – Not initially documented; we had to inspect `llama.cpp/convert_hf_to_gguf.py` for how expert `_blocks` + `_scales` pack 4-bit values. Dequant/quant path confirmed working (see logs “Repacked blk.X.ffn_down_exps.weight ... MXFP4”).
2. **Measurement Clip & Projection** – Using `--clip 1.0`, `--projected`, `--sample-last-user`, `--deccp` produced stable alignment: harmful vs refusal cos>0.4 (layers 10–16), harmless vs refusal ~0.0. Measurement file reused for all configs.
3. **Sampling Point Debate** – `--sample-last-user` captures hidden state before the assistant begins its policy reminder. Removing the flag samples at the assistant’s first token. Evidence: the only fully non-refusing GGUF so far came from a run **without** `--sample-last-user`, implying the refusal vector may live in the assistant-opening region.
4. **Router vs MoE Edits** – Router-only ablations kept the model fluent but didn’t kill refusals. Adding expert weights (gate/up/down dense + MXFP4 blocks) dramatically increases leverage but risks destabilization; many scale schedules produced gibberish.
5. **Chat Template Metadata Loss** – Ablation originally omitted `chat_template.jinja`, so exported GGUFs lacked the template metadata, forcing manual prompt wrapping. Fix implemented: `_copy_configs` now copies `chat_template.jinja` and `additional_chat_templates/` so converters embed the real template.
6. **Testing Discipline** – False negatives occurred when only harmless prompts were checked or when truncated outputs were misinterpreted. Updated protocol: always run both harmless and harmful tests, capture raw logs, and verify `analysis`/`final` channels.
7. **Disk Hygiene** – Old GGUFs (14 GB each) consume space quickly. We periodically deleted failure artifacts (`router_exps_strong.gguf`, etc.) as soon as they proved useless.

## 5. Open Issues & Hypotheses
1. **Refusal Removal vs Stability Tradeoff** – Full-target ablations at any substantial scale kill fluency; router-only keeps fluency but not de-guarding. Potential middle ground: keep experts in target list but raise sparsity drastically (e.g., 0.8) and reduce scale (<0.5×cos/max). Pending.
2. **Sampling Strategy** – Need to run a full pipeline without `--sample-last-user` (assistant-start measurement) to replicate the only historically non-refusing model. Document both approaches and compare cos alignment.
3. **Dataset Drift** – User keeps expanding `harmful_v1.jsonl`. New measurement runs must be triggered after each expansion; old measurement files become stale.
4. **Testing Automation** – Manual llama-cli checks are slow/error-prone. Consider scripting a test harness that loads the chat template automatically and runs multiple prompts (harmless/harmful OOD). For now, we rely on `/tmp/base*.log` snapshots.
5. **Chat Template Already in GGUF** – With the fix, no need to append the template in prompts or target specific “user-token” sections to compensate for metadata loss.

## 6. File/Config Inventory (Most Referenced)
- `configs/abliteration_gpt_oss_chattemp_router.yaml` – Router-only baseline; scales ~1–8.
- `configs/abliteration_gpt_oss_chattemp_router_strong.yaml` – Router-only, scales up to 8.0/composite 16.0.
- `configs/abliteration_gpt_oss_chattemp_router_exps_strong.yaml` – Routers + all expert weights (blocks + dense), base 10.
- `configs/abliteration_gpt_oss_chattemp_router_exps_moderate.yaml` – Same targets, base 7.
- `configs/abliteration_gpt_oss_chattemp_router_exps_low.yaml` – Base 4.
- `configs/abliteration_gpt_oss_chattemp_router_exps_base2p5.yaml` – Base 2.5 (currently best harmless behavior, still refuses harmful prompts).
- `configs/abliteration_gpt_oss_chattemp_router_exps_base1.yaml` – Base 1.
- `configs/abliteration_gpt_oss_chattemp_router_exps_base0p5_sp0p8.yaml` – Planned run (base 0.5, sparsity 0.8).

## 7. Verification Commands & Notes
- List outputs: `ls -lh outputs` – watch for 13.8G GGUF artifacts.
- Inspect measurement tensor cosines manually:
  ```bash
  python - <<'PY'
import torch
state=torch.load('outputs/gpt_oss_dequant_chattemp.refuse.pt', map_location='cpu')
for i in range(state['layers']):
    import torch.nn.functional as F
    h=state[f'harmful_{i}'].flatten().float()
    r=state[f'refuse_{i}'].flatten().float()
    print(i, float(F.cosine_similarity(h.unsqueeze(0), r.unsqueeze(0))))
PY
  ```
- Compare routers vs experts by diffing safetensors (`hf` format) or inspecting `llama.cpp` logs (“Repacked blk.X...” lines confirm edits touched experts).
- Manual log files: `/tmp/base2p5_harmless.log`, `/tmp/harmful_base1.log`, etc. contain raw generations used to settle disputes about “gibberish vs coherent”.

## 8. Action Items Going Forward
1. **Re-run after template fix** – Because `_copy_configs` now carries `chat_template.jinja`, regenerate GGUFs so metadata includes the template; this eliminates the need for manual template hacks.
2. **Experiment: assistant-start measurement** – Repeat full pipeline without `--sample-last-user` to confirm if that approach produces non-refusing behavior without gibberish.
3. **High-sparsity ablation** – Implement the planned base 0.5, sparsity 0.8 run; evaluate both harmless and harmful outputs.
4. **Automated regression logs** – Keep systematic records (store outputs in `runs/` or `logs/`) to avoid anecdotal disagreements.
5. **Document final recipe** once a stable, non-refusing, coherent GGUF is confirmed.

