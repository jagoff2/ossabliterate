## Workspace Training Roadmap

### 1. Capture Targets
- Residual tensors for each `hooked_layer` at the final token position.
- Decoder logits for the same step, before workspace modifications.
- Controller entropy heuristics, plan energy, and workspace slots.
- Generated tokens and attention metadata required for loss computation.

### 2. Storage Schema
- Persist batches as torch `.pt` blobs with keys:
  - `input_ids`, `attention_mask`
  - `residuals` (dict[layer] -> tensor)
  - `logits`
  - `plan_energy`, `slots`
  - `metadata` (prompt text, seed, toggles)
- Index with lightweight JSON manifest for streaming.

### 3. Training Objectives
- Cross-entropy loss on next-token logits with gradients flowing through probes, workspace, controller.
- Auxiliary broadcast loss measuring perplexity delta when virtual KV is appended.
- Optional KL term to regularise controller outputs toward heuristic baseline.

### 4. Runtime Integration
- Load trained weights via new `WorkspaceConfig` fields.
- Preserve backward-compatible defaults (fallback to heuristic behaviour when weights absent).

### 5. Performance Considerations (CPU Only)
- Prefer bf16 for probes/workspace to minimise RAM.
- Reuse preallocated buffers during capture/training loops.
- Gate optional features behind CLI flags for iterative experimentation.


### 6. Offline Capture & Training Workflow
- Capture prompts into `.pt` blobs: `python scripts/capture_workspace_data.py --prompts prompts.txt --output data/capture`
- Fine-tune probes/controller: `python scripts/train_workspace.py --manifest data/capture/manifest.jsonl --epochs 3 --device cpu`
- Load trained weights via config or `--workspace-state` flag when launching the CLI server.

### 7. Runtime Options
- `kv_plan_scale`, `kv_plan_bias`, `kv_projection_scale` let you regulate virtual KV strength.
- `log_virtual_kv_stats` enables per-layer norm telemetry retrievable from capture buffers and the profiling script.
- `chunk_size` caps active cache length; the server forwards this automatically when set in config.
- `enable_torch_compile`, `torch_compile_mode`, and `inference_threads` expose CPU-friendly optimisations.

### 8. Tooling
- `scripts/profile_workspace.py` reports latency and RSS deltas (uses `psutil` when available).
- `configs/cpu_small.yaml` provides a reduced-footprint preset for constrained CPU hosts.
`scripts/capture_workspace_data.py` and `scripts/train_workspace.py` share the same config loader, so they honour workspace tweaks automatically.

### 9. Upcoming Introspection Enhancements
- Introduce a **plan head** that verbalises the current workspace trace prior to final decoding, so the system explicitly states its intended steps.
- Persist workspace traces per rollout and feed them into offline critique rewards to strengthen diagnosis and planning behaviours.
- Add evaluation tasks that compare the stated diagnosis versus actual reward outcomes, ensuring self-reported correctness aligns with behaviour.

### 10. Plan-Aware Reasoning Dataset
- `scripts/generate_reasoning_dataset.py` synthesises hundreds of arithmetic, algebra, and story problems with strict instructions for `Plan`, `Step`, `Monitor`, and `Diagnosis` sections. Use:  
  `python scripts/generate_reasoning_dataset.py --count 500 --seed 2025 --output data/reasoning_plan_dataset.jsonl`
- Pass the resulting file to `training/train_gpt2_meta.py` via `--reasoning-tasks data/reasoning_plan_dataset.jsonl` to ensure RL episodes draw from the richer prompt set.
- Each entry records the ground-truth `answer`, keyword/marker requirements, and reward metadata so the trainer’s bonuses (plan, monitoring, diagnosis) have precise targets.
- Synthetic tasks also store a `validator` block (addition, subtraction, linear equation, division story, ratio) so `_run_validator` can confirm numeric correctness rather than relying on simple substring matches.
- To warm-start introspection, you can supply `gold_report` strings in your dataset and run `python scripts/pretrain_report_head.py --tasks path/to/tasks.jsonl --save-path checkpoints/report_pretrain.pt` before RL. This supervises the controller/report head against reference self-reports; during RL, `--report-supervision-weight` continues to apply cross-entropy loss whenever gold reports exist.

### 11. Introspection Reports in Training
- The reinforcement loop now calls `generate_introspection_report(...)` after every rollout, logs the textual self-report under `run_*/reports/episode_<N>.txt`, and assigns `--report-bonus` whenever the report references the requested markers.
- Workspace traces are stored alongside reports so future critique models can compare “what the network believed” with the emitted self-assessment.
- `episode_memory.jsonl` accumulates per-prompt outcomes; when a prompt reappears, agreeing with the stored diagnosis yields the `--memory-bonus`, while contradictions incur a penalty. This helps the agent maintain cross-episode consistency.
- Reports are also checked against the model’s actual attention: `--focus-bonus` rewards mentioning the most-attended tokens, while `--alignment-bonus` measures cosine similarity between the introspection summary and the report embedding. Missing focus tokens or misaligned summaries reduce reward automatically.
- A dedicated diagnosis classifier reads the introspection summary (`--diagnosis-pred-weight` controls its supervised loss). The textual `Diagnosis:` statement must agree with both the classifier and the validator to receive the `--diagnosis-pred-bonus`; inconsistent self-assessments are penalized.
- Each prompt now demands an explicit `Confidence:` level. `_extract_confidence` parses the final report, and `--confidence-bonus` rewards accurate calibration (high confidence when correct, low when wrong). Overconfident mistakes hurt reward automatically.
- `episode_memory.jsonl` now stores the introspection summary vector, focus terms, and workspace trace shapes alongside the report/outcome so downstream critique models can replay entire trajectories, not just scalar scores.
- `--trace-bonus` compares the report embedding against the mean workspace trace vector; reports that diverge from the actual workspace dynamics lose reward, reinforcing truthful narration of what the model actually did internally.
- Replay capability: when a prompt reappears, `--replay-probability` injects the previous completion/report into the new prompt, forcing the model to critique itself. Rewards `--replay-bonus`/`--replay-penalty` encourage fixing mistakes only when it actually improves correctness and references a `Correction:` block.
- Trace replay mode (`--trace-replay-probability`) mixes in the last run’s focus terms and instructs the agent to produce a `Trace Replay:` narrative; the `--trace-replay-bonus`/`--trace-replay-penalty` combo and `--trace-bonus` alignment metric ensure the written summary matches the actual workspace trajectory.
- Multi-pass reflection: `--reflection-passes > 1` runs an additional critique pass where the model receives its previous completion/report and must emit a `Reflection:` block before revising the plan/diagnosis/confidence. `--reflection-bonus` rewards successful self-corrections, while `--reflection-penalty` discourages regressions.
