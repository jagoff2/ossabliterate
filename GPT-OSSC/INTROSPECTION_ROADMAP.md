# Introspective Meta-Attention Roadmap

## Mission Statement
Build a GPT-2–based meta-attention system that exhibits genuine introspection, can explain its own focus of attention, diagnose past mistakes, and plan across many steps (toward an emergent theory of mind). Success requires three pillars: (1) high-quality supervised data with gold reflections and validators, (2) a distributed training pipeline for larger base models with reinforcement signals grounded in automated evaluators, and (3) a structured workspace (graph-of-thought) with multi-pass reflection and episodic memory.

## Phase 0 – Data + Evaluation Foundations (Week 1)
1. **Curated dataset expansion**
   - Target: 60–80 tasks across math, diagnostics, planning, multi-agent ToM, programming, and narrative reasoning.
   - Each entry must include:
     - Prompt, gold completion, validator/config, gold introspection report mentioning specific tokens and head/layer hints.
     - Reflection memo (à la Reflexion/Crystal) that critiques or confirms each step.
   - Store in `data/reasoning_plan_dataset_curated_v2.jsonl` with schema documentation in `docs/curated_dataset.md`.
2. **Automated evaluator harness**
   - Script: `scripts/evaluate_reports.py`.
   - Capabilities:
     - Prefix-quality check (perplexity + distance to first `Plan` token).
     - Plan/Monitor/Diagnosis coverage and repetition metrics.
     - Validator execution (math, story balance, ToM tasks) + fail-fast.
     - Reflection alignment score: measures references to prior focus terms or attention hints.
     - Tool-call sanity check once Toolformer-style APIs arrive.
   - Outputs JSON/CSV summary plus inline warnings for `progress.jsonl`.

## Phase 1 – Larger Base + Distributed Training (Week 1–2)
1. **Upgrade base model**
   - Use standard GPT-2 (124M) with gradient checkpointing.
   - Launch via `torchrun --nproc_per_node=2 ...` to consume both GPUs.
2. **Supervised warm-up**
   - Train on the expanded curated dataset until evaluator scores (prefix, plan, monitor, diagnosis, reflection) all ≥0.8.
   - Save checkpoint `runs/meta_warmup_curated_v2/meta_stack_warmup.pt`.
3. **Self-judging rewards**
   - Implement a self-critic head (“LLM-as-judge”) that scores each episode; combine with evaluator verdict to produce the RL reward.
   - Log judge scores in `progress.jsonl` for postmortem analysis.

## Phase 2 – Structured Graph Workspace (Week 2–3)
1. **Graph-of-Thought workspace**
   - Extend `MetaWorkspace` so each Plan/Step/Monitor becomes a node.
   - Edges encode dependencies (Plan→Step, Step→Monitor, Monitor→Diagnosis, cross-links for hypotheses).
   - Controller actions: `expand_node`, `merge`, `summarize`, `request_tool`.
2. **Graph rewards**
   - Reward nodes whose text matches attention tokens/validator evidence.
   - Penalize dangling nodes or cycles that never resolve.
   - Integrate log of graph operations into `progress.jsonl` so evaluators can detect missing branches.
3. **Toolformer-style API calls**
   - Provide stub calculator/search/QA tools; annotate curated data with tool calls.
   - Train the controller (self-labeling like Toolformer) to insert API usage when needed.

## Phase 3 – Reflection + Theory-of-Mind (Week 3–4)
1. **Multi-pass reflection (≥2 passes)**
   - Pass 0: base reasoning.
   - Pass 1: “Reflection” section referencing previous nodes, attention spans, validator outputs.
   - Reward when reflection fixes mistakes or acknowledges uncertainty; penalize when it repeats the prior answer verbatim.
2. **Episodic memory replay**
   - Store reflections per prompt in `episode_memory_rank*.jsonl`.
   - When a prompt reappears, feed the past reflection to the controller (Reflexion-style).
3. **Theory-of-mind tasks**
   - Add multi-agent ToM prompts that require reasoning about others’ beliefs.
   - Evaluator verifies that reflections reference other agents’ viewpoints.

## Phase 4 – Continuous Improvement and Scaling (Week 4+)
1. **Training loop cadence**
   - Alternate short RL chunks (≤5 episodes) with evaluation checkpoints.
   - Only run longer segments (1–5k episodes) when evaluator + judge scores stay ≥ thresholds for two consecutive chunks.
2. **Self-consistency enforcement**
   - Sample multiple reasoning paths, keep majority answer.
   - When disagreement occurs, trigger extra reflection or evaluator-guided rewrite.
3. **Final long run**
   - Once metrics are stable, run 20k+ episodes with `torchrun`.
   - Tail logs live; evaluator auto-aborts if metrics fall.

## Success Criteria
- Evaluator + judge metrics: plan/monitor coverage ≥0.85, diagnosis accuracy ≥0.9, reflection alignment ≥0.8.
- Introspection reports cite attention spans and past focus terms without scaffolding (verified by evaluator).
- Reports survive prefix-quality checks (no gibberish before Plan sections).
- Theory-of-mind benchmarks (custom multi-agent tasks + public datasets like ToMi) show ≥20% improvement over baseline.
- Long run completes without evaluator-triggered aborts, proving stability.

## Immediate Next Actions
1. Implement evaluator harness + prefix penalty (blocking).
2. Expand curated dataset generator to 60+ entries and regenerate warm-up checkpoint.
3. Integrate self-judge reward head and logging.
4. Prototype graph workspace extensions.
