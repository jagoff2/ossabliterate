# Introspective Meta-Attention – Action Checklist
Each step below is atomic and executable within a single interactive turn.

1. **Extend dataset script structure** – Update `scripts/build_curated_reasoning_data.py` to support richer metadata fields (reflection memo, attention hints, validator type tags). ✅
2. **Author 8 new curated tasks** – Append eight fully specified tasks (prompt, completion, report, reflection, validator config) to the script and regenerate `data/reasoning_plan_dataset_curated_v2.jsonl`.
3. **Document schema** – Create `docs/curated_dataset.md` describing every JSON field so future contributors can add tasks consistently. ✅
4. **Implement evaluator harness skeleton** – Add `scripts/evaluate_reports.py` with CLI argument parsing, file loading, and placeholders for checks.
5. **Add prefix-quality + coverage checks** – Inside the evaluator, implement prefix-perplexity scoring and plan/monitor/diagnosis coverage metrics; emit JSON summary. ✅
6. **Wire evaluator into training loop** – Modify `training/train_gpt2_meta.py` to call the evaluator after each episode, zeroing reward when the evaluator flags an invalid completion. ✅
7. **Add self-judge reward head** – Extend the meta model with a small classifier head that predicts “Pass/Fail” for each episode; log its score in `progress.jsonl`. ✅
8. **Integrate self-judge into loss** – Modify training so the judge loss contributes to the reward signal (e.g., via a KL or BCE term) and is recorded in episode summaries. ✅
9. **Prototype graph workspace nodes** – Extend `MetaWorkspace` to store node structures (Plan/Step/Monitor) and expose APIs to add/merge nodes. ✅
10. **Log graph operations** – Update the training loop to record every graph operation into `progress.jsonl` so the evaluator can verify graph completeness. ✅
