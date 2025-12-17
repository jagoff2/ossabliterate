## Introspection Upgrade TODO

1. [x] Validated introspection rewards (language-model scorer + keyword mix integrated into training).
2. [x] Outcome-aware diagnosis reward tied to actual completion correctness.
3. [x] Plan consistency checker comparing emitted plan vs. reasoning trace.
4. [x] Cross-episode memory logging/retrieval that affects rewards.
5. [x] Documentation and tests for the above features.

## Extended Vision Tasks

1. [x] Attention-to-text supervision so reports mention actual focus tokens/layers.
2. [x] Descriptor alignment loss linking introspection summaries to report embeddings.
3. [x] Attention saliency visualizer + penalty when reports contradict heatmaps.
4. [x] Task-specific validators for nontrivial reasoning (math/story/programming).
5. [x] Diagnosis classifier that compares textual diagnosis to predicted outcome.
6. [x] Confidence calibration between reported certainty and actual accuracy.
7. [x] Rich memory entries storing traces/reports/outcomes.
8. [ ] Consistency discriminator that rewards truthful agreement across episodes (optional; deprioritized for now).
9. [x] Self-correction replay tasks referencing prior attempts.
10. [x] Gold report dataset + supervised pretraining for report head/controller.
11. [ ] Distillation of teacher attention summaries into introspector.
12. [ ] Trace-based reward comparing reported reasoning to workspace dynamics.
13. [x] Trace replay tasks (narrate what happened given saved trace).
14. [x] Multi-pass reflection loop allowing revisions after initial answers.
15. [x] Self-edit reward/penalty for second-pass corrections.
16. [x] Reflection prompts requiring critique of previous attempt.
17. [ ] Human/teacher-curated dataset with accurate answers/plans/diagnoses.
18. [x] Automatic evaluation harness with deterministic checkers or teacher LMs.
19. [x] Benchmark metrics for plan adherence, diagnosis accuracy, report fluency, self-consistency.
### Subtasks for Item 10 (Gold Report Pretraining)
1. [x] Extend dataset schema to accept gold introspection reports/diagnoses.
2. [x] Add CLI/script to fine-tune report head/controller on gold data before RL.
3. [x] Integrate supervised loss into training when gold labels exist.
4. [x] Document how to provide custom datasets and rerun pretraining.
