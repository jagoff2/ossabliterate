## Next Actions TODO

0. **Imitation & Reward Gating**
   - [x] Add a pre-RL imitation phase that teacher-forces the entire gold completion so the model starts from a stable template.
   - [x] Gate text/plan/monitor rewards so no credit is given unless plan/monitor structure hits required thresholds.

1. **Reflection Enhancements**
   - Ensure reflection prompt enforces referencing prior Step/Monitor entries.
   - Add reward when Reflection includes a 'Trace Critique' mentioning focus terms.
   - Penalize reflections that repeat the previous completion verbatim.

2. **Trace-Based Supervision**
   - Build a lightweight classifier from attention traces -> predicted focus tokens.
   - Compare the classifier output with the introspection report, rewarding agreement.
   - Require a "Trace Summary:" section with layer/head references; supervise with synthetic labels.

3. **Memory & Evaluation**
   - Extend memory entries to store per-step descriptors and make reflection read them.
   - Implement an evaluator that scores plan adherence, diagnosis accuracy, confidence calibration, and report quality per chunk.
   - Run the evaluator every N episodes and log metrics for adaptive reward tuning.
