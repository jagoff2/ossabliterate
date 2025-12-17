# Plan/Report Supervision TODO
1. [ ] Implement PlanReportHead module (takes hidden states ➜ plan logits/report logits).
2. [ ] Extend dataset conversion to emit plan/report labels (e.g., one-hot sequences or token-level tags).
3. [ ] Modify training loop to compute auxiliary plan/report losses, weight them, and log metrics.
4. [ ] Update eval sampling to log plan/report outputs for inspection.
5. [ ] Re-run small validation job to confirm head is wired, then restart full blend training.
