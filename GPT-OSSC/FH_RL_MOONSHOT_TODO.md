# FH-RL Moonshot Checklist
(Status: [ ] pending, [~] in progress, [x] done)

1. [x] Build/expand introspection dataset to >=20k episodes -> data/introspection_full.jsonl
2. [x] Implement data mixer to blend C4 and introspection (training/data_mixers.py)
3. [x] Create blended data manifest configs/data_blend.json (weights, paths)
4. [x] Update long-run config (configs/fh_rl_dialogpt_blend.json) to use mixer and new paths; adjust steps/batch for big run
5. [x] Add ToM/IRR/ESRI/RDP logging in long-run trainer (reuse tiny-model metrics)
6. [x] Integrate ToM eval harness (plans/reports) for validation (using blended val mix w/ introspection weight)
7. [x] Launch long FH-RL run on blended data; save checkpoints + logs (PID 6810, logs: runs/fh_rl_dialogpt_blend/train.log)
