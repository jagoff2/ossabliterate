# Dataset Expansion TODO
1. [ ] Define blended dataset recipe in `docs/fh_rl_dataset_plan.md` (sources, mixing ratios, preprocessing).
2. [ ] Ingest a C4 subset (≈10M tokens) into `data/c4_chunk_train.jsonl` and `..._val.jsonl` (tokenized to GPT-2 ids).
3. [ ] Expand introspection dataset via generator scripts to ≥20k entries (`data/introspection_full.jsonl`).
4. [ ] Create mixing manifest `configs/data_blend.json` describing sampling weights between natural language and introspection data.
5. [ ] Add dataloader helper `training/data_mixers.py` to feed mixed batches.
6. [ ] Update training config (`configs/fh_rl_gpt2_moonshot.json`) to reference new data pipeline.
