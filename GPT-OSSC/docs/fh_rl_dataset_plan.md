# FH-RL Moonshot Dataset Plan

## Goals
- Provide enough raw text diversity for GPT-2 FH-RL to learn long-context fluency (C4 subset, 10M+ tokens).
- Supply rich introspection/ToM supervision so the reentry loop is actually trained to explain itself.
- Keep everything in GPT-2 tokenized JSONL for compatibility with the existing trainers.

## Components
1. **Natural Language Backbone (C4 Mini)**
   - Source: `allenai/c4` via HuggingFace (`c4/en` split).
   - Strategy: pull ~250k documents (~10M tokens after BPE), tokenize, chunk into 512-token windows.
   - Output: `data/c4_chunk_train.jsonl` (train) and `data/c4_chunk_val.jsonl` (val), each record `{input_ids, labels}`.

2. **Introspection Corpus**
   - Use `scripts/generate_reasoning_dataset.py` with expanded templates (plans, validator outputs, reflections).
   - Target size: ≥20k episodes, each with claim, gold completion, gold report, validator flags.
   - Save as `data/introspection_full.jsonl`.

3. **Blending**
   - Create `configs/data_blend.json` specifying sampling weights, e.g., 0.8 C4, 0.2 introspection, with optional temperature sampling for introspection difficulty tiers.
   - Training dataloader will randomly draw from either dataset per batch using these weights.

4. **Evaluation**
   - Natural-language eval: `data/c4_chunk_val.jsonl`
   - Introspection eval: sample 1k entries from `introspection_full` for validation.

## Next Steps
- Download/tokenize C4 chunk via new script `scripts/build_c4_subset.py`.
- Re-run `scripts/generate_reasoning_dataset.py --count 20000 --output data/introspection_full.jsonl` with high-verbosity prompts.
- Implement data mixer and update the FH-RL training config to use it.
