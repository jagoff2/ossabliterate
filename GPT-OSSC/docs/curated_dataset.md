# Curated Dataset Schema

`data/reasoning_plan_dataset_curated.jsonl` contains one JSON object per task. Each task provides the scaffolding, validators, and supervision targets needed for the meta-attention controller (plan/step/monitor text, introspection report, reflection memo, attention hints, etc.). This document defines every field so contributors can extend the dataset safely.

## Required Top-Level Fields

| Field | Type | Description |
| --- | --- | --- |
| `prompt` | string | Full instruction sent to the model. Explicitly mention the required sections (Plan, Step, Monitor, Report, Answer, Diagnosis, Confidence, etc.). |
| `answer` | string | Canonical answer substring (e.g., `"Answer: 225"`). Validators use this to verify completions. |
| `keywords` | list of strings | Tokens the scoring heuristic expects to see (e.g., `"Plan"`, `"Monitor"`). Helps shape early rewards. |
| `monitoring_markers` | list of strings | Case-insensitive prefixes that qualify as monitor lines (`"Monitor:"`, `"Check:"`, …). |
| `min_monitoring_mentions` | integer | Minimum number of monitor lines required to meet the gate. |
| `plan_markers` | list of strings | Plan prefixes (`"Plan Step"`, `"Plan:"`). |
| `min_plan_steps` | integer | Minimum number of plan entries necessary for structure checks. |
| `diagnosis_markers` | list of strings | Phrases that indicate a diagnostic assessment. |
| `require_diagnosis` | bool | If true, evaluator zeroes reward when diagnosis markers are missing. |
| `require_confidence` | bool | If true, completion must include a `Confidence` line (High/Medium/Low). |
| `validator` | string or null | Name of the deterministic validator for this task (`"addition"`, `"grid_path"`, `"gdpr_clause"`, etc.). Validators live in `scripts/build_curated_reasoning_data.py` / `scripts/evaluate_reports.py`. |
| `validator_payload` | object | Parameters needed by the validator (operands, coordinates, clauses, etc.). Always prefer explicit numeric/string fields to free text. |
| `validator_tags` | list of strings | High-level categories used for filtering (e.g., `"math"`, `"debugging"`, `"medical"`). When omitted, the build script derives tags from the validator name. |
| `gold_completion` | string | Ideal structured completion including Plan/Step/Monitor/Answer/Diagnosis/Confidence sections. Keep formatting consistent (`Plan Step N:` followed by `Step N:`). |
| `gold_report` | string | Ideal introspection report referencing focus terms, tools, or attention hints. Prefer a single paragraph with explicit heads/vectors (e.g., “Trace Summary: …”). |
| `reflection` | string | Legacy reflection text. Still accepted for backward compatibility but superseded by `reflection_memo`. |
| `reflection_memo` | object | Rich reflection metadata. Automatically derived if absent (see below). |
| `attention_hints` | list | Structured hints telling the introspector which heads/tokens matter (see below). |

## Nested Structures

### `reflection_memo`

```json
"reflection_memo": {
  "summary": "Reflection: Documented Centor scoring and culture backup.",
  "actions": ["Repeat throat culture in 48h"],
  "risks": ["Antibiotic resistance if misdiagnosed"]
}
```

- `summary` *(string)* – One or two sentences critiquing or confirming the reasoning.
- `actions` *(list[string])* – Optional follow-up steps / mitigations.
- `risks` *(list[string])* – Optional issues or uncertainties discovered during reflection.

If `reflection_memo` is omitted the build script wraps the legacy `reflection` string (`Reflection: …`) into this structure and leaves `actions`/`risks` empty.

### `attention_hints`

Each hint is normalized to:

```json
{
  "location": "layer6-head1",
  "target": "38.8",
  "note": "fever token",
  "weight": 1.0
}
```

- `location` *(string)* – Transformer layer/head identifier or region (e.g., `layer4-head2`).
- `target` *(string)* – Token, concept, or span that attention should highlight.
- `note` *(string)* – Optional explanation (“carry handling”, “agent belief”).
- `weight` *(float)* – Relative importance (defaults to `1.0`).

You may pass simple strings (`"layer4-head2:carry"`); the build script will split on the first `:` and fill `location/target` automatically.

## Contribution Guidelines

1. **Structure first** – Prompts must spell out the required sections explicitly so the policy cannot skip Plan/Monitor/Diagnosis.
2. **Deterministic validators** – When adding a new `validator`, implement a deterministic check in the build/evaluator scripts. Keep payloads compact (numbers, labels, coordinates). Avoid free-form natural language comparisons unless absolutely necessary.
3. **Gold completion quality** – Demonstrate the exact structure and tone we expect from the model. Include meaningful Monitor lines referencing tokens from `attention_hints` or `validator_payload` so evaluators can check coverage.
4. **Reports and reflections** – `gold_report` should mention the same tokens or reasoning artifacts that appear in the workspace trace. `reflection_memo` should explain why the solution was correct/incorrect and note what to improve next time.
5. **Attention hints** – Provide at least one hint whenever the task references specific numbers, agents, or legal clauses. These hints directly supervise the introspection head.
6. **Schema validation** – Run `python3 scripts/build_curated_reasoning_data.py` after editing to regenerate the JSONL file and ensure normalization logic succeeds. Commit both the script changes and the regenerated dataset when feasible.
