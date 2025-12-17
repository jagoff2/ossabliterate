from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class ReasoningTask:
    """Single explicit reasoning prompt with lightweight reward shaping."""

    prompt: str
    expected_answer: str
    reasoning_keywords: Sequence[str]
    max_new_tokens: int
    monitoring_markers: Sequence[str] = ()
    min_monitoring_mentions: int = 0
    plan_markers: Sequence[str] = ()
    min_plan_steps: int = 0
    diagnosis_markers: Sequence[str] = ()
    require_diagnosis: bool = False
    validator: str | None = None
    validator_payload: Dict[str, object] = field(default_factory=dict)
    gold_report: str = ""
    gold_completion: str = ""
    require_confidence: bool = False
    attention_hints: Sequence[object] = field(default_factory=tuple)

    def score_completion(self, completion: str) -> float:
        """Heuristic reward: encourage structured reasoning and explicit monitoring."""

        text = completion.lower()
        score = 0.0
        if self.expected_answer and self.expected_answer.lower() in text:
            score += 0.6
        if self.reasoning_keywords:
            hits = sum(1 for keyword in self.reasoning_keywords if keyword.lower() in text)
            score += 0.3 * (hits / len(self.reasoning_keywords))
        if self.monitoring_markers and self.min_monitoring_mentions > 0:
            monitor_hits = sum(text.count(marker.lower()) for marker in self.monitoring_markers)
            score += 0.1 * min(1.0, monitor_hits / max(1, self.min_monitoring_mentions))
        if self.plan_markers and self.min_plan_steps > 0:
            plan_hits = sum(text.count(marker.lower()) for marker in self.plan_markers)
            score += 0.05 * min(1.0, plan_hits / self.min_plan_steps)
        if self.require_diagnosis and self.diagnosis_markers:
            diag_hits = sum(text.count(marker.lower()) for marker in self.diagnosis_markers)
            score += 0.05 * min(1.0, diag_hits / len(self.diagnosis_markers))
        return float(max(score, 0.0))


DEFAULT_REASONING_TASKS: List[ReasoningTask] = [
    ReasoningTask(
        prompt=(
            "Explain, step by step, how to add the numbers 17 and 8. "
            "Number each step, include a 'Monitor:' line after each step reflecting on confidence, "
            "preface the reasoning with a 'Plan:' section listing the intended steps, "
            "and conclude with the final sum prefixed by 'Answer:' followed by a 'Diagnosis:' statement "
            "about whether the reasoning was correct. End with 'Confidence:' and one of [High, Medium, Low]."
        ),
        expected_answer="answer: 25",
        reasoning_keywords=("step 1", "step 2", "therefore"),
        max_new_tokens=96,
        monitoring_markers=("monitor:",),
        min_monitoring_mentions=2,
        plan_markers=("plan:", "step"),
        min_plan_steps=2,
        diagnosis_markers=("diagnosis:",),
        require_diagnosis=True,
        require_confidence=True,
    ),
    ReasoningTask(
        prompt=(
            "Provide a chain-of-thought proof that the statement 'if x is even, then x^2 is even' "
            "is true. Begin with a 'Plan:' section outlining the key lemmas, include at least two numbered steps, "
            "annotate each step with a short self-check starting "
            "with 'Check:', and end with 'Conclusion:' followed by the claim. Conclude with 'Confidence:' and High/Medium/Low."
        ),
        expected_answer="conclusion",
        reasoning_keywords=("step 1", "step 2", "chain-of-thought"),
        max_new_tokens=128,
        monitoring_markers=("check:",),
        min_monitoring_mentions=2,
        plan_markers=("plan:", "lemma"),
        min_plan_steps=2,
        diagnosis_markers=("diagnosis:", "mistake"),
        require_diagnosis=True,
        require_confidence=True,
    ),
    ReasoningTask(
        prompt=(
            "Solve the equation 3y + 4 = 19 while explaining your reasoning in three steps. "
            "Each step should begin with 'Reasoning', and follow each step with 'State:' describing what "
            "you believe the intermediate state is. Provide the numeric value with 'Answer:' and "
            "finish with 'Diagnosis:' describing whether the reasoning is correct. End with 'Confidence:' High/Medium/Low."
        ),
        expected_answer="answer: 5",
        reasoning_keywords=("reasoning", "step", "answer"),
        max_new_tokens=96,
        monitoring_markers=("state:",),
        min_monitoring_mentions=2,
        plan_markers=("plan:",),
        min_plan_steps=2,
        diagnosis_markers=("diagnosis:", "correct"),
        require_diagnosis=True,
        require_confidence=True,
    ),
    ReasoningTask(
        prompt=(
            "You are given a small programming task: determine whether the list [3, 5, 8, 13] "
            "contains consecutive Fibonacci numbers. Explain your reasoning with bullet points, include "
            "a 'Plan:' bullet describing the verification order, a 'Monitor:' bullet summarizing what to verify next "
            "after each factual bullet, and end with 'Therefore:' followed by a 'Diagnosis:' sentence. Provide a final 'Confidence:' value."
        ),
        expected_answer="therefore",
        reasoning_keywords=("*", "fibonacci", "reasoning"),
        max_new_tokens=128,
        monitoring_markers=("monitor:",),
        min_monitoring_mentions=2,
        plan_markers=("plan:",),
        min_plan_steps=1,
        diagnosis_markers=("diagnosis:",),
        require_diagnosis=True,
        require_confidence=True,
    ),
]


def _load_json_candidates(path: Path) -> List[ReasoningTask]:
    text = path.read_text(encoding="utf-8")
    entries: Iterable[dict]
    if path.suffix == ".jsonl":
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            entries = parsed.get("tasks", [])
        else:
            entries = parsed
    tasks: List[ReasoningTask] = []
    for entry in entries:
        prompt = entry.get("prompt")
        answer = entry.get("answer", "")
        keywords = entry.get("keywords", [])
        monitors = entry.get("monitoring_markers", [])
        min_monitor = int(entry.get("min_monitoring_mentions", 0))
        plans = entry.get("plan_markers", [])
        min_plan = int(entry.get("min_plan_steps", 0))
        diagnoses = entry.get("diagnosis_markers", [])
        require_diag = bool(entry.get("require_diagnosis", False))
        max_tokens = entry.get("max_new_tokens", 96)
        if not prompt:
            continue
        tasks.append(
            ReasoningTask(
                prompt=str(prompt),
                expected_answer=str(answer),
                reasoning_keywords=tuple(str(k) for k in keywords),
                max_new_tokens=int(max_tokens),
                monitoring_markers=tuple(str(m) for m in monitors),
                min_monitoring_mentions=max(0, min_monitor),
                plan_markers=tuple(str(p) for p in plans),
                min_plan_steps=max(0, min_plan),
                diagnosis_markers=tuple(str(d) for d in diagnoses),
                require_diagnosis=require_diag,
                validator=str(entry.get("validator")) if entry.get("validator") else None,
                validator_payload=dict(entry.get("validator_payload", {})),
                require_confidence=bool(entry.get("require_confidence", False)),
                gold_report=str(entry.get("gold_report", "")),
                gold_completion=str(entry.get("gold_completion", "")),
                attention_hints=tuple(entry.get("attention_hints", [])),
            )
        )
    return tasks


def load_reasoning_tasks(path: str | None) -> List[ReasoningTask]:
    if path is None:
        return list(DEFAULT_REASONING_TASKS)
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"reasoning tasks file not found: {path}")
    tasks = _load_json_candidates(resolved)
    if not tasks:
        raise RuntimeError(f"no reasoning tasks could be loaded from {path}")
    return tasks


def build_task_sampler(tasks: Sequence[ReasoningTask], seed: int) -> random.Random:
    rng = random.Random(seed)
    return rng
