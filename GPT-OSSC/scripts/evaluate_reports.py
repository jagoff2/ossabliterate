#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validator_utils import run_validator


DEFAULT_PLAN_MARKERS = ("plan step", "plan:")
DEFAULT_MONITOR_MARKERS = ("monitor:",)
DEFAULT_DIAG_MARKERS = ("diagnosis:", "analysis:")


@dataclass
class EpisodeMetrics:
    episode_id: Any
    prefix_quality: float
    prefix_perplexity: Optional[float]
    prefix_distance_score: float
    plan_coverage: float
    monitor_coverage: float
    diagnosis_coverage: float
    graph_plan_nodes: int
    graph_monitor_nodes: int
    graph_step_nodes: int
    invalid_reasons: List[str]
    validator_result: Optional[bool]
    teacher_score: Optional[float]
    graph_coverage: float


def load_reports(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def maybe_load_lm(model_name: str, device: str):
    if model_name.lower() == "none":
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model


def _extract_completion(entry: Dict[str, Any]) -> str:
    for key in ("completion", "completion_text", "completion_excerpt"):
        text = entry.get(key)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def _prefix_segment(text: str) -> Tuple[str, int]:
    lower = text.lower()
    idx = lower.find("plan:")
    if idx == -1:
        return text.strip(), len(text)
    return text[:idx].strip(), idx


def perplexity_score(text: str, tokenizer, model) -> Optional[float]:
    if not text or tokenizer is None or model is None:
        return None
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        loss = model(**encoded, labels=encoded["input_ids"]).loss
    return float(torch.exp(loss).item())


def normalize_perplexity(ppl: Optional[float], floor: float = 5.0, ceiling: float = 80.0) -> float:
    if ppl is None:
        return 0.0
    if ppl <= floor:
        return 1.0
    if ppl >= ceiling:
        return 0.0
    return 1.0 - (ppl - floor) / (ceiling - floor)


def distance_score(distance: int, cap: int) -> float:
    if distance <= 0:
        return 1.0
    if distance >= cap:
        return 0.0
    return 1.0 - (distance / cap)


def coverage_score(
    completion: str,
    markers: Iterable[str],
    minimum: int,
) -> float:
    if minimum <= 0:
        return 1.0
    lines = [ln.strip().lower() for ln in completion.splitlines() if ln.strip()]
    count = 0
    normalized_markers = tuple(m.lower() for m in markers)
    for ln in lines:
        if any(ln.startswith(marker) for marker in normalized_markers):
            count += 1
    return min(1.0, count / float(minimum))


def completion_teacher_score(text: str, tokenizer, model, max_length: int) -> Optional[float]:
    if not text or tokenizer is None or model is None:
        return None
    try:
        with torch.no_grad():
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            outputs = model(**encoded, labels=encoded["input_ids"])
            loss = outputs.loss
            return float(torch.exp(-loss).item())
    except Exception:
        return None


def graph_coverage_score(report_text: str, nodes: Iterable[Dict[str, Any]]) -> float:
    nodes = list(nodes or [])
    if not nodes or not report_text:
        return 0.0
    lowered = report_text.lower()
    hits = 0
    for node in nodes:
        label = str(node.get("label", "")).strip().lower()
        if label and label in lowered:
            hits += 1
    return hits / len(nodes)


def evaluate_episode(
    entry: Dict[str, Any],
    tokenizer,
    model,
    prefix_cap: int,
    plan_markers: Iterable[str],
    monitor_markers: Iterable[str],
    diag_markers: Iterable[str],
    plan_min: int,
    monitor_min: int,
    diag_min: int,
    teacher_tokenizer,
    teacher_model,
    teacher_max_length: int,
) -> Optional[EpisodeMetrics]:
    completion = _extract_completion(entry)
    if not completion:
        return None
    prefix_text, distance = _prefix_segment(completion)
    ppl = perplexity_score(prefix_text, tokenizer, model)
    ppl_score = normalize_perplexity(ppl)
    dist_score = distance_score(distance, prefix_cap)
    prefix_quality = 0.5 * (ppl_score + dist_score)
    plan_cov = coverage_score(completion, plan_markers, plan_min)
    monitor_cov = coverage_score(completion, monitor_markers, monitor_min)
    diag_cov = coverage_score(completion, diag_markers, diag_min)
    invalid = []
    if prefix_quality < 0.5:
        invalid.append("prefix-quality")
    if plan_cov < 0.75:
        invalid.append("plan-coverage")
    if monitor_cov < 0.75:
        invalid.append("monitor-coverage")
    if diag_min > 0 and diag_cov < 1.0:
        invalid.append("diagnosis-coverage")
    graph_nodes = entry.get("graph_nodes") or []
    validator_name = entry.get("validator")
    validator_payload = entry.get("validator_payload")
    validator_result = None
    if validator_name:
        validator_result = run_validator(validator_name, validator_payload, completion)
        if validator_result is False:
            invalid.append("validator")
    teacher_score = completion_teacher_score(
        completion,
        teacher_tokenizer,
        teacher_model,
        teacher_max_length,
    )
    graph_coverage = graph_coverage_score(entry.get("report"), graph_nodes)
    plan_node_count = sum(1 for node in graph_nodes if str(node.get("kind", "")).lower() == "plan")
    monitor_node_count = sum(1 for node in graph_nodes if str(node.get("kind", "")).lower() == "monitor")
    step_node_count = sum(1 for node in graph_nodes if str(node.get("kind", "")).lower() == "step")
    return EpisodeMetrics(
        episode_id=entry.get("episode"),
        prefix_quality=prefix_quality,
        prefix_perplexity=ppl,
        prefix_distance_score=dist_score,
        plan_coverage=plan_cov,
        monitor_coverage=monitor_cov,
        diagnosis_coverage=diag_cov,
        graph_plan_nodes=plan_node_count,
        graph_monitor_nodes=monitor_node_count,
        graph_step_nodes=step_node_count,
        invalid_reasons=invalid,
        validator_result=validator_result,
        teacher_score=teacher_score,
        graph_coverage=graph_coverage,
    )


def summarize(metrics: List[EpisodeMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {
            "total": 0,
            "invalid": 0,
            "averages": {},
            "episodes": [],
        }

    def avg(getter):
        values = [getter(m) for m in metrics if getter(m) is not None]
        return mean(values) if values else 0.0

    invalid_records = [
        {
            "episode": m.episode_id,
            "prefix_quality": m.prefix_quality,
            "plan_coverage": m.plan_coverage,
            "monitor_coverage": m.monitor_coverage,
            "diagnosis_coverage": m.diagnosis_coverage,
            "reasons": m.invalid_reasons,
        }
        for m in metrics
        if m.invalid_reasons
    ]

    report = {
        "total": len(metrics),
        "invalid": len(invalid_records),
        "averages": {
            "prefix_quality": avg(lambda m: m.prefix_quality),
            "prefix_perplexity": avg(lambda m: m.prefix_perplexity),
            "prefix_distance_score": avg(lambda m: m.prefix_distance_score),
            "plan_coverage": avg(lambda m: m.plan_coverage),
            "monitor_coverage": avg(lambda m: m.monitor_coverage),
            "diagnosis_coverage": avg(lambda m: m.diagnosis_coverage),
            "graph_plan_nodes": avg(lambda m: m.graph_plan_nodes),
            "graph_monitor_nodes": avg(lambda m: m.graph_monitor_nodes),
            "graph_step_nodes": avg(lambda m: m.graph_step_nodes),
            "validator_pass_rate": avg(lambda m: float(m.validator_result) if m.validator_result is not None else 0.0),
            "teacher_score": avg(lambda m: m.teacher_score),
            "graph_coverage": avg(lambda m: m.graph_coverage),
        },
        "invalid_records": invalid_records,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate introspection reports")
    parser.add_argument("progress", type=Path, help="Path to progress.jsonl")
    parser.add_argument("--output", type=Path, default=Path("reports_evaluation.json"))
    parser.add_argument("--plan-min", type=int, default=2, help="Min plan steps expected")
    parser.add_argument("--monitor-min", type=int, default=2, help="Min monitor mentions expected")
    parser.add_argument("--diagnosis-min", type=int, default=1, help="Min diagnosis mentions expected")
    parser.add_argument("--plan-marker", action="append", default=list(DEFAULT_PLAN_MARKERS))
    parser.add_argument("--monitor-marker", action="append", default=list(DEFAULT_MONITOR_MARKERS))
    parser.add_argument("--diagnosis-marker", action="append", default=list(DEFAULT_DIAG_MARKERS))
    parser.add_argument("--prefix-lm", default="gpt2", help="HF model name for prefix perplexity")
    parser.add_argument("--device", default="cpu", help="Device for LM (cpu or cuda)")
    parser.add_argument("--prefix-cap", type=int, default=160, help="Max chars allowed before Plan")
    parser.add_argument("--teacher-lm", default="none", help="Optional teacher LM for completion scoring")
    parser.add_argument("--teacher-device", default="cpu", help="Device for teacher LM")
    parser.add_argument("--teacher-max-length", type=int, default=512, help="Max length for teacher scoring")
    args = parser.parse_args()

    episodes = load_reports(args.progress)
    tokenizer, model = maybe_load_lm(args.prefix_lm, args.device)
    teacher_tokenizer, teacher_model = maybe_load_lm(args.teacher_lm, args.teacher_device)

    metrics: List[EpisodeMetrics] = []
    for entry in episodes:
        result = evaluate_episode(
            entry,
            tokenizer,
            model,
            args.prefix_cap,
            args.plan_marker,
            args.monitor_marker,
            args.diagnosis_marker,
            args.plan_min,
            args.monitor_min,
            args.diagnosis_min,
            teacher_tokenizer,
            teacher_model,
            args.teacher_max_length,
        )
        if result is not None:
            metrics.append(result)

    summary = summarize(metrics)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
