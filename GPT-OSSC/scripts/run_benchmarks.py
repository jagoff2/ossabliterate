#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_reports import (  # noqa: E402
    DEFAULT_DIAG_MARKERS,
    DEFAULT_MONITOR_MARKERS,
    DEFAULT_PLAN_MARKERS,
    EpisodeMetrics,
    evaluate_episode,
    load_reports,
    maybe_load_lm,
    summarize,
)


def run_single(
    path: Path,
    args: argparse.Namespace,
    tokenizer,
    model,
    teacher_tokenizer,
    teacher_model,
) -> Dict[str, object]:
    episodes = load_reports(path)
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
    summary["run"] = str(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark metrics across multiple progress logs.")
    parser.add_argument("progress", nargs="+", help="Path(s) to progress.jsonl files or directories containing them")
    parser.add_argument("--output", type=Path, default=Path("benchmark_summary.csv"))
    parser.add_argument("--plan-min", type=int, default=2)
    parser.add_argument("--monitor-min", type=int, default=2)
    parser.add_argument("--diagnosis-min", type=int, default=1)
    parser.add_argument("--plan-marker", action="append", default=list(DEFAULT_PLAN_MARKERS))
    parser.add_argument("--monitor-marker", action="append", default=list(DEFAULT_MONITOR_MARKERS))
    parser.add_argument("--diagnosis-marker", action="append", default=list(DEFAULT_DIAG_MARKERS))
    parser.add_argument("--prefix-lm", default="none")
    parser.add_argument("--prefix-device", default="cpu")
    parser.add_argument("--teacher-lm", default="none")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--teacher-max-length", type=int, default=512)
    parser.add_argument("--prefix-cap", type=int, default=160)
    parser.add_argument("--threshold-prefix", type=float, default=0.7)
    parser.add_argument("--threshold-plan", type=float, default=0.85)
    parser.add_argument("--threshold-monitor", type=float, default=0.85)
    parser.add_argument("--threshold-diagnosis", type=float, default=0.9)
    parser.add_argument("--threshold-validator", type=float, default=0.9)
    parser.add_argument("--threshold-teacher", type=float, default=0.8)
    parser.add_argument("--threshold-graph", type=float, default=0.75)
    parser.add_argument("--threshold-invalid-ratio", type=float, default=0.05)
    args = parser.parse_args()

    def expand_paths(spec: str) -> List[Path]:
        target = Path(spec)
        if target.is_dir():
            return list(target.glob("**/progress.jsonl"))
        return [target]

    all_paths: List[Path] = []
    for spec in args.progress:
        all_paths.extend(expand_paths(spec))
    if not all_paths:
        raise SystemExit("No progress.jsonl files found")

    tokenizer, model = maybe_load_lm(args.prefix_lm, args.prefix_device)
    teacher_tokenizer, teacher_model = maybe_load_lm(args.teacher_lm, args.teacher_device)

    rows: List[Dict[str, object]] = []
    for path in all_paths:
        summary = run_single(path, args, tokenizer, model, teacher_tokenizer, teacher_model)
        rows.append(summary)

    fieldnames = [
        "run",
        "total",
        "invalid",
        "avg_prefix",
        "avg_plan",
        "avg_monitor",
        "avg_diagnosis",
        "avg_graph_plan_nodes",
        "avg_graph_monitor_nodes",
        "avg_graph_step_nodes",
        "validator_pass_rate",
        "teacher_score",
        "avg_graph_coverage",
        "pass",
        "fail_reasons",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            averages = row.get("averages", {})
            failures = []
            def check(metric, threshold, label):
                value = averages.get(metric, 0.0) or 0.0
                if value < threshold:
                    failures.append(f"{label}<{threshold:.2f} ({value:.2f})")

            check("prefix_quality", args.threshold_prefix, "prefix")
            check("plan_coverage", args.threshold_plan, "plan")
            check("monitor_coverage", args.threshold_monitor, "monitor")
            check("diagnosis_coverage", args.threshold_diagnosis, "diagnosis")
            check("validator_pass_rate", args.threshold_validator, "validator")
            check("teacher_score", args.threshold_teacher, "teacher")
            check("graph_coverage", args.threshold_graph, "graph")
            total = row.get("total", 0) or 0
            invalid = row.get("invalid", 0) or 0
            if total > 0 and invalid / total > args.threshold_invalid_ratio:
                failures.append(f"invalid_ratio>{args.threshold_invalid_ratio:.2f}")
            passed = not failures
            writer.writerow(
                {
                    "run": row.get("run"),
                    "total": total,
                    "invalid": invalid,
                    "avg_prefix": averages.get("prefix_quality"),
                    "avg_plan": averages.get("plan_coverage"),
                    "avg_monitor": averages.get("monitor_coverage"),
                    "avg_diagnosis": averages.get("diagnosis_coverage"),
                    "avg_graph_plan_nodes": averages.get("graph_plan_nodes"),
                    "avg_graph_monitor_nodes": averages.get("graph_monitor_nodes"),
                    "avg_graph_step_nodes": averages.get("graph_step_nodes"),
                    "validator_pass_rate": averages.get("validator_pass_rate"),
                    "teacher_score": averages.get("teacher_score"),
                    "avg_graph_coverage": averages.get("graph_coverage"),
                    "pass": passed,
                    "fail_reasons": "; ".join(failures),
                }
            )
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {row.get('run')}")
    print(f"Wrote benchmark summary for {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
