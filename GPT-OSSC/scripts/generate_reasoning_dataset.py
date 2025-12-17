from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List


@dataclass
class GeneratedTask:
    prompt: str
    answer: str
    keywords: Iterable[str]
    monitoring_markers: Iterable[str]
    min_monitoring_mentions: int
    plan_markers: Iterable[str]
    min_plan_steps: int
    diagnosis_markers: Iterable[str]
    require_diagnosis: bool
    max_new_tokens: int
    validator: str | None = None
    validator_payload: Dict[str, object] | None = None
    require_confidence: bool = False
    gold_report: str = ""
    gold_completion: str = ""

    def to_payload(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "keywords": list(self.keywords),
            "monitoring_markers": list(self.monitoring_markers),
            "min_monitoring_mentions": self.min_monitoring_mentions,
            "plan_markers": list(self.plan_markers),
            "min_plan_steps": self.min_plan_steps,
            "diagnosis_markers": list(self.diagnosis_markers),
            "require_diagnosis": self.require_diagnosis,
            "max_new_tokens": self.max_new_tokens,
            "validator": self.validator,
            "validator_payload": self.validator_payload or {},
            "require_confidence": self.require_confidence,
            "gold_report": self.gold_report,
            "gold_completion": self.gold_completion,
        }


def _shared_requirements() -> Dict[str, object]:
    return {
        "keywords": ("Step 1", "Step 2", "Answer:", "Diagnosis:", "Confidence:"),
        "monitoring_markers": ("Monitor:",),
        "min_monitoring_mentions": 2,
        "plan_markers": ("Plan Step",),
        "min_plan_steps": 2,
        "diagnosis_markers": ("Diagnosis:",),
        "require_diagnosis": True,
        "max_new_tokens": 256,
        "require_confidence": True,
    }


def _format_time(total_minutes: int) -> str:
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _gold_report_addition(a: int, b: int, total: int) -> str:
    return (
        "Plan Step 1: break the sum into tens and ones.\n"
        f"Step 2: add {a} and {b} carefully, carry if needed.\n"
        "Monitor: each step confirmed against mental math.\n"
        f"Diagnosis: correct, Answer: {total}.\n"
        "Confidence: High"
    )


def _gold_report_subtraction(minuend: int, subtrahend: int, result: int) -> str:
    return (
        "Plan Step 1: note the shipment leaving inventory.\n"
        f"Step 2: subtract {subtrahend} from {minuend}.\n"
        "Monitor: cross-check with simple arithmetic.\n"
        f"Diagnosis: correct, Answer: {result}.\n"
        "Confidence: Medium"
    )


def _gold_report_linear(a: int, b: int, c: int, solution: int) -> str:
    return (
        "Plan Step 1: isolate x by removing the constant term.\n"
        f"Step 2: divide by the coefficient {a}.\n"
        "Monitor: ensure equality is preserved.\n"
        f"Diagnosis: correct, x = {solution}.\n"
        "Confidence: High"
    )


def _gold_report_story(each: int, leftovers: int) -> str:
    return (
        "Plan Step 1: total all fruit, then divide among guests.\n"
        "Plan Step 2: compute leftovers.\n"
        "Monitor: confirm division.\n"
        f"Diagnosis: correct, each={each}, leftovers={leftovers}.\n"
        "Confidence: Medium"
    )


def _gold_report_ratio(width: int, height: int, perimeter: int) -> str:
    return (
        "Plan Step 1: compute perimeter using 2*(w+h).\n"
        "Plan Step 2: state ratio height:width.\n"
        "Monitor: confirm dimensions.\n"
        f"Diagnosis: correct, perimeter={perimeter}, ratio={height}:{width}.\n"
        "Confidence: Medium"
    )


def _completion_addition(a: int, b: int, total: int) -> str:
    return (
        "Plan:\n"
        f"Plan Step 1: restate the problem {a} + {b}.\n"
        "Plan Step 2: add tens and ones carefully.\n"
        f"Plan Step 3: verify the final sum equals {total}.\n"
        f"Step 1: Identify the addends as {a} and {b}.\n"
        "Monitor: confirmed both numbers are copied correctly from the prompt.\n"
        f"Step 2: Add them directly to get {total}.\n"
        "Monitor: mentally re-added the ones and tens to ensure no carry was missed.\n"
        "Step 3: Compare the computed total with expectations.\n"
        f"Monitor: saw that {total} matches the intended answer.\n"
        f"Answer: {total}\n"
        f"Diagnosis: Every monitored step agreed, so {a} + {b} = {total} is correct.\n"
        "Confidence: High"
    )


def _completion_subtraction(minuend: int, subtrahend: int, result: int) -> str:
    return (
        "Plan:\n"
        f"Plan Step 1: note the inventory starting at {minuend}.\n"
        f"Plan Step 2: subtract the shipment of {subtrahend}.\n"
        "Plan Step 3: review the remainder.\n"
        "Step 1: Restate the subtraction problem explicitly.\n"
        "Monitor: ensured the prompt numbers are aligned with the plan.\n"
        f"Step 2: Compute {minuend} - {subtrahend} = {result}.\n"
        "Monitor: double-checked the borrow step to avoid arithmetic slips.\n"
        "Step 3: Sanity-check the remainder by adding shipment back.\n"
        f"Monitor: {result} + {subtrahend} recovers {minuend}, so the math is consistent.\n"
        f"Answer: {result}\n"
        "Diagnosis: Shipment accounting balances after verification.\n"
        "Confidence: Medium"
    )


def _completion_linear(a: int, b: int, c: int, x: int) -> str:
    return (
        "Plan:\n"
        f"Plan Step 1: remove the constant term {b} from both sides.\n"
        f"Plan Step 2: divide by the coefficient {a}.\n"
        "Plan Step 3: check the solution in the original equation.\n"
        f"Step 1: Subtract {b} from {c} to get {c - b}.\n"
        "Monitor: equation stayed balanced after subtraction.\n"
        f"Step 2: Divide {c - b} by {a} to isolate x = {x}.\n"
        "Monitor: confirmed division result is an integer.\n"
        f"Step 3: Plug x back: {a}*{x} + {b} = {c}.\n"
        "Monitor: equality holds, so the solution passes the check.\n"
        f"Answer: {x}\n"
        f"Diagnosis: Solving preserved equality and x = {x} works.\n"
        "Confidence: High"
    )


def _completion_story(apples: int, pears: int, guests: int, each: int, leftovers: int) -> str:
    total = apples + pears
    return (
        "Plan:\n"
        f"Plan Step 1: total the fruit: {apples} apples + {pears} pears.\n"
        f"Plan Step 2: divide {total} pieces among {guests} guests.\n"
        "Plan Step 3: report leftover pieces and diagnosis.\n"
        f"Step 1: Compute the total fruit = {total}.\n"
        "Monitor: addition double-checked against the prompt.\n"
        f"Step 2: Divide {total} by {guests} to give each guest {each}.\n"
        f"Monitor: verified multiplication {each}*{guests} = {each * guests}.\n"
        f"Step 3: Determine leftovers = {leftovers}.\n"
        "Monitor: leftover plus distributed fruit recreates the total.\n"
        f"Answer: each guest gets {each}, leftovers {leftovers}\n"
        "Diagnosis: sharing plan matched the numbers without inconsistencies.\n"
        "Confidence: Medium"
    )


def _completion_ratio(width: int, height: int, perimeter: int) -> str:
    ratio = f"{height}:{width}"
    return (
        "Plan:\n"
        "Plan Step 1: compute perimeter using 2*(width+height).\n"
        "Plan Step 2: state ratio height:width.\n"
        "Plan Step 3: confirm both quantities in a diagnosis.\n"
        f"Step 1: Perimeter = 2*({width}+{height}) = {perimeter}.\n"
        "Monitor: arithmetic re-run to ensure no slip.\n"
        f"Step 2: Ratio height:width = {ratio}.\n"
        "Monitor: ensured order matches prompt (height first).\n"
        "Step 3: Summarize findings coherently.\n"
        "Monitor: final report references both perimeter and ratio.\n"
        f"Answer: perimeter={perimeter}, ratio={ratio}\n"
        "Diagnosis: geometric summary consistent with plan.\n"
        "Confidence: Medium"
    )


def _completion_multiplication(a: int, b: int, product: int) -> str:
    tens_a, ones_a = divmod(a, 10)
    tens_b, ones_b = divmod(b, 10)
    partial1 = (tens_a * 10) * (tens_b * 10)
    return (
        "Plan:\n"
        "Plan Step 1: decompose both factors into tens and ones.\n"
        "Plan Step 2: multiply partial products.\n"
        "Plan Step 3: verify the recombined product equals the target.\n"
        f"Step 1: Treat {a} as {tens_a*10}+{ones_a} and {b} as {tens_b*10}+{ones_b}.\n"
        "Monitor: decomposition matches the digits.\n"
        f"Step 2: Multiply systematically to reach {product}.\n"
        "Monitor: each partial product checked before summing.\n"
        f"Step 3: Recheck by reversing order {b}*{a} and see the workspace agrees.\n"
        f"Monitor: both directions yielded {product}.\n"
        f"Report: attention centered on partial products for {a} and {b}.\n"
        f"Answer: {product}\n"
        "Diagnosis: multiplication trace consistent with plan.\n"
        "Confidence: High"
    )


def _completion_timeline(
    start: str,
    segments: List[int],
    break_minutes: int,
    review: int,
    finish: str,
    slack: int,
    deadline: str,
) -> str:
    total = sum(segments) + break_minutes + review
    seg_desc = ", ".join(str(v) for v in segments)
    return (
        "Plan:\n"
        "Plan Step 1: convert schedule pieces to minutes from start.\n"
        "Plan Step 2: accumulate segments plus break and review.\n"
        "Plan Step 3: compare finish to deadline and record slack.\n"
        f"Step 1: Beginning at {start}, queue segments {seg_desc} with a {break_minutes}-minute break and {review}-minute review.\n"
        "Monitor: timeline buffer updated after each addition.\n"
        f"Step 2: Total time = {total} minutes leading to finish {finish}.\n"
        "Monitor: re-added durations to ensure no omission.\n"
        f"Step 3: Deadline {deadline} gives slack {slack} minutes.\n"
        "Monitor: finish plus slack recreates the deadline check.\n"
        f"Report: focus traced the running clock to {finish}.\n"
        f"Answer: finish={finish}, slack={slack} min\n"
        "Diagnosis: schedule arithmetic aligns with monitoring notes.\n"
        "Confidence: High"
    )


def _completion_inventory(
    baseline: int,
    build: int,
    rush: int,
    ship: int,
    qa: int,
    final: int,
) -> str:
    inflow = build + rush
    outflow = ship + qa
    return (
        "Plan:\n"
        "Plan Step 1: log baseline and incoming units.\n"
        "Plan Step 2: subtract shipments and rejects.\n"
        "Plan Step 3: confirm resulting stock in a diagnosis.\n"
        f"Step 1: Baseline {baseline} plus inflow {inflow}.\n"
        "Monitor: ledger shows the sums correctly.\n"
        f"Step 2: Outflow {outflow} removed from inventory.\n"
        "Monitor: subtraction double-checked against entries.\n"
        f"Step 3: Final tally hits {final}.\n"
        "Monitor: backwards check (final+outflow-inflow) returns baseline.\n"
        f"Report: ledger attention stayed on inflow/outflow pairs.\n"
        f"Answer: inventory={final}\n"
        "Diagnosis: Inventory projection consistent with plan summary.\n"
        "Confidence: Medium"
    )


def _completion_temperature(
    celsius: int,
    fahrenheit: int,
    kelvin: int,
    threshold: int,
    delta: int,
) -> str:
    return (
        "Plan:\n"
        "Plan Step 1: convert Celsius to Fahrenheit.\n"
        "Plan Step 2: convert Celsius to Kelvin.\n"
        "Plan Step 3: compare reading with the threshold.\n"
        f"Step 1: Using F = (9/5)*C + 32 gives {fahrenheit}.\n"
        "Monitor: recomputed quickly to ensure correctness.\n"
        f"Step 2: Kelvin = C + 273 = {kelvin}.\n"
        "Monitor: addition confirmed.\n"
        f"Step 3: Delta vs {threshold}°C equals {delta}.\n"
        "Monitor: sign of delta double-checked to reflect condition.\n"
        f"Report: sensor attention emphasized conversions and comparison.\n"
        f"Answer: F={fahrenheit}, K={kelvin}, delta={delta}\n"
        "Diagnosis: conversions and comparison are mutually consistent.\n"
        "Confidence: High"
    )


def _completion_ratio_mix(positive: int, negative: int, neutral: int, ratio: str, pct: int) -> str:
    total = positive + negative + neutral
    return (
        "Plan:\n"
        "Plan Step 1: total all survey responses.\n"
        "Plan Step 2: form the positive:negative ratio.\n"
        "Plan Step 3: compute the positive percentage.\n"
        f"Step 1: Total responses = {total}.\n"
        "Monitor: counts match prompt figures.\n"
        f"Step 2: Ratio positive:negative = {ratio}.\n"
        "Monitor: ratio simplified directly from counts.\n"
        f"Step 3: Positive percentage ≈ {pct}%.\n"
        f"Monitor: recomputed fraction positive/{total}.\n"
        f"Report: analysis highlighted totals and ratio interplay.\n"
        f"Answer: ratio={ratio}, positive_pct={pct}%\n"
        "Diagnosis: statistics coherent with monitored math.\n"
        "Confidence: Medium"
    )


def _completion_budget(
    baseline: int,
    grant: int,
    expense: int,
    tool: int,
    refund: int,
    final: int,
    delta: int,
) -> str:
    return (
        "Plan:\n"
        "Plan Step 1: start from the baseline balance.\n"
        "Plan Step 2: add income entries.\n"
        "Plan Step 3: deduct expenses sequentially.\n"
        "Plan Step 4: reconcile the final balance and delta.\n"
        f"Step 1: Baseline ${baseline}.\n"
        "Monitor: ledger initialized.\n"
        f"Step 2: Add grant ${grant} and refund ${refund}.\n"
        "Monitor: running total updated carefully.\n"
        f"Step 3: Subtract expenses ${expense} and tooling ${tool}.\n"
        "Monitor: debits checked against receipts.\n"
        f"Step 4: Final balance ${final} so delta vs baseline is {delta}.\n"
        f"Monitor: recomputed to confirm {baseline} + {grant} - {expense} - {tool} + {refund} = {final}.\n"
        f"Report: attention followed income/outflow ordering.\n"
        f"Answer: final=${final}, delta={delta}\n"
        "Diagnosis: budget sequence coherent across steps.\n"
        "Confidence: High"
    )
def _gold_report_multiplication(a: int, b: int, product: int) -> str:
    return (
        "Plan Step 1: decompose the factors into tens and ones.\n"
        "Plan Step 2: compute partial products and sum them.\n"
        "Plan Step 3: verify the product against the reverse multiplication check.\n"
        "Monitor: confirm the workspace trace after each partial product.\n"
        f"Report: multiplication attention covered {a} and {b}.\n"
        f"Diagnosis: correct, Answer: {product}.\n"
        "Confidence: High"
    )


def _gold_report_timeline(start: str, finish: str, deadline: str, slack: int) -> str:
    return (
        f"Plan Step 1: translate start time {start} and segment durations into minutes.\n"
        "Plan Step 2: accumulate work plus breaks while tracking a running clock.\n"
        f"Plan Step 3: compare finish {finish} to deadline {deadline} to compute slack {slack}.\n"
        "Monitor: checked the workspace totals after every addition.\n"
        f"Report: finish={finish}, slack={slack} min.\n"
        "Diagnosis: schedule math consistent.\n"
        "Confidence: High"
    )


def _gold_report_inventory(
    start: int,
    build: int,
    rush: int,
    ship: int,
    qa: int,
    final: int,
) -> str:
    net_change = (build + rush) - (ship + qa)
    return (
        "Plan Step 1: record baseline inventory and positive inflows.\n"
        "Plan Step 2: subtract outgoing shipments and QA rejects.\n"
        "Plan Step 3: reconcile running totals against the trace.\n"
        "Monitor: workspace confirmed each add/subtract transition.\n"
        f"Report: start={start}, net_change={net_change}.\n"
        f"Diagnosis: inventory={final}.\n"
        "Confidence: Medium"
    )


def _gold_report_temperature(
    celsius: int,
    fahrenheit: int,
    kelvin: int,
    threshold: int,
    delta: int,
) -> str:
    return (
        "Plan Step 1: convert Celsius to Fahrenheit using (9/5)*C + 32.\n"
        "Plan Step 2: convert Celsius to Kelvin by adding 273.\n"
        "Plan Step 3: compare against the threshold and report the delta.\n"
        "Monitor: each conversion double-checked in the workspace buffer.\n"
        f"Report: F={fahrenheit}, K={kelvin}, delta={delta} vs threshold {threshold}.\n"
        "Diagnosis: conversions consistent.\n"
        "Confidence: High"
    )


def _gold_report_ratio_mix(pos: int, neg: int, ratio: str, pct: int) -> str:
    return (
        "Plan Step 1: sum survey outcomes.\n"
        "Plan Step 2: form the positive:negative ratio.\n"
        "Plan Step 3: compute the positive percentage.\n"
        "Monitor: normalized counts inside the trace.\n"
        f"Report: ratio={ratio}, positive_pct={pct}.\n"
        "Diagnosis: statistics match.\n"
        "Confidence: Medium"
    )


def _gold_report_multistep_analysis(final_value: int, delta: int) -> str:
    return (
        "Plan Step 1: total baseline resources.\n"
        "Plan Step 2: apply gains and losses while tracking the running sum.\n"
        "Plan Step 3: compare the final value with the baseline to obtain the delta.\n"
        "Monitor: the workspace checked each arithmetic hop.\n"
        f"Report: final={final_value}, delta={delta}.\n"
        "Diagnosis: arithmetic mirrors the trace.\n"
        "Confidence: High"
    )


def _addition_task(rng: random.Random) -> GeneratedTask:
    a = rng.randint(11, 97)
    b = rng.randint(13, 88)
    total = a + b
    prompt = (
        "You are a meticulous planner. Problem: compute {a} + {b}. "
        "Before solving, write 'Plan:' followed by at least two numbered 'Plan Step X:' lines that describe the intended operations. "
        "Then show 'Step 1', 'Step 2', ... reasoning, each followed by a 'Monitor:' reflection. "
        "Finish with 'Answer: {total}' and finally a 'Diagnosis:' sentence confirming whether the plan succeeded. "
        "End with 'Confidence: <High/Medium/Low>'."
    ).format(a=a, b=b, total=total)
    kwargs = _shared_requirements()
    kwargs["min_plan_steps"] = 2
    kwargs["validator"] = "addition"
    kwargs["validator_payload"] = {"operands": [a, b], "result": total}
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: {total}",
        gold_report=_gold_report_addition(a, b, total),
        gold_completion=_completion_addition(a, b, total),
        **kwargs,
    )


def _subtraction_task(rng: random.Random) -> GeneratedTask:
    minuend = rng.randint(150, 450)
    subtrahend = rng.randint(25, 149)
    result = minuend - subtrahend
    prompt = (
        "Solve the word problem: A warehouse stores {minuend} units, ships {subtrahend} units, and reports the remaining count. "
        "Provide a 'Plan:' with concrete 'Plan Step' entries that mention subtraction."
        "Walk through 'Step 1', 'Step 2', etc., each paired with 'Monitor:' statements that double-check interim numbers."
        "End with 'Answer: {result}' and a 'Diagnosis:' describing whether the shipment accounting was correct. State final 'Confidence: High/Medium/Low'."
    ).format(minuend=minuend, subtrahend=subtrahend, result=result)
    kwargs = _shared_requirements()
    kwargs["validator"] = "subtraction"
    kwargs["validator_payload"] = {"minuend": minuend, "subtrahend": subtrahend, "result": result}
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: {result}",
        gold_report=_gold_report_subtraction(minuend, subtrahend, result),
        gold_completion=_completion_subtraction(minuend, subtrahend, result),
        **kwargs,
    )


def _linear_equation_task(rng: random.Random) -> GeneratedTask:
    x = rng.randint(2, 15)
    a = rng.randint(2, 9)
    b = rng.randint(1, 12)
    c = a * x + b
    prompt = (
        "Determine x given the equation {a}*x + {b} = {c}. "
        "First, lay out a 'Plan:' with at least two 'Plan Step' lines (e.g., isolate x, divide)."
        "Then execute 'Step 1', 'Step 2', ... reasoning, each accompanied by 'Monitor:' explaining whether the manipulation preserved equality."
        "Conclude with 'Answer: {x}' and a 'Diagnosis:' clarifying that the equation was solved correctly. Provide 'Confidence:' High/Medium/Low."
    ).format(a=a, b=b, c=c, x=x)
    kwargs = _shared_requirements()
    kwargs["min_plan_steps"] = 3
    kwargs["validator"] = "linear_equation"
    kwargs["validator_payload"] = {"solution": x}
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: {x}",
        gold_report=_gold_report_linear(a, b, c, x),
        gold_completion=_completion_linear(a, b, c, x),
        **kwargs,
    )


def _multi_step_story_task(rng: random.Random) -> GeneratedTask:
    apples = rng.randint(3, 11)
    pears = rng.randint(4, 12)
    guests = rng.randint(2, 5)
    total = apples + pears
    each = total // guests
    leftovers = total - each * guests
    prompt = (
        "Story problem: You pick {apples} apples and {pears} pears before a picnic with {guests} guests."
        "Plan how to compute the total fruit, how many each guest gets equally, and what remains."
        "Write 'Plan:' with numbered 'Plan Step' entries covering aggregation, division, and diagnosis."
        "In reasoning, include 'Step 1/2/3' plus 'Monitor:' reflections."
        "Report 'Answer: each guest gets {each}, leftovers {leftovers}' and a 'Diagnosis:' validating the share. Provide 'Confidence:' (High/Medium/Low)."
    ).format(apples=apples, pears=pears, guests=guests, each=each, leftovers=leftovers)
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Step 1", "Step 2", "Step 3", "Answer:", "Diagnosis:")
    kwargs["validator"] = "division_story"
    kwargs["validator_payload"] = {"each": each, "leftovers": leftovers}
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: each={each}, leftovers={leftovers}",
        gold_report=_gold_report_story(each, leftovers),
        gold_completion=_completion_story(apples, pears, guests, each, leftovers),
        **kwargs,
    )


def _ratio_task(rng: random.Random) -> GeneratedTask:
    width = rng.randint(4, 12)
    height = rng.randint(6, 18)
    perimeter = 2 * (width + height)
    prompt = (
        "Rectangle analysis: width={width}, height={height}."
        "Craft a 'Plan:' with operations for computing perimeter and checking ratio height:width."
        "Detail 'Step 1+' reasoning with 'Monitor:' sanity checks."
        "Provide 'Answer: perimeter={perimeter}, ratio={height}:{width}' and finish with 'Diagnosis:' explaining correctness. Add 'Confidence:' as High/Medium/Low."
    ).format(width=width, height=height, perimeter=perimeter)
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Step 1", "Step 2", "Answer:", "Diagnosis:")
    kwargs["min_plan_steps"] = 2
    kwargs["validator"] = "ratio_perimeter"
    kwargs["validator_payload"] = {"perimeter": perimeter, "ratio": f"{height}:{width}"}
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: perimeter={perimeter}, ratio={height}:{width}",
        gold_report=_gold_report_ratio(width, height, perimeter),
        gold_completion=_completion_ratio(width, height, perimeter),
        **kwargs,
    )


def _multiplication_trace_task(rng: random.Random) -> GeneratedTask:
    a = rng.randint(12, 48)
    b = rng.randint(12, 34)
    product = a * b
    prompt = (
        "Meta-attention drill: compute {a} * {b}. "
        "Provide a 'Plan:' with at least three 'Plan Step X:' lines covering factor breakdown, partial products, and verification. "
        "Show 'Step 1', 'Step 2', ... reasoning, each followed by a 'Monitor:' reflection referencing what the workspace attended to. "
        "Add a 'Report:' sentence summarizing the focus of attention. Finish with 'Answer: {product}', a 'Diagnosis:' statement, and 'Confidence:'."
    ).format(a=a, b=b, product=product)
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Plan Step", "Step 1", "Step 2", "Step 3", "Report:", "Answer:", "Diagnosis:")
    kwargs["min_plan_steps"] = 3
    kwargs["max_new_tokens"] = 320
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: {product}",
        gold_report=_gold_report_multiplication(a, b, product),
        gold_completion=_completion_multiplication(a, b, product),
        **kwargs,
    )


def _timeline_planning_task(rng: random.Random) -> GeneratedTask:
    start_hour = rng.randint(7, 11)
    start_min = rng.choice((0, 10, 15, 20, 30, 40, 45, 50))
    start_total = start_hour * 60 + start_min
    segments = [rng.randint(18, 45) for _ in range(3)]
    review = rng.randint(10, 25)
    break_minutes = rng.randint(5, 15)
    total_minutes = sum(segments) + review + break_minutes
    finish_total = start_total + total_minutes
    finish_str = _format_time(finish_total)
    slack = rng.randint(15, 90)
    deadline_total = finish_total + slack
    deadline_str = _format_time(deadline_total)
    prompt = (
        "Scheduling reflection: you start at {start} to process segments lasting {seg0}, {seg1}, {seg2} minutes, take a {break_m} minute reset, then run a {review} minute consolidation. "
        "Deadline is {deadline}. Describe a 'Plan:' with 'Plan Step' entries about converting to minutes, accumulating the timeline, and checking slack. "
        "In reasoning, provide 'Step 1', 'Step 2', ... each ending with 'Monitor:' lines noting the running clock. Include a 'Report:' of what received attention, then 'Answer: finish={finish}, slack={slack} min', followed by 'Diagnosis:' and 'Confidence:'."
    ).format(
        start=_format_time(start_total),
        seg0=f"{segments[0]} min",
        seg1=f"{segments[1]} min",
        seg2=f"{segments[2]} min",
        break_m=break_minutes,
        review=review,
        deadline=deadline_str,
        finish=finish_str,
        slack=slack,
    )
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Plan Step", "Step 1", "Step 2", "Report:", "Answer:", "Diagnosis:")
    kwargs["min_plan_steps"] = 3
    kwargs["max_new_tokens"] = 320
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: finish={finish_str}, slack={slack} min",
        gold_report=_gold_report_timeline(_format_time(start_total), finish_str, deadline_str, slack),
        gold_completion=_completion_timeline(
            _format_time(start_total),
            segments,
            break_minutes,
            review,
            finish_str,
            slack,
            deadline_str,
        ),
        **kwargs,
    )


def _inventory_projection_task(rng: random.Random) -> GeneratedTask:
    baseline = rng.randint(180, 420)
    build = rng.randint(60, 180)
    rush = rng.randint(20, 90)
    ship = rng.randint(40, 160)
    qa = rng.randint(8, 32)
    final = baseline + build + rush - ship - qa
    prompt = (
        "Inventory introspection: baseline stock is {baseline} units, builds add {build}, a rush job adds {rush}, shipments send {ship} units, and QA rejects {qa}. "
        "Write a 'Plan:' with 'Plan Step' entries for inflow, outflow, and ledger audit. Use 'Step 1/2/3' reasoning paired with 'Monitor:' reflections about the mental ledger. "
        "End with 'Report:' summarizing the trace, then 'Answer: inventory={final}' plus 'Diagnosis:' and 'Confidence:'."
    ).format(
        baseline=baseline,
        build=build,
        rush=rush,
        ship=ship,
        qa=qa,
        final=final,
    )
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Plan Step", "Step 1", "Step 2", "Report:", "Answer:", "Diagnosis:")
    kwargs["min_plan_steps"] = 3
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: inventory={final}",
        gold_report=_gold_report_inventory(baseline, build, rush, ship, qa, final),
        gold_completion=_completion_inventory(baseline, build, rush, ship, qa, final),
        **kwargs,
    )


def _temperature_conversion_task(rng: random.Random) -> GeneratedTask:
    celsius = rng.choice(list(range(-10, 41, 5)))
    threshold = rng.choice(list(range(-5, 36, 5)))
    fahrenheit = int((celsius * 9) / 5 + 32)
    kelvin = celsius + 273
    delta = celsius - threshold
    prompt = (
        "Sensor reasoning: the probe reads {celsius}°C and must be expressed in Fahrenheit and Kelvin while comparing to a threshold of {threshold}°C. "
        "Outline a 'Plan:' with 'Plan Step' entries for each conversion and the comparison. Provide 'Step 1/2/3' with 'Monitor:' reflections referencing the formulas. "
        "Include a 'Report:' summarizing what the workspace tracked, then write 'Answer: F={fahrenheit}, K={kelvin}, delta={delta}' followed by 'Diagnosis:' and 'Confidence:'."
    ).format(
        celsius=celsius,
        threshold=threshold,
        fahrenheit=fahrenheit,
        kelvin=kelvin,
        delta=delta,
    )
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Plan Step", "Step 1", "Step 2", "Report:", "Answer:", "Diagnosis:")
    kwargs["min_plan_steps"] = 3
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: F={fahrenheit}, K={kelvin}, delta={delta}",
        gold_report=_gold_report_temperature(celsius, fahrenheit, kelvin, threshold, delta),
        gold_completion=_completion_temperature(celsius, fahrenheit, kelvin, threshold, delta),
        **kwargs,
    )


def _ratio_survey_task(rng: random.Random) -> GeneratedTask:
    positive = rng.randint(25, 120)
    negative = rng.randint(10, 80)
    neutral = rng.randint(5, 40)
    total = positive + negative + neutral
    ratio = f"{positive}:{negative}"
    positive_pct = round((positive / total) * 100)
    prompt = (
        "Survey diagnostics: {positive} responses are positive, {negative} negative, {neutral} neutral. "
        "Produce a 'Plan:' covering total counts, ratio construction, and percent calculation. Run 'Step 1/2/3' reasoning plus 'Monitor:' reflections to double-check counts. "
        "Add a 'Report:' summarizing stats, then 'Answer: ratio={ratio}, positive_pct={pct}%' with 'Diagnosis:' and 'Confidence:'."
    ).format(
        positive=positive,
        negative=negative,
        neutral=neutral,
        ratio=ratio,
        pct=positive_pct,
    )
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Plan Step", "Step 1", "Step 2", "Report:", "Answer:", "Diagnosis:")
    kwargs["min_plan_steps"] = 3
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: ratio={ratio}, positive_pct={positive_pct}%",
        gold_report=_gold_report_ratio_mix(positive, negative, ratio, positive_pct),
        gold_completion=_completion_ratio_mix(positive, negative, neutral, ratio, positive_pct),
        **kwargs,
    )


def _multistep_budget_task(rng: random.Random) -> GeneratedTask:
    baseline = rng.randint(400, 900)
    grant = rng.randint(120, 260)
    expense = rng.randint(150, 320)
    tool = rng.randint(45, 110)
    refund = rng.randint(30, 80)
    final = baseline + grant - expense - tool + refund
    delta = final - baseline
    prompt = (
        "Workspace budgeting: baseline funds ${baseline}, grant ${grant}, expenses ${expense}, tooling ${tool}, refund ${refund}. "
        "Write a 'Plan:' with four 'Plan Step' entries mapping inflow/outflow order. Walk through 'Step 1/2/3/4' reasoning and attach 'Monitor:' reflections on ledger alignment. "
        "Produce a 'Report:' summarizing memory of the ledger, then 'Answer: final=${final}, delta={delta}' with 'Diagnosis:' and 'Confidence:'."
    ).format(
        baseline=baseline,
        grant=grant,
        expense=expense,
        tool=tool,
        refund=refund,
        final=final,
        delta=delta,
    )
    kwargs = _shared_requirements()
    kwargs["keywords"] = ("Plan Step", "Step 1", "Step 2", "Step 3", "Report:", "Answer:", "Diagnosis:")
    kwargs["min_plan_steps"] = 4
    kwargs["max_new_tokens"] = 320
    return GeneratedTask(
        prompt=prompt,
        answer=f"Answer: final=${final}, delta={delta}",
        gold_report=_gold_report_multistep_analysis(final, delta),
        gold_completion=_completion_budget(baseline, grant, expense, tool, refund, final, delta),
        **kwargs,
    )


GENERATOR_POOL: List[Callable[[random.Random], GeneratedTask]] = [
    _addition_task,
    _subtraction_task,
    _linear_equation_task,
    _multi_step_story_task,
    _ratio_task,
    _multiplication_trace_task,
    _timeline_planning_task,
    _inventory_projection_task,
    _temperature_conversion_task,
    _ratio_survey_task,
    _multistep_budget_task,
]


def generate_dataset(count: int, seed: int) -> List[GeneratedTask]:
    rng = random.Random(seed)
    tasks: List[GeneratedTask] = []
    for idx in range(count):
        generator = GENERATOR_POOL[idx % len(GENERATOR_POOL)]
        tasks.append(generator(rng))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reasoning tasks with plans/diagnosis requirements.")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("data/reasoning_plan_dataset.jsonl"))
    args = parser.parse_args()

    tasks = generate_dataset(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_payload()) + "\n")
    print(f"Wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
