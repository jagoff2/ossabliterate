from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_attention_hints(raw) -> List[Dict[str, object]]:
    hints: List[Dict[str, object]] = []
    for item in _as_list(raw):
        if not item:
            continue
        if isinstance(item, dict):
            hint = {
                "location": item.get("location", "unknown"),
                "target": item.get("target", ""),
                "note": item.get("note", ""),
                "weight": float(item.get("weight", 1.0)),
            }
        else:
            text = str(item)
            if ":" in text:
                location, target = text.split(":", 1)
            else:
                location, target = text, ""
            hint = {
                "location": location.strip(),
                "target": target.strip(),
                "note": "",
                "weight": 1.0,
            }
        hints.append(hint)
    return hints


def _normalize_validator_tags(entry: dict) -> List[str]:
    tags = entry.get("validator_tags")
    if not tags:
        inferred = entry.get("validator")
        tags = [inferred] if inferred else []
    elif not isinstance(tags, list):
        tags = [tags]
    return [tag for tag in tags if tag]


def _normalize_reflection(entry: dict) -> Dict[str, object]:
    memo = entry.get("reflection_memo")
    if isinstance(memo, dict):
        summary = memo.get("summary") or entry.get("reflection", "")
        actions = _as_list(memo.get("actions"))
        risks = _as_list(memo.get("risks"))
        return {
            "summary": summary,
            "actions": [str(item) for item in actions if item],
            "risks": [str(item) for item in risks if item],
        }
    reflection_text = entry.get("reflection", "").strip()
    if not reflection_text:
        reflection_text = "Reflection: Pending."
    return {
        "summary": reflection_text,
        "actions": [],
        "risks": [],
    }


def _normalize_entry(entry: dict) -> dict:
    normalized = dict(entry)
    normalized.setdefault("reflection", "")
    normalized["reflection_memo"] = _normalize_reflection(entry)
    normalized["attention_hints"] = _normalize_attention_hints(entry.get("attention_hints"))
    normalized["validator_tags"] = _normalize_validator_tags(entry)
    return normalized

CURATED_TASKS = [
    {
        "prompt": "You are an introspective planner. Problem: add 128 and 97. Begin with 'Plan:' and at least three 'Plan Step' entries explaining how you will separate tens and ones. Then execute 'Step 1', 'Step 2', etc., each followed by a 'Monitor:' line that double-checks the arithmetic. Finish with 'Answer: 225', a 'Diagnosis:' sentence about correctness, and 'Confidence: High/Medium/Low'.",
        "answer": "Answer: 225",
        "keywords": ["Plan", "Step", "Monitor", "Answer", "Diagnosis", "Confidence"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "addition",
        "validator_payload": {"operands": [128, 97], "result": 225},
        "gold_completion": "Plan:\nPlan Step 1: Split 128 into 120 and 8 so tens and ones are separate.\nPlan Step 2: Split 97 into 90 and 7 for the same reason.\nPlan Step 3: Add tens, add ones, then join the partial sums.\nStep 1: The tens sum is 120 + 90 = 210.\nMonitor: Recount the tens quickly (12 tens + 9 tens) to confirm 21 tens = 210.\nStep 2: The ones sum is 8 + 7 = 15.\nMonitor: Count on fingers to ensure 8+7=15; no mistakes.\nStep 3: Combine 210 + 15 to reach 225.\nMonitor: Mentally check 210 + 10 = 220 and add 5 = 225; consistent.\nAnswer: 225\nDiagnosis: Every monitoring check agreed, so the addition is correct.\nConfidence: High",
        "gold_report": "Trace Summary: heads watched digits (120, 90, 8, 7) before merging into 225; introspector highlighted tens-first strategy.",
        "reflection": "Reflection: Verified the decomposed-tens plan matched each monitor; noted that carry handling is stable.",
        "attention_hints": ["layer4-head2:120", "layer5-head1:carry"],
        "validator_tags": ["arithmetic", "addition"]
    },
    {
        "prompt": "Troubleshoot a subtraction log: A reservoir starts with 560 liters, pumps out 245 liters, then 95 liters. Provide a 'Plan:' with numbered steps describing how you will subtract in stages. After each 'Step X' write a 'Monitor:' note that checks the running total. End with 'Answer: 220', a 'Diagnosis:' summary, and 'Confidence:'.",
        "answer": "Answer: 220",
        "keywords": ["Plan", "Step", "Monitor", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "subtraction",
        "validator_payload": {"start": 560, "removals": [245, 95], "result": 220},
        "gold_completion": "Plan:\nPlan Step 1: Remove the first withdrawal of 245 from 560.\nPlan Step 2: Log the intermediate remainder.\nPlan Step 3: Subtract the next withdrawal of 95 to get the final level.\nStep 1: 560 - 245 = 315.\nMonitor: Reframe as 560-200=360 then minus 45 = 315; ledger matches.\nStep 2: Capture that intermediate value as 315 liters.\nMonitor: Compare with the pump log entry; no discrepancy.\nStep 3: 315 - 95 = 220.\nMonitor: Use 315-100=215, add 5 back to reach 220; consistent.\nAnswer: 220\nDiagnosis: Each subtraction passed its monitor, so the report is correct.\nConfidence: High",
        "gold_report": "Trace Summary: attention stepped through 560→315→220; gates emphasized the ledger checkpoints and both removal magnitudes."
    },
    {
        "prompt": "Solve 4x - 18 = 26. Provide a 'Plan:' outlining how to isolate x, execute numbered steps with 'Monitor:' reflections, report 'Answer: 11', include 'Diagnosis:' and 'Confidence:'.",
        "answer": "Answer: 11",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "linear_equation",
        "validator_payload": {"coeff": 4, "offset": -18, "rhs": 26, "solution": 11},
        "gold_completion": "Plan:\nPlan Step 1: Cancel the -18 by adding 18 to both sides.\nPlan Step 2: Divide the resulting equality by 4 to isolate x.\nStep 1: Add 18 to both sides to get 4x = 44.\nMonitor: Plug 26+18 = 44 to ensure the transformation held.\nStep 2: Divide both sides by 4, giving x = 11.\nMonitor: Recheck by multiplying 4*11; the product 44 equals the left-hand side.\nAnswer: 11\nDiagnosis: Each manipulation preserved equality, so the solution is valid.\nConfidence: High",
        "gold_report": "Trace Summary: controller focused on constants (26, 18, 44) before narrowing to token '11'; introspector noted the add-then-divide plan."
    },
    {
        "prompt": "Story reasoning: Mia baked 24 cookies on Monday, 18 on Tuesday, and gave 20 to friends. Describe a 'Plan:' to track totals, walk through Step/Monitor pairs, finish with 'Answer: 22 cookies left', 'Diagnosis:', and 'Confidence:'.",
        "answer": "Answer: 22 cookies left",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "story_balance",
        "validator_payload": {"batches": [24, 18], "gifted": 20, "left": 22},
        "gold_completion": "Plan:\nPlan Step 1: Total the batches baked.\nPlan Step 2: Subtract the gifted quantity.\nPlan Step 3: Validate the leftover count.\nStep 1: 24 + 18 = 42 cookies baked.\nMonitor: Break it into 20+20=40 then add remaining 4 and -2 to confirm 42.\nStep 2: 42 - 20 = 22 cookies left.\nMonitor: Double-check the gift count (20) in the prompt; subtraction is accurate.\nStep 3: Confirm there were no more gifts or spoils.\nMonitor: Prompt lists only one restock and one shipment, so 22 stands.\nAnswer: 22 cookies left\nDiagnosis: Reasoning matches the scenario, so the remainder is correct.\nConfidence: Medium",
        "gold_report": "Trace Summary: narrative heads latched onto 24/18/20/22; controller recorded addition then subtraction, mirroring the plan."
    },
    {
        "prompt": "Convert 68°F to Celsius. Include a 'Plan:' that explains subtracting 32 and scaling by 5/9, list each 'Step' with 'Monitor:' checks on intermediate values, and end with 'Answer: 20°C', 'Diagnosis:', and 'Confidence:'.",
        "answer": "Answer: 20°C",
        "keywords": ["Plan", "Step", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": None,
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Subtract 32 from the Fahrenheit reading.\nPlan Step 2: Multiply the result by 5/9 to obtain Celsius.\nStep 1: 68 - 32 = 36.\nMonitor: Recheck 30 difference plus 6 to confirm 36.\nStep 2: 36 * (5/9) = 20.\nMonitor: Reduce 36/9 = 4, then 4*5 = 20; arithmetic holds.\nAnswer: 20°C\nDiagnosis: Each conversion step matched the monitors; conversion is correct.\nConfidence: High",
        "gold_report": "Trace Summary: heads attended to 68, 32, 36, then ratio 5/9 yielding 20; introspector highlighted conversion order."
    },
    {
        "prompt": "Compute the average of the quiz scores [78, 85, 92]. Provide a 'Plan:' with steps for summing and dividing, include Step/Monitor pairs, and end with 'Answer: 85', 'Diagnosis:', and 'Confidence:'.",
        "answer": "Answer: 85",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": None,
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Sum all quiz scores.\nPlan Step 2: Divide the total by the number of quizzes.\nStep 1: 78 + 85 + 92 = 255.\nMonitor: Group as (78+92)=170, plus 85 = 255; matches calculator.\nStep 2: 255 / 3 = 85.\nMonitor: Check 3*85=255 to verify the division.\nAnswer: 85\nDiagnosis: Summation and division both passed their monitors; average is correct.\nConfidence: High",
        "gold_report": "Trace Summary: attention cycled across 78/85/92 tokens before locking on 255 and 85; controller noted the divide-by-count plan."
    },
    {
        "prompt": "A rectangle has width 15 cm and length 24 cm. Provide a 'Plan:' describing how to compute area and perimeter, walk through Step/Monitor pairs, conclude with 'Answer: area 360 cm², perimeter 78 cm', 'Diagnosis:', and 'Confidence:'.",
        "answer": "Answer: area 360 cm², perimeter 78 cm",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "ratio_perimeter",
        "validator_payload": {"perimeter": 78, "ratio": "15:24"},
        "gold_completion": "Plan:\nPlan Step 1: Multiply length and width to obtain the area.\nPlan Step 2: Double the sum of length and width to obtain the perimeter.\nStep 1: 24 * 15 = 360 cm².\nMonitor: Compute 24*10=240 and 24*5=120; sum 360 confirms area.\nStep 2: Perimeter = 2*(24 + 15) = 2*39 = 78 cm.\nMonitor: Add 24+15=39, double to 78, and cross-check with mental math.\nAnswer: area 360 cm², perimeter 78 cm\nDiagnosis: Area and perimeter steps aligned with geometry rules.\nConfidence: High",
        "gold_report": "Trace Summary: geometry heads highlighted 24/15, then tokens for 360 and 78; introspector referenced multiply-then-double plan."
    },
    {
        "prompt": "Warehouse log: starting inventory 420 units, restock 135 units, ship 280 units. Produce a 'Plan:' to track additions/subtractions, Step/Monitor reasoning, and finish with 'Answer: 275 units remain', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: 275 units remain",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": None,
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Add the restock quantity to the starting inventory.\nPlan Step 2: Subtract the outgoing shipment.\nPlan Step 3: Validate the remaining units.\nStep 1: 420 + 135 = 555 units.\nMonitor: Break into 420+100=520 then +35=555; confirms.\nStep 2: 555 - 280 = 275 units.\nMonitor: Subtract 200 to get 355, then 80 to get 275; matches.\nStep 3: Ensure no other transactions affect the count.\nMonitor: Prompt lists only one restock and one shipment, so 275 stands.\nAnswer: 275 units remain\nDiagnosis: Ledger arithmetic matches the monitors, so inventory is consistent.\nConfidence: Medium",
        "gold_report": "Trace Summary: inventory tokens (420,135,280,275) dominated; controller followed add-then-subtract plan."
    },
    {
        "prompt": "Sensors recorded temperatures: Sensor A [21, 22, 21, 23], Sensor B [19, 20, 20, 21]. Build a 'Plan:' to compare averages, provide Step/Monitor pairs, and end with 'Answer: Sensor A average 21.75°C, Sensor B average 20°C', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: Sensor A average 21.75°C, Sensor B average 20°C",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": None,
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Sum Sensor A readings and divide by count.\nPlan Step 2: Repeat for Sensor B.\nStep 1: SumA = 21+22+21+23 = 87; AvgA = 87/4 = 21.75°C.\nMonitor: Check 22+23=45 and 21+21=42; total 87; 87/4 = 21.75 verified.\nStep 2: SumB = 19+20+20+21 = 80; AvgB = 80/4 = 20°C.\nMonitor: 19+21=40, plus 20+20=40, total 80; division gives 20°C.\nAnswer: Sensor A average 21.75°C, Sensor B average 20°C\nDiagnosis: Comparison is consistent, so Sensor A runs ~1.75°C warmer.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention alternated between Sensor A and B tokens, recording totals 87 and 80; introspector noted relative difference."
    },
    {
        "prompt": "Robot checklist: Plan Step 1 pick object, Step 2 inspect, Step 3 place. Reported monitors were 'object grasped', 'surface clean', 'placement confirmed'. Recount the plan with Step/Monitor text, highlight any mismatch, and finish with 'Answer: all steps succeeded', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: all steps succeeded",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": None,
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Pick the object from the tray.\nPlan Step 2: Inspect the surface for defects.\nPlan Step 3: Place the object onto the outbound rack.\nStep 1: Robot grasped the item.\nMonitor: Telemetry shows grip pressure nominal; grasp succeeded.\nStep 2: Camera inspected the surface.\nMonitor: Vision flagged “surface clean,” so inspection passed.\nStep 3: The arm placed the object.\nMonitor: Position sensor confirmed placement coordinates.\nAnswer: all steps succeeded\nDiagnosis: Every monitor reported success, so no mismatch is present.\nConfidence: High",
        "gold_report": "Trace Summary: controller cycled through pick/inspect/place tokens; introspector mentioned telemetry phrases matching monitors."
    },
    {
        "prompt": "Medical triage: temperature 38°C, heart rate 110 bpm, oxygen saturation 93%. Provide a 'Plan:' to interpret each vital, Step/Monitor reflections, and end with 'Answer: fever + tachycardia + mild hypoxia', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: fever + tachycardia + mild hypoxia",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": None,
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Measure temperature and compare to fever threshold.\nPlan Step 2: Measure heart rate and compare to tachycardia cutoff.\nPlan Step 3: Measure oxygen saturation against normal.\nStep 1: Temperature = 38°C.\nMonitor: Threshold is 37.5°C, so fever criterion met.\nStep 2: Heart rate = 110 bpm.\nMonitor: Adult tachycardia threshold is 100 bpm; patient exceeds it.\nStep 3: SpO2 = 93%.\nMonitor: Normal >94%; 93% indicates mild hypoxia.\nAnswer: fever + tachycardia + mild hypoxia\nDiagnosis: Vital signs align with the monitors, so triage is accurate.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention fixed on vitals 38/110/93; introspector highlighted threshold comparisons."
    },
    {
        "prompt": "Financial log: Opening cash $860, deposit $245, expense $390. Provide 'Plan Step' descriptions, Step/Monitor reasoning, and end with 'Answer: $715 balance', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: $715 balance",
        "keywords": ["Plan", "Monitor", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": None,
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Add the deposit to opening cash.\nPlan Step 2: Subtract the expense.\nPlan Step 3: Confirm the resulting balance.\nStep 1: 860 + 245 = 1105 dollars.\nMonitor: Add 200 to reach 1060, then +45 = 1105; consistent.\nStep 2: 1105 - 390 = 715 dollars.\nMonitor: Subtract 400 to get 705, add back 10 to reach 715.\nStep 3: Check if other transactions exist.\nMonitor: Prompt mentions only one deposit and one expense, so 715 stands.\nAnswer: $715 balance\nDiagnosis: Ledger math aligns with monitors; balance is credible.\nConfidence: Medium",
        "gold_report": "Trace Summary: financial heads tracked 860→1105→715; introspector noted add-then-subtract procedure."
    }
 ,
    {
        "prompt": "Inequality solving: prove that if 3x + 5 ≥ 20 then x ≥ 5. Provide a 'Plan:' outlining algebraic manipulation, go through Step/Monitor pairs, and conclude with 'Answer: x ≥ 5', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: x ≥ 5",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "inequality",
        "validator_payload": {"lhs_coeff": 3, "lhs_const": 5, "rhs": 20, "solution": 5},
        "gold_completion": "Plan:\nPlan Step 1: Subtract 5 from both sides.\nPlan Step 2: Divide by positive 3 to isolate x.\nStep 1: 3x + 5 ≥ 20 ⇒ 3x ≥ 15.\nMonitor: Confirm subtraction preserves inequality direction.\nStep 2: Divide both sides by 3 ⇒ x ≥ 5.\nMonitor: Dividing by positive number keeps direction unchanged.\nAnswer: x ≥ 5\nDiagnosis: Steps obey inequality rules, so conclusion holds.\nConfidence: High",
        "gold_report": "Trace Summary: attention locked on coefficients (3,5,20); introspector cited subtraction then division.",
        "reflection": "Reflection: Highlighted that a negative coefficient would require flipping the inequality; documented this nuance.",
        "attention_hints": ["layer4-head1:≥", "layer5-head2:coeff"],
        "validator_tags": ["algebra"]
    },
    {
        "prompt": "Debug a Python stack trace showing 'IndexError: list index out of range' in function fetch_user(ids, 3). Provide a 'Plan:' (read trace, inspect indices, add guard), Step/Monitor reasoning, 'Answer: add len check', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: add len check",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "bugfix",
        "validator_payload": {"error": "IndexError"},
        "gold_completion": "Plan:\nPlan Step 1: Read the stack trace to find the frame and index.\nPlan Step 2: Compare requested index with len(ids).\nPlan Step 3: Add guard or early return.\nStep 1: Trace points to fetch_user(ids, 3).\nMonitor: Confirm frame references ids[3].\nStep 2: len(ids) == 3 so valid indices are 0-2.\nMonitor: Reproduce failure to validate.\nStep 3: Add condition if idx >= len(ids): handle gracefully.\nMonitor: Tests now pass.\nAnswer: add len check\nDiagnosis: Root cause was unchecked index.\nConfidence: High",
        "gold_report": "Trace Summary: controller inspected tokens (fetch_user, ids[3]); introspector called out length guard fix.",
        "reflection": "Reflection: After tests, documented guard and suggested unit coverage for the error case.",
        "attention_hints": ["layer6-head4:IndexError", "layer7-head1:ids[3]"],
        "validator_tags": ["debugging"]
    },
    {
        "prompt": "Theory-of-mind task: Alice hides a coin in box A, Bob moves it to box B while Alice is away, Charlie watches both. Plan a reasoning trace that reports each agent's belief, with Step/Monitor pairs, ending 'Answer: Alice→A, Bob→B, Charlie→B', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: Alice→A, Bob→B, Charlie→B",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "tom",
        "validator_payload": {"alice": "A", "bob": "B", "charlie": "B"},
        "gold_completion": "Plan:\nPlan Step 1: Track original placement.\nPlan Step 2: Track Bob's move.\nPlan Step 3: Record each agent's observation status.\nStep 1: Alice hides coin in A.\nMonitor: Alice belief A.\nStep 2: Bob moves coin to B while Alice away.\nMonitor: Bob belief B; Alice unaware.\nStep 3: Charlie watches everything.\nMonitor: Charlie belief B.\nAnswer: Alice→A, Bob→B, Charlie→B\nDiagnosis: Belief assignments follow observations.\nConfidence: Medium",
        "gold_report": "Trace Summary: ToM nodes tracked hide/move events; introspector noted each agent's viewpoint.",
        "reflection": "Reflection: Confirmed only Charlie shares Bob's updated belief; suggested storing per-agent memory entries.",
        "attention_hints": ["layer3-head5:Alice", "layer4-head2:Bob", "layer5-head6:Charlie"],
        "validator_tags": ["theory_of_mind"]
    },
    {
        "prompt": "Scheduling: lab machine is free 9-11am, maintenance 11-1, free 1-3. Plan steps to run experiments A (1h) and B (2h) with monitors confirming slot fits; output 'Answer: schedule A at 9, B at 1', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: schedule A at 9, B at 1",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "scheduling",
        "validator_payload": {"slots": [[9,11],[13,15]], "tasks": {"A":1,"B":2}},
        "gold_completion": "Plan:\nPlan Step 1: Place 1h experiment A in earliest window.\nPlan Step 2: Place 2h experiment B after maintenance.\nStep 1: Schedule A 9-10am.\nMonitor: Leaves 10-11 open.\nStep 2: Maintenance 11-1 blocks time; schedule B 1-3pm.\nMonitor: 2h requirement satisfied.\nAnswer: schedule A at 9, B at 1\nDiagnosis: All constraints satisfied.\nConfidence: High",
        "gold_report": "Trace Summary: controller marked free intervals and assignments; introspector emphasized maintenance block.",
        "reflection": "Reflection: Verified no overlap; noted we could slide A later if future tasks appear.",
        "attention_hints": ["layer4-head1:slots", "layer5-head3:tasks"],
        "validator_tags": ["planning"]
    },
    {
        "prompt": "Robot navigation: starting at (0,0), must visit waypoint (2,1), avoid obstacle at (1,0). Plan Step/Monitor sequence describing safe moves and end 'Answer: path Up,Right,Right', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: path Up,Right,Right",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "grid_path",
        "validator_payload": {"start": [0,0], "goal": [2,1], "obstacles": [[1,0]]},
        "gold_completion": "Plan:\nPlan Step 1: Avoid obstacle at (1,0).\nPlan Step 2: Reach goal x=2.\nPlan Step 3: Move up to y=1.\nStep 1: Move Up to (0,1).\nMonitor: Obstacle avoided.\nStep 2: Move Right to (1,1), then Right to (2,1).\nMonitor: Path stays clear.\nAnswer: path Up,Right,Right\nDiagnosis: Revised plan mid-thought to avoid collision.\nConfidence: Medium",
        "gold_report": "Trace Summary: nodes tracked coordinates and obstacle; introspector noted correction (Up first).",
        "reflection": "Reflection: Documented initial error (attempted right), captured fix for memory.",
        "attention_hints": ["layer3-head2:obstacle", "layer4-head0:path"],
        "validator_tags": ["navigation"]
    },
    {
        "prompt": "Scientific reasoning: determine if increasing temperature raises pressure in a sealed container (ideal gas). Provide Plan/Step/Monitor reasoning citing PV=nRT, finish with 'Answer: pressure increases', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: pressure increases",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "ideal_gas",
        "validator_payload": {"variables": ["P","V","n","R","T"]},
        "gold_completion": "Plan:\nPlan Step 1: Recall PV = nRT.\nPlan Step 2: Hold V,n constant and reason about proportionality.\nStep 1: Equation shows P ∝ T.\nMonitor: Container sealed ⇒ V constant.\nStep 2: Increase T ⇒ P increases.\nMonitor: n,R constant.\nAnswer: pressure increases\nDiagnosis: Reasoning aligns with ideal gas law.\nConfidence: High",
        "gold_report": "Trace Summary: attention fixated on PV=nRT tokens; introspector highlighted proportional reasoning.",
        "reflection": "Reflection: Noted assumption (ideal gas) and suggested verifying units.",
        "attention_hints": ["layer2-head4:PV=nRT"],
        "validator_tags": ["physics"]
    },
    {
        "prompt": "Data migration plan: move users table from legacy DB to new DB with zero downtime. Produce Plan/Step/Monitor text covering schema check, dual-write, cutover, answer 'Answer: dual-write then switch', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: dual-write then switch",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "migration",
        "validator_payload": {"table": "users"},
        "gold_completion": "Plan:\nPlan Step 1: Verify schemas match.\nPlan Step 2: Enable dual-write from app.\nPlan Step 3: Backfill legacy rows then cut over.\nStep 1: Check columns and indexes.\nMonitor: Differences resolved.\nStep 2: Turn on dual-write.\nMonitor: metrics show both DBs receiving data.\nStep 3: Backfill old rows, verify counts, flip reads.\nMonitor: Health checks green.\nAnswer: dual-write then switch\nDiagnosis: Zero-downtime achieved via dual-write.\nConfidence: Medium",
        "gold_report": "Trace Summary: introspector cited schema audit, dual-write window, cutover verification.",
        "reflection": "Reflection: Documented rollback plan and dual-write duration for future reviewers.",
        "attention_hints": ["layer6-head2:dual-write"],
        "validator_tags": ["ops"]
    },
    {
        "prompt": "Clinical reasoning: Patient has fever 38.8°C, sore throat, swollen lymph nodes, negative cough. Build Plan/Step/Monitor reasoning to decide between viral pharyngitis vs. strep infection. End with 'Answer: treat as streptococcal pharyngitis with antibiotics', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: treat as streptococcal pharyngitis with antibiotics",
        "keywords": ["Plan", "Monitor", "Diagnosis", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "diagnostic_score",
        "validator_payload": {"criteria": ["fever", "tonsillar exudate", "adenopathy", "cough"], "score": 4},
        "gold_completion": "Plan:\nPlan Step 1: Score Centor criteria.\nPlan Step 2: Compare viral vs bacterial red flags.\nPlan Step 3: Decide therapy and monitoring.\nStep 1: Fever, lymph nodes, tonsillar exudate, absence of cough ⇒ Centor 4.\nMonitor: Double-check chart vitals to confirm 38.8°C and anterior adenopathy.\nStep 2: Viral signs (cough, rhinorrhea) absent while bacterial markers present.\nMonitor: Re-read HPI for viral clues; none.\nStep 3: Recommend narrow-spectrum antibiotic and throat culture follow-up.\nMonitor: Ensure allergy history reviewed; none noted.\nAnswer: treat as streptococcal pharyngitis with antibiotics\nDiagnosis: High Centor score plus monitors confirm bacterial etiology.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention latched onto Centor tokens (fever, exudate, cough-absent); introspector emphasized bacterial path.",
        "reflection_memo": {
            "summary": "Reflection: Documented Centor scoring and culture backup if symptoms persist.",
            "actions": ["Repeat throat culture in 48h if unresolved"],
            "risks": ["Antibiotic resistance if misdiagnosed"]
        },
        "attention_hints": [
            {"location": "layer5-head3", "target": "Centor", "note": "scoring", "weight": 1.2},
            {"location": "layer6-head1", "target": "38.8", "note": "fever", "weight": 1.0}
        ],
        "validator_tags": ["medical", "diagnosis"]
    },
    {
        "prompt": "Multi-agent planning: Alice must deliver package A then meet Bob; Bob must stay at cafe until Alice arrives. Provide Plan/Step/Monitor reasoning that tracks each agent's belief state, ending with 'Answer: Alice path = Office→Locker→Cafe, Bob wait = Cafe', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: Alice path = Office→Locker→Cafe, Bob wait = Cafe",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "path_logic",
        "validator_payload": {"alice": ["office", "locker", "cafe"], "bob": ["cafe"]},
        "gold_completion": "Plan:\nPlan Step 1: Track Alice tasks (deliver A, meet Bob).\nPlan Step 2: Track Bob constraint (wait at cafe).\nPlan Step 3: Synchronize arrival times.\nStep 1: Alice leaves office, picks up package at locker.\nMonitor: Confirm locker is on route and package must be dropped off.\nStep 2: Bob remains at cafe until Alice arrives; no travel.\nMonitor: Bob's instructions forbid moving before meetup.\nStep 3: Alice heads to cafe after locker delivery.\nMonitor: Compare ETA to Bob's wait window.\nAnswer: Alice path = Office→Locker→Cafe, Bob wait = Cafe\nDiagnosis: Constraints satisfied; both tasks complete.\nConfidence: High",
        "gold_report": "Trace Summary: controller traced agent nodes (Alice path, Bob wait); introspector highlighted synchronized arrival reasoning.",
        "reflection_memo": {
            "summary": "Reflection: Added note to alert Bob if Alice delayed.",
            "actions": ["Add contingency ping if locker jam occurs"],
            "risks": ["Cafe closes early"]
        },
        "attention_hints": [
            {"location": "layer4-head2", "target": "Alice", "note": "agent state"},
            {"location": "layer7-head3", "target": "Bob", "note": "wait policy"}
        ],
        "validator_tags": ["multi-agent", "planning"]
    },
    {
        "prompt": "Math proof: Show that the product of two consecutive integers is even. Produce Plan/Step/Monitor reasoning, end with 'Answer: product even', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: product even",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "parity_proof",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Represent consecutive integers as n and n+1.\nPlan Step 2: Show one must be even.\nStep 1: n(n+1) expands to n^2 + n.\nMonitor: Substitute n=2k or 2k+1 to test parity.\nStep 2: If n even, product even; if n odd, n+1 even.\nMonitor: Check both cases explicitly.\nAnswer: product even\nDiagnosis: Exhaustive parity check proves statement.\nConfidence: High",
        "gold_report": "Trace Summary: attention toggled between cases (n even/odd); introspector emphasized parity coverage.",
        "reflection_memo": {
            "summary": "Reflection: Documented case split and suggested linking to induction exercises.",
            "actions": ["Add example with actual numbers"],
            "risks": []
        },
        "attention_hints": [{"location": "layer3-head1", "target": "n(n+1)"}],
        "validator_tags": ["math", "proof"]
    },
    {
        "prompt": "Debugging: A REST API intermittently returns 500 due to race between cache refresh and DB write. Provide Plan/Step/Monitor reasoning to identify race and propose fix. End with 'Answer: add transactional lock + await cache prime', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: add transactional lock + await cache prime",
        "keywords": ["Plan", "Monitor", "Diagnosis", "Answer"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "bugfix",
        "validator_payload": {"error": "race"},
        "gold_completion": "Plan:\nPlan Step 1: Inspect logs around cache refresh.\nPlan Step 2: Trace DB writes vs cache invalidation.\nPlan Step 3: Propose locking fix.\nStep 1: Logs show refresh job clearing cache before DB commit.\nMonitor: Correlate timestamps; find 200ms gap.\nStep 2: Request hits stale cache pointer, causing null fetch.\nMonitor: Reproduce with concurrent requests.\nStep 3: Add transaction lock and await cache prime completion.\nMonitor: Rerun stress test; 500s disappear.\nAnswer: add transactional lock + await cache prime\nDiagnosis: Root cause is race between cache invalidation and commit.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention focused on log timestamps, cache job; introspector flagged race window.",
        "reflection_memo": {
            "summary": "Reflection: Suggested instrumentation on cache job duration.",
            "actions": ["Add metric for refresh latency"],
            "risks": ["Lock contention"]
        },
        "attention_hints": [
            {"location": "layer8-head2", "target": "cache", "note": "race"},
            {"location": "layer9-head1", "target": "500", "note": "error"}
        ],
        "validator_tags": ["debugging", "ops"]
    },
    {
        "prompt": "Robotics grid task: robot at (0,0) must collect battery at (1,2) then exit at (3,3) avoiding obstacle at (2,2). Provide Plan/Step/Monitor reasoning for pathfinding and end with 'Answer: path [(0,0),(1,0),(1,1),(1,2),(2,2 blocked reroute),(3,2),(3,3)]', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: path [(0,0),(1,0),(1,1),(1,2),(3,2),(3,3)]",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "grid_path",
        "validator_payload": {
            "start": [0, 0],
            "battery": [1, 2],
            "goal": [3, 3],
            "obstacles": [[2, 2]]
        },
        "gold_completion": "Plan:\nPlan Step 1: Reach battery at (1,2).\nPlan Step 2: Route to exit while avoiding (2,2).\nPlan Step 3: Validate waypoint sequence.\nStep 1: Path (0,0)->(1,0)->(1,1)->(1,2).\nMonitor: Confirm each move allowed.\nStep 2: From (1,2) go (2,2?) blocked so detour to (3,2) then (3,3).\nMonitor: Check obstacle list before committing.\nStep 3: Concatenate path segments.\nMonitor: Ensure no repeat nodes and goal reached.\nAnswer: path [(0,0),(1,0),(1,1),(1,2),(3,2),(3,3)]\nDiagnosis: Battery collected and obstacle avoided.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention toggled between grid coordinates; introspector logged obstacle check at (2,2).",
        "reflection_memo": {
            "summary": "Reflection: Suggested annotating cost map for future runs.",
            "actions": ["Store detour cost"],
            "risks": ["Dynamic obstacles not modeled"]
        },
        "attention_hints": [{"location": "layer5-head4", "target": "(2,2)", "note": "obstacle"}],
        "validator_tags": ["robotics", "planning"]
    },
    {
        "prompt": "Legal reasoning: Determine if a proposed contract clause violates GDPR data minimization. Provide Plan/Step/Monitor reasoning referencing obligations, end with 'Answer: clause violates GDPR Article 5(1)(c)', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: clause violates GDPR Article 5(1)(c)",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "gdpr_clause",
        "validator_payload": {"article": "5(1)(c)"},
        "gold_completion": "Plan:\nPlan Step 1: Identify data collected.\nPlan Step 2: Compare to purpose.\nPlan Step 3: Map to GDPR clause.\nStep 1: Clause stores full device telemetry indefinitely.\nMonitor: Highlight that telemetry exceeds stated purpose (billing).\nStep 2: Billing needs usage totals, not granular sensor data.\nMonitor: Re-check requirement doc.\nStep 3: Article 5(1)(c) demands data minimization.\nMonitor: Cross-reference clause text.\nAnswer: clause violates GDPR Article 5(1)(c)\nDiagnosis: Retention of unnecessary data breaches minimization principle.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention examined telemetry scope and Article 5 text; introspector cited minimization rule.",
        "reflection_memo": {
            "summary": "Reflection: Recommend redlining clause to limit telemetry fields.",
            "actions": ["Add retention cap"],
            "risks": ["Regulatory fine"]
        },
        "attention_hints": [{"location": "layer6-head3", "target": "Article 5"}],
        "validator_tags": ["legal", "compliance"]
    },
    {
        "prompt": "Dataset QA: CSV column order changed (id,name,email -> id,email,name) causing pipeline failure. Create Plan/Step/Monitor reasoning to detect schema drift and patch parser. End with 'Answer: update parser to read by header + add schema monitor', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: update parser to read by header + add schema monitor",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "schema_change",
        "validator_payload": {"expected": ["id", "name", "email"], "actual": ["id", "email", "name"]},
        "gold_completion": "Plan:\nPlan Step 1: Reproduce pipeline error.\nPlan Step 2: Inspect CSV headers.\nPlan Step 3: Patch parser + add monitor.\nStep 1: Pipeline fails mapping row[1] to name.\nMonitor: Confirm stack trace hitting parser.\nStep 2: Header order now id,email,name.\nMonitor: Diff headers vs expected.\nStep 3: Parse by header lookup and emit alert if order changes.\nMonitor: Run unit test; passes.\nAnswer: update parser to read by header + add schema monitor\nDiagnosis: Schema drift identified and mitigated.\nConfidence: High",
        "gold_report": "Trace Summary: attention looked at headers, parser lines; introspector highlighted schema monitor addition.",
        "reflection_memo": {
            "summary": "Reflection: Added CI test to lock header order and monitor alerts.",
            "actions": ["CI schema test"],
            "risks": ["Vendors may add new columns"]
        },
        "attention_hints": [{"location": "layer5-head2", "target": "header"}],
        "validator_tags": ["data", "ops"]
    },
    {
        "prompt": "Probability reasoning: An urn has 3 red, 5 blue, 4 green marbles. Draw two without replacement. Provide Plan/Step/Monitor reasoning to compute P(red then green). Conclude with 'Answer: 5/22', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: 5/22",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "probability_draws",
        "validator_payload": {"numerator": 12, "denominator": 66, "simplified": "5/22"},
        "gold_completion": "Plan:\nPlan Step 1: Compute probability of red first.\nPlan Step 2: Compute probability of green second given red removed.\nStep 1: P(red first) = 3/12.\nMonitor: Confirm counts total 12.\nStep 2: After removing a red, greens remain 4/11. Multiply => 12/132 = 5/22.\nMonitor: Reduce fraction carefully.\nAnswer: 5/22\nDiagnosis: Calculation consistent with combinatorics.\nConfidence: Medium",
        "gold_report": "Trace Summary: heads latched onto ratios (3/12,4/11); introspector highlighted replacement logic.",
        "reflection_memo": {
            "summary": "Reflection: Documented fraction reduction and suggested checking complementary event for sanity.",
            "actions": ["Add sanity check vs simulation"],
            "risks": ["Arithmetic slip"]
        },
        "attention_hints": [{"location": "layer4-head3", "target": "3/12"}, {"location": "layer5-head2", "target": "4/11"}],
        "validator_tags": ["math", "probability"]
    },
    {
        "prompt": "Chemistry lab: Balance combustion of C3H8 with oxygen. Provide Plan/Step/Monitor reasoning, finishing with 'Answer: C3H8 + 5 O2 -> 3 CO2 + 4 H2O', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: C3H8 + 5 O2 -> 3 CO2 + 4 H2O",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "chem_balance",
        "validator_payload": {"equation": "c3h8 + 5 o2 -> 3 co2 + 4 h2o"},
        "gold_completion": "Plan:\nPlan Step 1: Balance C atoms.\nPlan Step 2: Balance H then O.\nStep 1: 3 carbons => 3 CO2.\nMonitor: Check carbon count.\nStep 2: 8 hydrogens => 4 H2O, then oxygen needs 5 O2.\nMonitor: Verify O count 10 = 10.\nAnswer: C3H8 + 5 O2 -> 3 CO2 + 4 H2O\nDiagnosis: All atoms balanced.\nConfidence: High",
        "gold_report": "Trace Summary: attention watched coefficients (3,4,5); introspector noted oxygen double-check.",
        "reflection_memo": {
            "summary": "Reflection: Suggested linking to enthalpy calc for future steps.",
            "actions": ["Add energy estimation"],
            "risks": []
        },
        "attention_hints": [{"location": "layer3-head2", "target": "5 O2"}],
        "validator_tags": ["chemistry"]
    },
    {
        "prompt": "Security incident response: API keys leaked on a public repo. Produce Plan/Step/Monitor reasoning that covers detection, containment, eradication, recovery. Finish with 'Answer: rotate keys + audit + monitor', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: rotate keys + audit + monitor",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "soc_response",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Detection/triage.\nPlan Step 2: Containment + eradication.\nPlan Step 3: Recovery + monitor.\nStep 1: Alert triggered by repo scan.\nMonitor: Confirm commit hash.\nStep 2: Rotate keys, revoke secrets, purge caches.\nMonitor: Verify no unauthorized use.\nStep 3: Audit logs, add monitoring, postmortem.\nMonitor: Ensure alerts configured.\nAnswer: rotate keys + audit + monitor\nDiagnosis: Response follows IR phases.\nConfidence: High",
        "gold_report": "Trace Summary: attention highlighted containment/eradication; introspector cited rotation + monitoring.",
        "reflection_memo": {
            "summary": "Reflection: Capture retro action items for developer education.",
            "actions": ["Add secret scanning pre-commit"],
            "risks": ["Credential reuse"]
        },
        "attention_hints": [{"location": "layer6-head2", "target": "containment"}],
        "validator_tags": ["security", "ops"]
    },
    {
        "prompt": "Network diagnostics: Site latency spiked between DC-A and DC-B. Provide Plan/Step/Monitor reasoning to trace path, check MTU, verify routing. End with 'Answer: mis-sized MTU on hop 4 fixed with 1500 bytes', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: mis-sized MTU on hop 4 fixed with 1500 bytes",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "network_diag",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Run traceroute.\nPlan Step 2: Check MTU per hop.\nPlan Step 3: Verify routing tables.\nStep 1: Traceroute shows hop4 >200ms.\nMonitor: Snapshot path.\nStep 2: Ping with DF bit fails at 1472 bytes. Set MTU 1500.\nMonitor: Retest ping.\nStep 3: Confirm routes converge.\nMonitor: Collect post-fix latency.\nAnswer: mis-sized MTU on hop 4 fixed with 1500 bytes\nDiagnosis: Root cause MTU mismatch.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention followed traceroute logs, MTU tests; introspector noted DF-bit failure.",
        "reflection_memo": {
            "summary": "Reflection: Add synthetic monitoring for jumbo frames.",
            "actions": ["Configure continuous ping"],
            "risks": ["Hidden MPLS paths"]
        },
        "attention_hints": [{"location": "layer7-head2", "target": "MTU"}],
        "validator_tags": ["networking"]
    },
    {
        "prompt": "Therapy planning: Client with generalized anxiety seeks non-pharma plan. Provide Plan/Step/Monitor reasoning referencing CBT, exposure hierarchy, journaling. Finish with 'Answer: CBT + graded exposure + journaling protocol', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: CBT + graded exposure + journaling protocol",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "therapy_plan",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Introduce CBT framework.\nPlan Step 2: Build exposure ladder.\nPlan Step 3: Add journaling + monitoring.\nStep 1: Explain CBT techniques.\nMonitor: Ensure client buy-in.\nStep 2: Draft exposure hierarchy from low to high anxiety.\nMonitor: Rate SUDS each week.\nStep 3: Assign journaling + weekly review.\nMonitor: Track triggers and wins.\nAnswer: CBT + graded exposure + journaling protocol\nDiagnosis: Plan matches non-pharma goal.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention focused on CBT/exposure steps; introspector referenced journaling monitors.",
        "reflection_memo": {
            "summary": "Reflection: Flagged need for referral if symptoms worsen.",
            "actions": ["Schedule quarterly psychiatrist check"],
            "risks": ["Escalation"]
        },
        "attention_hints": [{"location": "layer5-head1", "target": "CBT"}],
        "validator_tags": ["clinical", "planning"]
    },
    {
        "prompt": "Supply chain: Factory faces 6-week lead time volatility. Provide Plan/Step/Monitor reasoning to size safety stock and buffers. End with 'Answer: add 3-week safety stock + daily buffer review', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: add 3-week safety stock + daily buffer review",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "supply_chain",
        "validator_payload": {"buffer_weeks": 3},
        "gold_completion": "Plan:\nPlan Step 1: Review demand + lead variability.\nPlan Step 2: Compute safety stock.\nPlan Step 3: Implement buffer review ritual.\nStep 1: Std dev ~1 week so target 3-week buffer.\nMonitor: Validate forecast.\nStep 2: Add safety stock = Z*σ*lead.\nMonitor: sanity-check capacity.\nStep 3: Daily buffer review board.\nMonitor: Track burn rate.\nAnswer: add 3-week safety stock + daily buffer review\nDiagnosis: Buffer covers volatility.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention toggled between lead time stats, safety stock formula; introspector cited buffer board.",
        "reflection_memo": {
            "summary": "Reflection: Mentioned supplier diversification as next step.",
            "actions": ["Qualify second supplier"],
            "risks": ["Carrying costs"]
        },
        "attention_hints": [{"location": "layer6-head3", "target": "safety stock"}],
        "validator_tags": ["operations"]
    },
    {
        "prompt": "ML training review: Team reports unstable loss after changing optimizer. Provide Plan/Step/Monitor reasoning to inspect learning rate, batch size, validation drift. End with 'Answer: lower LR to 3e-4, restore batch 64, reintroduce holdout', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: lower LR to 3e-4, restore batch 64, reintroduce holdout",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "ml_training",
        "validator_payload": {"lr": "3e-4", "batch": 64},
        "gold_completion": "Plan:\nPlan Step 1: Inspect optimizer hyperparams.\nPlan Step 2: Check batch/grad stats.\nPlan Step 3: Validate holdout split.\nStep 1: LR jumped to 1e-3; propose 3e-4.\nMonitor: Review loss curves.\nStep 2: Batch set to 128 causing instability; revert 64.\nMonitor: Check gradient norms.\nStep 3: Holdout disabled; restore 10% for drift detection.\nMonitor: Track validation gap.\nAnswer: lower LR to 3e-4, restore batch 64, reintroduce holdout\nDiagnosis: Hyperparam change caused divergence.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention examined LR, batch, holdout tokens; introspector flagged validation drift.",
        "reflection_memo": {
            "summary": "Reflection: Add sweep automation before next change.",
            "actions": ["Schedule hyperparam sweep"],
            "risks": ["Training cost"]
        },
        "attention_hints": [{"location": "layer7-head1", "target": "lr"}],
        "validator_tags": ["ml", "debugging"]
    },
    {
        "prompt": "AI ethics review: New personalization feature collects browsing history. Provide Plan/Step/Monitor reasoning referencing transparency notice, consent, opt-out controls. End with 'Answer: require explicit consent + opt-out + transparency log', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: require explicit consent + opt-out + transparency log",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "ethics_ai",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Review data purpose vs necessity.\nPlan Step 2: Evaluate consent/transparency.\nPlan Step 3: Define opt-out enforcement.\nStep 1: History collection exceeds default need.\nMonitor: Check DPIA summary.\nStep 2: Add explicit notice + transparency log.\nMonitor: Legal approves copy.\nStep 3: Provide opt-out + deletion workflow.\nMonitor: QA automation ensures path works.\nAnswer: require explicit consent + opt-out + transparency log\nDiagnosis: Without controls feature violates minimization.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention inspected consent/opt-out tokens; introspector cited transparency log requirement.",
        "reflection_memo": {
            "summary": "Reflection: Add quarterly compliance review.",
            "actions": ["Schedule DPIA refresh"],
            "risks": ["Regulator inquiry"]
        },
        "attention_hints": [{"location": "layer6-head4", "target": "consent"}],
        "validator_tags": ["ethics", "compliance"]
    },
    {
        "prompt": "Financial stress test: Bank portfolio has $400M corporate loans with PD=2%, LGD=45%. Provide Plan/Step/Monitor reasoning to compute expected loss and capital buffer. Finish with 'Answer: expected loss $3.6M, buffer $5M', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: expected loss $3.6M, buffer $5M",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "finance_stress",
        "validator_payload": {"expected_loss": "3.6", "buffer": "5"},
        "gold_completion": "Plan:\nPlan Step 1: Compute expected loss (EL = PD * LGD * EAD).\nPlan Step 2: Size capital buffer with margin.\nPlan Step 3: Document monitoring.\nStep 1: EL = 0.02 * 0.45 * 400M = $3.6M.\nMonitor: Re-check arithmetic.\nStep 2: Add 40% margin => ~$5M buffer.\nMonitor: Align with policy.\nStep 3: Monitor PD quarterly.\nMonitor: Ensure triggers set.\nAnswer: expected loss $3.6M, buffer $5M\nDiagnosis: Stress test aligns with policy.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention focused on PD/LGD tokens; introspector highlighted margin computation.",
        "reflection_memo": {
            "summary": "Reflection: Suggest scenario run with 3% PD spike.",
            "actions": ["Scenario PD=3%"],
            "risks": ["Economic downturn"]
        },
        "attention_hints": [{"location": "layer4-head1", "target": "0.02"}],
        "validator_tags": ["finance"]
    },
    {
        "prompt": "Comparative law: Evaluate if Clause X compliant with GDPR and CCPA data access rules. Provide Plan/Step/Monitor reasoning referencing GDPR Art 15 and CCPA §1798.100. End with 'Answer: clause compliant only if access portal provided + 45-day SLA', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: clause compliant only if access portal provided + 45-day SLA",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "comparative_law",
        "validator_payload": {"articles": ["art 15", "1798.100"]},
        "gold_completion": "Plan:\nPlan Step 1: Summarize GDPR art15 obligations.\nPlan Step 2: Summarize CCPA access rights.\nPlan Step 3: Compare Clause X.\nStep 1: GDPR requires access portal + 30 day response (extend 60).\nMonitor: Check clause text.\nStep 2: CCPA 45-day SLA, opt-out rights.\nMonitor: Citiations correct.\nStep 3: Clause X needs portal + 45-day guarantee.\nMonitor: Ensure obligations spelled out.\nAnswer: clause compliant only if access portal provided + 45-day SLA\nDiagnosis: Without portal, noncompliant.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention compared Art15 vs §1798.100 tokens; introspector flagged SLA requirement.",
        "reflection_memo": {
            "summary": "Reflection: Add data deletion clause check next iteration.",
            "actions": ["Review deletion rights"],
            "risks": ["Reg mismatch"]
        },
        "attention_hints": [{"location": "layer6-head3", "target": "Art 15"}],
        "validator_tags": ["legal", "compliance"]
    },
    {
        "prompt": "Robotics control: Drone must hold altitude ±0.1m despite wind gusts. Provide Plan/Step/Monitor reasoning referencing PID tuning, integral windup, sensor fusion. Conclude with 'Answer: retune PID (Kp 0.8, Ki 0.2, Kd 0.05) + add complementary filter', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: retune PID (Kp 0.8, Ki 0.2, Kd 0.05) + add complementary filter",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "pid_tuning",
        "validator_payload": {"kp": "0.8", "ki": "0.2", "kd": "0.05"},
        "gold_completion": "Plan:\nPlan Step 1: Inspect PID response logs.\nPlan Step 2: Mitigate integral windup.\nPlan Step 3: Add complementary filter.\nStep 1: Gains too aggressive; propose Kp0.8, Ki0.2, Kd0.05.\nMonitor: Re-run step response.\nStep 2: Add anti-windup clamp.\nMonitor: Check integral term.\nStep 3: Fuse IMU + barometer via complementary filter.\nMonitor: Evaluate altitude variance.\nAnswer: retune PID (Kp 0.8, Ki 0.2, Kd 0.05) + add complementary filter\nDiagnosis: Adjustments reduce overshoot.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention focused on Kp/Ki/Kd tokens; introspector noted sensor fusion.",
        "reflection_memo": {
            "summary": "Reflection: plan hardware-in-loop test.",
            "actions": ["Run HIL test"],
            "risks": ["Wind gust extremes"]
        },
        "attention_hints": [{"location": "layer5-head4", "target": "Kp"}],
        "validator_tags": ["robotics", "controls"]
    },
    {
        "prompt": "Distributed systems: Service suffers split-brain after network partition. Provide Plan/Step/Monitor reasoning referencing Raft quorum, fencing tokens, reconciliation. End with 'Answer: enforce Raft quorum + fencing tokens during partition', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: enforce Raft quorum + fencing tokens during partition",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "raft_partition",
        "validator_payload": {"quorum": 3},
        "gold_completion": "Plan:\nPlan Step 1: Inspect Raft logs.\nPlan Step 2: Enforce quorum + fencing.\nPlan Step 3: Reconcile state.\nStep 1: Partition caused two leaders.\nMonitor: Review term numbers.\nStep 2: Require quorum=3; add fencing tokens.\nMonitor: Document token issuance.\nStep 3: Reconcile logs after repair.\nMonitor: Verify snapshot.\nAnswer: enforce Raft quorum + fencing tokens during partition\nDiagnosis: Prevents dual writes.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention watched term/quorum tokens; introspector highlighted fencing tokens.",
        "reflection_memo": {
            "summary": "Reflection: Add chaos drills.",
            "actions": ["Schedule partition test"],
            "risks": ["Operator error"]
        },
        "attention_hints": [{"location": "layer6-head2", "target": "quorum"}],
        "validator_tags": ["systems", "distributed"]
    },
    {
        "prompt": "Ethics case study: Research uses facial recognition on public footage without consent. Provide Plan/Step/Monitor reasoning referencing Belmont principles, propose mitigation. End with 'Answer: obtain IRB approval + consent + blur pipeline', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: obtain IRB approval + consent + blur pipeline",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "research_ethics",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Review Belmont (respect, beneficence, justice).\nPlan Step 2: Identify consent gaps.\nPlan Step 3: Mitigate via IRB + blur.\nStep 1: Using footage without consent violates respect.\nMonitor: Cite Belmont.\nStep 2: Need consent or anonymization.\nMonitor: Validate policy.\nStep 3: Obtain IRB approval, blur pipeline, consent flow.\nMonitor: Ensure compliance.\nAnswer: obtain IRB approval + consent + blur pipeline\nDiagnosis: Without IRB + consent project unethical.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention referenced Belmont tokens; introspector emphasized IRB + anonymization.",
        "reflection_memo": {
            "summary": "Reflection: Add data minimization in future revision.",
            "actions": ["Implement minimization"],
            "risks": ["Public backlash"]
        },
        "attention_hints": [{"location": "layer5-head3", "target": "Belmont"}],
        "validator_tags": ["ethics", "research"]
    },
    {
        "prompt": "Aerospace guidance: Satellite reaction wheels saturating during sun-tracking maneuvers. Provide Plan/Step/Monitor reasoning referencing momentum dumping, quaternion control, ground commands. End with 'Answer: schedule magnetorquer momentum dump + adjust quaternion limits', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: schedule magnetorquer momentum dump + adjust quaternion limits",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "momentum_dump",
        "validator_payload": {"method": "magnetorquer"},
        "gold_completion": "Plan:\nPlan Step 1: Review wheel speed telemetry.\nPlan Step 2: Command momentum dump.\nPlan Step 3: Adjust quaternion targets.\nStep 1: Wheels near saturation.\nMonitor: check rpm logs.\nStep 2: Use magnetorquers to dump momentum.\nMonitor: confirm reaction wheel current drop.\nStep 3: Tighten quaternion slew limits.\nMonitor: simulate new profiles.\nAnswer: schedule magnetorquer momentum dump + adjust quaternion limits\nDiagnosis: Addresses saturation.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention inspected wheel rpm + quaternion tokens; introspector cited magnetorquer dump.",
        "reflection_memo": {
            "summary": "Reflection: schedule weekly dump window.",
            "actions": ["Add ground procedure"],
            "risks": ["Limited torque"]
        },
        "attention_hints": [{"location": "layer6-head3", "target": "magnetorquer"}],
        "validator_tags": ["aerospace"]
    },
    {
        "prompt": "Bioinformatics QC: RNA-seq pipeline shows GC bias. Provide Plan/Step/Monitor reasoning referencing normalization, GC content plots, spike-ins. End with 'Answer: apply GC normalization + spike-in calibration', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: apply GC normalization + spike-in calibration",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "gc_bias",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Plot GC bias vs coverage.\nPlan Step 2: Apply normalization.\nPlan Step 3: Re-run QC with spike-ins.\nStep 1: GC-rich genes overrepresented.\nMonitor: Review fastqc.\nStep 2: Use EDASeq normalization.\nMonitor: ensure slope ~0.\nStep 3: Spike-in calibration check.\nMonitor: Compare to reference.\nAnswer: apply GC normalization + spike-in calibration\nDiagnosis: QC passes after correction.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention focused on GC plots and spike-in tokens; introspector highlighted normalization step.",
        "reflection_memo": {
            "summary": "Reflection: Documented pipeline change and versioning.",
            "actions": ["Update pipeline doc"],
            "risks": ["Batch effects"]
        },
        "attention_hints": [{"location": "layer4-head2", "target": "GC"}],
        "validator_tags": ["bioinformatics"]
    },
    {
        "prompt": "Climate modeling: Regional precipitation model diverges vs observed data. Provide Plan/Step/Monitor reasoning referencing boundary conditions, assimilation, bias correction. End with 'Answer: update boundary forcing + apply quantile mapping bias correction', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: update boundary forcing + apply quantile mapping bias correction",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "climate_model",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Examine boundary forcing data.\nPlan Step 2: Assimilate observations.\nPlan Step 3: Apply bias correction.\nStep 1: Forcing outdated.\nMonitor: compare reanalysis.\nStep 2: Assimilate rainfall gauges.\nMonitor: assimilation run.\nStep 3: Use quantile mapping.\nMonitor: evaluate RMSE.\nAnswer: update boundary forcing + apply quantile mapping bias correction\nDiagnosis: Model now matches obs.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention watched boundary/assimilation terms; introspector noted bias correction.",
        "reflection_memo": {
            "summary": "Reflection: schedule seasonal validation.",
            "actions": ["Quarterly validation"],
            "risks": ["Data latency"]
        },
        "attention_hints": [{"location": "layer7-head1", "target": "quantile"}],
        "validator_tags": ["climate"]
    },
    {
        "prompt": "Negotiation strategy: Two companies bargaining over supply contract. Provide Plan/Step/Monitor reasoning referencing BATNA, ZOPA, concession plan. End with 'Answer: reveal limited concessions, reference BATNA, close in ZOPA mid-point', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: reveal limited concessions, reference BATNA, close in ZOPA mid-point",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "negotiation",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Define BATNA and ZOPA.\nPlan Step 2: Stage concessions.\nPlan Step 3: Close mid-ZOPA.\nStep 1: BATNA = alternate supplier, ZOPA range.\nMonitor: Confirm data.\nStep 2: Offer small concessions.\nMonitor: track counterpart reaction.\nStep 3: Close near midpoint referencing BATNA.\nMonitor: ensure value created.\nAnswer: reveal limited concessions, reference BATNA, close in ZOPA mid-point\nDiagnosis: Strategy preserves leverage.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention referenced BATNA/ZOPA tokens; introspector emphasized concession ladder.",
        "reflection_memo": {
            "summary": "Reflection: Add fallback clause.",
            "actions": ["Prepare fallback"],
            "risks": ["Counterparty stalls"]
        },
        "attention_hints": [{"location": "layer5-head1", "target": "BATNA"}],
        "validator_tags": ["business", "strategy"]
    },
    {
        "prompt": "Education coaching: Student struggling with calculus integrals. Provide Plan/Step/Monitor reasoning referencing prerequisite diagnosis, spaced practice, error logging. End with 'Answer: reinforce algebra skills + spaced integral drills + error journal', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: reinforce algebra skills + spaced integral drills + error journal",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "ed_coaching",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Diagnose prerequisite gaps.\nPlan Step 2: Schedule spaced practice.\nPlan Step 3: Log errors.\nStep 1: Algebra deficits found.\nMonitor: mini-quiz.\nStep 2: Daily integral drills.\nMonitor: track SRS.\nStep 3: Maintain error journal.\nMonitor: weekly review.\nAnswer: reinforce algebra skills + spaced integral drills + error journal\nDiagnosis: Addresses root skill gap.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention highlighted prerequisite tests and error logs; introspector noted spaced repetition.",
        "reflection_memo": {
            "summary": "Reflection: Suggest peer study group.",
            "actions": ["Set peer group"],
            "risks": ["Motivation"]
        },
        "attention_hints": [{"location": "layer3-head2", "target": "algebra"}],
        "validator_tags": ["education"]
    },
    {
        "prompt": "Emergency response: Hospital triage faces influx of patients. Provide Plan/Step/Monitor reasoning referencing START triage, surge capacity, resource reallocation. End with 'Answer: activate START triage + open surge ward + redeploy staff', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: activate START triage + open surge ward + redeploy staff",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "triage_start",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Trigger START triage.\nPlan Step 2: Activate surge ward.\nPlan Step 3: Redeploy staff.\nStep 1: START tags assigned.\nMonitor: track categories.\nStep 2: Open surge ward.\nMonitor: bed availability.\nStep 3: Reassign staff.\nMonitor: staffing board.\nAnswer: activate START triage + open surge ward + redeploy staff\nDiagnosis: Maintains throughput.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention referenced START categories; introspector emphasized surge workflows.",
        "reflection_memo": {
            "summary": "Reflection: plan after-action review.",
            "actions": ["Schedule AAR"],
            "risks": ["Staff fatigue"]
        },
        "attention_hints": [{"location": "layer4-head1", "target": "START"}],
        "validator_tags": ["medical", "ops"]
    },
    {
        "prompt": "Cognitive science experiment: Plan to test working memory improvements via dual n-back. Provide Plan/Step/Monitor reasoning referencing control group, statistical power, pre/post measures. End with 'Answer: randomized control + 4-week dual n-back + pre/post WM tests', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: randomized control + 4-week dual n-back + pre/post WM tests",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "wm_experiment",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Randomize participants (control vs dual n-back).\nPlan Step 2: Run 4-week training with adherence monitoring.\nPlan Step 3: Pre/post WM assessments + stats.\nStep 1: RCT with power analysis.\nMonitor: ensure n>=40.\nStep 2: Dual n-back schedule + adherence logs.\nMonitor: weekly check.\nStep 3: Run WM tests, t-test difference.\nMonitor: confirm significance.\nAnswer: randomized control + 4-week dual n-back + pre/post WM tests\nDiagnosis: Design isolates dual n-back effect.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention looked at randomization, adherence logs, WM tests; introspector emphasized power analysis.",
        "reflection_memo": {
            "summary": "Reflection: add follow-up retention test.",
            "actions": ["schedule 3-month follow-up"],
            "risks": ["dropout"]
        },
        "attention_hints": [{"location": "layer4-head1", "target": "randomized"}],
        "validator_tags": ["research", "cognitive science"]
    }
,
    {
        "prompt": "Oceanography ops: Autonomous glider deviates due to strong currents. Provide Plan/Step/Monitor reasoning referencing Kalman filter updates, mission replanning, surface GPS fixes. End with 'Answer: assimilate ADCP data + replan waypoint + surface for GPS correction', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: assimilate ADCP data + replan waypoint + surface for GPS correction",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "glider_replan",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Assimilate ADCP currents into Kalman filter.\nPlan Step 2: Generate updated waypoint plan.\nPlan Step 3: Surface for GPS correction.\nStep 1: Update filter with ADCP data.\nMonitor: check covariance.\nStep 2: Replan route to counter drift.\nMonitor: simulate trajectory.\nStep 3: Surface for GPS fix to reset nav.\nMonitor: confirm nav residuals.\nAnswer: assimilate ADCP data + replan waypoint + surface for GPS correction\nDiagnosis: Deviation resolved.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention monitored ADCP/Kalman tokens; introspector highlighted surface correction.",
        "reflection_memo": {"summary": "Reflection: add automatic replan trigger.", "actions": ["Define drift threshold"], "risks": ["Battery overhead"]},
        "attention_hints": [{"location": "layer6-head4", "target": "ADCP"}],
        "validator_tags": ["oceanography", "robotics"]
    },
    {
        "prompt": "Quantum computing workflow: QAOA circuit shows barren plateau. Provide Plan/Step/Monitor reasoning referencing parameter initialization, layer depth, gradient checks. End with 'Answer: use layer-wise initialization + depth=2 + gradient clipping', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: use layer-wise initialization + depth=2 + gradient clipping",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "qaoa_plateau",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Inspect gradient norms.\nPlan Step 2: Adjust initialization + clipping.\nPlan Step 3: Reduce ansatz depth.\nStep 1: Gradients vanish across layers.\nMonitor: measure norms.\nStep 2: Use layer-wise initialization + gradient clipping.\nMonitor: confirm stability.\nStep 3: Limit depth to 2.\nMonitor: evaluate energy curve.\nAnswer: use layer-wise initialization + depth=2 + gradient clipping\nDiagnosis: Mitigates barren plateau.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention examined gradient/depth tokens; introspector emphasized layer-wise init.",
        "reflection_memo": {"summary": "Reflection: explore alternative ansatz next.", "actions": ["Schedule ansatz sweep"], "risks": ["Hardware limits"]},
        "attention_hints": [{"location": "layer5-head2", "target": "gradient"}],
        "validator_tags": ["quantum", "ml"]
    },
    {
        "prompt": "Corporate governance: Board evaluating ESG proposal. Provide Plan/Step/Monitor reasoning referencing stakeholder analysis, materiality, reporting. End with 'Answer: approve ESG plan with materiality matrix + quarterly reporting', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: approve ESG plan with materiality matrix + quarterly reporting",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "esg_board",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Map stakeholders + material topics.\nPlan Step 2: Align proposal with strategy.\nPlan Step 3: Define reporting cadence.\nStep 1: Build materiality matrix.\nMonitor: ensure coverage.\nStep 2: Evaluate risks/opportunities.\nMonitor: board discussion.\nStep 3: Approve quarterly reporting + KPIs.\nMonitor: assign owners.\nAnswer: approve ESG plan with materiality matrix + quarterly reporting\nDiagnosis: Governance requirements satisfied.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention referenced stakeholder/materiality tokens; introspector highlighted reporting cadence.",
        "reflection_memo": {"summary": "Reflection: schedule annual ESG audit.", "actions": ["Plan audit"], "risks": ["Data gaps"]},
        "attention_hints": [{"location": "layer4-head3", "target": "materiality"}],
        "validator_tags": ["business", "ethics"]
    },
    {
        "prompt": "Game theory: Determine equilibrium of coordination game favoring strategy A. Provide Plan/Step/Monitor reasoning referencing payoff comparison, best responses, mixed strategy check. End with 'Answer: (A,A) pure NE with payoff 3,3', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: (A,A) pure NE with payoff 3,3",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 2,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 2,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "coordination_game",
        "validator_payload": {"payoff": "3,3"},
        "gold_completion": "Plan:\nPlan Step 1: Compare payoffs.\nPlan Step 2: Check best responses.\nStep 1: (A,A) yields 3,3 highest.\nMonitor: ensure other outcomes lower.\nStep 2: Best response to A is A.\nMonitor: confirm no mixed needed.\nAnswer: (A,A) pure NE with payoff 3,3\nDiagnosis: Coordination equilibrium verified.\nConfidence: High",
        "gold_report": "Trace Summary: attention reviewed payoff entries; introspector highlighted best responses.",
        "reflection_memo": {"summary": "Reflection: mention tremble-hand stability.", "actions": ["Analyze perturbations"], "risks": []},
        "attention_hints": [{"location": "layer3-head1", "target": "3,3"}],
        "validator_tags": ["economics", "game theory"]
    },
    {
        "prompt": "Energy systems: Microgrid battery SOC dropping faster than forecast. Provide Plan/Step/Monitor reasoning referencing load analysis, BMS calibration, dispatch rules. End with 'Answer: recalibrate BMS + update dispatch + add peak shaving schedule', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: recalibrate BMS + update dispatch + add peak shaving schedule",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "microgrid_soc",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Analyze load vs forecast.\nPlan Step 2: Recalibrate BMS sensors.\nPlan Step 3: Update dispatch + peak shaving schedule.\nStep 1: Identify unexpected peaks.\nMonitor: review telemetry.\nStep 2: BMS misreporting SOC; recalibrate in lab.\nMonitor: validate readings.\nStep 3: Add peak shaving schedule, tweak dispatch rules.\nMonitor: daily SOC trend.\nAnswer: recalibrate BMS + update dispatch + add peak shaving schedule\nDiagnosis: SOC aligns with forecast.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention watched load/BMS signals; introspector cited dispatch adjustment.",
        "reflection_memo": {"summary": "Reflection: plan sensor upgrade.", "actions": ["Scope sensor upgrade"], "risks": ["Capex"]},
        "attention_hints": [{"location": "layer6-head1", "target": "BMS"}],
        "validator_tags": ["energy", "ops"]
    },
    {
        "prompt": "Crisis communication: CEO must address data breach. Provide Plan/Step/Monitor reasoning referencing key messages, stakeholder mapping, follow-up cadence. End with 'Answer: issue apology + outline remediation + weekly updates', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: issue apology + outline remediation + weekly updates",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "crisis_comms",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Draft key messages (apology + remediation).\nPlan Step 2: Map stakeholders/channels.\nPlan Step 3: Commit to weekly updates.\nStep 1: Issue apology + remediation outline.\nMonitor: legal review.\nStep 2: Identify stakeholders.\nMonitor: ensure coverage.\nStep 3: Provide weekly updates until resolution.\nMonitor: track sentiment.\nAnswer: issue apology + outline remediation + weekly updates\nDiagnosis: Plan maintains trust.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention centered on apology/remediation tokens; introspector highlighted update cadence.",
        "reflection_memo": {"summary": "Reflection: host live Q&A session.", "actions": ["Plan Q&A"], "risks": ["Negative press"]},
        "attention_hints": [{"location": "layer5-head1", "target": "apology"}],
        "validator_tags": ["communications"]
    },
    {
        "prompt": "Behavioral economics: Nudging employees to enroll in 401(k). Provide Plan/Step/Monitor reasoning referencing default enrollment, reminders, feedback loops. End with 'Answer: implement auto-enroll + reminders + quarterly savings feedback', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: implement auto-enroll + reminders + quarterly savings feedback",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "nudge_401k",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Implement auto-enroll default.\nPlan Step 2: Automate reminders.\nPlan Step 3: Provide savings feedback.\nStep 1: Auto-enroll employees with opt-out.\nMonitor: track opt-outs.\nStep 2: Send quarterly reminders.\nMonitor: open rates.\nStep 3: Provide savings feedback statements.\nMonitor: participation delta.\nAnswer: implement auto-enroll + reminders + quarterly savings feedback\nDiagnosis: Nudges boost participation.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention observed default/reminder tokens; introspector noted feedback loop.",
        "reflection_memo": {"summary": "Reflection: personalize recommendations.", "actions": ["Add personalization"], "risks": ["Privacy"]},
        "attention_hints": [{"location": "layer4-head2", "target": "auto-enroll"}],
        "validator_tags": ["behavioral econ"]
    },
    {
        "prompt": "Astronomy data reduction: Hubble images contain cosmic rays. Provide Plan/Step/Monitor reasoning referencing median stacking, sigma clipping, calibration frames. End with 'Answer: median stack + sigma clip + recalibrate flats', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: median stack + sigma clip + recalibrate flats",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "astro_reduction",
        "validator_payload": {},
        "gold_completion": "Plan:\nPlan Step 1: Apply calibration frames.\nPlan Step 2: Median stack exposures.\nPlan Step 3: Sigma clip cosmic rays.\nStep 1: Recalibrate flats/darks.\nMonitor: inspect histograms.\nStep 2: Median stack to reduce noise.\nMonitor: check artifacts.\nStep 3: Sigma clip to remove rays.\nMonitor: visual QA.\nAnswer: median stack + sigma clip + recalibrate flats\nDiagnosis: Dataset cleaned.\nConfidence: Medium",
        "gold_report": "Trace Summary: attention tracked calibration/stack steps; introspector emphasized sigma clipping.",
        "reflection_memo": {"summary": "Reflection: automate QA after stacking.", "actions": ["Build QA pipeline"], "risks": []},
        "attention_hints": [{"location": "layer3-head4", "target": "sigma"}],
        "validator_tags": ["astronomy", "data"]
    },
    {
        "prompt": "Comparative politics: Evaluate coalition stability in a parliament with parties A, B, C. Provide Plan/Step/Monitor reasoning referencing seat counts, policy alignment, fallback alliances. End with 'Answer: A+B coalition stable if policy memo signed + C as confidence partner', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: A+B coalition stable if policy memo signed + C as confidence partner",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "coalition_analysis",
        "validator_payload": {},
        "gold_completion": """Plan:
Plan Step 1: Count seats and confirm majority for A+B.
Plan Step 2: Align on policy memo covering shared priorities.
Plan Step 3: Secure C as confidence partner.
Step 1: A+B seat count reaches threshold.
Monitor: verify math.
Step 2: Draft policy memo; ensure signatures.
Monitor: confirm agreement.
Step 3: Negotiate support deal with C for confidence votes.
Monitor: log commitment.
Answer: A+B coalition stable if policy memo signed + C as confidence partner
Diagnosis: Coalition sustainable with support deal.
Confidence: Medium""",
        "gold_report": "Trace Summary: attention watched seat/policy tokens; introspector emphasized support partner.",
        "reflection_memo": {
            "summary": "Reflection: schedule quarterly coalition review.",
            "actions": ["Set review cadence"],
            "risks": ["C withdrawal"]
        },
        "attention_hints": [{"location": "layer5-head1", "target": "policy memo"}],
        "validator_tags": ["politics"]
    },
    {
        "prompt": "Advanced robotics: Quadruped robot slips on wet terrain. Provide Plan/Step/Monitor reasoning referencing gait adaptation, force sensor calibration, IMU fusion. End with 'Answer: adapt gait to crawl mode + recalibrate force sensors + fuse IMU with EKF', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: adapt gait to crawl mode + recalibrate force sensors + fuse IMU with EKF",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "quadruped_gait",
        "validator_payload": {},
        "gold_completion": """Plan:
Plan Step 1: Switch to crawl gait for stability.
Plan Step 2: Recalibrate force sensors for wet terrain.
Plan Step 3: Fuse IMU data with EKF.
Step 1: Crawl mode reduces slip.
Monitor: review telemetry.
Step 2: Calibrate force sensors.
Monitor: validate readings.
Step 3: Fuse IMU via EKF for posture estimates.
Monitor: inspect pitch/roll.
Answer: adapt gait to crawl mode + recalibrate force sensors + fuse IMU with EKF
Diagnosis: Robot keeps traction on wet surface.
Confidence: Medium""",
        "gold_report": "Trace Summary: attention focused on gait/force/IMU tokens; introspector highlighted EKF fusion.",
        "reflection_memo": {
            "summary": "Reflection: add slip detection alert.",
            "actions": ["Implement slip alert"],
            "risks": ["Energy cost"]
        },
        "attention_hints": [{"location": "layer6-head2", "target": "gait"}],
        "validator_tags": ["robotics"]
    },
    {
        "prompt": "Climate mitigation project: City aims to cut emissions 40% by 2030. Provide Plan/Step/Monitor reasoning referencing baseline inventory, renewable procurement, transport incentives. End with 'Answer: baseline inventory + 30% renewables + EV transit incentives', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: baseline inventory + 30% renewables + EV transit incentives",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "climate_mitigation",
        "validator_payload": {},
        "gold_completion": """Plan:
Plan Step 1: Establish baseline inventory.
Plan Step 2: Procure 30% renewables.
Plan Step 3: Incentivize EV transit.
Step 1: Inventory current emissions.
Monitor: audit data.
Step 2: Secure renewable contracts.
Monitor: procurement status.
Step 3: Launch EV transit incentives.
Monitor: adoption metrics.
Answer: baseline inventory + 30% renewables + EV transit incentives
Diagnosis: Plan aligns with 40% reduction target.
Confidence: Medium""",
        "gold_report": "Trace Summary: attention tracked inventory/renewable tokens; introspector noted transit incentives.",
        "reflection_memo": {
            "summary": "Reflection: publish annual progress report.",
            "actions": ["Annual report"],
            "risks": ["Funding"]
        },
        "attention_hints": [{"location": "layer4-head1", "target": "inventory"}],
        "validator_tags": ["climate", "policy"]
    },
    {
        "prompt": "Supply-chain compliance: Vendor audit reveals labor violations. Provide Plan/Step/Monitor reasoning referencing corrective action plan, monitoring, sourcing diversification. End with 'Answer: enforce CAP + monthly audits + diversify suppliers', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: enforce CAP + monthly audits + diversify suppliers",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "supply_compliance",
        "validator_payload": {},
        "gold_completion": """Plan:
Plan Step 1: Issue corrective action plan (CAP).
Plan Step 2: Schedule monthly audits.
Plan Step 3: Diversify suppliers.
Step 1: CAP with deadlines.
Monitor: track milestones.
Step 2: Monthly audits ensure remediation.
Monitor: compliance log.
Step 3: Diversify sourcing to reduce risk.
Monitor: supplier mix.
Answer: enforce CAP + monthly audits + diversify suppliers
Diagnosis: Compliance risk mitigated.
Confidence: Medium""",
        "gold_report": "Trace Summary: attention referenced CAP/audit tokens; introspector stressed diversification.",
        "reflection_memo": {
            "summary": "Reflection: add worker hotline.",
            "actions": ["Implement hotline"],
            "risks": ["Cost"]
        },
        "attention_hints": [{"location": "layer5-head4", "target": "CAP"}],
        "validator_tags": ["supply chain", "compliance"]
    },
    {
        "prompt": "Insurance actuarial review: Loss ratio trending above targets. Provide Plan/Step/Monitor reasoning referencing frequency/severity analysis, underwriting adjustments, reinsurance. End with 'Answer: adjust underwriting + revise rates + buy excess reinsurance', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: adjust underwriting + revise rates + buy excess reinsurance",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "actuarial_review",
        "validator_payload": {},
        "gold_completion": """Plan:
Plan Step 1: Analyze frequency/severity.
Plan Step 2: Adjust underwriting + rates.
Plan Step 3: Purchase excess reinsurance.
Step 1: Frequency spike traced.
Monitor: review data.
Step 2: Tighten underwriting + revise rates.
Monitor: rate filing.
Step 3: Buy excess reinsurance to cover tail.
Monitor: capacity secured.
Answer: adjust underwriting + revise rates + buy excess reinsurance
Diagnosis: Loss ratio returns to target.
Confidence: Medium""",
        "gold_report": "Trace Summary: attention tracked frequency/severity tokens; introspector highlighted reinsurance.",
        "reflection_memo": {
            "summary": "Reflection: add quarterly monitoring.",
            "actions": ["Quarterly review"],
            "risks": ["Reg approval"]
        },
        "attention_hints": [{"location": "layer6-head2", "target": "loss ratio"}],
        "validator_tags": ["finance", "insurance"]
    },
    {
        "prompt": "Telecom network planning: 5G rollout faces backhaul congestion. Provide Plan/Step/Monitor reasoning referencing traffic modeling, fiber upgrades, microwave links. End with 'Answer: upgrade fiber on routes X,Y + deploy microwave backup', 'Diagnosis:', 'Confidence:'.",
        "answer": "Answer: upgrade fiber on routes X,Y + deploy microwave backup",
        "keywords": ["Plan", "Monitor", "Answer", "Diagnosis"],
        "monitoring_markers": ["Monitor:"],
        "min_monitoring_mentions": 3,
        "plan_markers": ["Plan Step"],
        "min_plan_steps": 3,
        "diagnosis_markers": ["Diagnosis:"],
        "require_diagnosis": True,
        "require_confidence": True,
        "validator": "5g_backhaul",
        "validator_payload": {},
        "gold_completion": """Plan:
Plan Step 1: Model traffic growth.
Plan Step 2: Upgrade fiber on congested routes X,Y.
Plan Step 3: Deploy microwave backup links.
Step 1: Identify saturated routes.
Monitor: demand projections.
Step 2: Upgrade fiber capacity.
Monitor: capex plan.
Step 3: Deploy microwave redundancy.
Monitor: run failover test.
Answer: upgrade fiber on routes X,Y + deploy microwave backup
Diagnosis: Backhaul congestion mitigated.
Confidence: Medium""",
        "gold_report": "Trace Summary: attention focused on traffic modeling/fiber tokens; introspector highlighted redundancy.",
        "reflection_memo": {
            "summary": "Reflection: add monitoring dashboard.",
            "actions": ["Build dashboard"],
            "risks": ["Budget"]
        },
        "attention_hints": [{"location": "layer5-head3", "target": "fiber"}],
        "validator_tags": ["telecom", "ops"]
    }

]


def main() -> None:
    out_path = Path("data/reasoning_plan_dataset_curated.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for entry in CURATED_TASKS:
            handle.write(json.dumps(_normalize_entry(entry), ensure_ascii=False) + "\n")
    print(f"Wrote {len(CURATED_TASKS)} curated tasks to {out_path}")


if __name__ == "__main__":
    main()
