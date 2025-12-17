from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _numbers(text: str) -> List[int]:
    return [int(match.group()) for match in re.finditer(r"-?\d+", text)]


def _contains_all(text: str, tokens: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(tok.lower() in lowered for tok in tokens if tok)


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(tok.lower() in lowered for tok in tokens if tok)


def run_validator(kind: Optional[str], payload: Optional[Dict[str, Any]], completion: str) -> Optional[bool]:
    if not kind:
        return None
    text = completion.lower()
    payload = payload or {}

    if kind == "addition":
        result = int(payload.get("result", 0))
        operands = payload.get("operands") or []
        if len(operands) >= 2:
            if not all(str(op) in completion for op in operands):
                return False
        reported = _numbers(completion)
        return result in reported

    if kind == "subtraction":
        result = int(payload.get("result", 0))
        return str(result) in completion

    if kind == "linear_equation":
        solution = payload.get("solution")
        if solution is None:
            return None
        return str(solution) in completion or f"x = {solution}" in completion.lower()

    if kind == "division_story":
        each = payload.get("each")
        leftovers = payload.get("leftovers")
        tokens = [str(each), str(leftovers)]
        return _contains_all(text, tokens)

    if kind == "ratio_perimeter":
        perimeter = payload.get("perimeter")
        ratio = payload.get("ratio", "")
        return str(perimeter) in completion and str(ratio).lower() in text

    if kind == "diagnostic_score":
        score = str(payload.get("score", ""))
        criteria = [str(c) for c in payload.get("criteria", [])]
        return score in completion and _contains_all(text, criteria)

    if kind == "path_logic":
        alice = payload.get("alice", [])
        bob = payload.get("bob", [])
        alice_ok = all(loc.lower() in text for loc in alice)
        bob_ok = all(loc.lower() in text for loc in bob)
        return alice_ok and bob_ok

    if kind == "parity_proof":
        return _contains_all(text, ["even", "odd"]) and "n+1" in completion.replace(" ", "")

    if kind == "bugfix":
        error = str(payload.get("error", "")).lower()
        return error in text and _contains_any(text, ["fix", "guard", "lock", "await"])

    if kind == "grid_path":
        coords = []
        for label in ("start", "battery", "goal"):
            point = payload.get(label)
            if point:
                coords.append(f"({point[0]},{point[1]})")
        for obstacle in payload.get("obstacles", []) or []:
            coords.append(f"({obstacle[0]},{obstacle[1]})")
        return all(coord in completion.replace(" ", "") for coord in coords)

    if kind == "gdpr_clause":
        article = str(payload.get("article", "")).lower()
        return article in text and "minimization" in text

    if kind == "schema_change":
        expected = payload.get("expected", []) or []
        actual = payload.get("actual", []) or []
        expected_ok = all(col.lower() in text for col in expected)
        actual_ok = all(col.lower() in text for col in actual)
        return expected_ok and actual_ok and "header" in text

    if kind == "migration":
        return _contains_all(text, ["dual-write", "backfill", "schema"])

    if kind == "probability_draws":
        simplified = str(payload.get("simplified", "")).lower()
        return simplified in text or "5/22" in text

    if kind == "chem_balance":
        equation = str(payload.get("equation", "")).lower().replace(" ", "")
        return equation in completion.replace(" ", "").lower()

    if kind == "soc_response":
        return _contains_all(text, ["containment", "eradication", "recovery"])

    if kind == "network_diag":
        return _contains_all(text, ["traceroute", "mtu", "df", "hop"])

    if kind == "therapy_plan":
        return _contains_all(text, ["cbt", "exposure", "journaling"])

    if kind == "supply_chain":
        buffer_weeks = str(payload.get("buffer_weeks", ""))
        return _contains_all(text, ["safety stock", "lead", "buffer"]) and buffer_weeks in text

    if kind == "ml_training":
        lr = str(payload.get("lr", "")).lower()
        batch = str(payload.get("batch", ""))
        return lr in text and batch in text and "holdout" in text

    if kind == "ethics_ai":
        return _contains_all(text, ["consent", "opt-out", "transparency"])

    if kind == "momentum_dump":
        return _contains_all(text, ["magnetorquer", "momentum", "quaternion"])

    if kind == "gc_bias":
        return _contains_all(text, ["gc", "normalization", "spike"])

    if kind == "climate_model":
        return _contains_all(text, ["boundary", "assimilation", "bias"])

    if kind == "negotiation":
        return _contains_all(text, ["batna", "zopa", "concession"])

    if kind == "ed_coaching":
        return _contains_all(text, ["algebra", "spaced", "journal"])

    if kind == "triage_start":
        return _contains_all(text, ["start", "surge", "redeploy"])

    if kind == "wm_experiment":
        return _contains_all(text, ["randomized", "dual", "pre/post"])

    if kind == "glider_replan":
        return _contains_all(text, ["adcp", "kalman", "gps"])

    if kind == "qaoa_plateau":
        return _contains_all(text, ["gradient", "depth", "layer"])

    if kind == "esg_board":
        return _contains_all(text, ["materiality", "stakeholder", "report"])

    if kind == "coordination_game":
        payoff = str(payload.get("payoff", "")).lower()
        return payoff in text and "best response" in text

    if kind == "microgrid_soc":
        return _contains_all(text, ["bms", "dispatch", "peak"])

    if kind == "crisis_comms":
        return _contains_all(text, ["apology", "remediation", "weekly"])

    if kind == "nudge_401k":
        return _contains_all(text, ["auto-enroll", "reminder", "feedback"])

    if kind == "astro_reduction":
        return _contains_all(text, ["median", "sigma", "calibration"])

    if kind == "coalition_analysis":
        return _contains_all(text, ["coalition", "policy", "confidence"])

    if kind == "quadruped_gait":
        return _contains_all(text, ["gait", "force", "imu"])

    if kind == "climate_mitigation":
        return _contains_all(text, ["inventory", "renewable", "incentive"])

    if kind == "supply_compliance":
        return _contains_all(text, ["cap", "audit", "diversify"])

    if kind == "actuarial_review":
        return _contains_all(text, ["loss", "underwriting", "reinsurance"])

    if kind == "5g_backhaul":
        return _contains_all(text, ["fiber", "microwave", "traffic"])

    return None
