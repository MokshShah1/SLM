"""Objective scorer: grade a model's structured tutor move against the policy.

Everything here is programmatic (no LLM judge). The message *text* quality is the
only thing that needs a judge; that is optional and lives in evaluate.py.
"""
from __future__ import annotations

import json
import re

from tutor.policy import (
    STATES, DIAGNOSES, MOVES, LEGAL_MOVES, ALLOWED_DIAGNOSES, REVEALS_ANSWER,
)

REQUIRED_KEYS = {"expected_answer", "student_state", "diagnosis", "move", "message", "reveals_answer"}

METRIC_KEYS = [
    "parse_ok", "schema_ok", "answer_correct", "state_correct", "diagnosis_exact", "diagnosis_legal",
    "move_exact", "move_legal", "flag_correct", "leak_ok", "policy_ok", "structured_exact",
]


def parse_json(text: str):
    """Best-effort extract a single JSON object from model output. Returns dict or None."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    if "{" in s and "}" in s:
        s = s[s.find("{"): s.rfind("}") + 1]
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


_NUM_TOKEN = re.compile(r"\d+/\d+|\d+\.\d+|\d+")


def answer_in_text(answer: str, text: str) -> bool:
    """True if the final answer appears as a standalone number/fraction in the text.

    Extracts number tokens so "8" matches "The answer is 8." but not "18" or "8.5".
    """
    a = str(answer).strip().lower().lstrip("$")
    t = (text or "").lower().replace(",", "").replace("$", "")
    return a in set(_NUM_TOKEN.findall(t))


def schema_validate(obj: dict):
    reasons = []
    if set(obj.keys()) != REQUIRED_KEYS:
        missing = REQUIRED_KEYS - set(obj)
        extra = set(obj) - REQUIRED_KEYS
        if missing:
            reasons.append("missing:" + ",".join(sorted(missing)))
        if extra:
            reasons.append("extra:" + ",".join(sorted(extra)))
    if obj.get("student_state") not in STATES:
        reasons.append("bad-state")
    if obj.get("diagnosis") not in DIAGNOSES:
        reasons.append("bad-diagnosis")
    if obj.get("move") not in MOVES:
        reasons.append("bad-move")
    if not isinstance(obj.get("expected_answer"), (str, int, float)):
        reasons.append("expected_answer-bad-type")
    if not isinstance(obj.get("message"), str):
        reasons.append("message-not-string")
    if not isinstance(obj.get("reveals_answer"), bool):
        reasons.append("reveals-not-bool")
    return (len(reasons) == 0), reasons


def score_one(pred_text: str, gold: dict) -> dict:
    """gold requires keys: student_state, diagnosis, move, reveals_answer, answer."""
    m = {k: False for k in METRIC_KEYS}
    obj = parse_json(pred_text)
    if obj is None:
        return m
    m["parse_ok"] = True
    ok, _ = schema_validate(obj)
    m["schema_ok"] = ok
    m["answer_correct"] = str(gold["answer"]).strip() in _NUM_TOKEN.findall(str(obj.get("expected_answer", "")))

    gstate = gold["student_state"]
    # Label correctness (only meaningful once we at least have valid enum values).
    m["state_correct"] = obj.get("student_state") == gstate
    m["diagnosis_exact"] = obj.get("diagnosis") == gold["diagnosis"]
    m["diagnosis_legal"] = obj.get("diagnosis") in ALLOWED_DIAGNOSES.get(gstate, set())
    m["move_exact"] = obj.get("move") == gold["move"]
    m["move_legal"] = obj.get("move") in LEGAL_MOVES.get(gstate, set())
    m["flag_correct"] = obj.get("reveals_answer") == REVEALS_ANSWER[gstate]

    leaks = answer_in_text(gold["answer"], obj.get("message", ""))
    illegal_leak = leaks and not REVEALS_ANSWER[gstate]
    m["leak_ok"] = not illegal_leak

    m["policy_ok"] = m["schema_ok"] and m["move_legal"] and m["flag_correct"] and m["leak_ok"]
    m["structured_exact"] = (
        m["schema_ok"] and m["state_correct"] and m["diagnosis_exact"]
        and m["move_exact"] and m["flag_correct"]
    )
    return m


def aggregate(rows: list[dict], cats: list[str] | None = None) -> dict:
    n = len(rows) or 1
    overall = {k: round(sum(r[k] for r in rows) / n, 3) for k in METRIC_KEYS}
    out = {"n": len(rows), "overall": overall}
    if cats:
        by = {}
        for c in sorted(set(cats)):
            idx = [i for i, cc in enumerate(cats) if cc == c]
            by[c] = {k: round(sum(rows[i][k] for i in idx) / len(idx), 3) for k in METRIC_KEYS}
        out["by_category"] = by
    return out


def format_table(agg: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    o = agg["overall"]
    lines.append(f"  n={agg['n']}")
    for k in METRIC_KEYS:
        lines.append(f"  {k:<18} {o[k]:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import random
    from tutor.policy import build_scenario

    rng = random.Random(1)
    rows, cats = [], []
    for _ in range(60):
        s = build_scenario(rng)
        gold = {**s.gold_labels(), "answer": s.answer}
        rows.append(score_one(json.dumps(s.gold_json()), gold))
        cats.append(s.category)
    print("GOLD scored against itself (should be ~1.0 everywhere):")
    print(format_table(aggregate(rows, cats)))

    # A deliberately bad model: always the same wrong move + leaks the answer.
    bad_rows, bad_cats = [], []
    for _ in range(60):
        s = build_scenario(rng)
        gold = {**s.gold_labels(), "answer": s.answer}
        bad = json.dumps({
            "student_state": "stuck", "diagnosis": "none", "move": "affirm_and_confirm",
            "message": f"The answer is {s.answer}.", "reveals_answer": True,
        })
        bad_rows.append(score_one(bad, gold))
        bad_cats.append(s.category)
    print("\nBAD model (should be low on policy_ok / leak_ok / structured_exact):")
    print(format_table(aggregate(bad_rows, bad_cats)))
