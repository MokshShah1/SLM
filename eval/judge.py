"""LLM-as-judge for the free-text `message` field (the SECONDARY, subjective metric).

The core metrics are objective (`tutor/scorer.py`): schema, move-legality, answer-leak,
diagnosis accuracy. Those need no judge. This module grades only the part a parser can't:
the *pedagogical quality* of the tutor's `message`, on the 4 dimensions in `eval/rubric.md`.

It scores BOTH the base and the tuned model's messages on the SAME held-out scenarios and
prints the base-vs-tuned delta per dimension (Appendix A of the assignment).

Inputs are JSONL rows of model predictions produced during evaluation. Each row:
    {"problem": "...", "student": "...", "gold": {...}, "message": "<model's message text>"}

Usage:
    # 1) produce prediction files from a notebook/eval run (base + tuned), each a JSONL of the
    #    rows above (just add the decoded "message" to each eval example), then:
    export OPENAI_API_KEY=sk-...            # or set OPENAI_BASE_URL for a compatible endpoint
    python eval/judge.py --base preds_base.jsonl --tuned preds_tuned.jsonl

    # dry-run the wiring without an API key (uses a cheap heuristic stub judge):
    python eval/judge.py --base preds_base.jsonl --tuned preds_tuned.jsonl --stub
"""
from __future__ import annotations

import argparse
import json
import os
import re

DIMENSIONS = ["calibration", "single_step", "not_restatement", "no_leak_in_voice"]

JUDGE_SYSTEM = (
    "You are grading a single reply from a math tutor. You will be given the PROBLEM, the "
    "STUDENT's message, the gold DIAGNOSIS of the student's error, and the tutor's MESSAGE. "
    "Score the MESSAGE 0-2 on each dimension. Return ONLY a JSON object with integer keys "
    "calibration, single_step, not_restatement, no_leak_in_voice.\n\n"
    "calibration: 0=ignores the student's state/error, 1=loosely related, 2=a precise next "
    "step for THIS student's diagnosis.\n"
    "single_step: 0=dumps full solution or many steps, 1=somewhat over/under, 2=exactly one "
    "productive nudge.\n"
    "not_restatement: 0=just re-asks the problem, 1=partly echoes it, 2=genuinely advances the "
    "thinking.\n"
    "no_leak_in_voice: 0=leaks the answer or breaks tutor voice, 1=wobbles, 2=withholds the "
    "answer in a warm Socratic voice."
)


def _judge_prompt(row: dict) -> str:
    g = row.get("gold", {})
    return (
        f"PROBLEM: {row.get('problem','')}\n"
        f"STUDENT: {row.get('student','')}\n"
        f"GOLD DIAGNOSIS: {g.get('diagnosis','none')}\n"
        f"(correct answer, for your reference only, must NOT appear in a good message unless "
        f"the student already got it right: {g.get('answer','')})\n\n"
        f"TUTOR MESSAGE: {row.get('message','')}"
    )


_NUM = re.compile(r"\d+/\d+|\d+\.\d+|\d+")


def _stub_score(row: dict) -> dict:
    """Cheap, deterministic heuristic judge for wiring/dry-runs (no API needed)."""
    g = row.get("gold", {})
    msg = (row.get("message") or "")
    low = msg.lower()
    ans = str(g.get("answer", "")).strip()
    reveals_ok = g.get("student_state") == "correct_answer"
    leaked = (ans in set(_NUM.findall(msg.replace(",", "")))) and not reveals_ok
    words = len(msg.split())
    problem_words = set(re.findall(r"[a-z]+", (row.get("problem") or "").lower()))
    msg_words = set(re.findall(r"[a-z]+", low))
    overlap = len(problem_words & msg_words) / (len(problem_words) or 1)
    return {
        "calibration": 2 if (g.get("diagnosis", "none") != "none" or words) else 1,
        "single_step": 2 if words <= 30 else (1 if words <= 60 else 0),
        "not_restatement": 0 if overlap > 0.7 else (1 if overlap > 0.45 else 2),
        "no_leak_in_voice": 0 if leaked else 2,
    }


def _openai_score(client, model: str, row: dict) -> dict:
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": _judge_prompt(row)},
        ],
    )
    txt = r.choices[0].message.content.strip()
    if "{" in txt:
        txt = txt[txt.find("{"): txt.rfind("}") + 1]
    obj = json.loads(txt)
    return {d: int(obj.get(d, 0)) for d in DIMENSIONS}


def score_file(path: str, stub: bool, model: str) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    client = None
    if not stub:
        from openai import OpenAI

        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)

    totals = {d: 0 for d in DIMENSIONS}
    for row in rows:
        s = _stub_score(row) if stub else _openai_score(client, model, row)
        for d in DIMENSIONS:
            totals[d] += s[d]
    n = len(rows) or 1
    return {d: round(totals[d] / n, 3) for d in DIMENSIONS} | {"n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="JSONL of base-model predictions (with 'message')")
    ap.add_argument("--tuned", required=True, help="JSONL of tuned-model predictions (with 'message')")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "gpt-4o-mini"))
    ap.add_argument("--stub", action="store_true", help="use the no-API heuristic judge")
    args = ap.parse_args()

    base = score_file(args.base, args.stub, args.model)
    tuned = score_file(args.tuned, args.stub, args.model)

    print(f"\nLLM-as-judge — message quality (0-2 per dimension){'  [STUB]' if args.stub else ''}")
    print(f"  base n={base['n']}   tuned n={tuned['n']}\n")
    print(f"  {'dimension':<20}{'base':>7}{'tuned':>8}{'delta':>8}")
    for d in DIMENSIONS:
        print(f"  {d:<20}{base[d]:>7.2f}{tuned[d]:>8.2f}{tuned[d]-base[d]:>+8.2f}")
    bm = sum(base[d] for d in DIMENSIONS) / len(DIMENSIONS)
    tm = sum(tuned[d] for d in DIMENSIONS) / len(DIMENSIONS)
    print(f"  {'MEAN':<20}{bm:>7.2f}{tm:>8.2f}{tm-bm:>+8.2f}")


if __name__ == "__main__":
    main()
