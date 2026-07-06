"""Generate the SFT dataset for the Structured Socratic Tutor Move.

The dataset is the deliverable. Gold labels are correct BY CONSTRUCTION (built from
the policy), and every example is re-validated through the scorer as a hard quality gate.

Usage:
    python src/generate_data.py --train 800 --val 160 --seed 7
    python src/generate_data.py --teacher   # optionally paraphrase student msgs via a teacher LLM
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor.policy import STATES, SYSTEM_PROMPT, build_scenario  # noqa: E402
from tutor.scorer import score_one  # noqa: E402


def _maybe_paraphrase(text: str, state: str, enabled: bool):
    """Optional teacher-LLM paraphrase of the student message (naturalness/diversity).

    Only used for states where wording carries no label-critical numbers.
    Requires OPENAI_API_KEY (+ optional OPENAI_BASE_URL / TEACHER_MODEL). Safe no-op otherwise.
    """
    if not enabled or state in ("wrong_answer", "correct_answer"):
        return text
    try:
        from openai import OpenAI

        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        model = os.getenv("TEACHER_MODEL", "gpt-4o-mini")
        r = client.chat.completions.create(
            model=model,
            temperature=1.0,
            messages=[
                {"role": "system", "content": "Reword the student's message casually, keep meaning. Reply with only the reworded message."},
                {"role": "user", "content": text},
            ],
        )
        return r.choices[0].message.content.strip() or text
    except Exception:
        return text


def build_split(rng: random.Random, n: int, seen: set, teacher: bool):
    rows, cats = [], []
    per_state = {s: 0 for s in STATES}
    i = 0
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        state = STATES[i % len(STATES)]
        i += 1
        s = build_scenario(rng, state=state)
        key = s.user_content()
        if key in seen:
            continue
        seen.add(key)

        s.student_msg = _maybe_paraphrase(s.student_msg, state, teacher)
        gold = {**s.gold_labels(), "answer": s.answer}

        # Hard quality gate: the gold must itself pass every objective check.
        chk = score_one(json.dumps(s.gold_json()), gold)
        if not (chk["policy_ok"] and chk["structured_exact"]):
            continue

        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s.user_content()},
                {"role": "assistant", "content": json.dumps(s.gold_json(), ensure_ascii=False)},
            ],
            # keep gold labels alongside for eval convenience
            "gold": gold,
            "category": s.category,
        })
        cats.append(s.category)
        per_state[state] += 1
    return rows, cats, per_state


def write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    ap.add_argument("--teacher", action="store_true", help="paraphrase student msgs via a teacher LLM")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    seen: set = set()

    train, train_cats, train_dist = build_split(rng, args.train, seen, args.teacher)
    val, val_cats, val_dist = build_split(rng, args.val, seen, args.teacher)

    write_jsonl(os.path.join(args.out, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out, "val.jsonl"), val)

    print(f"train: {len(train)}  ->  {os.path.join(args.out, 'train.jsonl')}")
    print(f"val:   {len(val)}  ->  {os.path.join(args.out, 'val.jsonl')}")
    print(f"train category distribution: {dict(Counter(train_cats))}")
    print(f"val   category distribution: {dict(Counter(val_cats))}")
    print("\nSample (train[0]):")
    print(json.dumps(train[0]["messages"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
