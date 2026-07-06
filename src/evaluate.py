"""Evaluate a model on the Structured Socratic Tutor Move task.

Objective metrics only (the scorer). Import `run_eval` from a notebook to build the
base-vs-tuned table, or run this as a CLI on one model.

Examples:
    python src/evaluate.py --check-gold --data eval/tutor_eval.jsonl   # validate the eval file (no model)
    python src/evaluate.py --model Qwen/Qwen3-1.7B --data eval/tutor_eval.jsonl
    python src/evaluate.py --model Qwen/Qwen3-1.7B --adapter outputs/tutor-lora --data eval/tutor_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor.policy import SYSTEM_PROMPT  # noqa: E402
from tutor.scorer import score_one, aggregate, format_table, METRIC_KEYS  # noqa: E402


def load_rows(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "problem" in r:
                user = f"PROBLEM: {r['problem']}\nSTUDENT: {r['student']}"
            else:
                user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            gold = r["gold"]
            rows.append({"user": user, "gold": gold,
                         "category": r.get("category", gold["student_state"])})
    return rows


def build_prompt(tok, user: str, thinking: bool = False) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def run_eval(model, tok, rows, max_new_tokens=200, thinking=False, verbose=False):
    import torch

    metrics, cats = [], []
    for r in rows:
        text = build_prompt(tok, r["user"], thinking)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        m = score_one(reply, r["gold"])
        metrics.append(m)
        cats.append(r["category"])
        if verbose:
            print("\n" + "-" * 72)
            print(f"[{r['category']}] {r['user'].splitlines()[-1]}")
            print(f"PRED: {reply[:200]}")
            print(f"policy_ok={m['policy_ok']} structured_exact={m['structured_exact']} "
                  f"leak_ok={m['leak_ok']} diagnosis_exact={m['diagnosis_exact']}")
    return aggregate(metrics, cats)


def check_gold(rows):
    """Validate the eval file's internal consistency (no model)."""
    metrics, cats = [], []
    for r in rows:
        g = r["gold"]
        pred = json.dumps({
            "student_state": g["student_state"], "diagnosis": g["diagnosis"],
            "move": g["move"], "message": "placeholder next step, no number here",
            "reveals_answer": g["reveals_answer"],
        })
        metrics.append(score_one(pred, g))
        cats.append(r["category"])
    return aggregate(metrics, cats)


def print_report(agg: dict, title: str):
    print("\n" + "=" * 72)
    print(format_table(agg, title))
    if "by_category" in agg:
        print("\n  by category (policy_ok / structured_exact / diagnosis_exact):")
        for c, mm in agg["by_category"].items():
            print(f"    {c:<18} {mm['policy_ok']:.2f} / {mm['structured_exact']:.2f} / {mm['diagnosis_exact']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter (the tuned model)")
    ap.add_argument("--data", default="data/val.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--check-gold", action="store_true", help="validate eval file, no model")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[:args.limit]

    if args.check_gold:
        print_report(check_gold(rows), f"GOLD self-check: {args.data}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    title = f"{args.model}{' + ' + args.adapter if args.adapter else ' (base)'}  on  {args.data}"
    print_report(run_eval(model, tok, rows, args.max_new_tokens, args.thinking, args.verbose), title)


if __name__ == "__main__":
    main()
