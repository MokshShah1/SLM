# SLM — Structured Socratic Math Tutor (Small Learning Model)

Fine-tune a small open model (Qwen3-1.7B) into a reliable **Socratic math tutor** that emits a
**structured coaching move** — diagnose the student's misconception, pick the pedagogically-correct
next move, write one calibrated hint, and **never reveal the answer** — following a fixed pedagogy
policy a prompted base model can't apply reliably.

> **The dataset is the deliverable.** Gold labels are generated *from the policy* (`tutor/policy.py`),
> and every example is re-validated by the scorer (`tutor/scorer.py`). Training is a downstream button-press.

- **Behavior spec (the gate):** [`behavior_spec.md`](behavior_spec.md)
- **Policy (human-readable):** [`policy.md`](policy.md)
- **Base model:** `Qwen/Qwen3-1.7B` (Instruct) · **Framework:** Unsloth QLoRA · **Compute:** Colab T4

## Why this target (the litmus-test journey)
We empirically killed easier targets because a *well-prompted* Qwen3-1.7B already nailed them:
| Candidate | Base result | Verdict |
|-----------|-------------|---------|
| Tutor "never reveal answer" | 0/22 leaks (even under jailbreaks) | prompt-solvable |
| Strict JSON on messy input | 11/12 | prompt-solvable |
| NL → custom DSL (easy cases) | 10/10 | prompt-solvable |
| NL → DSL **policy** cases | 1/8 | hard — but not education |
| **Structured tutor move (policy/judgment)** | base misdiagnoses & picks illegal moves | **locked** ✅ |

The pattern: this base is a strong instruction-follower, so **format is easy; policy/judgment is hard.**
The tutor-move target is education-related, objectively gradeable, and fails the prompt test.

## Repo layout
```
behavior_spec.md            # the gate: spec + policy table + grading
policy.md                   # human-readable pedagogy policy
tutor/
  policy.py                 # taxonomy, policy, problem generators, gold-scenario builder (source of truth)
  scorer.py                 # schema validation + OBJECTIVE scorer (labels, no-leak, move-legality)
src/
  generate_data.py          # policy-driven train/val JSONL (chat format) + hard quality gate
  evaluate.py               # run base/tuned model -> objective metrics + base-vs-tuned table
eval/
  tutor_eval.jsonl          # HELD-OUT, hand-written, novel-phrasing benchmark (24 cases)
  rubric.md                 # optional LLM-judge rubric for message-text quality
notebooks/
  01_train_qlora.ipynb      # Unsloth QLoRA SFT on Qwen3-1.7B + base-vs-tuned eval + demo
data/                       # generated train.jsonl / val.jsonl (the artifact)
outputs/                    # LoRA adapter / checkpoints (gitignored)
```

## Run it

### 1. Locally (no GPU needed) — generate data + validate the pipeline
```bash
python -m tutor.policy            # peek at generated scenarios
python -m tutor.scorer            # scorer self-test (gold=1.0, bad model tanks)
python src/generate_data.py --train 800 --val 160 --seed 7
python src/evaluate.py --check-gold --data eval/tutor_eval.jsonl   # validate the benchmark
```

### 2. In Colab (T4 GPU) — train + measure
Open [`notebooks/01_train_qlora.ipynb`](notebooks/01_train_qlora.ipynb): it installs Unsloth,
gets this repo, generates data, evaluates the **base** model, QLoRA-fine-tunes, then evaluates the
**tuned** model on the same held-out set — printing the base-vs-tuned table.

## Results (held-out eval, base vs tuned)

| metric | base | tuned | delta |
|---|---|---|---|
| structured_exact | 0.167 | **0.833** | +0.666 |
| diagnosis_exact | 0.458 | **0.917** | +0.459 |
| policy_ok | 0.542 | **1.000** | +0.458 |
| leak_ok (never reveals answer) | 0.750 | **1.000** | +0.250 |

**Message-text quality** (secondary, LLM-as-judge on the same held-out set — 0–2 per dimension, `eval/judge.py`):

| dimension | base | tuned | delta |
|---|---|---|---|
| calibration | 2.00 | 2.00 | +0.00 |
| single_step | 1.92 | 2.00 | +0.08 |
| not_restatement | 1.12 | **1.62** | +0.50 |
| no_leak_in_voice | 1.50 | **2.00** | +0.50 |
| **mean** | 1.64 | **1.91** | +0.27 |

Full write-up + litmus journey + the v1→v2 data iteration: [`BRAINLIFT.md`](BRAINLIFT.md).

## Deliverables (map to the brief)
1. **Dataset** — `data/train.jsonl` (published) ← the real artifact
2. **Model** — LoRA adapter on Qwen3-1.7B (push to HF Hub) + demo in the notebook
3. **Eval harness + results** — `src/evaluate.py` + base-vs-tuned table on `eval/tutor_eval.jsonl`
4. **Brainlift** — the litmus journey above: data → behavior, with numbers
5. **Demo video** — the notebook's interactive cell doing what the base can't

## Rules we hold to
No training before the eval exists ✅ · one target/one context ✅ · fix data not hyperparameters ·
measure the target behavior (policy adherence), not math-trivia accuracy.
