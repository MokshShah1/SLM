# Brainlift — Behavior from Data

**Thesis:** a tiny open model (Qwen3-1.7B) can be made a *reliable* structured Socratic math
tutor — one that diagnoses a student's misconception, picks the pedagogically-correct move, and
**never reveals the answer** — purely by controlling the training data. A well-prompted base model
can't do this reliably; a fine-tune on policy-generated data can.

## The spiky POV
Everyone reaches for "a tutor that won't give the answer." We *measured* that and it's a trap:
a well-prompted Qwen3-1.7B already refuses to leak answers **0/22 times, even under jailbreaks**.
Format and refusal are prompt-solvable on modern small models. **What isn't prompt-solvable is
reliable pedagogical judgment** — diagnosing the error and choosing the right move, every time,
in a machine-checkable structure. That's the behavior worth training.

## The litmus journey (why this target)
| Candidate behavior | Well-prompted base result | Verdict |
|---|---|---|
| Tutor "never reveal the answer" | 0/22 leaks (incl. jailbreaks) | prompt-solvable |
| Strict JSON on messy input | 11/12 | prompt-solvable |
| NL → custom DSL (clean cases) | 10/10 | prompt-solvable |
| NL → DSL under *policy* (scenes/aliases/clamps) | 1/8 | hard, but not education |
| **Structured Socratic tutor move (policy + judgment)** | base misdiagnoses, picks illegal moves | **chosen** |

## Behavior spec (the gate)
One structured "coaching move" per turn: `{expected_answer, student_state, diagnosis, move,
message, reveals_answer}`, following a fixed pedagogy policy (see `behavior_spec.md`). Graded
**objectively** by `tutor/scorer.py` — no fuzzy judging needed for the core metrics.

## Method — the dataset is the deliverable
- **Policy as source of truth** (`tutor/policy.py`): taxonomy + legal-move table + math problem
  generators. Gold labels are correct *by construction*.
- **Generate → hard-filter**: every example is re-scored by the grader before it's kept.
- **Eval before training**: a hand-written, novel-phrasing held-out set (`eval/tutor_eval.jsonl`)
  so we measure generalization, not template memorization.
- **Train**: QLoRA SFT (Unsloth) on Qwen3-1.7B.

## Results (held-out, base vs tuned)

**v1 (structured move, no reasoning step)**
| metric | base | tuned | delta |
|---|---|---|---|
| structured_exact | 0.167 | 0.542 | +0.375 |
| diagnosis_exact | 0.417 | 0.708 | +0.291 |
| leak_ok | 0.750 | 1.000 | +0.250 |

**Diagnosed failure:** `wrong_answer` stuck at policy_ok 0.20 — the model never *computed* the
answer, so it rubber-stamped wrong numbers as "correct" (e.g. "1000" for 20% of 50 → "correct!").

**v2 fix (in the DATA, not hyperparameters):** add an `expected_answer` field the model computes
**first** (private working) — compute-then-classify.

| metric | base | tuned | delta |
|---|---|---|---|
| **structured_exact** | 0.167 | **0.750** | **+0.583** |
| **diagnosis_exact** | 0.458 | **0.917** | **+0.459** |
| **policy_ok** | 0.542 | **0.917** | **+0.375** |
| **leak_ok** | 0.750 | **1.000** | **+0.250** |
| answer_correct | 0.750 | 0.958 | +0.208 |

`wrong_answer` policy_ok: **0.20 → 1.00**. The demo case now returns `wrong_answer` + a calibrated
hint instead of a false "correct!".

## Error analysis (where the tuned model still fails)
- `asking_for_answer` policy_ok 0.50 — occasionally picks a non-`redirect` move (data-balance fix).
- `wrong_answer` exact misconception label ~0.60 — always a *legal* move + no leak, but the precise
  error type is ambiguous for some wrong numbers.
- `stuck` low `structured_exact` is a **metric artifact** (a different *legal* move than the
  canonical gold; policy adherence is still 1.00).
All are data-shaped, not hyperparameter problems.

## Did data → behavior hold? 
Yes. On a held-out set with phrasings never seen in training, controlling only the data moved a
1.7B model from unreliable (17% fully-correct moves, leaks 25% of the time) to reliable (75%
fully-correct, 0% leaks, 92% correct misconception diagnosis) — and a single targeted data change
(the reasoning step) fixed the one diagnosed failure mode. The win is **reliable, constrained,
on-device pedagogical behavior**, not raw capability.
