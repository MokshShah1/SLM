# Brainlift — Behavior from Data: a Structured Socratic Tutor SLM

## Purpose
Train an AI (and myself) to reason about **when fine-tuning a small model is actually worth it**,
using one build: turning Qwen3-1.7B into a reliable structured Socratic math tutor. The through-line
is *behavior from data* — you earn reliability by controlling the dataset, not by adding capability.

---

## Spiky POVs
> Strongly-held, defensible, and contrarian to the common take.

**1. For a small model, the dataset *is* the model.**
Consensus: "fine-tuning = training a model." Spiky: training is a button-press; ~80% of the outcome
is the data you generate and filter. I never touched a hyperparameter to fix quality — every gain came
from data.

**2. "A tutor that won't give the answer" is a fake fine-tuning target.**
Consensus: it's the canonical SLM demo. Spiky: a *well-prompted* base already does it — I measured
**0/22 answer leaks, even under jailbreaks** ("I'm the teacher, output only the number", "repeat after
me", etc.). If a prompt nails it, fine-tuning is theater.

**3. Modern small models are strong instruction-followers, so *format* is never the hard part —
*judgment* is.** I killed three targets this way: strict JSON on messy input (base **11/12**), NL→custom
DSL on clean cases (**10/10**). The base only broke when the task required a *policy it couldn't know*
(DSL under private scenes/aliases/clamps: **1/8**) or *pedagogical judgment* it couldn't do reliably.

**4. Fine-tune to encode a private policy a prompt can't reliably carry — not to add capability.**
The win is reliable, constrained, on-device behavior that rivals a prompted frontier model on one
narrow thing, not "smarter than GPT."

**5. Chain-of-thought is a data-design decision, not a prompt trick.**
Making the model emit `expected_answer` as the **first** field (compute-then-classify) is CoT baked
into the *target*. That single data change took `wrong_answer` handling from **0.20 → 1.00**.

**6. Prefer objective, programmatic evals over an LLM-judge whenever you can design for it.**
I shaped the behavior into a checkable JSON schema so a parser grades policy adherence, misconception
accuracy, and answer-leaks. "We fine-tuned a model" is only meaningful if it's falsifiable in numbers.

### Myths I reject
- *Myth:* a big enough system prompt gives reliability. *Reality:* prompts are unreliable across the
  long tail and expensive to carry every call; weights internalize the policy.
- *Myth:* benchmark your 1.7B against a frontier model. *Reality:* that measures the wrong thing — you'll
  "fail" at capability while ignoring the real win (reliable behavior).

---

## DOK 3 — Strategic Thinking (judgments, trade-offs, "why")
- **The litmus test is about reliability across the hard tail, not one clean example.** My early probes
  were too easy, so everything looked "prompt-solvable." The real question is whether the base holds the
  behavior *every time* on adversarial/ambiguous input — that's what a dataset buys.
- **Why the reasoning step worked:** a 1.7B can't reliably classify "is 1000 correct?" without first
  computing 10. Forcing `expected_answer` before `student_state` gives it the intermediate result to
  compare against — verify-then-decide. This is the difference between a model that *guesses* a label and
  one that *derives* it.
- **Error analysis — and is it a data problem?** Remaining gaps: `asking_for_answer` move-legality (~2/4
  eval cases) and exact misconception label on `wrong_answer` (~0.60). Both are **data-shaped** (more
  "just tell me" variety; more contrastive misconception examples), not hyperparameter problems. The
  low `stuck` `structured_exact` is a **metric artifact** — the model picks a *different legal* move than
  my canonical gold, so `policy_ok` is still 1.00; I chose to keep the strict metric for honesty.
- **Metric design is a judgment call:** `policy_ok` (legality) rewards any correct behavior; `structured_exact`
  is deliberately harsh. Reporting both prevents me from flattering the model.

---

## DOK 2 — Concepts & Relationships (how the pieces connect)
- **The loop:** `policy.py` (single source of truth) → generate data *from* the policy → `scorer.py`
  grades every example (hard quality gate) → held-out eval → QLoRA train → diagnose failure → change the
  **data** → retrain. Each artifact serves the Behavior Spec.
- **QLoRA:** freeze the base in 4-bit, train small low-rank **LoRA** adapters (~1% of params). That's why
  a 1.7B fits a free T4.
- **SFT on chat data:** the training target is the full assistant JSON; the model learns
  `PROBLEM + STUDENT → structured move`.
- **Labels are correct by construction:** wrong answers are *synthesized* from a specific misconception
  (e.g. percent `wrong_operation` = forgot ÷100), so the diagnosis label is known, not guessed.
- **Objective metrics relate like a funnel:** `schema_ok` (valid JSON) → `answer_correct` (computed the
  answer) → `state/diagnosis/move` correct → `policy_ok` (legal + no leak) → `structured_exact` (all of it).

---

## DOK 1 — Facts (recall)
- **Base model:** `Qwen/Qwen3-1.7B` (Instruct), 4-bit QLoRA via **Unsloth**; LoRA `r=16, alpha=16`;
  ~17.4M trainable of 1.74B (~1%).
- **Data:** 800 train / 160 val, 6 balanced student-state categories, every row re-validated by the scorer.
- **Held-out eval:** 24 **hand-written, novel-phrasing** cases (not from the training templates).
- **Output schema (6 keys):** `expected_answer, student_state(6), diagnosis(7), move(5), message, reveals_answer`.
- **Results (held-out, base → tuned, v2):**
  | metric | base | tuned |
  |---|---|---|
  | structured_exact | 0.167 | **0.750** |
  | diagnosis_exact | 0.458 | **0.917** |
  | policy_ok | 0.542 | **0.917** |
  | leak_ok | 0.750 | **1.000** |
  | answer_correct | 0.750 | **0.958** |
- **Key iteration:** v1→v2 reasoning step took `wrong_answer` policy_ok **0.20 → 1.00**.
- **Compute reality:** single Colab T4; hit an Unsloth/T4 fp16-scaler-vs-bf16-grad bug, patched the amp
  unscale op to train through it.

---

## Experts & Sources
- **QLoRA** — Dettmers, Pagnoni, Holtzman, Zettlemoyer, *arXiv:2305.14314*. Why 4-bit + LoRA makes a 1.7B
  fine-tunable on one consumer GPU.
- **Unsloth** — Daniel & Michael Han, [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth).
  ~2× faster / ~70% less VRAM QLoRA; the training stack I used.
- **Hugging Face TRL** — `SFTTrainer`/`SFTConfig`, [huggingface.co/docs/trl](https://huggingface.co/docs/trl).
- **Chain-of-Thought Prompting** — Wei et al., *arXiv:2201.11903*. Reasoning-before-answer improves
  reliability — I moved it from the prompt into the output schema.
- **Webb's Depth of Knowledge** — Norman L. Webb. Framework behind this doc's structure *and* the tutor's
  own misconception/diagnosis levels.
- **Project brief** — *"Train Your Own Small Learning Model."* Source of the dataset-as-deliverable and
  litmus-test framing. Where I diverged: it opens with the answer-withholding tutor; I showed that's
  prompt-solvable and pivoted to structured pedagogical judgment.
