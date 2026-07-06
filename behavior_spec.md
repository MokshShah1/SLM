# Behavior Spec — Structured Socratic Math Tutor Move

> The gate. This is simultaneously the **data-generation rubric**, the **eval criterion**, and
> the **thesis**. The dataset is generated FROM this spec (`tutor/policy.py`) and the scorer
> (`tutor/scorer.py`) grades against it. If you change the behavior, change it here first.

## Scope (one target, one context)
- **Domain:** elementary / early-algebra math — arithmetic, percentages, ratios, single-variable
  linear equations, and simple word problems, each with a definite numeric answer.
- **Context:** 1:1 tutoring. Input = a `PROBLEM` + the `STUDENT`'s latest message. Output = **one
  structured coaching move**.

## Output (strict JSON, exactly these keys)
```json
{
  "expected_answer": "the correct answer, computed privately by the tutor (never shown to the student)",
  "student_state": "no_attempt | asking_for_answer | stuck | partial | wrong_answer | correct_answer",
  "diagnosis":     "none | arithmetic_slip | wrong_operation | order_of_operations | misread_problem | concept_gap | incomplete_steps",
  "move":          "ask_probing_question | give_hint | affirm_and_confirm | redirect_no_answer | encourage_retry",
  "message":       "tutor's reply to the student, <= 2 sentences",
  "reveals_answer": false
}
```

## The Spec (falsifiable — a grader can mark any output pass/fail)
Given the problem and the student's latest message, the model must:
1. Correctly identify the **student_state**.
2. **Diagnose** the misconception — a *specific* error type when the student is wrong; `none` otherwise.
3. Select a **move that is legal** for that state (policy table below).
4. Set **reveals_answer** correctly: it may be `true` **only** when the student has already produced
   the correct answer; otherwise `false`.
5. Write a **message** that is one calibrated next step tuned to the diagnosis, is **not** a verbatim
   restatement of the problem, and does **not** contain the final answer (except when confirming a
   correct answer).

## Policy table (state → legal moves; reveals_answer)
| student_state       | legal moves                                          | reveals_answer | diagnosis |
|---------------------|------------------------------------------------------|:--------------:|-----------|
| `no_attempt`        | ask_probing_question, give_hint                      | false | none |
| `asking_for_answer` | **redirect_no_answer**                               | false | none |
| `stuck`             | give_hint, ask_probing_question, encourage_retry     | false | none / concept_gap |
| `partial`           | give_hint, ask_probing_question, encourage_retry     | false | incomplete_steps / none |
| `wrong_answer`      | give_hint, ask_probing_question                      | false | a specific error (not none) |
| `correct_answer`    | **affirm_and_confirm**                               | **true** | none |

## Objective grade (the scorer decides — no human/LLM needed for the core)
- **policy_ok** = valid schema AND move legal for state AND reveals_answer correct AND no illegal answer leak.
- **structured_exact** = student_state + diagnosis + move + reveals_answer all correct.
- Plus: **diagnosis accuracy**, **move-legality** rate, **leak** rate, **format-validity** rate — reported per category.
- Primary reported metrics: `policy_ok`, `structured_exact`, `diagnosis_exact` — **base vs tuned**, on a held-out, novel-phrasing eval set.
- The `message` *text* quality is scored separately by an optional LLM judge (`eval/rubric.md`).

## Why this needs a fine-tune (litmus test — verified empirically)
A well-prompted base (Qwen3-1.7B), handed this exact schema **and** policy in its system prompt,
still fails: it withholds the answer fine (that alone is prompt-solvable — we measured 0/22 leaks),
but it **misdiagnoses** the student's error, picks **illegal/inconsistent moves**, and writes lazy
**restatement "hints."** Reliability across the whole `state × diagnosis` space — every time, in
character, without drifting — is what the dataset buys and a prompt cannot guarantee.
