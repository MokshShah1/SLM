# HomeCtl → Tutor: the pedagogy policy (human-readable)

`tutor/policy.py` is the **source of truth**; this file describes it. The dataset is generated
from the policy, so every training/eval label is correct by construction.

## Student states
| state | meaning |
|-------|---------|
| `no_attempt` | hasn't tried / asks how to start |
| `asking_for_answer` | demands the final answer |
| `stuck` | tried but stuck, no concrete work shown |
| `partial` | correct work so far, not finished |
| `wrong_answer` | produced an incorrect final answer |
| `correct_answer` | produced the correct final answer |

## Misconception taxonomy (diagnosis)
`none`, `arithmetic_slip` (right method, bad computation), `wrong_operation`,
`order_of_operations`, `misread_problem` (solved for the wrong quantity),
`concept_gap` (underlying concept), `incomplete_steps` (on track, stopped early).

## Moves
`ask_probing_question`, `give_hint`, `affirm_and_confirm`, `redirect_no_answer`, `encourage_retry`.

## Hard rules (what makes this hard to prompt, easy to grade)
- **Never reveal the final answer** unless the student already stated the correct one
  (`reveals_answer` may be `true` only for `correct_answer`).
- Each state has a fixed set of **legal moves** (see `behavior_spec.md`). `asking_for_answer`
  must `redirect_no_answer`; `correct_answer` must `affirm_and_confirm`.
- `wrong_answer` must carry a **specific** diagnosis; the hint is tuned to that diagnosis.
- The `message` is one calibrated next step, never a restatement, never leaks the answer.

## Problem generators (elementary / early-algebra)
`arith` (+, −, ×), `percent` (p% of n), `linear` (ax + b = c), `word_total`
(k× ratio, find the base), `word_ratio` (recipe scaling). Each yields a definite integer answer,
so the answer-leak check is programmatic.

## How wrong answers are synthesized (so the diagnosis label is known)
For each problem type we perturb the true answer in a way that matches a specific misconception —
e.g. percent `wrong_operation` = `p*n` (forgot ÷100); linear `order_of_operations` = `c//a`
(divided before subtracting b); word `misread_problem` = the *other* quantity. This gives us
labeled misconceptions for free.

## Data generation & quality gate
`src/generate_data.py` samples balanced scenarios across states/types, builds the gold JSON, and
**re-validates every example through the scorer** (`policy_ok` and `structured_exact` must hold)
before it is written. Optional `--teacher` reworts student phrasings via a teacher LLM for
diversity (numbers are never touched).
