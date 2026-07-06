# Message-quality rubric (optional LLM-as-judge — the SECONDARY metric)

The core metrics are objective (`tutor/scorer.py`): policy adherence, diagnosis accuracy,
move-legality, no-answer-leak. This rubric grades only the **free-text `message`** — the part a
parser can't judge. Use it to show the tuned model's hints are *pedagogically better*, not just
structurally correct. Score base and tuned on the same held-out outputs; report the delta.

| Dimension | 0 | 1 | 2 |
|-----------|---|---|---|
| **Calibration** | Ignores the student's state/error | Loosely related | A precise next step for THIS student's diagnosis |
| **Single-step** | Dumps full solution or many steps | Somewhat over/under | Exactly one productive nudge |
| **Not-a-restatement** | Just re-asks the problem | Partly echoes it | Genuinely advances the thinking |
| **No leak & in-voice** | Leaks answer or breaks tutor voice | Wobbles | Withholds answer, warm Socratic voice |

Required outputs: mean score per dimension (base vs tuned) on the same scenarios, plus a short
error-analysis paragraph — where does the tuned model still fail, and is it a data problem?

Judge prompt (sketch): *"You are grading a math tutor's single reply. Given the PROBLEM, the
STUDENT message, and the gold diagnosis, score the tutor MESSAGE 0–2 on each dimension above.
Return JSON {calibration, single_step, not_restatement, no_leak_in_voice}."*
