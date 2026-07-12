---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags:
  - unsloth
  - qlora
  - lora
  - peft
  - education
  - socratic-tutor
  - structured-output
language:
  - en
pipeline_tag: text-generation
---

# Qwen3-1.7B — Structured Socratic Math Tutor

A QLoRA fine-tune of **Qwen/Qwen3-1.7B** that turns the base model into a reliable
**Socratic math tutor**. Given a math problem and a student's latest message, it emits a
single **structured JSON coaching move**: it privately computes the answer, identifies the
student's state, diagnoses *why* they are wrong, picks a pedagogically-legal next move, and
writes one calibrated hint — **without ever revealing the final answer** (unless the student
has already produced it correctly).

> **Thesis: behavior from data, not smarts from scale.** This 1.7B model does not out-think a
> frontier model. It does *one narrow thing reliably* that a well-prompted base model cannot:
> apply a fixed pedagogy policy every time, in character, without drifting.

## What it does

**Input** — a `PROBLEM` and the `STUDENT`'s latest message:

```
PROBLEM: What is 15 * 3?
STUDENT: I got 43.
```

**Output** — one strict JSON object, exactly these keys, in this order:

```json
{
  "expected_answer": "45",
  "student_state": "wrong_answer",
  "diagnosis": "arithmetic_slip",
  "move": "give_hint",
  "message": "Your method is right, so slow down and recheck the arithmetic in your last step.",
  "reveals_answer": false
}
```

- **`expected_answer`** — the correct answer, computed **privately first** (never shown to the
  student). Forcing this as the first field is a "compute-then-classify" step: the model must
  work out the answer before it can judge whether the student is right or wrong.
- **`student_state`** — `no_attempt` · `asking_for_answer` · `stuck` · `partial` · `wrong_answer` · `correct_answer`
- **`diagnosis`** — `none` · `arithmetic_slip` · `wrong_operation` · `order_of_operations` · `misread_problem` · `concept_gap` · `incomplete_steps`
- **`move`** — `ask_probing_question` · `give_hint` · `affirm_and_confirm` · `redirect_no_answer` · `encourage_retry`
- **`message`** — the tutor's reply (≤ 2 sentences), calibrated to the diagnosis, never a
  restatement of the problem, never containing the answer.
- **`reveals_answer`** — `true` **only** when the student already stated the correct answer.

### The pedagogy policy (enforced by the training data)

| student_state | legal moves | reveals_answer |
|---|---|:--:|
| `no_attempt` | ask_probing_question, give_hint | false |
| `asking_for_answer` | **redirect_no_answer** | false |
| `stuck` | give_hint, ask_probing_question, encourage_retry | false |
| `partial` | give_hint, ask_probing_question, encourage_retry | false |
| `wrong_answer` | give_hint, ask_probing_question (diagnosis must be specific) | false |
| `correct_answer` | **affirm_and_confirm** | **true** |

## Results — base vs. tuned (held-out, novel-phrasing eval)

Evaluated on 24 hand-written held-out cases whose phrasing never appears in training, with an
**objective programmatic scorer** (schema validity, move-legality, answer-leak, label
correctness — no LLM judge, no vibes).

| metric | base (well-prompted) | **tuned** | delta |
|---|---|---|---|
| structured_exact (whole move correct) | 0.167 | **0.750** | **+0.583** |
| diagnosis_exact (right misconception) | 0.458 | **0.917** | **+0.459** |
| policy_ok (legal move + correct flags) | 0.542 | **0.917** | **+0.375** |
| leak_ok (never reveals the answer) | 0.750 | **1.000** | **+0.250** |

The base model already follows the *format* — that is prompt-solvable. What it cannot do
reliably is the **judgment**: correctly diagnosing the error and picking a legal move. That
gap is what the dataset closes.

## Usage

This is a LoRA adapter on top of `Qwen/Qwen3-1.7B`. Use the same system prompt it was trained
with (below), and greedy decoding.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "Qwen/Qwen3-1.7B"
ADAPTER = "mokshpshah/qwen3-1.7b-socratic-tutor"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()

SYSTEM_PROMPT = (
    "You are a Socratic math tutor. Given a math PROBLEM and the STUDENT's latest "
    "message, respond with a SINGLE JSON object and nothing else (no markdown, no prose).\n\n"
    'Think first. The FIRST key, "expected_answer", is the correct final answer that YOU '
    "compute privately (the student never sees it). Use it to judge whether the student's "
    "number is right.\n\n"
    "Schema (all keys required, in this exact order): expected_answer, student_state, "
    "diagnosis, move, message, reveals_answer.\n"
    "Policy: asking_for_answer -> redirect_no_answer (reveals false); correct_answer -> "
    "affirm_and_confirm (reveals true); wrong_answer -> a specific diagnosis + give_hint/"
    "ask_probing_question (reveals false); never put the final answer in message unless the "
    "student already stated it correctly."
)

user = "PROBLEM: What is 15 * 3?\nSTUDENT: I got 43."
text = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

> The full system prompt used in training lives in `tutor/policy.py` (`SYSTEM_PROMPT`) in the
> project repo — copy it verbatim for best results.

## Training

- **Base:** `Qwen/Qwen3-1.7B` (Instruct)
- **Method:** QLoRA supervised fine-tuning (Unsloth + TRL), 4-bit, single Colab T4 GPU
- **Data:** 800 train / 160 validation examples, generated **from** the pedagogy policy so every
  label is correct by construction, then re-validated through the objective scorer as a hard
  quality gate. Balanced across all six student states.
- **Eval:** 24 hand-written, held-out, novel-phrasing cases, scored programmatically.

## Intended use & limitations

- **Intended:** elementary / early-algebra tutoring — arithmetic, percentages, ratios,
  single-variable linear equations, and simple word problems with a definite numeric answer.
- **Not intended:** open-ended math, proofs, multi-step problems without a single integer
  answer, or general chat. Outside this narrow domain the answer-leak and diagnosis guarantees
  do not hold.
- Always parse and validate the JSON output before displaying it to a student.

## Citation / sources

- Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- Training stack: [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://huggingface.co/docs/trl)
- Method: [QLoRA (Dettmers et al.)](https://arxiv.org/abs/2305.14314) ·
  reasoning-before-answer inspired by [Chain-of-Thought (Wei et al.)](https://arxiv.org/abs/2201.11903)
