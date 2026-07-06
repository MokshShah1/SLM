"""The pedagogy policy + problem generators + gold-scenario builder.

This module is the single source of truth for the tutor's behavior. Data is
generated FROM this policy, so every training/eval label is correct by construction.
The scorer (tutor/scorer.py) grades a model against exactly these rules.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------------------
STATES = [
    "no_attempt",         # student hasn't tried / asks how to start
    "asking_for_answer",  # student demands the final answer
    "stuck",              # tried but stuck, no concrete work shown
    "partial",            # correct work so far but not finished
    "wrong_answer",       # produced an incorrect final answer
    "correct_answer",     # produced the correct final answer
]

DIAGNOSES = [
    "none",
    "arithmetic_slip",     # right method, computation error
    "wrong_operation",     # used the wrong operation
    "order_of_operations", # PEMDAS / sequencing error
    "misread_problem",     # solved for the wrong quantity
    "concept_gap",         # underlying concept misunderstood
    "incomplete_steps",    # on track but stopped early
]

MOVES = [
    "ask_probing_question",
    "give_hint",
    "affirm_and_confirm",
    "redirect_no_answer",
    "encourage_retry",
]

# --------------------------------------------------------------------------------------
# Policy: what is legal / required for each student_state
# --------------------------------------------------------------------------------------
LEGAL_MOVES = {
    "no_attempt": {"ask_probing_question", "give_hint"},
    "asking_for_answer": {"redirect_no_answer"},
    "stuck": {"give_hint", "ask_probing_question", "encourage_retry"},
    "partial": {"give_hint", "ask_probing_question", "encourage_retry"},
    "wrong_answer": {"give_hint", "ask_probing_question"},
    "correct_answer": {"affirm_and_confirm"},
}

ALLOWED_DIAGNOSES = {
    "no_attempt": {"none"},
    "asking_for_answer": {"none"},
    "stuck": {"none", "concept_gap"},
    "partial": {"none", "incomplete_steps"},
    "wrong_answer": {"arithmetic_slip", "wrong_operation", "order_of_operations",
                     "misread_problem", "concept_gap"},
    "correct_answer": {"none"},
}

# reveals_answer is allowed to be true ONLY when the student already produced the answer.
REVEALS_ANSWER = {s: (s == "correct_answer") for s in STATES}

# A single deterministic "canonical" move per state (used as the SFT target label).
CANONICAL_MOVE = {
    "no_attempt": "ask_probing_question",
    "asking_for_answer": "redirect_no_answer",
    "stuck": "give_hint",
    "partial": "give_hint",
    "wrong_answer": "give_hint",
    "correct_answer": "affirm_and_confirm",
}

CANONICAL_DIAGNOSIS = {
    "no_attempt": "none",
    "asking_for_answer": "none",
    "stuck": "none",
    "partial": "incomplete_steps",
    "correct_answer": "none",
    # wrong_answer's diagnosis is set when the wrong answer is synthesized
}

SYSTEM_PROMPT = (
    "You are a Socratic math tutor. Given a math PROBLEM and the STUDENT's latest "
    "message, respond with a SINGLE JSON object and nothing else (no markdown, no prose).\n\n"
    "Schema (all keys required):\n"
    '  "student_state": one of ["no_attempt","asking_for_answer","stuck","partial","wrong_answer","correct_answer"]\n'
    '  "diagnosis": one of ["none","arithmetic_slip","wrong_operation","order_of_operations","misread_problem","concept_gap","incomplete_steps"]\n'
    '  "move": one of ["ask_probing_question","give_hint","affirm_and_confirm","redirect_no_answer","encourage_retry"]\n'
    '  "message": your reply to the student, at most 2 sentences\n'
    '  "reveals_answer": boolean\n\n'
    "Pedagogy policy (follow exactly):\n"
    "  - Never reveal or compute the final answer unless the student has already stated the correct answer.\n"
    '  - asking_for_answer -> move MUST be "redirect_no_answer", reveals_answer MUST be false.\n'
    '  - correct_answer   -> move MUST be "affirm_and_confirm", reveals_answer MUST be true, diagnosis MUST be "none".\n'
    '  - wrong_answer      -> diagnosis MUST be a specific error (not "none"); move "give_hint" or "ask_probing_question"; reveals_answer false.\n'
    '  - no_attempt        -> diagnosis "none"; move "ask_probing_question" or "give_hint"; reveals_answer false.\n'
    '  - stuck             -> move "give_hint", "ask_probing_question", or "encourage_retry"; reveals_answer false.\n'
    '  - partial           -> diagnosis "incomplete_steps" or "none"; move "give_hint"/"ask_probing_question"/"encourage_retry"; reveals_answer false.\n'
    "  - message must be ONE calibrated next step tuned to the diagnosis, never a restatement "
    "of the question, and must not contain the final answer (except when confirming a correct one)."
)

# --------------------------------------------------------------------------------------
# Math problem generators  ->  (problem_text, answer_int, info)
# --------------------------------------------------------------------------------------
def _arith(rng):
    op = rng.choice(["+", "-", "*"])
    a, b = rng.randint(2, 20), rng.randint(2, 20)
    if op == "+":
        ans, text = a + b, f"What is {a} + {b}?"
    elif op == "-":
        if b > a:
            a, b = b, a
        ans, text = a - b, f"What is {a} - {b}?"
    else:
        ans, text = a * b, f"What is {a} * {b}?"
    return text, ans, {"type": "arith", "a": a, "b": b, "op": op}


def _percent(rng):
    p = rng.choice([10, 20, 25, 50, 75])
    n = rng.choice([20, 40, 60, 80, 120, 200])
    ans = p * n // 100
    return f"What is {p}% of {n}?", ans, {"type": "percent", "p": p, "n": n}


def _linear(rng):
    x = rng.randint(1, 12)
    a = rng.randint(2, 9)
    b = rng.randint(1, 20)
    c = a * x + b
    return f"Solve for x: {a}x + {b} = {c}", x, {"type": "linear", "a": a, "b": b, "c": c, "x": x}


def _word_total(rng):
    k = rng.choice([2, 3, 4])
    tom = rng.randint(2, 10)
    total = tom * (k + 1)
    text = (f"Sam has {k} times as many apples as Tom. Together they have {total} apples. "
            "How many apples does Tom have?")
    return text, tom, {"type": "word_total", "k": k, "total": total, "tom": tom}


def _word_ratio(rng):
    cups = rng.choice([2, 3, 4])
    per = rng.choice([2, 3, 4])
    mult = rng.choice([2, 3, 4, 5])
    cookies = per * mult
    ans = cups * mult
    text = (f"A recipe uses {cups} cups of flour to make {per} cookies. "
            f"How much flour is needed for {cookies} cookies?")
    return text, ans, {"type": "word_ratio", "cups": cups, "per": per, "cookies": cookies}


GENERATORS = [_arith, _percent, _linear, _word_total, _word_ratio]


def make_wrong(rng, ans, info):
    """Return (wrong_answer_str, diagnosis) consistent with the problem type."""
    t = info["type"]
    opts = {}
    if t == "arith":
        opts["arithmetic_slip"] = ans + rng.choice([-2, -1, 1, 2])
        a, b, op = info["a"], info["b"], info["op"]
        opts["wrong_operation"] = (a + b) if op == "*" else (a * b if op == "+" else a + b)
    elif t == "percent":
        p, n = info["p"], info["n"]
        opts["arithmetic_slip"] = ans + rng.choice([-2, -1, 1, 2])
        opts["concept_gap"] = p                 # thinks the answer is just the percent
        opts["wrong_operation"] = p * n         # forgot to divide by 100
    elif t == "linear":
        opts["arithmetic_slip"] = ans + rng.choice([-1, 1, 2])
        opts["misread_problem"] = info["b"]     # answered a coefficient
        opts["order_of_operations"] = info["c"] // info["a"]  # divided before subtracting b
    elif t == "word_total":
        opts["misread_problem"] = info["k"] * info["tom"]     # gave Sam's count
        opts["arithmetic_slip"] = ans + rng.choice([-1, 1])
        opts["concept_gap"] = info["total"] // 2
    elif t == "word_ratio":
        opts["arithmetic_slip"] = ans + rng.choice([-1, 1, 2])
        opts["wrong_operation"] = info["cookies"]             # just echoed the cookie count
        opts["concept_gap"] = info["cups"] + info["per"]
    # keep only positive, genuinely-wrong options
    opts = {d: v for d, v in opts.items() if isinstance(v, int) and v > 0 and v != ans}
    if not opts:
        opts["arithmetic_slip"] = ans + 1
    diagnosis = rng.choice(list(opts))
    return str(opts[diagnosis]), diagnosis


# --------------------------------------------------------------------------------------
# Student message + gold tutor message synthesis
# --------------------------------------------------------------------------------------
_NO_ATTEMPT = ["I don't know where to start.", "How do I even begin this one?",
               "I have no idea what to do here."]
_ASKING = ["Just tell me the answer.", "Can you just give me the answer?",
           "What's the answer? I don't feel like working it out."]
_STUCK = ["I tried but I'm stuck.", "I don't get it.", "I'm confused and can't figure this out."]


def _partial_msg(rng, info):
    t = info["type"]
    if t == "linear":
        return f"I subtracted {info['b']} from both sides, but then I got confused."
    if t == "percent":
        return "I know 10% is easy to find, but I wasn't sure what to do next."
    if t == "word_total":
        return "I think I need to split the total into equal parts, but I stopped there."
    if t == "word_ratio":
        return "I found how many batches it is, but wasn't sure what to do with that."
    return "I started setting it up but didn't finish."


_HINTS = {
    "arithmetic_slip": "Your method is right, so slow down and recheck the arithmetic in your last step.",
    "wrong_operation": "Think about which operation this situation really calls for - does it combine, separate, or scale the numbers?",
    "order_of_operations": "Check the order here - which part do you need to handle before you divide or add?",
    "misread_problem": "Reread the question carefully - are you solving for the exact quantity it asks about?",
    "concept_gap": "Let's revisit the key idea - what does the main term in this problem actually mean?",
    "incomplete_steps": "Good progress - what is the very next step from where you stopped?",
    "none": "You've got this - start by writing down what you know and what you need to find.",
}


def gold_message(rng, state, diagnosis, answer):
    if state == "asking_for_answer":
        return "I won't just hand over the answer, but I'll help you get there yourself. What part feels trickiest so far?"
    if state == "correct_answer":
        return f"Yes - {answer} is correct! Nice work reasoning all the way through it yourself."
    if state == "no_attempt":
        return "Let's start by understanding the problem - in your own words, what is it asking you to find?"
    if state == "encourage_retry" or CANONICAL_MOVE.get(state) == "encourage_retry":
        return "You're close - give it one more try, taking it one small step at a time."
    # stuck / partial / wrong_answer -> a calibrated give_hint keyed to the diagnosis
    return _HINTS.get(diagnosis, _HINTS["none"])


# --------------------------------------------------------------------------------------
# Scenario assembly
# --------------------------------------------------------------------------------------
@dataclass
class Scenario:
    category: str
    problem: str
    answer: str
    student_msg: str
    student_state: str
    diagnosis: str
    move: str
    reveals_answer: bool
    message: str

    def user_content(self) -> str:
        return f"PROBLEM: {self.problem}\nSTUDENT: {self.student_msg}"

    def gold_labels(self) -> dict:
        return {
            "student_state": self.student_state,
            "diagnosis": self.diagnosis,
            "move": self.move,
            "reveals_answer": self.reveals_answer,
        }

    def gold_json(self) -> dict:
        d = self.gold_labels()
        d["message"] = self.message
        return d


def build_scenario(rng: random.Random, state: str | None = None,
                   ptype: str | None = None) -> Scenario:
    gen = next((g for g in GENERATORS if g.__name__ == f"_{ptype}"), None) if ptype else None
    gen = gen or rng.choice(GENERATORS)
    problem, ans_int, info = gen(rng)
    answer = str(ans_int)
    state = state or rng.choice(STATES)

    if state == "wrong_answer":
        wrong, diagnosis = make_wrong(rng, ans_int, info)
        student_msg = rng.choice([f"I think the answer is {wrong}.", f"Is it {wrong}?",
                                  f"I got {wrong}."])
    elif state == "correct_answer":
        diagnosis = "none"
        student_msg = rng.choice([f"I got {answer}.", f"Is it {answer}?",
                                  f"I worked it out and got {answer}."])
    elif state == "partial":
        diagnosis = "incomplete_steps"
        student_msg = _partial_msg(rng, info)
    elif state == "no_attempt":
        diagnosis = "none"
        student_msg = rng.choice(_NO_ATTEMPT)
    elif state == "asking_for_answer":
        diagnosis = "none"
        student_msg = rng.choice(_ASKING)
    else:  # stuck
        diagnosis = "none"
        student_msg = rng.choice(_STUCK)

    move = CANONICAL_MOVE[state]
    reveals = REVEALS_ANSWER[state]
    message = gold_message(rng, state, diagnosis, answer)
    return Scenario(
        category=state, problem=problem, answer=answer, student_msg=student_msg,
        student_state=state, diagnosis=diagnosis, move=move,
        reveals_answer=reveals, message=message,
    )


if __name__ == "__main__":
    rng = random.Random(0)
    for st in STATES:
        s = build_scenario(rng, state=st)
        print("=" * 80)
        print(f"[{st}] {s.problem}  (answer={s.answer})")
        print(f"  STUDENT: {s.student_msg}")
        print(f"  GOLD   : {s.gold_json()}")
