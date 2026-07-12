# Sample Dataset — Structured Socratic Math Tutor

**Full dataset:** 800 train · 160 validation · 24 held-out eval examples.  
**This sample:** 2 examples from each of the 6 student states (12 total).

Every example is generated *from* the pedagogy policy in `tutor/policy.py` and re-validated by the objective scorer (`tutor/scorer.py`) before being written — so every label is correct by construction.

**Format:** chat-style JSONL. Input = `PROBLEM` + `STUDENT` message. Output = one strict-JSON coaching move.

**Train split balance (all 6 states ~evenly represented):**  
`asking_for_answer` 137, `correct_answer` 138, `no_attempt` 128, `partial` 122, `stuck` 134, `wrong_answer` 141

---

## The output schema (every assistant reply)

```json
{
  "expected_answer": "computed privately by the tutor, never shown to the student",
  "student_state":   "no_attempt | asking_for_answer | stuck | partial | wrong_answer | correct_answer",
  "diagnosis":       "none | arithmetic_slip | wrong_operation | order_of_operations | misread_problem | concept_gap | incomplete_steps",
  "move":            "ask_probing_question | give_hint | affirm_and_confirm | redirect_no_answer | encourage_retry",
  "message":         "tutor reply, <= 2 sentences, no answer leak",
  "reveals_answer":  false
}
```

---

## State: `no_attempt`

**Example 1**
- **Problem:** Solve for x: 8x + 2 = 26
- **Student:** "I don't know where to start."
- **Tutor move:** state=`no_attempt` · diagnosis=`none` · move=`ask_probing_question` · reveals_answer=`false`
- **Message:** "Let's start by understanding the problem - in your own words, what is it asking you to find?"
- *(private) expected_answer:* `3`

**Example 2**
- **Problem:** What is 10% of 120?
- **Student:** "I don't know where to start."
- **Tutor move:** state=`no_attempt` · diagnosis=`none` · move=`ask_probing_question` · reveals_answer=`false`
- **Message:** "Let's start by understanding the problem - in your own words, what is it asking you to find?"
- *(private) expected_answer:* `12`


## State: `asking_for_answer`

**Example 3**
- **Problem:** A recipe uses 2 cups of flour to make 3 cookies. How much flour is needed for 6 cookies?
- **Student:** "What's the answer? I don't feel like working it out."
- **Tutor move:** state=`asking_for_answer` · diagnosis=`none` · move=`redirect_no_answer` · reveals_answer=`false`
- **Message:** "I won't just hand over the answer, but I'll help you get there yourself. What part feels trickiest so far?"
- *(private) expected_answer:* `4`

**Example 4**
- **Problem:** Solve for x: 4x + 18 = 46
- **Student:** "Just tell me the answer."
- **Tutor move:** state=`asking_for_answer` · diagnosis=`none` · move=`redirect_no_answer` · reveals_answer=`false`
- **Message:** "I won't just hand over the answer, but I'll help you get there yourself. What part feels trickiest so far?"
- *(private) expected_answer:* `7`


## State: `stuck`

**Example 5**
- **Problem:** What is 10% of 20?
- **Student:** "I don't get it."
- **Tutor move:** state=`stuck` · diagnosis=`none` · move=`give_hint` · reveals_answer=`false`
- **Message:** "You've got this - start by writing down what you know and what you need to find."
- *(private) expected_answer:* `2`

**Example 6**
- **Problem:** A recipe uses 3 cups of flour to make 4 cookies. How much flour is needed for 12 cookies?
- **Student:** "I tried but I'm stuck."
- **Tutor move:** state=`stuck` · diagnosis=`none` · move=`give_hint` · reveals_answer=`false`
- **Message:** "You've got this - start by writing down what you know and what you need to find."
- *(private) expected_answer:* `9`


## State: `partial`

**Example 7**
- **Problem:** Sam has 2 times as many apples as Tom. Together they have 15 apples. How many apples does Tom have?
- **Student:** "I think I need to split the total into equal parts, but I stopped there."
- **Tutor move:** state=`partial` · diagnosis=`incomplete_steps` · move=`give_hint` · reveals_answer=`false`
- **Message:** "Good progress - what is the very next step from where you stopped?"
- *(private) expected_answer:* `5`

**Example 8**
- **Problem:** A recipe uses 4 cups of flour to make 4 cookies. How much flour is needed for 12 cookies?
- **Student:** "I found how many batches it is, but wasn't sure what to do with that."
- **Tutor move:** state=`partial` · diagnosis=`incomplete_steps` · move=`give_hint` · reveals_answer=`false`
- **Message:** "Good progress - what is the very next step from where you stopped?"
- *(private) expected_answer:* `12`


## State: `wrong_answer`

**Example 9**
- **Problem:** What is 15 * 3?
- **Student:** "I got 43."
- **Tutor move:** state=`wrong_answer` · diagnosis=`arithmetic_slip` · move=`give_hint` · reveals_answer=`false`
- **Message:** "Your method is right, so slow down and recheck the arithmetic in your last step."
- *(private) expected_answer:* `45`

**Example 10**
- **Problem:** Solve for x: 3x + 19 = 25
- **Student:** "I think the answer is 8."
- **Tutor move:** state=`wrong_answer` · diagnosis=`order_of_operations` · move=`give_hint` · reveals_answer=`false`
- **Message:** "Check the order here - which part do you need to handle before you divide or add?"
- *(private) expected_answer:* `2`


## State: `correct_answer`

**Example 11**
- **Problem:** A recipe uses 2 cups of flour to make 4 cookies. How much flour is needed for 20 cookies?
- **Student:** "I got 10."
- **Tutor move:** state=`correct_answer` · diagnosis=`none` · move=`affirm_and_confirm` · reveals_answer=`true`
- **Message:** "Yes - 10 is correct! Nice work reasoning all the way through it yourself."
- *(private) expected_answer:* `10`

**Example 12**
- **Problem:** Sam has 4 times as many apples as Tom. Together they have 50 apples. How many apples does Tom have?
- **Student:** "Is it 10?"
- **Tutor move:** state=`correct_answer` · diagnosis=`none` · move=`affirm_and_confirm` · reveals_answer=`true`
- **Message:** "Yes - 10 is correct! Nice work reasoning all the way through it yourself."
- *(private) expected_answer:* `10`

