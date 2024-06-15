HUSKY_MATH_GENERATOR_PROMPT = """Given the original question, the solution history that consists of steps for solving the input question and their corresponding outputs, and the current step that must be addressed to solve the input question, answer the current step by reasoning step-by-step.
- Make sure to answer the subquestion and not the original question.
- Do not attempt to directly answer the original question unless the subquestion asks for the same thing as the original question.
- Present the answer "ANS" to the subquestion in LaTeX using the format 'The answer is \\boxed{ANS}.' without any units in the box.
---
Question: %s
Solution history:
%s
Current step: %s
Solution: """

HUSKY_MATH_GENERATOR_TULU_PROMPT = f"""<|user|>
{HUSKY_MATH_GENERATOR_PROMPT}
<|assistant|>\n"""

HUSKY_MATH_GENERATOR_MISTRAL_PROMPT = f"""<s>[INST]
{HUSKY_MATH_GENERATOR_PROMPT}
[/INST]"""