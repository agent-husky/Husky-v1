HUSKY_ACTION_GENERATOR_PROMPT = """Given the input question and the solution history that consists of steps for solving the input question and their corresponding outputs, decide whether the solution history has already solved the original question. If the original question has not been solved yet, assign a tool (either [math], [code], [search], [match_info] or [commonsense]) and generate the next step that needs to be answered to solve the original question. Do not generate a step that has already been written in the solution history. Otherwise, if the original question has already been solved, return the [finish] tool, along with the final answer to the original question based on the solution history.
- [math] is for: 1) solving math questions, writing or re-organizing equations, performing abstract reasoning such as case-by-case analysis, or identifying the conditions given in the question.
- [code] is for: 1) computing large numbers (at least 100), fractions or decimals. 2) counting or averaging long lists of numbers. 3) performing date-related operations, such as counting the number of days between two dates.
- [search] is for: retrieving specific knowledge from the Web to answer questions related to history, sports, culture, geography, medicine, science, etc.
- [match_info] is for: retrieving information about a specific NFL football game.
- [commonsense] is for: applying commonsense knowledge to reason about a relatively simple step, such as comparing two numbers or recalling a widely-known fact.
- [finish] is for: indicating that the question has been solved, and it is followed by the answer to the question.
---
Question: %s
Solution history:
%s
Next step or final answer: """

HUSKY_ACTION_GENERATOR_TULU_PROMPT = f"""<|user|>
{HUSKY_ACTION_GENERATOR_PROMPT}
<|assistant|>\n"""

HUSKY_ACTION_GENERATOR_MISTRAL_PROMPT = f"""<s>[INST]
{HUSKY_ACTION_GENERATOR_PROMPT}
[/INST]\n"""
