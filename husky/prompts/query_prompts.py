HUSKY_QUERY_GENERATOR_PROMPT = """Given the input question, the solution history that consists of steps for solving the input question and their corresponding outputs, and the current step that needs to be taken to solve the question, write a concise, informative Google Search query for obtaining information regarding the current step.
---
Question: %s
Solution history:
%s
Current step: %s
Search query: """

HUSKY_QUERY_GENERATOR_TULU_PROMPT = f"""<|user|>
{HUSKY_QUERY_GENERATOR_PROMPT}
<|assistant|>\n"""

HUSKY_QUERY_GENERATOR_MISTRAL_PROMPT = f"""<s>[INST]
{HUSKY_QUERY_GENERATOR_PROMPT}
[/INST]\n"""