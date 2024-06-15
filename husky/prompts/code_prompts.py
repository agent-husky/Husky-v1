HUSKY_CODE_GENERATOR_PROMPT = """Given the input question, the solution history that consists of steps for solving the input question and their corresponding outputs, and the current step that must be addressed to solve the input question, write code that solves the current step.
- Write the code in Python.
- Do not attempt to write code that directly answers the question. Write code that answers the given step.
- For math questions, utilize the 'pi' symbol and 'Rational' from the sympy package for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.
- Example imports are provided below. Import any of these packages, as well as additional packages as needed.
import math
import numpy as np
import sympy
from datetime import datetime
from math import comb, gcd, lcm
from scipy.optimize import minimize
from sympy import symbols, Eq, solve, expand, factor, Matrix
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.core.relational import LessThan
---
Question: %s
Solution history:
%s
Current step: %s
Code:
"""

HUSKY_CODE_GENERATOR_TULU_PROMPT = f"""<|user|>
{HUSKY_CODE_GENERATOR_PROMPT}
<|assistant|>\n"""

HUSKY_CODE_GENERATOR_MISTRAL_PROMPT = f"""<s>[INST]
{HUSKY_CODE_GENERATOR_PROMPT}
[/INST]"""