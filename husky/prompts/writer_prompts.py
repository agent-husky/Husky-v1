HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_PROMPT = """Given the input question, the current step that is helpful for solving the original question, the code addressing this step and the corresponding output, rewrite the output into a complete sentence that integrates information from the original question and the current step.
Below are a few examples of the generated output. Adhere to the format shown in the examples.
---
Question: What is the degree of the polynomial $(4 +5x^3 +100 +2\\pi x^4 + \\sqrt{10}x^4 +9)$?
Step: Simplify the polynomial and identify its highest power.
Code:
from sympy import symbols, simplify, pi, sqrt

# define the variable
x = symbols('x')

# define the polynomial
polynomial = 4 + 5*x**3 + 100 + 2*pi*x**4 + sqrt(10)*x**4 + 9

# simplify the polynomial
simplified_polynomial = simplify(polynomial)

# identify the highest power
degree = simplified_polynomial.as_poly().degree()

# print the output
out = degree
print(out)
Code output: 4.0
Rewrite: The highest power of the simplified polynomial is 4.
---
Question: The town's annual budget totals $32 million. If half of the budget goes towards policing and $12 million goes towards education. How much money is left for managing public spaces?
Step: Compute the amount of money left for managing public spaces after the budget usage on policing and education.
Code:
# define the total budget
total_budget = 32

# calculate the amount allocated for policing (half of the budget)
policing_budget = total_budget / 2

# define the amount allocated for education
education_budget = 12

# calculate the amount left for managing public spaces
amount_left = total_budget - (policing_budget + education_budget)

# print the output
out = amount_left
print(out)
Code output: 4.0
Rewrite: $4 million is left for managing public spaces.
---
Question: Solve for the sum of all possible values of $x$ when $3^{x^2+4x+4}=9^{x+2}$.
Step: Solve the simplified equation for $x$.
Code:
from sympy import symbols, Eq, solve

# define the variable
x = symbols('x')

# simplified equation from step 1
simplified_equation = Eq(x**2 + 4*x + 4, 2*x + 4)

# solve the simplified equation for x
solution_x = solve(simplified_equation, x)

# print the output
out = solution_x
print(out)
Code output: [-2, 0]
Rewrite: The possible values for $x$ after solving the simplified equation are -2 and 0.
---
Question: A dance studio offers lessons to students. It costs $25 per session to rent the studio plus $1.50 per student per session. At present, the dance studio has 10 students and is rented 3 days a week. How much are the expected earnings of the dance studio in a month?
Step: Find the total earnings for renting the studio and the total earnings from the students in one session.
Code:
# define the cost for renting the studio per session
cost_studio_per_session = 25

# define the cost for each student per session
cost_student_per_session = 1.5

# define the number of students
num_students = 10

# determine the earnings from all 10 students per session
cost_students_per_session = cost_student_per_session * num_students

# determine the total earnings in one session
total_earnings_per_session = cost_studio_per_session + cost_students_per_session

# print the output
out = total_earnings_per_session
print(out)
Code output: 40
Rewrite: The total earnings for renting the studio and from the students in one session together is $40.
---
Question: %s
Step: %s
Code:
%s
Code output: %s
Rewrite: """

HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_TULU_PROMPT = f"""<|user|>
{HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_PROMPT}
<|assistant|>\n"""

HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_MISTRAL_PROMPT = f"""<s>[INST]
{HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_PROMPT}
[/INST]"""

HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_PROMPT = """Given an instruction and the corresponding search result from Google, rewrite the search result into a complete sentence that answers the instruction.
Below are a few examples of the generated outputs. Adhere to the format shown in the examples.
---
Instruction: Find the release date of "Harry Potter and the Sorcerer's Stone".
Output: November 16, 2001
Answer: The first Harry Potter movie was released on November 16, 2001.
---
Instruction: Search for Lionel Messi's birthday.
Output: Lionel Andrés "Leo" Messi (Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi]; born 24 June 1987) is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team.
Answer: Lionel Messi's birthday is on June 24, 1987.
---
Instruction: Find the height of Mount Baker.
Output: Baker Summit - Climbing. The most prominent feature of the Mt. Baker Wilderness is the 10,781 foot [3,286 meters] active volcano from which the wilderness takes its name.
Answer: Mount Baker is 10,781 feet tall.
---
Instruction: Identify who the fourteenth President of the United States is.
Output: Franklin Pierce became 14th President of the United States at a time of apparent tranquility (1853-1857). By pursuing the recommendations of southern advisers, Pierce — a New Englander — hoped to ease the divisions that led eventually to Civil War. Franklin Pierce became President at a time of apparent tranquility.
Answer: The fourteenth President of the United States is Franklin Pierce.
---
Instruction: %s
Output: %s
Answer: """

HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_TULU_PROMPT = f"""<|user|>
{HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_PROMPT}
<|assistant|>\n"""

HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_MISTRAL_PROMPT = f"""<s>[INST]
{HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_PROMPT}
[/INST]"""
