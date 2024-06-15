HUSKY_REASONING_GENERATOR_FEWSHOT_PROMPT = """Given the input question, the solution history that consists of steps for solving the input question and their corresponding outputs, and the current step that needs to be taken to solve the question, solve the current step using logical reasoning, basic math and commonsense knowledge.
Below are a few examples of the generated output. Adhere to the format shown in the examples.
---
Question: Which type of currency would one be advised to bring if visiting that has Mae Hong Son Province within its borders?
Solution history:
Step: Identify the country that has Mae Hong Son Province within its borders.
Output: The country that has Mae Hong Son Province within its borders is Thailand.
Current step: Determine the type of currency used in Thailand.
Output: The currency used in Thailand is the Thai Baht.
---
Question: What happened first, the Selective Draft Law Cases in the United States or Eugene Debs' run for president for the last time?
Solution history:
Step: Identify the year the Selective Draft Law Cases were decided in the United States.
Output: The Selective Draft Law Cases were decided in the year 1918.
Step: Identify the year Eugene Debs ran for president for the last time.
Output: Eugene Debs ran for president for the last time in the year 1920.
Current step: Determine which event happened first.
Output: Since the Selective Draft Law Cases were decided in 1918 and Eugene Debs ran for president in 1920, the Selective Draft Law Cases happened first.
---
Question: Who was the president of the body that petitioned the king of England to recognize the rights of colonists when the declaration of independence was adopted?
Solution history:
None
Current step: Identify the body that petitioned the king of England to recognize the rights of colonists when the Declaration of Independence was adopted.
Output: The body that petitioned the king of England to recognize the rights of colonists when the Declaration of Independence was adopted was the Second Continental Congress.
---
Question: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80 percent more of those in purple. There are only 25 percent as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?
Solution history:
Step: Compute the number of purple flowers in the garden.
Output: There are 80/100 * 10 = 8 more purple flowers than yellow flowers. Therefore, there are 10 + 8 = 18 purple flowers.
Current step: Compute the number of green flowers in the garden.
Output: The yellow and purple flowers sum up to 10 + 18 = 28 flowers. There are 25/1000 * 28 = 7 green flowers in Mark's garden.
---
Question: Who threw the most touchdown passes in the game between the Seattle Seahawks and the Washington Redskins where Shaun Suisham made a 26-yard field goal in the third quarter?
Solution history:
Step: Identify the date of the game between the Seattle Seahawks and the Washington Redskins where Shaun Suisham made a 26-yard field goal in the third quarter.
Output: The game between the Seattle Seahawks and the Washington Redskins, where Shaun Suisham made a 26-yard field goal in the third quarter, took place on November 23, 2008.
Step: Retrieve information about the game.
Output:
['Quarter', 'Time', 'Tm', 'Detail', 'WAS', 'SEA']
['1', '7:19', 'Seahawks', 'Olindo Mare 45 yard field goal', '0', '3']
['2', '4:32', 'Redskins', 'Ladell Betts 1 yard rush (Shaun Suisham kick)', '7', '3']
['2', '0:29', 'Seahawks', 'Maurice Morris 4 yard pass from Matt Hasselbeck (Olindo Mare kick)', '7', '10']
['3', '7:21', 'Redskins', 'Shaun Suisham 26 yard field goal', '10', '10']
['3', '3:25', 'Redskins', 'Antwaan Randle El 8 yard pass from Jason Campbell (Shaun Suisham kick)', '17', '10']
['4', '13:24', 'Seahawks', 'John Carlson 10 yard pass from Matt Hasselbeck (Olindo Mare kick)', '17', '17']
['4', '9:19', 'Redskins', 'Shaun Suisham 22 yard field goal', '20', '17']
Current step: Compare the number of touchdown passes thrown by each quarterback in the game.
Output: According to the game information, Matt Hasselbeck threw 2 touchdown passes for the Seattle Seahawks, and Jason Campbell threw 1 touchdown pass for the Washington Redskins. Therefore, Matt Hasselbeck threw the most touchdown passes in the game.
---
Question: There were 12 balloons that were either green, blue, yellow, or red. They were divided into equal groups and then Anya took half of the yellow ones home. How many balloons did Anya take home?
Solution history:
None
Current step: Identify the number of yellow balloons.
Output: Since the balloons are divided into equal groups and there are four colors, the number of yellow balloons is $\\frac{12}{4} = 3$.
---
Question: Nathan will roll two six-sided dice. What is the probability that he will roll a number less than three on the first die and a number greater than three on the second die? Express your answer as a common fraction.
Solution history:
Step: Compute the probability that Nate rolls a number less than three on the first die.
Output: For the first die to be less than three, it must be a 1 or a 2, which occurs with probability $\\frac{1}{3}$.
Current step: Compute the probability that Nate rolls a number greater than three on the second die.
Output: For the second die to be greater than 3, it must be a 4 or a 5 or a 6, which occurs with probability $\\frac{1}{2}$.
---
Question: What major river runs through the city where Al-Maqrizi was born?
Solution history:
Step: Identify the city where Al-Maqrizi was born.
Output: Al-Maqrizi was born in Cairo.
Current step: Identify the major river that runs through Cairo.
Output: The major river that runs through Cairo is the Nile River.
---
Question: %s
Solution history:
%s
Current step: %s
Output: """

HUSKY_REASONING_GENERATOR_FEWSHOT_TULU_PROMPT = f"""<|user|>
{HUSKY_REASONING_GENERATOR_FEWSHOT_PROMPT}
<|assistant|>\n"""

HUSKY_REASONING_GENERATOR_FEWSHOT_MISTRAL_PROMPT = f"""<s>[INST]
{HUSKY_REASONING_GENERATOR_FEWSHOT_PROMPT}
[/INST]"""
