import json
import os
import re
import time

def add_to_file(output_data, output_path):
    with open(output_path, "a") as f:
        json_data = json.dumps(output_data)
        f.write(json_data + '\n')

def batch_list(input_list, batch_size):
    batched_list = []
    for i in range(0, len(input_list), batch_size):
        batch = input_list[i:i + batch_size]
        batched_list.append(batch)
    return batched_list

def read_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist")
        return []
    with open(file_path, "r") as f:
        json_list = list(f)
        outputs = []
        for json_str in json_list:
            try:
                data = json.loads(json_str)
            except:
                print(json_str)
                exit(0)
            outputs.append(data)
    return outputs

def read_json_safe(file_path, max_retries=5, retry_delay=3):
    for _ in range(max_retries):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return data
        except PermissionError:
            time.sleep(retry_delay)

def write_json_safe(data, file_path, max_retries=5, retry_delay=3):
    for _ in range(max_retries):
        try:
            with open(file_path, "w") as file:
                json.dump(data, file)
            return  # Exit the loop if writing succeeds
        except PermissionError:
            time.sleep(retry_delay)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = ""
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return ""
    
def simplify_answer(answer, convert_to_str=False):
    if 'relational' in str(type(answer)):
        return str(answer)
    elif 'numpy' in str(type(answer)):
        if answer.shape == ():
            # scalar value
            answer = round(float(answer), 2)
        else:
            # array value
            answer = round(float(answer[0]), 2)
        return str(answer) if convert_to_str else answer
    elif not answer:
        return "[FAIL]"
    else:
        if type(answer) in [list, tuple]:
            if 'sympy' in str(type(answer[0])):
                try:
                    answer = [round(float(x), 2) for x in answer]
                except Exception:
                    answer = [str(x) for x in answer]
            else:
                answer = [str(x) for x in answer]
            if len(answer) == 1:
                answer = answer[0]
            return answer
        else:
            if 'sympy' in str(type(answer)):
                try:
                    answer = round(float(answer), 2)
                except Exception:
                    answer = str(answer)
                return answer
            elif 'int' in str(type(answer)):
                return str(answer) if convert_to_str else answer
            else:
                try:
                    answer = round(float(answer), 4)
                    return str(answer) if convert_to_str else answer
                except:
                    return str(answer) if convert_to_str else answer

def extract_number(output):
    if output is None:
        return None
    elif isinstance(output, int) or isinstance(output, float):
        return output
    try:
        output_list = re.findall(r'\d+\.\d+|\d+', output)
    except TypeError:
        return None
    return float(output_list[-1]) if len(output_list) > 0 else None

def postprocess_search_solution(solution_text):
    # Split the text into sections based on "Step N:" pattern
    steps = re.split(r'(Step \d+: )', solution_text)[1:]
    step_output_list = [s.split('\n', 1)[1:] for i, s in enumerate(steps) if i % 2 == 1]
    # Combine step headers with their content
    steps = [steps[i] + steps[i + 1] for i in range(0, len(steps), 2)]
    # Organize information from the solution
    step_number_list = []
    step_info = {}
    assert(len(steps) == len(step_output_list))
    for step, step_output in zip(steps, step_output_list):
        step_number = int(re.search(r'Step (\d+):', step).group(1))
        if step_number not in step_info:
            step_info[step_number] = {}
            step_number_list.append(step_number)
        if "step_txt" not in step_info[step_number]:
            step_txt = re.search(r'Step \d+: ([^\n]+)', step).group(1)
            step_info[step_number]["step_txt"] = step_txt
        step_info[step_number]["step_output"] = step_output[-1]
    # Build the modified solution
    final_content_list = []
    for step_idx in step_number_list:
        step_txt = step_info[step_idx]["step_txt"]
        step_output = step_info[step_idx]["step_output"]
        final_content = f"Step {step_idx}: {step_txt}\n{step_output}".strip()
        final_content_list.append(final_content)
    modified_solution = "\n\n".join(final_content_list)
    return modified_solution

def extract_number_gsm8k(output_string):
    output_string = re.sub(r"(\d),(\d)", r"\1\2", output_string)
    # Regular expression to match an optional '-' for negative numbers,
    # followed by digits, an optional decimal point, and optional more digits
    # It captures the numeric part of the string, ignoring units or other text
    match = re.search(r'-?\d+\.?\d*', output_string)
    if match:
        return match.group()  # Return the matched numeric part
    else:
        return ""  # Return None if no numeric part is found
    
def extract_number_math(string):
    pattern = r'^(\d+(?:\.\d+)?)\s*[a-zA-Z]+$'  # Regular expression pattern to match "[NUMBER] [UNIT]"
    match = re.match(pattern, string)
    if match:
        return match.group(1)  # Extracting the number part and converting it to float
    else:
        return string
