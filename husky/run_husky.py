import argparse
import os
import pickle
import re
import subprocess
import sys
import time

from tqdm import tqdm

from generator import OpenAIGenerator, VLLMGenerator
from model_classes import MODEL_CLASSES
from prompts.action_prompts import *
from prompts.code_prompts import *
from prompts.math_prompts import *
from prompts.query_prompts import *
from prompts.reasoning_prompts import *
from prompts.writer_prompts import *
from search import GoogleSearchAPI
from utils import batch_list, read_json_safe, read_jsonl, simplify_answer, write_json_safe

# regex patterns to match for extracting code and their outputs
code_pattern = r'```python\n(.*?)```'
output_pattern = r'```output\n(.*?)```'

TOOL_LIST = ["math", "code", "search", "commonsense", "finish"]

# import prefix for code
code_prefix = """import math
import numpy as np
import sympy
from datetime import datetime
from math import comb, gcd, lcm
from scipy.optimize import minimize
from sympy import symbols, Eq, solve, expand, factor, simplify, Matrix
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy.core.relational import LessThan
"""

def parse_args():
    parser = argparse.ArgumentParser(description="performing inference")
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--subtask", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--use_action", action="store_true", help="")
    parser.add_argument("--use_code", action="store_true", help="")
    parser.add_argument("--use_math", action="store_true", help="")
    parser.add_argument("--use_reason", action="store_true", help="")
    parser.add_argument("--use_search", action="store_true", help="")
    parser.add_argument("--use_update", action="store_true", help="")
    args = parser.parse_args()
    return args

def extract_answer(output_str: str):
    answer = output_str.split("The answer is")[-1]
    if len(answer) > 0:
        answer = answer[:-1] if answer[-1] == '.' else answer
    else:
        answer = ""
    return answer

def extract_tool_and_step(input_string):
    # Regular expression to match the pattern [TOOL] STEP
    match = re.match(r'\[(?P<tool>\w+)\]\s*(?P<step>.+)', input_string)
    if match:
        tool = match.group('tool')
        step = match.group('step')
    else:
        tool = "commonsense"
        step = "Solve the question using the given information."
    return tool, step

def wait_for_file(file_path, timeout=36000):
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            return False
        time.sleep(1)
    return True

def wait_for_files(file_path_list, timeout=36000):
    start_time = time.time()
    while not any([os.path.exists(fp) for fp in file_path_list]):
        if time.time() - start_time > timeout:
            return False
        time.sleep(1)
    return True

def load_data(dataset_name, subtask=""):
    if dataset_name == "lila":
        dataset = read_jsonl(f"dataset/lila/{subtask}/test.jsonl")
    else:
        dataset = read_jsonl(f"dataset/{dataset_name}/test.jsonl")
    return dataset

class HuskyPredictor:

    def __init__(self, model_id, model_config, root_dir, save_dir, use_action=False, use_code=False, use_math=False, use_reason=False, use_search=False, use_update=False, num_gpus=1):
        self.model_id = model_id
        self.model_config = model_config
        self.root_dir = root_dir
        self.save_dir = os.path.join(self.root_dir, save_dir)
        self.cache_dir = os.path.join(self.root_dir, save_dir, "cache")
        self.num_gpus = num_gpus
        self.use_action = use_action
        self.use_code = use_code
        self.use_math = use_math
        self.use_reason = use_reason
        self.use_search = use_search
        self.use_update = use_update
        assert(int(use_action) + int(use_code) + int(use_math) + int(use_reason) + int(use_search) + int(use_update) <= 1)
        if self.use_search:
            self.browser = GoogleSearchAPI(answer_only=False, top_k=1)
        if not self.use_update:
            self.__init_model__(model_id, model_config, num_gpus=num_gpus)

    def __init_model__(self, model_id, model_config, num_gpus=1):
        self.model = OpenAIGenerator(model_id, model_config) if model_id in MODEL_CLASSES["openai"] else VLLMGenerator(model_id, model_config, num_gpus=num_gpus)

    def init_dataset(self, dataset):
        remain_indices = []
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)  # this will also create 'save_dir'
        for data in tqdm(dataset, desc="loading data"):
            save_path = os.path.join(self.save_dir, f"q{data['index']}.json")
            initial_data = {"index": data["index"], "question": data["question"], "previous_outputs": "None", "step": None, "tool": None, "reason_output": None, "math_output": None, "code": None, "code_exec": None, "code_output": None, "search_query": None, "search_result": None, "search_output": None, "answer": None, "label": data["answer"], "history": []}
            remain_indices.append(data["index"])
            write_json_safe(initial_data, save_path)
        with open(self.get_remain_indices_path(step_idx=0), "wb") as f:
            pickle.dump(remain_indices, f)

    def get_code_indices_path(self, step_idx: int):
        return os.path.join(self.cache_dir, f"_code_indices_step{step_idx}.pkl")

    def get_search_indices_path(self, step_idx: int):
        return os.path.join(self.cache_dir, f"_search_indices_step{step_idx}.pkl")

    def get_remain_indices_path(self, step_idx: int):
        return os.path.join(self.cache_dir, f"_remain_indices_step{step_idx}.pkl")

    def get_status_file_path(self, step_idx: int, question_idx: int, tool: str):
        return os.path.join(self.cache_dir, f"step{step_idx}_qst{question_idx}_{tool}.json")

    def run(self, batch_size=32, max_iterations=6):
        # STEP 1: generate the next tool + action and identify final answers
        if self.use_action:
            step_index = 0
            if "mistral" in self.model_id:
                input_prompt = HUSKY_ACTION_GENERATOR_MISTRAL_PROMPT
            elif "llama" in self.model_id or "tulu" in self.model_id:
                input_prompt = HUSKY_ACTION_GENERATOR_TULU_PROMPT
            else:
                input_prompt = HUSKY_ACTION_GENERATOR_PROMPT
            while step_index < (max_iterations + 1):
                with open(self.get_remain_indices_path(step_index), "rb") as f:
                    remain_indices = pickle.load(f)
                action_indices_batched = batch_list(remain_indices, batch_size=batch_size)
                remain_indices_new = []
                code_indices, search_indices = [], []                
                for action_indices_batch in tqdm(action_indices_batched, desc="generating action"):
                    action_prompts_batch = []
                    remain_dataset = {}
                    for index in action_indices_batch:
                        # do not wait for anything during the first step
                        if step_index > 0:
                            # wait for the previous update to finish
                            wait_for_file(self.get_status_file_path(step_index, index, tool="update"))
                        try:
                            wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                            remain_dataset[index] = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                        except:
                            print("error from action generator")
                            print(index)
                            exit(0)
                        # build the prompt
                        question = remain_dataset[index]["question"]
                        previous_outputs = remain_dataset[index]["previous_outputs"]
                        prompt = input_prompt % (question, previous_outputs)
                        action_prompts_batch.append(prompt)
                    # generate the actions for each batch
                    action_outputs_batch = self.model.generate(action_prompts_batch)
                    for action_index, action_output in zip(action_indices_batch, action_outputs_batch):
                        # the action generator decides that the answer has been reached
                        tool_output, step_output = extract_tool_and_step(action_output)
                        if tool_output not in TOOL_LIST:
                            tool_output = "commonsense"
                        remain_dataset[action_index]["step"] = step_output.strip()
                        remain_dataset[action_index]["tool"] = f"[{tool_output.strip()}]"
                        if "finish" in tool_output:
                            answer = extract_answer(step_output)
                            remain_dataset[action_index]["answer"] = answer.strip()
                        # the action generator returns the next step to take
                        else:
                            remain_indices_new.append(action_index)
                        save_path = os.path.join(self.save_dir, f"q{action_index}.json")
                        write_json_safe(remain_dataset[action_index], save_path)
                        write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index+1, action_index, tool="action"))
                        if "code" in tool_output.strip():
                            code_indices.append(action_index)
                        if "search" in tool_output.strip():
                            search_indices.append(action_index)
                # save the new remaining indices after the answers are extracted from some of the questions
                with open(self.get_remain_indices_path(step_index + 1), "wb") as f:
                    pickle.dump(remain_indices_new, f)
                # save the indices that require code usage for the current step
                with open(self.get_code_indices_path(step_index + 1), "wb") as f:
                    pickle.dump(code_indices, f)
                # save the indices that require search usage for the current step
                with open(self.get_search_indices_path(step_index + 1), "wb") as f:
                    pickle.dump(search_indices, f)
                step_index += 1
        # STEP 2A: generate the math-based solutions
        if self.use_math:
            step_index = 1
            if "mistral" in self.model_id:
                input_prompt = HUSKY_MATH_GENERATOR_MISTRAL_PROMPT
            elif "llama" in self.model_id or "tulu" in self.model_id:
                input_prompt = HUSKY_MATH_GENERATOR_TULU_PROMPT
            else:
                input_prompt = HUSKY_MATH_GENERATOR_PROMPT
            while step_index < (max_iterations + 1):
                wait_for_file(self.get_remain_indices_path(step_index))
                with open(self.get_remain_indices_path(step_index), "rb") as f:
                    remain_indices = pickle.load(f)
                index_i, batch_count = 0, 0                
                progress_bar = tqdm(total=len(remain_indices), desc="generating solution")
                while index_i < len(remain_indices):
                    remain_dataset = {}
                    math_indices_batch, math_prompts_batch = [], []
                    index_i_prev = index_i
                    while batch_count < batch_size and index_i < len(remain_indices):
                        index = remain_indices[index_i]
                        wait_for_file(self.get_status_file_path(step_index, index, tool="action"))
                        try:
                            wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                            remain_dataset[index] = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                        except:
                            print("error from math generator")
                            print(index)
                            exit(0)
                        tool = remain_dataset[index]["tool"]
                        if "math" in tool:
                            question = remain_dataset[index]["question"]
                            previous_outputs = remain_dataset[index]["previous_outputs"]
                            current_step = remain_dataset[index]["step"]
                            prompt = input_prompt % (question, previous_outputs, current_step)
                            math_indices_batch.append(index)
                            math_prompts_batch.append(prompt)
                            batch_count += 1
                        index_i += 1
                    # generate the self-made solution for each batch
                    math_outputs_batch = self.model.generate(math_prompts_batch)
                    # update progress bar
                    progress_bar.update(index_i - index_i_prev)
                    # re-initialize batch count to 0
                    batch_count = 0
                    # save the output from each batch
                    for math_index, math_output in zip(math_indices_batch, math_outputs_batch):
                        # some models attempt to predict the next sequence of subquestions as well
                        if "\nStep:" in math_output:
                            math_output = math_output.split("\nStep:", 1)[0]
                        if "\nCurrent step:" in math_output:
                            math_output = math_output.split("\nCurrent step:", 1)[0]
                        if "\nThe answer is" in math_output:
                            math_output = math_output.split("\nThe answer is", 1)[0]
                        if "\nNext solution:" in math_output:
                            math_output = math_output.split("\nNext solution", 1)[0]
                        remain_dataset[math_index]["math_output"] = math_output.strip()
                        save_path = os.path.join(self.save_dir, f"q{math_index}.json")
                        write_json_safe(remain_dataset[math_index], save_path)
                        write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, math_index, tool="math"))
                step_index += 1
                progress_bar.close()
        # STEP 2B: generate and execute the code solutions
        if self.use_code:
            step_index = 1
            # define the prompt template to be used
            if "mistral" in self.model_id:
                input_prompt = HUSKY_CODE_GENERATOR_MISTRAL_PROMPT
            elif "tulu" in self.model_id:
                input_prompt = HUSKY_CODE_GENERATOR_TULU_PROMPT
            else:
                input_prompt = HUSKY_CODE_GENERATOR_PROMPT
            # iterate until the max number of iterations is reached
            while step_index < (max_iterations + 1):
                wait_for_file(self.get_remain_indices_path(step_index))
                with open(self.get_remain_indices_path(step_index), "rb") as f:
                    remain_indices = pickle.load(f)
                index_i, batch_count = 0, 0
                progress_bar = tqdm(total=len(remain_indices), desc="generating code")
                while index_i < len(remain_indices):
                    remain_dataset = {}
                    code_indices_batch, code_prompts_batch = [], []
                    index_i_prev = index_i
                    while batch_count < batch_size and index_i < len(remain_indices):
                        index = remain_indices[index_i]
                        wait_for_file(self.get_status_file_path(step_index, index, tool="action"))
                        try:
                            wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                            remain_dataset[index] = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                        except:
                            print("error from code generator")
                            print(index)
                            exit(0)
                        tool = remain_dataset[index]["tool"]
                        if "code" in tool:
                            question = remain_dataset[index]["question"]
                            previous_outputs = remain_dataset[index]["previous_outputs"]
                            current_step = remain_dataset[index]["step"]
                            prompt = input_prompt % (question, previous_outputs, current_step)
                            code_indices_batch.append(index)
                            code_prompts_batch.append(prompt)
                            batch_count += 1
                        index_i += 1
                    # generate the self-made solution for each batch
                    code_outputs_batch = self.model.generate(code_prompts_batch)
                    # update progress bar
                    progress_bar.update(index_i - index_i_prev)
                    # re-initialize batch count to 0
                    batch_count = 0
                    # execute each code batch and save the output from each batch
                    for code_index, code_output in zip(code_indices_batch, code_outputs_batch):
                        remain_dataset[code_index]["code"] = code_output.strip()
                        try:
                            result = subprocess.run([sys.executable, "-c", code_prefix + "\n" + code_output.strip()], capture_output=True, text=True, timeout=10)
                            if result.stderr.strip() == "":
                                code_exec = simplify_answer(result.stdout, convert_to_str=True).strip()
                            else:
                                code_exec = ""
                        except:
                            code_exec = ""
                        remain_dataset[code_index]["code_exec"] = code_exec.strip()
                        save_path = os.path.join(self.save_dir, f"q{code_index}.json")
                        write_json_safe(remain_dataset[code_index], save_path)
                        write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, code_index, tool="code"))
                step_index += 1
                progress_bar.close()
        # STEP 2C: generate and execute the search-based solutions
        if self.use_search:
            step_index = 1
            # define the prompt template to be used
            if "mistral" in self.model_id:
                input_prompt = HUSKY_QUERY_GENERATOR_MISTRAL_PROMPT
            elif "llama" in self.model_id or "tulu" in self.model_id:
                input_prompt = HUSKY_QUERY_GENERATOR_TULU_PROMPT
            else:
                input_prompt = HUSKY_QUERY_GENERATOR_PROMPT
            while step_index < (max_iterations + 1):
                wait_for_file(self.get_remain_indices_path(step_index))
                with open(self.get_remain_indices_path(step_index), "rb") as f:
                    remain_indices = pickle.load(f)
                index_i, batch_count = 0, 0
                progress_bar = tqdm(total=len(remain_indices), desc="generating queries")
                while index_i < len(remain_indices):
                    remain_dataset = {}
                    search_indices_batch, search_prompts_batch = [], []
                    index_i_prev = index_i
                    while batch_count < batch_size and index_i < len(remain_indices):
                        index = remain_indices[index_i]
                        wait_for_file(self.get_status_file_path(step_index, index, tool="action"))
                        try:
                            wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                            remain_dataset[index] = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                        except:
                            print("error from query generator")
                            print(index)
                            exit(0)
                        tool = remain_dataset[index]["tool"]
                        if "search" in tool or "match_info" in tool:
                            question = remain_dataset[index]["question"]
                            previous_outputs = remain_dataset[index]["previous_outputs"]
                            current_step = remain_dataset[index]["step"]
                            prompt = input_prompt % (question, previous_outputs, current_step)
                            search_indices_batch.append(index)
                            search_prompts_batch.append(prompt)
                            batch_count += 1
                        index_i += 1
                    # generate the search queries for each batch
                    search_queries_batch = self.model.generate(search_prompts_batch)
                    # update progress bar
                    progress_bar.update(index_i - index_i_prev)
                    # re-initialize batch count to 0
                    batch_count = 0
                    # execute each code batch and save the output from each batch
                    for search_index, search_query in zip(search_indices_batch, search_queries_batch):
                        # print(f"Search query: {search_query}")
                        # x = input("continue: ")
                        remain_dataset[search_index]["search_query"] = search_query.strip()
                        search_tool = remain_dataset[search_index]["tool"]
                        if "match_info" in search_tool:
                            search_result = self.browser.search(search_query, use_date=False, use_match_info=True)
                            remain_dataset[search_index]["search_output"] = search_result.strip() if search_result is not None else "None"
                        else:
                            search_result = self.browser.search(search_query, use_date=False, use_match_info=False)
                            remain_dataset[search_index]["search_result"] = search_result.strip() if search_result is not None else "Search failed."
                        save_path = os.path.join(self.save_dir, f"q{search_index}.json")
                        write_json_safe(remain_dataset[search_index], save_path)
                        if "match_info" in search_tool:
                            write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, search_index, tool="match_info"))
                        else:
                            write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, search_index, tool="search"))
                step_index += 1
                progress_bar.close()
        # STEP 2D + 3: generate and execute the reasoning-based solutions, and rewrite the code outputs into natural language
        if self.use_reason:
            step_index = 1
            # define the prompt template to be used
            if "mistral" in self.model_id:
                input_reasoning_prompt = HUSKY_REASONING_GENERATOR_FEWSHOT_MISTRAL_PROMPT
                input_code_prompt = HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_MISTRAL_PROMPT
                input_search_prompt = HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_MISTRAL_PROMPT
            elif "tulu" in self.model_id:
                input_reasoning_prompt = HUSKY_REASONING_GENERATOR_FEWSHOT_TULU_PROMPT
                input_code_prompt = HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_TULU_PROMPT
                input_search_prompt = HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_TULU_PROMPT
            else:
                input_reasoning_prompt = HUSKY_REASONING_GENERATOR_FEWSHOT_PROMPT
                input_code_prompt = HUSKY_CODE_OUTPUT_WRITER_FEWSHOT_PROMPT
                input_search_prompt = HUSKY_SEARCH_OUTPUT_WRITER_FEWSHOT_PROMPT
            while step_index < (max_iterations + 1):
                wait_for_file(self.get_remain_indices_path(step_index))
                with open(self.get_remain_indices_path(step_index), "rb") as f:
                    remain_indices = pickle.load(f)
                # first, perform any reasoning that must be completed by the base model
                index_i, batch_count = 0, 0
                progress_bar = tqdm(total=len(remain_indices), desc="generating reasoning")
                while index_i < len(remain_indices):
                    remain_dataset = {}
                    reason_indices_batch, reason_prompts_batch = [], []
                    index_i_prev = index_i
                    while batch_count < batch_size and index_i < len(remain_indices):
                        index = remain_indices[index_i]
                        wait_for_file(self.get_status_file_path(step_index, index, tool="action"))
                        try:
                            wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                            remain_dataset[index] = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                        except:
                            print("error from reasoning generator")
                            print(index)
                            exit(0)
                        tool = remain_dataset[index]["tool"]
                        if "commonsense" in tool:
                            question = remain_dataset[index]["question"]
                            previous_outputs = remain_dataset[index]["previous_outputs"]
                            current_step = remain_dataset[index]["step"]
                            prompt = input_reasoning_prompt % (question, previous_outputs, current_step)
                            reason_indices_batch.append(index)
                            reason_prompts_batch.append(prompt)
                            batch_count += 1
                        index_i += 1
                    reason_outputs_batch = self.model.generate(reason_prompts_batch)
                    # update progress bar
                    progress_bar.update(index_i - index_i_prev)
                    # re-initialize batch count to 0
                    batch_count = 0
                    # execute each code batch and save the output from each batch
                    for reason_index, reason_output in zip(reason_indices_batch, reason_outputs_batch):
                        if "\nStep:" in reason_output:
                            reason_output = reason_output.split("\nStep:", 1)[0]
                        if "\nCurrent step:" in reason_output:
                            reason_output = reason_output.split("\nCurrent step:", 1)[0]
                        if "\nThe answer is" in reason_output:
                            reason_output = reason_output.split("\nThe answer is", 1)[0]
                        if "\nNext solution:" in reason_output:
                            reason_output = reason_output.split("\nNext solution", 1)[0]
                        remain_dataset[reason_index]["reason_output"] = reason_output.strip()
                        save_path = os.path.join(self.save_dir, f"q{reason_index}.json")
                        write_json_safe(remain_dataset[reason_index], save_path)
                        write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, reason_index, tool="reason"))
                progress_bar.close()
                # second, perform any rewriting that must be completed by the base model
                progress_bar = tqdm(total=len(remain_indices), desc="rewriting code+search outputs")
                # rewrite the code and search outputs together
                rewrite_data_batch, rewrite_indices_batch, rewrite_prompts_batch = [], [], []
                index_i_prev = 0
                for index_i, index in enumerate(remain_indices):
                    # wait until the given question at the given step is executed by one of the tools
                    wait_for_files([
                        self.get_status_file_path(step_index, index, tool="math"),
                        self.get_status_file_path(step_index, index, tool="reason"),
                        self.get_status_file_path(step_index, index, tool="code"),
                        self.get_status_file_path(step_index, index, tool="search"),
                        self.get_status_file_path(step_index, index, tool="match_info"),
                        ])
                    try:
                        wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                        remain_data = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                    except Exception as e:
                        print("error from rewriter")
                        print(index)
                        print(e)
                        exit(0)
                    question = remain_data["question"]
                    current_step = remain_data["step"]
                    tool = remain_data["tool"]
                    if "code" in tool:
                        code = remain_data["code"]
                        code_output = remain_data["code_exec"]
                        prompt = input_code_prompt % (question, current_step, code, code_output)
                        rewrite_data_batch.append(remain_data)
                        rewrite_indices_batch.append(index)
                        rewrite_prompts_batch.append(prompt)
                    elif "search" in tool:
                        search_result = remain_data["search_result"]
                        prompt = input_search_prompt % (current_step, search_result)
                        rewrite_data_batch.append(remain_data)
                        rewrite_indices_batch.append(index)
                        rewrite_prompts_batch.append(prompt)
                    # perform generation if batch size is reached, or the end of the current round of indices has been reached
                    if len(rewrite_prompts_batch) == batch_size or index_i == len(remain_indices) - 1:
                        rewrite_outputs_batch = self.model.generate(rewrite_prompts_batch)
                        for rewrite_index, rewrite_data, rewrite_output in zip(rewrite_indices_batch, rewrite_data_batch, rewrite_outputs_batch):
                            # print(rewrite_output)
                            # x = input("continue: ")
                            save_path = os.path.join(self.save_dir, f"q{rewrite_index}.json")
                            if "code" in rewrite_data["tool"]:
                                rewrite_data["code_output"] = rewrite_output.strip()
                                write_json_safe(rewrite_data, save_path)
                                write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, rewrite_index, tool="code_write"))
                            elif "search" in rewrite_data["tool"]:
                                rewrite_data["search_output"] = rewrite_output.strip()
                                write_json_safe(rewrite_data, save_path)
                                write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, rewrite_index, tool="search_write"))
                        # update the progress bar
                        progress_bar.update((index_i + 1) - index_i_prev)
                        index_i_prev = index_i + 1
                        # re-initialize the individual batches
                        rewrite_data_batch = []
                        rewrite_indices_batch = []
                        rewrite_prompts_batch = []
                # handle remainder cases
                if len(rewrite_prompts_batch) > 0:
                    rewrite_outputs_batch = self.model.generate(rewrite_prompts_batch)
                    for rewrite_index, rewrite_data, rewrite_output in zip(rewrite_indices_batch, rewrite_data_batch, rewrite_outputs_batch):
                        save_path = os.path.join(self.save_dir, f"q{rewrite_index}.json")
                        if "code" in rewrite_data["tool"]:
                            rewrite_data["code_output"] = rewrite_output.strip()
                            write_json_safe(rewrite_data, save_path)
                            write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, rewrite_index, tool="code_write"))
                        elif "search" in rewrite_data["tool"]:
                            rewrite_data["search_output"] = rewrite_output.strip()
                            write_json_safe(rewrite_data, save_path)
                            write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, rewrite_index, tool="search_write"))
                    progress_bar.update((index_i + 1) - index_i_prev)
                progress_bar.close()
                step_index += 1
        # STEP 5: Update previous outputs and history
        if self.use_update:
            step_index = 1
            while step_index < (max_iterations + 1):
                wait_for_file(self.get_code_indices_path(step_index))
                wait_for_file(self.get_search_indices_path(step_index))
                wait_for_file(self.get_remain_indices_path(step_index))
                with open(self.get_remain_indices_path(step_index), "rb") as f:
                    remain_indices = pickle.load(f)
                for index in tqdm(remain_indices, desc="updating outputs"):
                    wait_for_files([
                        self.get_status_file_path(step_index, index, tool="math"),
                        self.get_status_file_path(step_index, index, tool="reason"),
                        self.get_status_file_path(step_index, index, tool="code_write"),
                        self.get_status_file_path(step_index, index, tool="search_write"),
                        self.get_status_file_path(step_index, index, tool="match_info"),
                        ])
                    try:
                        wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                        remain_data = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                    except Exception as e:
                        print("error from updater")
                        print(index)
                        print(e)
                        exit(0)
                    previous_outputs = remain_data["previous_outputs"]
                    if previous_outputs == "None":
                        previous_outputs = ""
                    current_step = remain_data["step"]
                    tool = remain_data["tool"]
                    if "code" in tool:
                        if remain_data["code_output"] is not None:
                            current_output = remain_data["code_output"].strip()
                        else:
                            current_output = "This step has failed."
                    elif "search" in tool or "match_info" in tool:
                        if remain_data["search_output"] is not None:
                            current_output = remain_data["search_output"].strip()
                        else:
                            current_output = "This step has failed."
                    elif "math" in tool:
                        if remain_data["math_output"] is not None:
                            current_output = remain_data["math_output"].strip()
                        else:
                            current_output = "This step has failed."
                    else:
                        if remain_data["reason_output"] is not None:
                            current_output = remain_data["reason_output"].strip()
                        else:
                            current_output = "This step has failed."
                    history = {"step": current_step, "tool": tool, "output": current_output}
                    remain_data["history"].append(history)
                    previous_outputs = '\n'.join([previous_outputs, f"Step: {current_step.strip()}\nOutput: {current_output.strip()}"]).strip()
                    remain_data["previous_outputs"] = previous_outputs
                    # save the bookkeeping data structures
                    save_path = os.path.join(self.save_dir, f"q{index}.json")
                    write_json_safe(remain_data, save_path)
                    write_json_safe({"complete": "yes"}, self.get_status_file_path(step_index, index, tool="update"))
                step_index += 1
        # STEP 6: assign answers to the unanswered questions
        if self.use_action:
            remain_dataset = {}
            with open(self.get_remain_indices_path(max_iterations), "rb") as f:
                remain_indices = pickle.load(f)
            for index in tqdm(remain_indices, desc="extracting answers"):
                try:
                    wait_for_file(os.path.join(self.save_dir, f"q{index}.json"))
                    remain_dataset[index] = read_json_safe(os.path.join(self.save_dir, f"q{index}.json"))
                except:
                    print("error from answer extractor")
                    print(index)
                    exit(0)
                tool = remain_dataset[index]["tool"]
                if "code" in tool:  # use the latest code-executed output
                    code_output = remain_dataset[index]["code_exec"]
                    remain_dataset[index]["answer"] = code_output
                else:  # answer is unknown
                    remain_dataset[index]["answer"] = ""
                save_path = os.path.join(self.save_dir, f"q{index}.json")
                write_json_safe(remain_dataset[index], save_path)


if __name__ == "__main__":

    args = parse_args()

    dataset_name = args.dataset_name
    subtask = args.subtask
    if subtask == "none":
        subtask = ""
    split = args.split
    num_samples = args.num_samples
    if num_samples == 0:
        num_samples = None

    model_id = args.model_id
    model_config = {
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": 1,
        "n": 1,
        "logprobs": None,
        "quote": "---"
    }

    root_dir = args.root_dir
    save_dir = args.save_dir
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    max_iterations = args.max_iterations

    use_action = args.use_action
    use_code = args.use_code
    use_math = args.use_math
    use_reason = args.use_reason
    use_search = args.use_search
    use_update = args.use_update

    dataset = load_data(dataset_name, subtask=subtask, split=split, num_samples=num_samples)
    predictor = HuskyPredictor(model_id, model_config, root_dir, save_dir, use_action=use_action, use_code=use_code, use_math=use_math, use_reason=use_reason, use_search=use_search, use_update=use_update, num_gpus=num_gpus)
    if use_action:
        predictor.init_dataset(dataset)
    predictor.run(batch_size=batch_size, max_iterations=max_iterations)
