import backoff
import json
import openai
import os

from copy import deepcopy
from vllm import LLM, SamplingParams

from model_to_path import MODEL_TO_PATH

CACHE_DIR="./models/.cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

OPENAI_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]

with open("keys/openai_key.json", "r") as f:
    openai_data = json.load(f)
    openai.api_key = openai_data["api_key"]
    openai.organization = openai_data["organization"]

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def loop_generate(inputs, model: LLM, sampling_params):
    output_resp = []
    for input in inputs:
        resp = model.generate([input], sampling_params, use_tqdm=False)
        if len(resp) == 0 or len(resp[0].outputs[0].text.strip()) == 0:
            output_resp += ["[NULL]"]
        else:
            output_resp += resp
    return output_resp

class CompletionOutput:

    def __init__(self, text):
        self.text = text
        self.token_ids = []
        self.logprobs = {}

class OpenAIGenerator:

    def __init__(self, model_id, hyperparam_info):
        self.model_id = model_id
        self.hyperparam_info = hyperparam_info
    
    def generate(self, inputs: list[str], max_new_tokens=None) -> list[str]:
        resp = []
        if len(inputs) == 0:
            return resp
        if self.model_id in OPENAI_MODELS:
            outputs = []
            for input in inputs:
                try:  # perform chat completion
                    res = chat_completions_with_backoff(
                        model=self.model_id,
                        messages=[{"role": "user", "content": input}],
                        max_tokens=max_new_tokens if max_new_tokens is not None else self.hyperparam_info["max_new_tokens"],
                        temperature=self.hyperparam_info["temperature"],
                        top_p=self.hyperparam_info["top_p"],
                        n=self.hyperparam_info["n"],
                        stop=self.hyperparam_info["quote"]
                    )
                    if self.hyperparam_info["n"] == 1:
                        outputs.append(res["choices"][0]["message"]["content"])
                    else:
                        output_list = [r["message"]["content"] for r in res["choices"]]
                        outputs.append(output_list)
                except:  # in this case, the length of the input sequence exceeded the maximum length
                    try:
                        res = chat_completions_with_backoff(
                            model=self.model_id if self.model_id in ["gpt-4", "gpt-4-1106-preview"] else "gpt-3.5-turbo-16k",
                            messages=[{"role": "user", "content": input}],
                            max_tokens=max_new_tokens if max_new_tokens is not None else self.hyperparam_info["max_new_tokens"],
                            temperature=self.hyperparam_info["temperature"],
                            top_p=self.hyperparam_info["top_p"],
                            n=self.hyperparam_info["n"],
                            stop=self.hyperparam_info["quote"]
                        )
                        if self.hyperparam_info["n"] == 1:
                            outputs.append(res["choices"][0]["message"]["content"])
                        else:
                            output_list = [r["message"]["content"] for r in res["choices"]]
                            outputs.append(output_list)
                    except:
                        outputs.append("")
        else:
            try:
                resp = completions_with_backoff(
                    model=self.model_id,
                    prompt=inputs,
                    max_tokens=max_new_tokens if max_new_tokens is not None else self.hyperparam_info["max_new_tokens"],
                    temperature=self.hyperparam_info["temperature"],
                    top_p=self.hyperparam_info["top_p"],
                    n=self.hyperparam_info["n"],
                    stop=self.hyperparam_info["quote"],
                    logprobs = self.hyperparam_info["logprobs"]
                )
                outputs = [x["text"] for x in resp['choices']]
            except:
                outputs = []
                for input in inputs:
                    try:
                        res = chat_completions_with_backoff(
                            model=self.model_id,
                            messages=[{"role": "user", "content": input}],
                            max_tokens=max_new_tokens if max_new_tokens is not None else self.hyperparam_info["max_new_tokens"],
                            temperature=self.hyperparam_info["temperature"],
                            top_p=self.hyperparam_info["top_p"],
                            n=self.hyperparam_info["n"],
                            stop=self.hyperparam_info["quote"]
                        )
                        outputs.append(res["choices"][0]["message"]["content"])
                    except:
                        outputs.append("")
        return outputs

class VLLMGenerator:

    def __init__(self, model_id, hyperparam_info, sampling_params=None, use_fast_tokenizer=True, num_gpus=1):
        if use_fast_tokenizer:
            if num_gpus > 1:
                self.model = LLM(model=MODEL_TO_PATH[model_id], download_dir=CACHE_DIR, tensor_parallel_size=num_gpus)
            else:
                self.model = LLM(model=MODEL_TO_PATH[model_id], download_dir=CACHE_DIR)
        else:
            if num_gpus > 1:
                self.model = LLM(model=MODEL_TO_PATH[model_id], tokenizer="hf-internal-testing/llama-tokenizer", download_dir=CACHE_DIR, tensor_parallel_size=num_gpus)
            else:
                self.model = LLM(model=MODEL_TO_PATH[model_id], tokenizer="hf-internal-testing/llama-tokenizer", download_dir=CACHE_DIR)
        if sampling_params is None:
            self.sampling_params = SamplingParams(
                max_tokens=hyperparam_info["max_new_tokens"],
                n=hyperparam_info["n"], 
                temperature=hyperparam_info["temperature"],
                top_p=hyperparam_info["top_p"],
                stop=hyperparam_info["quote"],
                logprobs=hyperparam_info["logprobs"] if "logprobs" in hyperparam_info else None
            )
        else:
            self.sampling_params = sampling_params
    
    def generate(self, inputs, max_new_tokens=None, sampling_params=None):
        sampling_params = self.sampling_params if sampling_params is None else sampling_params
        sampling_params = deepcopy(sampling_params)
        if max_new_tokens is not None:
            sampling_params.max_tokens = max_new_tokens
        resp = self.model.generate(inputs, sampling_params, use_tqdm=False)
        if len(resp) < len(inputs):
            resp = loop_generate(inputs, self.model, self.sampling_params)
        answers = [x.outputs[0].text.strip() if x != "[NULL]" else x for x in resp]
        return answers
