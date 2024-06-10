import argparse
import numpy as np
import os
import re
import string

from collections import Counter
from tqdm import tqdm
from utils import add_to_file, read_json_safe

from math_utils import eval_math, eval_last_single_answer

def parse_args():
    parser = argparse.ArgumentParser(description="performing evaluation")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    args = parser.parse_args()
    return args

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

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
    
def evaluate_husky(dataset_name, root_dir, save_dir):
    save_dir = os.path.join(root_dir, save_dir)
    file_list = [f for f in os.listdir(save_dir) if f[0] == 'q']
    file_list = sorted(file_list, key=lambda x: int(x.split('q')[1].split('.')[0]))
    acc_list, em_list, f1_list = [], [], []
    for f in tqdm(file_list, desc="evaluating predictions"):
        data = read_json_safe(os.path.join(save_dir, f))
        if dataset_name == "gsm8k":
            if data["answer"] is not None:
                pred = extract_number_gsm8k(data["answer"])
                acc = eval_last_single_answer({"prediction": pred, "answer": data["label"]})
            else:
                acc = 0
            acc_list.append(float(acc))
        elif dataset_name == "MATH":
            if data["answer"] is not None:
                pred = extract_number_math(data["answer"])
                acc = eval_math({"prediction": pred, "answer": data["label"]})
            else:
                acc = 0
            acc_list.append(float(acc))
        elif dataset_name[:4] == "lila":
            if data["answer"] is not None:
                if data["label"].lower() in ["true", "false"]:
                    pred = data["answer"]
                    acc_str = (data["answer"].lower() == data["label"].lower())
                    acc_yn = (data["answer"].lower() == "yes" and data["label"].lower() == "true") or (data["answer"].lower() == "no" and data["label"].lower() == "false")
                    acc = int(acc_str or acc_yn)
                else:
                    pred = extract_number_math(data["answer"])
                    acc = eval_math({"prediction": pred, "answer": data["label"]})
            else:
                acc = 0
            acc_list.append(float(acc))
        elif dataset_name in ["finqa", "tatqa"]:
            if data["answer"] is not None:
                try:
                    try:
                        pred = float(extract_number_gsm8k(data["answer"]))
                    except TypeError:
                        pred = float(data["answer"])
                    try:
                        answer = float(extract_number_gsm8k(data["label"]))
                    except TypeError:
                        answer = float(data["label"])
                    pred, answer = str(pred), str(answer)
                except ValueError:
                    pred, answer = str(str(data["answer"]).lower().strip()), str(str(data["label"]).lower().strip())
                em = compute_exact_match_score(pred, answer)
                f1 = compute_f1_score(pred, answer)[0]
            else:
                em, f1 = 0, 0
            em_list.append(float(em))
            f1_list.append(float(f1))
        elif dataset_name in ["tabmwp", "mmqa"]:
            if data["answer"] is not None:
                try:
                    pred = str(float(extract_number_gsm8k(data["answer"])))
                    label = str(float(extract_number_gsm8k(str(data["label"]))))
                except ValueError:
                    pred, label = str(data["answer"].lower().strip()), str(str(data["label"]).lower().strip())
                em = compute_exact_match_score(pred, label)
                f1 = compute_f1_score(pred, label)[0]
            else:
                em, f1 = 0, 0
            em_list.append(float(em))
            f1_list.append(float(f1))
        elif dataset_name == "strategyqa":
            pred, label = normalize_answer(data["answer"]), data["label"]
            em = int(pred.startswith('yes') and label == True or pred.startswith('no') and label == False)
            em_list.append(float(em))
        elif dataset_name in ["bamboogle", "hotpotqa"]:
            pred, label = data["answer"], data["label"]
            em = compute_exact_match_score(pred, label)
            f1 = compute_f1_score(pred, label)[0]
            em_list.append(float(em))
            f1_list.append(float(f1))
        elif dataset_name in ["drop", "iirc"]:
            if data["answer"] is not None:
                try:
                    pred = str(float(extract_number_gsm8k(data["answer"])))
                    answer = str(float(extract_number_gsm8k(data["label"])))
                except ValueError:
                    pred, answer = str(data["answer"]), str(data["label"])
                em = compute_exact_match_score(pred, answer)
                f1 = compute_f1_score(pred, answer)[0]
            else:
                em, f1 = 0, 0
            em_list.append(float(em))
            f1_list.append(float(f1))
        elif dataset_name == "huskyqa":
            if data["answer"] is not None:
                try:
                    pred = str(float(extract_number_gsm8k(data["answer"])))
                    answer = str(float(extract_number_gsm8k(data["label"])))
                except ValueError:
                    pred, answer = str(data["answer"]), str(data["label"])
                acc = eval_math({"prediction": pred, "answer": answer})
            else:
                acc = 0
            acc_list.append(float(acc))
    if len(acc_list) > 0:
        acc_score = round(np.mean(acc_list, axis=0) * 100, 2)
        print(f"Acc: {acc_score}")
    if len(em_list) > 0:
        em_score = round(np.mean(em_list, axis=0) * 100, 2)
        print(f"EM: {em_score}")
    if len(f1_list) > 0:
        f1_score = round(np.mean(f1_list, axis=0) * 100, 2)
        print(f"F1: {f1_score}")
    return


if __name__ == "__main__":

    args = parse_args()

    dataset_name = args.dataset_name
    root_dir = args.root_dir
    save_dir = args.save_dir

    evaluate_husky(dataset_name, root_dir, save_dir)
