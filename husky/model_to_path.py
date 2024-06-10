BASE_MODEL_TO_PATH = {
    "llama-2-7b": "MODEL_DIRECTORY_HERE",
    "llama-3-8b": "MODEL_DIRECTORY_HERE",
    "tulu-2-7b": "MODEL_DIRECTORY_HERE",
}

ACTION_MODEL_TO_PATH = {
    "action-generator-unified-llama2-7b": "MODEL_DIRECTORY_HERE",
    "action-generator-unified-llama2-13b": "MODEL_DIRECTORY_HERE",
    "action-generator-unified-llama3-8b": "MODEL_DIRECTORY_HERE",
    "action-generator-numeric-llama2-7b": "MODEL_DIRECTORY_HERE",
    "action-generator-knowledge-llama2-7b": "MODEL_DIRECTORY_HERE",
    "action-generator-tabular-llama2-7b": "MODEL_DIRECTORY_HERE",
}

CODE_MODEL_TO_PATH = {
    "code-generator-deepseekcoder-instruct": "MODEL_DIRECTORY_HERE",
    "code-generator-codetulu-7b": "MODEL_DIRECTORY_HERE",
    "code-generator-llama3-8b": "MODEL_DIRECTORY_HERE",
}

MATH_MODEL_TO_PATH = {
    "math-generator-deepseekmath-instruct": "MODEL_DIRECTORY_HERE",
    "math-generator-tulu2-7b": "MODEL_DIRECTORY_HERE",
    "math-generator-llama3-8b": "MODEL_DIRECTORY_HERE",
}

QUERY_MODEL_TO_PATH = {
    "query-generator-llama2-7b": "MODEL_DIRECTORY_HERE",
    "query-generator-llama3-8b": "MODEL_DIRECTORY_HERE",
}

MODEL_TO_PATH = {
    **BASE_MODEL_TO_PATH,
    **ACTION_MODEL_TO_PATH,
    **CODE_MODEL_TO_PATH,
    **MATH_MODEL_TO_PATH,
    **QUERY_MODEL_TO_PATH,
}