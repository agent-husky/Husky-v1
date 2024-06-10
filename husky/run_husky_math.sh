export CUDA_VISIBLE_DEVICES=2
export HF_DATASETS_CACHE="cache/huggingface"
export HF_EVALUATE_CACHE="cache/huggingface"
export HF_METRICS_CACHE="cache/huggingface"
export HF_MODULES_CACHE="cache/huggingface"
export CACHE_DIR="cache"
export TOKENIZERS_PARALLELISM=true

NUM_GPUS=1

MODEL_ID="math-generator-deepseekmath-instruct"

BATCH_SIZE=16

MAX_LENGTH=2048
MAX_NEW_TOKENS=512
TEMPERATURE=0

DATASET_NAME="gsm8k"
SUBTASK="none"

SPLIT="test"
NUM_SAMPLES=0

if [[ $DATASET_NAME == "gsm8k" ]] || [[ $DATASET_NAME == "MATH" ]]; then
    MAX_ITERATIONS=10
elif [[ $DATASET_NAME == "huskyqa" ]]; then
    MAX_ITERATIONS=8
else
    MAX_ITERATIONS=5
fi

if [[ $SUBTASK == "none" ]]; then
    ROOT_DIR="evals/${DATASET_NAME}"
else
    ROOT_DIR="evals/${DATASET_NAME}_${SUBTASK}"
fi

SAVE_DIR="DIRECTORY_TO_SAVE_RESULTS"

python run_husky.py \
    --model_id $MODEL_ID \
    --root_dir $ROOT_DIR \
    --save_dir $SAVE_DIR \
    --dataset_name $DATASET_NAME \
    --subtask $SUBTASK \
    --split $SPLIT \
    --num_samples $NUM_SAMPLES \
    --max_iterations $MAX_ITERATIONS \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --num_gpus $NUM_GPUS \
    --use_math
