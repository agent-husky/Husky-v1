export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_CACHE="cache/huggingface"
export HF_EVALUATE_CACHE="cache/huggingface"
export HF_METRICS_CACHE="cache/huggingface"
export HF_MODULES_CACHE="cache/huggingface"
export CACHE_DIR="cache"
export TOKENIZERS_PARALLELISM=true

DATASET_NAME="gsm8k"
SUBTASK="none"
if [[ $SUBTASK == "none" ]]; then
    ROOT_DIR="evals/${DATASET_NAME}"
else
    ROOT_DIR="evals/${DATASET_NAME}_${SUBTASK}"
fi

SAVE_DIR="DIRECTORY_TO_FETCH_SAVED_RESULTS"

python eval_husky.py \
    --dataset_name $DATASET_NAME \
    --root_dir $ROOT_DIR \
    --save_dir $SAVE_DIR
