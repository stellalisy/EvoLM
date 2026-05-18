HF_CHECKPOINT=Qwen/Qwen3-1.7B
VERBOSE=true

# Use local debug dataset (5k samples)
# To generate this dataset, run:
#   python scripts/data/create_debug_dataset.py scottgeng00/dpo_model_ladder data/debug_dataset_5k.jsonl 5000
DATASET_MIXER_LIST="data/debug_dataset_5k.jsonl 1.0"
DATASET_MIXER_LIST_SPLITS="train"
DATASET_MIXER_EVAL_LIST="data/debug_dataset_5k.jsonl 1.0"
DATASET_MIXER_EVAL_LIST_SPLITS="train"

# Use same column keys as dpo_model_ladder
QUESTION_KEY=prompt
ACCEPTED_ANSWER_KEY=qwen-2.5-72b-instruct
REJECTED_ANSWER_KEY=qwen-2.5-1.5b-instruct

# TRAINER_TENSOR_PARALLEL_SIZE=4
