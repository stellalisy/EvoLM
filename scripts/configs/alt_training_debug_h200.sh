#!/bin/bash
# Debug configuration for alternating/iterative training
# Single node setup with:
#   - 2 vLLM engines for policy
#   - 2 vLLM instances for reward (via rubric judge server)
#   - 2 GPUs for training

# Lock debug runs to the 1.7B policy checkpoint.
HF_CHECKPOINT=Qwen/Qwen3-1.7B
# Keep output/model metadata aligned with the debug policy checkpoint.
MODEL_ID=Qwen3-1.7B

# Append "debug" to experiment name
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="debug"
else
    # Append "_debug" to existing EXP_NAME_BASE if it doesn't already contain it
    if [[ "$EXP_NAME_BASE" != *"debug"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_debug"
    fi
fi

VERBOSE=true

# Enable joint training mode (two-model setup)
JOINT_TRAINING=true

# Set training script for joint training
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# Policy vLLM configuration
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-1}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}

# Training configuration
NUM_LEARNERS_PER_NODE=${NUM_LEARNERS_PER_NODE:-1}

RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-1}

# Inference model configuration (for question inference when using inferred_question method)
INFERENCE_MODEL=${INFERENCE_MODEL:-"meta-llama/Llama-3.2-3B-Instruct"}
INFERENCE_NUM_ENGINES=${INFERENCE_NUM_ENGINES:-1}
INFERENCE_TENSOR_PARALLEL_SIZE=1
INFERENCE_GPU_MEMORY_UTILIZATION=0.9
INFERENCE_MAX_MODEL_LEN=10000

# # Reward vLLM configuration: 2 instances for reward (via rubric judge server)
# # Override the default from rubric_judge_server.sh to use 2 instances
# VLLM_DATA_PARALLEL_SIZE=2

# Rubric model - use same model as policy for debugging
# RUBRIC_MODEL=${RUBRIC_MODEL:-"Qwen/Qwen3-1.7B"}
RUBRIC_MODEL="Qwen/Qwen3-1.7B"

# Reduce batch sizes for debugging
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=${POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT:-4}
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=${POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT:-2}
POLICY_ARGS_RESPONSE_LENGTH=${POLICY_ARGS_RESPONSE_LENGTH:-256}
POLICY_ARGS_ASYNC_STEPS=${POLICY_ARGS_ASYNC_STEPS:-1}
POLICY_ARGS_NUM_EPOCHS=${POLICY_ARGS_NUM_EPOCHS:-1}

# Alternating loop parameters - reduced for debugging
ALTERNATING_CYCLES=999
ALTERNATING_STEPS_PER_PHASE=2

# Declare prefixed variables as known (reserved for future GRPO policy session routing)
KNOWN_CONFIG_VARIABLES=(
    POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT
    POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT
    POLICY_ARGS_RESPONSE_LENGTH
    POLICY_ARGS_ASYNC_STEPS
    POLICY_ARGS_NUM_EPOCHS
    TOKENIZER_ARGS_TOKENIZER_NAME_OR_PATH
    MODEL_ARGS_MODEL_NAME_OR_PATH
    ACTIVE_SAMPLING
)

# Announce config summary
echo "[alt_training_debug] Configuration:"
echo "  Policy vLLM engines: $VLLM_NUM_ENGINES"
echo "  Training GPUs: $NUM_LEARNERS_PER_NODE"
echo "  Reward vLLM instances: $VLLM_DATA_PARALLEL_SIZE (via rubric_judge_server)"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
echo "  Policy model: $HF_CHECKPOINT"
echo "  Rubric model: $RUBRIC_MODEL"

VERBOSE=true

# Use local debug dataset (5k samples)
# To generate this dataset, run:
#   python scripts/data/create_debug_dataset.py scottgeng00/dpo_model_ladder data/debug_dataset_5k.jsonl 5000
DATASET_MIXER_LIST="data/debug_dataset_500.jsonl 1.0"
DATASET_MIXER_LIST_SPLITS="train"
DATASET_MIXER_EVAL_LIST="data/debug_dataset_500.jsonl 1.0"
DATASET_MIXER_EVAL_LIST_SPLITS="train"

# Use same column keys as dpo_model_ladder
QUESTION_KEY=prompt
ACCEPTED_ANSWER_KEY=qwen-2.5-72b-instruct
REJECTED_ANSWER_KEY=qwen-2.5-1.5b-instruct

# TRAINER_TENSOR_PARALLEL_SIZE=4

ASYNC_STEPS=2
ACTIVE_SAMPLING=false
FILTER_ZERO_STD_SAMPLES=false

# Enable world padding to handle cases where B < world_size
# ALLOW_WORLD_PADDING=true

NUM_SAMPLES_PER_PROMPT_ROLLOUT=2
NUM_UNIQUE_PROMPTS_ROLLOUT=4
