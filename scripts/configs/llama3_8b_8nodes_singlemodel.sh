#!/bin/bash
# Configuration for alternating training with 8-node setup in SINGLE MODEL MODE
# Using Llama-3.1-8B-Instruct as both policy and rubric generator
# Mirrors qwen3_8b_8nodes_singlemodel.sh but with Llama-3.1-8B-Instruct

# Assert that single model mode is enabled when using this config
if [ "${SINGLE_MODEL_MODE:-false}" != "true" ]; then
    echo "ERROR: This config file (llama3_8b_8nodes_singlemodel.sh) requires SINGLE_MODEL_MODE=true"
    echo "Please include 'single_model' config before this config in your launch command:"
    echo "  ./scripts/launch.sh alternating_training single_model ... llama3_8b_8nodes_singlemodel"
    exit 1
fi

HF_CHECKPOINT=${HF_CHECKPOINT:-meta-llama/Llama-3.1-8B-Instruct}

# Set experiment name
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="llama3_8b_8nodes_alt_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *"\${NUM_UNIQUE_PROMPTS_ROLLOUT}"* ]] && [[ "$EXP_NAME_BASE" != *"_n"* ]] && [[ "$EXP_NAME_BASE" != *"nup"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
    fi
fi

JOINT_TRAINING=true
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py
NUM_LEARNERS_PER_NODE=8

# vLLM engines: same budget as Qwen3-8B single-model config
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}

# Rubric model = same as policy (single model mode)
RUBRIC_MODEL=${RUBRIC_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}

# Policy training arguments (match Qwen3-8B main run)
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=64
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=8
POLICY_ARGS_RESPONSE_LENGTH=16384
POLICY_ARGS_ASYNC_STEPS=4
POLICY_ARGS_NUM_EPOCHS=1

ALTERNATING_CYCLES=${ALTERNATING_CYCLES:-9999}

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

echo "[llama3_8b_8nodes_singlemodel] Configuration:"
echo "  Single model mode: ENABLED"
echo "  Policy vLLM engines (before patch): $VLLM_NUM_ENGINES"
echo "  Rubric judge engines (before patch): $RUBRIC_JUDGE_NUM_ENGINES"
echo "  Final vLLM engines (after patch): $((VLLM_NUM_ENGINES + RUBRIC_JUDGE_NUM_ENGINES)) (combined)"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
echo "  Policy model: $HF_CHECKPOINT"
echo "  Rubric model: $RUBRIC_MODEL"
echo "  Rejected answer method: $REJECTED_ANSWER_METHOD"
echo "  Replay buffer age range: [$REPLAY_BUFFER_MIN_AGE, ${REPLAY_BUFFER_MAX_AGE:-inf}]"

ASYNC_STEPS=4
NUM_UNIQUE_PROMPTS_ROLLOUT=64
