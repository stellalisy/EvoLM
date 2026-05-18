#!/bin/bash
# Configuration for alternating training with 8-node setup (TWO-MODEL MODE)
# Using Llama-3.1-8B-Instruct as both policy and rubric generator (separate stacks)
# Mirrors qwen3_8b_8nodes_alt_vllm_judge.sh but with Llama-3.1-8B-Instruct

HF_CHECKPOINT=${OVERRIDDEN_VALUES[HF_CHECKPOINT]:-"meta-llama/Llama-3.1-8B-Instruct"}

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="llama3_8b_8nodes_alt_nup\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *"\${NUM_UNIQUE_PROMPTS_ROLLOUT}"* ]] && [[ "$EXP_NAME_BASE" != *"nup"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_nup\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
    fi
fi

JOINT_TRAINING=true
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# 8-node two-model configuration: resources split between policy and rubric stacks
NUM_LEARNERS_PER_NODE="8"

# vLLM engines: halved vs single-model (16 instead of 40) because two model stacks
VLLM_NUM_ENGINES=${OVERRIDDEN_VALUES[VLLM_NUM_ENGINES]:-16}
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}

RUBRIC_MODEL=${OVERRIDDEN_VALUES[RUBRIC_MODEL]:-"meta-llama/Llama-3.1-8B-Instruct"}

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

echo "[llama3_8b_8nodes_alt_vllm_judge] Configuration:"
echo "  Policy vLLM engines: $VLLM_NUM_ENGINES (two-model setup)"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE"
echo "  Rubric judge engines: $RUBRIC_JUDGE_NUM_ENGINES"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
echo "  Policy model: $HF_CHECKPOINT"
echo "  Rubric model: $RUBRIC_MODEL"

ASYNC_STEPS=4
NUM_UNIQUE_PROMPTS_ROLLOUT=64
