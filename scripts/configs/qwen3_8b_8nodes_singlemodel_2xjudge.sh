#!/bin/bash
# Configuration for alternating training with 8-node setup in SINGLE MODEL MODE
# with 2x rubric judge capacity relative to qwen3_8b_8nodes_singlemodel.sh.

# Assert that single model mode is enabled when using this config
# This config requires single_model.sh to be loaded first
if [ "${SINGLE_MODEL_MODE:-false}" != "true" ]; then
    echo "ERROR: This config file (qwen3_8b_8nodes_singlemodel_2xjudge.sh) requires SINGLE_MODEL_MODE=true"
    echo "Please include 'single_model' config before this config in your launch command:"
    echo "  ./scripts/launch.sh alternating_training single_model ... qwen3_8b_8nodes_singlemodel_2xjudge"
    exit 1
fi

HF_CHECKPOINT=${HF_CHECKPOINT:-Qwen/Qwen3-8B}

# Set experiment name to include NUM_UNIQUE_PROMPTS_ROLLOUT
# Use lazy evaluation (${NUM_UNIQUE_PROMPTS_ROLLOUT}) so it's expanded later in launch.sh
# NUM_UNIQUE_PROMPTS_ROLLOUT defaults to 8 if not set
# Use short abbreviation: n = number of unique prompts
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_8nodes_alt_j2x_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *"\${NUM_UNIQUE_PROMPTS_ROLLOUT}"* ]] && [[ "$EXP_NAME_BASE" != *"_n"* ]] && [[ "$EXP_NAME_BASE" != *"nup"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_j2x_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
    elif [[ "$EXP_NAME_BASE" != *"j2x"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_j2x"
    fi
fi

# Enable joint training mode (alternating rubric/policy training)
# In single_model_mode, both actors share the same vLLM engines
JOINT_TRAINING=true

# Set training script for joint training
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# 8-node configuration for alternating training
# Each node will have DeepSpeed learner processes
# Default: 8 learners per node (can be overridden for multi-policy co-evolution)
NUM_LEARNERS_PER_NODE=8

# vLLM engines configuration for single_model_mode
# Keep the full shared engine budget at 40 (same as the standard singlemodel
# config) so the policy phase does not deadlock under load.  The previous
# 24+32=56 split starved the shared engines and caused systematic vLLM
# deadlocks during policy training.  The 1.7B judge model is small enough
# (~3.4 GB vs ~16 GB for 8B) that 16 extra engines fit comfortably in the
# remaining GPU memory (total 72 vs 56).
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}

# Rubric judge engines configuration (used for patch calculation)
# In single_model_mode, this is added to VLLM_NUM_ENGINES to get total engine count
# 2x the standard 16 judge engines to support RLCER's heavy judge workload
RUBRIC_JUDGE_NUM_ENGINES=32

# Scale per-process judge request concurrency linearly with judge engine count.
# The baseline 8-node config effectively uses 128 / 8 = 16 requests per judge.
MAX_CONCURRENT_JUDGE_REQUESTS=${MAX_CONCURRENT_JUDGE_REQUESTS:-$((RUBRIC_JUDGE_NUM_ENGINES * 16))}
register_command_env_var MAX_CONCURRENT_JUDGE_REQUESTS

# Rubric model configuration
RUBRIC_MODEL="${RUBRIC_MODEL:-Qwen/Qwen3-8B}"

# Policy training arguments
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=64
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=8
POLICY_ARGS_RESPONSE_LENGTH=16384
POLICY_ARGS_ASYNC_STEPS=2
POLICY_ARGS_NUM_EPOCHS=1

# Alternating loop parameters
ALTERNATING_CYCLES=${ALTERNATING_CYCLES:-9999}

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
    MAX_CONCURRENT_JUDGE_REQUESTS
)

# Announce config summary
echo "[qwen3_8b_8nodes_singlemodel_2xjudge] Configuration:"
echo "  Single model mode: ENABLED"
echo "  Policy vLLM engines (before patch): $VLLM_NUM_ENGINES"
echo "  Rubric judge engines (before patch): $RUBRIC_JUDGE_NUM_ENGINES"
echo "  Final vLLM engines (after patch): $((VLLM_NUM_ENGINES + RUBRIC_JUDGE_NUM_ENGINES)) (combined)"
echo "  MAX_CONCURRENT_JUDGE_REQUESTS: $MAX_CONCURRENT_JUDGE_REQUESTS"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE (unchanged in single_model_mode)"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
echo "  Policy model: $HF_CHECKPOINT"
echo "  Rubric model: $RUBRIC_MODEL"
echo "  Rejected answer method: $REJECTED_ANSWER_METHOD"
echo "  Replay buffer age range: [$REPLAY_BUFFER_MIN_AGE, ${REPLAY_BUFFER_MAX_AGE:-inf}]"

ASYNC_STEPS=2

NUM_UNIQUE_PROMPTS_ROLLOUT=64

# Note: This config is designed for SINGLE MODEL MODE
