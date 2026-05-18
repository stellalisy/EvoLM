#!/bin/bash
# Configuration for alternating training with 8-node setup using Llama-3 policy
# Llama-3.1-8B has 8192 max_position_embeddings, so we need shorter sequences
# This config sets up NUM_LEARNERS_PER_NODE for alternating training
# VLLM_NUM_ENGINES and NUM_LEARNERS_PER_NODE are doubled for two-model setup

HF_CHECKPOINT=Qwen/Qwen3-8B

# Set experiment name to include NUM_UNIQUE_PROMPTS_ROLLOUT
# Use lazy evaluation (${NUM_UNIQUE_PROMPTS_ROLLOUT}) so it's expanded later in launch.sh
# NUM_UNIQUE_PROMPTS_ROLLOUT defaults to 8 if not set
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_8nodes_llama_pol_nup\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    # Append NUM_UNIQUE_PROMPTS_ROLLOUT to existing EXP_NAME_BASE if it doesn't already contain it
    if [[ "$EXP_NAME_BASE" != *"\${NUM_UNIQUE_PROMPTS_ROLLOUT}"* ]] && [[ "$EXP_NAME_BASE" != *"nup"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_nup\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
    fi
fi

# Enable joint training mode (two-model setup)
JOINT_TRAINING=true

# Set training script for joint training
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# 8-node configuration: doubled for alternating training (two models)
# Each node will have DeepSpeed learner processes
# Total: 16 learners across 8 nodes (doubled from base config)
NUM_LEARNERS_PER_NODE="8"

# vLLM engines for two-model alternating training setup
# Must hard-assign (not conditional) because rubric_judge/* configs load earlier
# and set VLLM_NUM_ENGINES=40 (appropriate for single-model mode but too many here).
# GPU budget: 64 total = 4 judge + 2*(8 learner + 16 vLLM) + 12 headroom
# Previous: 16 judge engines → 0 headroom → engine crashes → stalls
VLLM_NUM_ENGINES=${OVERRIDDEN_VALUES[VLLM_NUM_ENGINES]:-16}

# Rubric judge engines: 4 is sufficient and leaves GPU headroom for stability
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-4}

# Rubric model configuration (conditional so CLI overrides can point to a checkpoint)
RUBRIC_MODEL=${RUBRIC_MODEL:-"Qwen/Qwen3-8B"}

# Policy training arguments
# REDUCED for Llama-3.1-8B's 8K context limit
# Max context = 8192, so max_len = max_prompt + response = 2048 + 6144 = 8192
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=64
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=8
POLICY_ARGS_RESPONSE_LENGTH=6144  # Reduced from 16384 for Llama-3's 8K limit
POLICY_ARGS_ASYNC_STEPS=4
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
)

# Announce config summary
echo "[qwen3_8b_8nodes_llama3_policy] Configuration:"
echo "  Policy vLLM engines: $VLLM_NUM_ENGINES (doubled for alternating training)"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE (doubled for alternating training)"
echo "  Rubric judge engines: $RUBRIC_JUDGE_NUM_ENGINES"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
echo "  Policy model: $HF_CHECKPOINT"
echo "  Rubric model: $RUBRIC_MODEL"
echo "  **Response length: $POLICY_ARGS_RESPONSE_LENGTH (reduced for Llama-3 8K limit)**"

ASYNC_STEPS=4

NUM_UNIQUE_PROMPTS_ROLLOUT=64

# Note: This config is specifically designed for cross-model transfer experiments
# with Llama-3.1-8B policy model, which has 8192 max_position_embeddings.
# The response_length is reduced to 6144 to fit within the 8K context limit
# (max_model_len = max_prompt_token_length + response_length = 2048 + 6144 = 8192)
