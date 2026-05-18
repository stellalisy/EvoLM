#!/bin/bash
# Configuration for alternating training with 8-node setup in SINGLE MODEL MODE.
# This variant fixes node/engine initialization semantics by using an explicit
# shared engine count and NOT relying on RUBRIC_JUDGE_NUM_ENGINES as an additive
# patch term.

# Assert single model mode.
if [ "${SINGLE_MODEL_MODE:-false}" != "true" ]; then
    echo "ERROR: This config file (qwen3_8b_8nodes_singlemodel_fixed.sh) requires SINGLE_MODEL_MODE=true"
    echo "Please include 'single_model' config before this config in your launch command:"
    echo "  ./scripts/launch.sh alternating_training single_model ... qwen3_8b_8nodes_singlemodel_fixed"
    exit 1
fi

HF_CHECKPOINT=Qwen/Qwen3-8B

# Set experiment name to include NUM_UNIQUE_PROMPTS_ROLLOUT.
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_8nodes_smfix_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
else
    if [[ "$EXP_NAME_BASE" != *"\${NUM_UNIQUE_PROMPTS_ROLLOUT}"* ]] && [[ "$EXP_NAME_BASE" != *"smfix"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_smfix_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
    fi
fi

# Alternating setup.
JOINT_TRAINING=true
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py
NUM_LEARNERS_PER_NODE="8"
RUBRIC_MODEL="Qwen/Qwen3-8B"

# Explicit shared vLLM engine count for single_model_mode.
# This is the total shared pool; no additive patching is assumed.
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-56}

# Keep rubric judge engines disabled by default in this fixed config.
# Set >0 only if you intentionally want separate auxiliary judge engines.
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-0}

# Policy training arguments.
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=64
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=8
POLICY_ARGS_RESPONSE_LENGTH=16384
POLICY_ARGS_ASYNC_STEPS=4
POLICY_ARGS_NUM_EPOCHS=1

# Alternating loop parameters.
ALTERNATING_CYCLES=${ALTERNATING_CYCLES:-9999}

# Declare prefixed variables as known (reserved for future GRPO policy session routing).
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

echo "[qwen3_8b_8nodes_singlemodel_fixed] Configuration:"
echo "  Single model mode: ENABLED"
echo "  Shared vLLM engines (explicit total): $VLLM_NUM_ENGINES"
echo "  Rubric judge auxiliary engines: $RUBRIC_JUDGE_NUM_ENGINES"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
echo "  Policy model: $HF_CHECKPOINT"
echo "  Rubric model: $RUBRIC_MODEL"

ASYNC_STEPS=4
NUM_UNIQUE_PROMPTS_ROLLOUT=64

