#!/bin/bash
# Configuration for alternating training with 8-node setup in SINGLE MODEL MODE
# WITH INFERENCE MODEL for inferred_question method
# This config sets up NUM_LEARNERS_PER_NODE for alternating training
# In single_model_mode, VLLM_NUM_ENGINES combines policy + rubric engines (via single_model.sh patch)
# NUM_LEARNERS_PER_NODE remains unchanged (trainers are separate)

# Assert that single model mode is enabled when using this config
# This config requires single_model.sh to be loaded first
if [ "${SINGLE_MODEL_MODE:-false}" != "true" ]; then
    echo "ERROR: This config file (qwen3_8b_8nodes_singlemodel_inferred.sh) requires SINGLE_MODEL_MODE=true"
    echo "Please include 'single_model' config before this config in your launch command:"
    echo "  ./scripts/launch.sh alternating_training single_model ... qwen3_8b_8nodes_singlemodel_inferred"
    exit 1
fi

HF_CHECKPOINT=Qwen/Qwen3-8B

# Set experiment name to include NUM_UNIQUE_PROMPTS_ROLLOUT
# Use lazy evaluation (${NUM_UNIQUE_PROMPTS_ROLLOUT}) so it's expanded later in launch.sh
# NUM_UNIQUE_PROMPTS_ROLLOUT defaults to 8 if not set
# Use short abbreviation: n = number of unique prompts
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_8nodes_alt_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}_inferred"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    # Append NUM_UNIQUE_PROMPTS_ROLLOUT to existing EXP_NAME_BASE if it doesn't already contain it
    if [[ "$EXP_NAME_BASE" != *"\${NUM_UNIQUE_PROMPTS_ROLLOUT}"* ]] && [[ "$EXP_NAME_BASE" != *"_n"* ]] && [[ "$EXP_NAME_BASE" != *"nup"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_n\${NUM_UNIQUE_PROMPTS_ROLLOUT}_inferred"
    fi
fi

# Enable joint training mode (alternating rubric/policy training)
# In single_model_mode, both actors share the same vLLM engines
JOINT_TRAINING=true

# Set training script for joint training
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# 8-node configuration for alternating training
# Each node will have DeepSpeed learner processes
# Total: 8 learners per node (unchanged in single_model_mode)
NUM_LEARNERS_PER_NODE="8"

# vLLM engines configuration for single_model_mode
# These values will be combined by single_model.sh patch function:
# Final VLLM_NUM_ENGINES = VLLM_NUM_ENGINES + RUBRIC_JUDGE_NUM_ENGINES = 16 + 16 = 32
# This represents the combined capacity needed for both policy and rubric workloads
VLLM_NUM_ENGINES=24

# Rubric judge engines configuration (used for patch calculation)
# In single_model_mode, this is added to VLLM_NUM_ENGINES to get total engine count
RUBRIC_JUDGE_NUM_ENGINES=16

# Rubric model configuration
RUBRIC_MODEL="Qwen/Qwen3-8B"

# Inference model configuration (for question inference when using inferred_question method)
# This model is used to infer/generate questions from responses
INFERENCE_MODEL="Qwen/Qwen3-8B"
INFERENCE_NUM_ENGINES=16

# Policy training arguments
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=64
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=8
POLICY_ARGS_RESPONSE_LENGTH=16384
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
echo "[qwen3_8b_8nodes_singlemodel_inferred] Configuration:"
echo "  Single model mode: ENABLED"
echo "  Policy vLLM engines (before patch): $VLLM_NUM_ENGINES"
echo "  Rubric judge engines (before patch): $RUBRIC_JUDGE_NUM_ENGINES"
echo "  Final vLLM engines (after patch): $((VLLM_NUM_ENGINES + RUBRIC_JUDGE_NUM_ENGINES)) (combined)"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE (unchanged in single_model_mode)"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
echo "  Policy model: $HF_CHECKPOINT"
echo "  Rubric model: $RUBRIC_MODEL"
echo "  Inference model: $INFERENCE_MODEL"
echo "  Inference engines: $INFERENCE_NUM_ENGINES"
echo "  Rejected answer method: $REJECTED_ANSWER_METHOD"
echo "  Replay buffer age range: [$REPLAY_BUFFER_MIN_AGE, ${REPLAY_BUFFER_MAX_AGE:-inf}]"

ASYNC_STEPS=4

NUM_UNIQUE_PROMPTS_ROLLOUT=64

# Note: This config is designed for SINGLE MODEL MODE with INFERENCE MODEL
# Usage:
#   ./scripts/launch.sh alternating_training single_model dpo_model_ladder rubric_judge_server_vllm qwen3_8b_8nodes_singlemodel_inferred
#
# Or with tensor parallelism:
#   ./scripts/launch.sh alternating_training single_model dpo_model_ladder rubric_judge_server_vllm qwen3_8b_8nodes_singlemodel_inferred DEEPSPEED_TENSOR_PARALLEL_SIZE=2
#
# This will create:
#   - Alternating training setup in SINGLE MODEL MODE (shared vLLM engines)
#   - Combined vLLM engines: VLLM_NUM_ENGINES + RUBRIC_JUDGE_NUM_ENGINES = 56 total
#   - Inference model engines: INFERENCE_NUM_ENGINES = 8 (separate from policy/rubric)
#   - Total GPUs needed: NUM_LEARNERS_PER_NODE * num_nodes * DEEPSPEED_TENSOR_PARALLEL_SIZE
#   - vLLM engines: 56 (shared between rubric and policy actors) + 8 inference engines
