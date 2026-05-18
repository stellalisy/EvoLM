#!/bin/bash
# Multi-policy frozen configuration: 3 policies
# Uses Qwen3-8B, Llama-3-8B, and Mistral-7B as frozen policies

# Multi-policy frozen settings
MULTI_POLICY_MODELS="Qwen/Qwen3-8B,meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"
MULTI_POLICY_NUM_ENGINES_PER_MODEL=${MULTI_POLICY_NUM_ENGINES_PER_MODEL:-2}
MULTI_POLICY_TENSOR_PARALLEL_SIZE=1
MULTI_POLICY_SAMPLING_STRATEGY="uniform"

# Must freeze policy when using multi-policy frozen (overridable for co-evolve experiments)
FREEZE_POLICY_MODEL=${FREEZE_POLICY_MODEL:-true}

# Default rejected answer method to multi_policy_frozen (overridable for combined experiments)
REJECTED_ANSWER_METHOD=${REJECTED_ANSWER_METHOD:-multi_policy_frozen}

# Declare FREEZE_POLICY_MODEL as known (will be used in run_training via base_config.sh)
KNOWN_CONFIG_VARIABLES=(
    FREEZE_POLICY_MODEL
)

# Note: This requires rubric_judge engines for judging responses
# Make sure to configure RUBRIC_JUDGE_MODEL and RUBRIC_JUDGE_NUM_ENGINES
