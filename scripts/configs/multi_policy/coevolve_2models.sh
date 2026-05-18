#!/bin/bash
# Multi-policy CO-EVOLUTION configuration: 2 policies (1 main + 1 extra)
# Main: Qwen3-8B (standard resources), Extra: Llama-3.1-8B
# BOTH policies receive gradient updates and alternate with rubric training.

# Extra co-evolving policy (main policy is HF_CHECKPOINT, listed separately)
# Note: Use Llama-3.1 (128K context) instead of Llama-3 (8K context) to fit response_length=16384
MULTI_POLICY_COEVOLVE_MODELS="meta-llama/Llama-3.1-8B-Instruct"

# Resources per extra co-evolving policy model
MULTI_POLICY_COEVOLVE_VLLM_ENGINES=${MULTI_POLICY_COEVOLVE_VLLM_ENGINES:-4}
MULTI_POLICY_COEVOLVE_NUM_LEARNERS=${MULTI_POLICY_COEVOLVE_NUM_LEARNERS:-1}

# All policies train (not frozen)
FREEZE_POLICY_MODEL=false

# Default rejected answer method (replay buffer works well since all policies contribute)
REJECTED_ANSWER_METHOD=${REJECTED_ANSWER_METHOD:-replay_buffer}

# Declare variables as known
KNOWN_CONFIG_VARIABLES=(
    FREEZE_POLICY_MODEL
)
