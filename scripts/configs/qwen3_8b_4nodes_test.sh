#!/bin/bash
# Test configuration for combined multi-policy + multi-judge with 4 nodes
# Simplified config for comprehensive testing

HF_CHECKPOINT=Qwen/Qwen3-8B

# Set experiment name
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_4nodes_test"
fi

# Enable joint training mode
JOINT_TRAINING=true
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# 4-node test configuration
NUM_LEARNERS_PER_NODE="4"

# Engines for 4 nodes (32 GPUs total)
# Policy learners: 4 GPUs
# Rubric learners: 4 GPUs
# Policy vLLM: 8 engines
# Rubric vLLM: 8 engines
# Multi-policy frozen: 3 models × 2 engines = 6 GPUs
# Multi-judge: 2 judges × 2 engines = 4 GPUs
# Total: ~26-28 GPUs
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-8}

# Rubric model
RUBRIC_MODEL="Qwen/Qwen3-8B"

# Policy training arguments (reduced for testing)
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=8
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=4
POLICY_ARGS_RESPONSE_LENGTH=2048
POLICY_ARGS_ASYNC_STEPS=2
POLICY_ARGS_NUM_EPOCHS=1

# Alternating loop parameters (small for testing)
ALTERNATING_CYCLES=${ALTERNATING_CYCLES:-2}
ALTERNATING_STEPS_PER_PHASE=${ALTERNATING_STEPS_PER_PHASE:-10}

NUM_UNIQUE_PROMPTS_ROLLOUT=8
ASYNC_STEPS=2

# Disable HuggingFace push for testing (avoids network calls)
PUSH_TO_HUB=false

echo "[qwen3_8b_4nodes_test] Test configuration:"
echo "  Nodes: 4"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE"
echo "  vLLM engines: $VLLM_NUM_ENGINES"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
