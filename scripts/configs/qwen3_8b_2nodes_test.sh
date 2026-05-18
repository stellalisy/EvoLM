#!/bin/bash
# Test configuration for multi-judge with 2 nodes
# Simplified config for quick testing

HF_CHECKPOINT=Qwen/Qwen3-8B

# Set experiment name
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_2nodes_test"
fi

# Enable joint training mode
JOINT_TRAINING=true
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# 2-node test configuration
NUM_LEARNERS_PER_NODE="4"

# Reduced engines for testing (2 nodes = 16 GPUs total)
# Policy learners: 4 GPUs
# Rubric learners: 4 GPUs
# Policy vLLM: 4 engines
# Rubric vLLM: 4 engines (shared with multi-judge in single model mode)
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-4}

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

echo "[qwen3_8b_2nodes_test] Test configuration:"
echo "  Nodes: 2"
echo "  Training GPUs per node: $NUM_LEARNERS_PER_NODE"
echo "  vLLM engines: $VLLM_NUM_ENGINES"
echo "  Alternating cycles: $ALTERNATING_CYCLES"
echo "  Steps per phase: $ALTERNATING_STEPS_PER_PHASE"
