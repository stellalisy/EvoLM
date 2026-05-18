#!/bin/bash
# Base configuration for multi-policy co-evolution
# Sets model checkpoint and uses conditional assignments for overridable values

HF_CHECKPOINT=Qwen/Qwen3-8B

# Use conditional assignments (:-) so command-line overrides work
# These provide sensible defaults but can be overridden
NUM_LEARNERS_PER_NODE=${NUM_LEARNERS_PER_NODE:-8}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-6}
NUM_UNIQUE_PROMPTS_ROLLOUT=${NUM_UNIQUE_PROMPTS_ROLLOUT:-64}

# Set experiment name if not provided
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_mp_nup\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
fi

echo "[qwen3_8b_multipolicy] HF_CHECKPOINT=$HF_CHECKPOINT"
echo "[qwen3_8b_multipolicy] NUM_LEARNERS_PER_NODE=$NUM_LEARNERS_PER_NODE (overridable)"
echo "[qwen3_8b_multipolicy] VLLM_NUM_ENGINES=$VLLM_NUM_ENGINES (overridable)"
