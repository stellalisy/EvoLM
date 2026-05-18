#!/bin/bash
# Configuration for 2-node training setup
# This config sets up NUM_LEARNERS_PER_NODE for 2 nodes


HF_CHECKPOINT=Qwen/Qwen3-8B

# Set experiment name to include NUM_UNIQUE_PROMPTS_ROLLOUT
# Use lazy evaluation (${NUM_UNIQUE_PROMPTS_ROLLOUT}) so it's expanded later in launch.sh
# NUM_UNIQUE_PROMPTS_ROLLOUT defaults to 8 if not set
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qwen3_8b_8nodes_nup\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    # Append NUM_UNIQUE_PROMPTS_ROLLOUT to existing EXP_NAME_BASE if it doesn't already contain it
    if [[ "$EXP_NAME_BASE" != *"\${NUM_UNIQUE_PROMPTS_ROLLOUT}"* ]] && [[ "$EXP_NAME_BASE" != *"nup"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_nup\${NUM_UNIQUE_PROMPTS_ROLLOUT}"
    fi
fi

# 2-node configuration: 4 learners per node (8 total learners)
# Each node will have 4 DeepSpeed learner processes
# Total: 8 learners across 2 nodes
NUM_LEARNERS_PER_NODE="8"

# Adjust vLLM engines for 2-node setup if needed
# Default is 4, but you may want to scale this based on your needs
# With 2 nodes, you might want more engines: VLLM_NUM_ENGINES=8
VLLM_NUM_ENGINES=40

RUBRIC_JUDGE_NUM_ENGINES=16

# Note: This config can be combined with other configs:
#   ./scripts/launch.sh two_node rubric_judge dpo_model_ladder
# 
# Or with tensor parallelism:
#   ./scripts/launch.sh two_node rubric_judge dpo_model_ladder DEEPSPEED_TENSOR_PARALLEL_SIZE=2
#
# This will create:
#   - Node 1: 4 learners (each using DEEPSPEED_TENSOR_PARALLEL_SIZE GPUs)
#   - Node 2: 4 learners (each using DEEPSPEED_TENSOR_PARALLEL_SIZE GPUs)
#   - Total GPUs needed: 8 * DEEPSPEED_TENSOR_PARALLEL_SIZE

NUM_UNIQUE_PROMPTS_ROLLOUT=64