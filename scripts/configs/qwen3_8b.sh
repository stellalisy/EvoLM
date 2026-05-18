#!/bin/bash
# Configuration for 2-node training setup
# This config sets up NUM_LEARNERS_PER_NODE for 2 nodes


HF_CHECKPOINT=Qwen/Qwen3-8B

# 2-node configuration: 4 learners per node (8 total learners)
# Each node will have 4 DeepSpeed learner processes
# Total: 8 learners across 2 nodes
NUM_LEARNERS_PER_NODE="6"

# Adjust vLLM engines for 2-node setup if needed
# Default is 4, but you may want to scale this based on your needs
# With 2 nodes, you might want more engines: VLLM_NUM_ENGINES=8
VLLM_NUM_ENGINES=10

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

