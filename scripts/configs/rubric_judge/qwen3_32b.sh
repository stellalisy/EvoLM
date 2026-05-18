#!/bin/bash
# Rubric judge model config: Qwen3-32B
# Largest model - highest quality, requires significant GPU resources

RUBRIC_JUDGE_MODEL="Qwen/Qwen3-32B"
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=4
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.9
RUBRIC_JUDGE_MAX_MODEL_LEN=32768
# With TP=4, each engine uses 4 GPUs. 4 engines = 16 GPUs
# This leaves room for 40 policy engines (40 GPUs) on a 64-GPU cluster
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-8}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-24}


# Add suffix to experiment name (rjq332b = rubric judge qwen3 32b)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjq332b"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rjq332b"* ]] && [[ "$EXP_NAME_BASE" != *"rjq"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_rjq332b"
fi

echo "[rubric_judge/qwen3_32b] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, EXP_NAME_BASE=${EXP_NAME_BASE}"

