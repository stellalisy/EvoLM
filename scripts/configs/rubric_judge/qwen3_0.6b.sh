#!/bin/bash
# Rubric judge model config: Qwen3-0.6B
# Smallest model - fast inference, lower quality

RUBRIC_JUDGE_MODEL="Qwen/Qwen3-0.6B"
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=1
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.9
RUBRIC_JUDGE_MAX_MODEL_LEN=32768
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}

# Add suffix to experiment name (rjq306b = rubric judge qwen3 0.6b)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjq306b"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rjq306b"* ]] && [[ "$EXP_NAME_BASE" != *"rjq"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_rjq306b"
fi

echo "[rubric_judge/qwen3_0.6b] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, EXP_NAME_BASE=${EXP_NAME_BASE}"

