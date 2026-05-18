#!/bin/bash
# Rubric judge model config: Llama-3-8B-Instruct
# 8B single-GPU judge with the native 8k context window.

RUBRIC_JUDGE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=1
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.9
RUBRIC_JUDGE_MAX_MODEL_LEN=8192
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}

# Add suffix to experiment name (rjl38b = rubric judge llama 3 8b)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjl38b"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rjl38b"* ]] && [[ "$EXP_NAME_BASE" != *"rj"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_rjl38b"
fi

echo "[rubric_judge/llama3_8b_instruct] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, EXP_NAME_BASE=${EXP_NAME_BASE}"
