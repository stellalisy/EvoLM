#!/bin/bash
# Rubric judge model config: Llama 3.2 1B Instruct
# Small text-only Llama judge on a single GPU.

RUBRIC_JUDGE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=1
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.9
RUBRIC_JUDGE_MAX_MODEL_LEN=32768
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}

# Add suffix to experiment name (rjll321b = rubric judge llama 3.2 1b)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjll321b"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rjll321b"* ]] && [[ "$EXP_NAME_BASE" != *"rjll"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_rjll321b"
fi

echo "[rubric_judge/llama3_2_1b_instruct] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, EXP_NAME_BASE=${EXP_NAME_BASE}"
