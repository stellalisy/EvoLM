#!/bin/bash
# Rubric judge model config: Gemma 3 1B IT
# Small Gemma 3 text judge on a single GPU.

RUBRIC_JUDGE_MODEL="google/gemma-3-1b-it"
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=1
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.9
RUBRIC_JUDGE_MAX_MODEL_LEN=32768
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}

# Add suffix to experiment name (rjg31b = rubric judge gemma 3 1b)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjg31b"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rjg31b"* ]] && [[ "$EXP_NAME_BASE" != *"rjg"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_rjg31b"
fi

echo "[rubric_judge/gemma3_1b_it] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, EXP_NAME_BASE=${EXP_NAME_BASE}"
