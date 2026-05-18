#!/bin/bash
# Rubric judge model config: OLMo 2 0425 1B Instruct
# Small text-only OLMo 2 judge on a single GPU.

RUBRIC_JUDGE_MODEL="allenai/OLMo-2-0425-1B-Instruct"
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=1
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.9
RUBRIC_JUDGE_MAX_MODEL_LEN=4096
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}

# Add suffix to experiment name (rjolm21b = rubric judge olmo 2 1b)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjolm21b"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rjolm21b"* ]] && [[ "$EXP_NAME_BASE" != *"rjo"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_rjolm21b"
fi

echo "[rubric_judge/olmo2_0425_1b_instruct] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, EXP_NAME_BASE=${EXP_NAME_BASE}"
