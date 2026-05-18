#!/bin/bash
# Rubric judge model config: OLMo-3-7B-Think
# 7B think model judge on a single GPU.

RUBRIC_JUDGE_MODEL="allenai/Olmo-3-7B-Think"
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=1
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=0.9
RUBRIC_JUDGE_MAX_MODEL_LEN=32768
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-16}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-40}

# Add suffix to experiment name (rjol37b = rubric judge olmo3 7b)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjol37b"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rjol37b"* ]] && [[ "$EXP_NAME_BASE" != *"rjo"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_rjol37b"
fi

echo "[rubric_judge/olmo3_7b_think] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, EXP_NAME_BASE=${EXP_NAME_BASE}"
