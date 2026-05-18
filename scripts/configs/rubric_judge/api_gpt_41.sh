#!/bin/bash
# API rubric judge/proposer config for RRD.
# Uses LiteLLM API path (no local rubric_judge vLLM engines).

RUBRIC_JUDGE_MODEL="gpt-4.1"
RUBRIC_JUDGE_NUM_ENGINES=0

# Keep an explicit suffix in experiment names.
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjapi41"
else
    if [[ "$EXP_NAME_BASE" != *"rjapi41"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rjapi41"
    fi
fi

echo "[rubric_judge/api_gpt_41] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES}, EXP_NAME_BASE=${EXP_NAME_BASE}"
