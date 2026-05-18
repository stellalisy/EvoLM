#!/bin/bash
# API rubric judge/proposer config for RRD.
# Uses LiteLLM API path (no local rubric_judge vLLM engines).

RUBRIC_JUDGE_MODEL="gpt-4.1-mini"
RUBRIC_JUDGE_NUM_ENGINES=0

# Keep an explicit suffix in experiment names.
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rjapi41mini"
else
    if [[ "$EXP_NAME_BASE" != *"rjapi41mini"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rjapi41mini"
    fi
fi

echo "[rubric_judge/api_gpt_41_mini] RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL}, RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES}, EXP_NAME_BASE=${EXP_NAME_BASE}"
