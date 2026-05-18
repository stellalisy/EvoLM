#!/bin/bash
# Inference model config: policy
# Uses the policy model for question inference
# Reuses policy vLLM engines, no additional model loading needed

INFERENCE_MODEL_FOR_QUESTION_INFERENCE="policy"

# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"infpol"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_infpol"
fi

echo "[inference_model/policy] INFERENCE_MODEL_FOR_QUESTION_INFERENCE=${INFERENCE_MODEL_FOR_QUESTION_INFERENCE}, EXP_NAME_BASE=${EXP_NAME_BASE}"

# Set rejected answer method to inferred_question
REJECTED_ANSWER_METHOD="inferred_question"