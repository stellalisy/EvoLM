#!/bin/bash
# Inference model config: rubric_judge
# Reuses the rubric judge model for question inference
# This saves GPU memory by not loading an additional model

INFERENCE_MODEL_FOR_QUESTION_INFERENCE="rubric_judge"

# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"infrj"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_infrj"
fi

echo "[inference_model/rubric_judge] INFERENCE_MODEL_FOR_QUESTION_INFERENCE=${INFERENCE_MODEL_FOR_QUESTION_INFERENCE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
