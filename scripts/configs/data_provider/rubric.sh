#!/bin/bash
# Data provider config: rubric
# First generates a rubric for the question, then:
# - Chosen answer: conditioned on question + rubric (should be higher quality)
# - Rejected answer: conditioned on question only (no rubric)

REJECTED_ANSWER_METHOD="rubric"
# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rub"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_rub"
fi

echo "[data_provider/rubric] REJECTED_ANSWER_METHOD=${REJECTED_ANSWER_METHOD}, EXP_NAME_BASE=${EXP_NAME_BASE}"

