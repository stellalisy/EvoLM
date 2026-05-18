#!/bin/bash
# Data provider config: combined
# Combines multiple data provider methods with configurable weights
# 
# Set COMBINED_DATA_PROVIDER_WEIGHTS before sourcing this config:
#   export COMBINED_DATA_PROVIDER_WEIGHTS="replay_buffer:0.5,inferred_question:0.25,rubric:0.25"
#
# To disable replay buffer completely:
#   export COMBINED_DATA_PROVIDER_WEIGHTS="inferred_question:0.5,rubric:0.5"
#
# Note: If inferred_question has positive weight, INFERENCE_MODEL_FOR_QUESTION_INFERENCE must also be set.

REJECTED_ANSWER_METHOD="combined"

# Default weights (equal distribution) - override by setting COMBINED_DATA_PROVIDER_WEIGHTS before sourcing
# Weights are auto-normalized, so 1,1,1 becomes 0.33,0.33,0.34 internally
COMBINED_DATA_PROVIDER_WEIGHTS="${COMBINED_DATA_PROVIDER_WEIGHTS:-replay_buffer:1,inferred_question:1,rubric:1}"

# If inferred_question is used, set inference model (override if needed)
# Default to inference_engine (requires inference_engine/*.sh config to be loaded)
if [[ "$COMBINED_DATA_PROVIDER_WEIGHTS" == *"inferred_question"* ]]; then
    INFERENCE_MODEL_FOR_QUESTION_INFERENCE="${INFERENCE_MODEL_FOR_QUESTION_INFERENCE:-inference_engine}"
fi

# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"combined"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_combined"
fi

echo "[data_provider/combined] REJECTED_ANSWER_METHOD=${REJECTED_ANSWER_METHOD}, COMBINED_DATA_PROVIDER_WEIGHTS=${COMBINED_DATA_PROVIDER_WEIGHTS}, EXP_NAME_BASE=${EXP_NAME_BASE}"

