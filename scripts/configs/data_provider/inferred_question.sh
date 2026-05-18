#!/bin/bash
# Data provider config: inferred_question
# Generates rejected answers by inferring what question the model thinks it's answering
# from the accepted answer, then generating a new rollout to that inferred question
#
# Requires inference engine to be configured (use inference_engine/qwen3_8b.sh or similar)
# Or set INFERENCE_MODEL_FOR_QUESTION_INFERENCE="rubric_judge" to use the rubric judge model

REJECTED_ANSWER_METHOD="inferred_question"
# Default to inference_engine (requires inference_engine/*.sh config to be loaded)
INFERENCE_MODEL_FOR_QUESTION_INFERENCE="${INFERENCE_MODEL_FOR_QUESTION_INFERENCE:-inference_engine}"
# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"_iq"* ]] && [[ "$EXP_NAME_BASE" != *"iq_"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_iq"
fi

echo "[data_provider/inferred_question] REJECTED_ANSWER_METHOD=${REJECTED_ANSWER_METHOD}, INFERENCE_MODEL_FOR_QUESTION_INFERENCE=${INFERENCE_MODEL_FOR_QUESTION_INFERENCE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
