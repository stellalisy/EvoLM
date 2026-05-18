#!/bin/bash
# Enable single model mode - share vLLM engines between rubric and policy actors
# Usage: ./scripts/launch.sh alternating_training single_model dpo_model_ladder rubric_judge_server_vllm alt_training_debug

SINGLE_MODEL_MODE=true

# Add suffix to experiment name (sm = single model)
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="sm"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"sm"* ]] && [[ "$EXP_NAME_BASE" != *"single_model"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_sm"
fi
