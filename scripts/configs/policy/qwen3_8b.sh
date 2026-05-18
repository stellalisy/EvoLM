#!/bin/bash
# Policy model: Qwen3-8B (default)
# This is the same model used in main experiments

HF_CHECKPOINT=Qwen/Qwen3-8B

# Append policy model identifier to experiment name if not already present
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -n "$EXP_NAME_BASE" ] && [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *"_qwen8b"* ]] && [[ "$EXP_NAME_BASE" != *"qwen"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_qwen8b"
    fi
fi
