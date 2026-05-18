#!/bin/bash
# Policy model: Qwen3-14B
# Larger Qwen3 variant, used for transfer experiments

HF_CHECKPOINT=Qwen/Qwen3-14B

# Append policy model identifier to experiment name if not already present
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -n "$EXP_NAME_BASE" ] && [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *"_qwen14b"* ]] && [[ "$EXP_NAME_BASE" != *"qwen14"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_qwen14b"
    fi
fi
