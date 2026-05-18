#!/bin/bash
# Policy model: Mistral-7B-Instruct-v0.3
# Different architecture from Qwen3 and Llama-3, used for transfer experiments

HF_CHECKPOINT=mistralai/Mistral-7B-Instruct-v0.3

# Append policy model identifier to experiment name if not already present
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -n "$EXP_NAME_BASE" ] && [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *"_mistral7b"* ]] && [[ "$EXP_NAME_BASE" != *"mistral"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_mistral7b"
    fi
fi
