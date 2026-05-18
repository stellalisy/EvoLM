#!/bin/bash
# Policy model: Llama-3.1-8B
# Different architecture from Qwen3, used for transfer experiments
# NOTE: Llama-3.1-8B has max_position_embeddings=8192, same as Llama-3.0
# For longer contexts, use the qwen3_8b_8nodes_llama3_policy node config
# which sets POLICY_ARGS_RESPONSE_LENGTH=6144 to fit within the 8K limit

HF_CHECKPOINT=meta-llama/Llama-3.1-8B-Instruct

# Append policy model identifier to experiment name if not already present
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -n "$EXP_NAME_BASE" ] && [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *"_llama8b"* ]] && [[ "$EXP_NAME_BASE" != *"llama"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_llama8b"
    fi
fi
