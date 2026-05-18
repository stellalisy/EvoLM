#!/bin/bash
# Reward mode config: rrd_llm
# Uses full RRD procedure with LLM-assigned rubric weights.

RUBRIC_REWARD_MODE="rrd_llm"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rrdllm"
else
    if [[ "$EXP_NAME_BASE" != *"_rrdllm"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rrdllm"
    fi
fi

echo "[reward_mode/rrd_llm] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
