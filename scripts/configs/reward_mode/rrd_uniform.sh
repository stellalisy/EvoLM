#!/bin/bash
# Reward mode config: rrd_uniform
# Uses full RRD procedure with uniform rubric weights.

RUBRIC_REWARD_MODE="rrd_uniform"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rrduni"
else
    if [[ "$EXP_NAME_BASE" != *"_rrduni"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rrduni"
    fi
fi

echo "[reward_mode/rrd_uniform] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
