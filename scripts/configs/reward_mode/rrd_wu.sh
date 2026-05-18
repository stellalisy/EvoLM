#!/bin/bash
# Reward mode config: rrd_wu
# Uses paper-style binary rubric judging with whitened-uniform weighting.

RUBRIC_REWARD_MODE="rrd_wu"

# Add suffix to experiment name unless already set
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rrdwu"
else
    if [[ "$EXP_NAME_BASE" != *"_rrdwu"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rrdwu"
    fi
fi

echo "[reward_mode/rrd_wu] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
