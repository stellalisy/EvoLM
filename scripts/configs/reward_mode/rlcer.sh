#!/bin/bash
# Reward mode config: rlcer
# RLCER (Sheng et al., 2026): Reinforcement Learning with CoT Supervision
# via Self-Evolving Rubrics. Uses correlation-filtered rubrics for CoT rewards
# combined with outcome rewards.
#
# Key hyperparameters (env vars):
#   RLCER_CORRELATION_THRESHOLD - rubric validity threshold (default: 0.2)
#   RLCER_OUTCOME_WEIGHT - weight for outcome reward component (default: 1.0)
#   RLCER_COT_WEIGHT - weight for CoT reward component (default: 1.0)

RUBRIC_REWARD_MODE="rlcer"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rlcer"
else
    if [[ "$EXP_NAME_BASE" != *"_rlcer"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rlcer"
    fi
fi

echo "[reward_mode/rlcer] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
