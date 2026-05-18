#!/bin/bash
# Reward mode config: rlcer_evolving
# RLCER "with evolving" (Sheng et al., 2026): Full RLCER method where both the
# Reasoner AND the Rubricator are trained.
#
# Reasoner reward:  outcome + CoT (same as rlcer)
# Rubricator reward: K_valid / K + r_format  (Eq. 10 in paper)
#   where K_valid = rubric items correlated with answer correctness
#
# Key hyperparameters (env vars):
#   RLCER_CORRELATION_THRESHOLD - rubric validity threshold (default: 0.2)
#   RLCER_OUTCOME_WEIGHT - weight for outcome reward component (default: 1.0)
#   RLCER_COT_WEIGHT - weight for CoT reward component (default: 1.0)
#
# Unlike reward_mode/rlcer (without evolving), this mode does NOT skip rubric
# model training.  The rubric model is updated in alternating phases with the
# policy model.

RUBRIC_REWARD_MODE="rlcer_evolving"
ALLOW_WORLD_PADDING=true

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rlcer_evo"
else
    if [[ "$EXP_NAME_BASE" != *"_rlcer_evo"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rlcer_evo"
    fi
fi

echo "[reward_mode/rlcer_evolving] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
