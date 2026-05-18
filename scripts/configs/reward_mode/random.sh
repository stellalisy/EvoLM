#!/bin/bash
# Reward mode config: random
# Random rewards baseline: each rollout receives a random 0 or 1 score.
# Tests whether GRPO training signal alone (without meaningful reward content)
# can improve the policy. No judge calls are made.

RUBRIC_REWARD_MODE="random"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="random"
else
    if [[ "$EXP_NAME_BASE" != *"_random"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_random"
    fi
fi

echo "[reward_mode/random] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
