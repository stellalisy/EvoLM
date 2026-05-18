#!/bin/bash
# Reward mode config: rubric_arm
# Rubric-ARM (Xu et al., 2026) — Alternating RL for Rubric-Based Reward Modeling.
#
# Two models trained in alternation:
#   - Rubric generator: generates [Hard Rule]/[Principle] rubrics from prompt only.
#     Reward R_r = I[judge predicts correct preference] (Eq. 8).
#   - Policy: evaluated via debiased pairwise judging against a reference.
#     R = 0.5 * (I[fwd picks policy] + I[rev picks policy]) (Eq. 16).
#
# No correlation filter (unlike RLCER). No pointwise scoring — holistic pairwise.
# Requires preference data (accepted/rejected answers in the dataset).

RUBRIC_REWARD_MODE="rubric_arm"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rubric_arm"
else
    if [[ "$EXP_NAME_BASE" != *"_rubric_arm"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rubric_arm"
    fi
fi

echo "[reward_mode/rubric_arm] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
