#!/bin/bash
# Reward mode config: direct_likert
# RaR paper baseline (Gunjal et al., 2025): Direct Likert-scale scoring.
# An LLM-as-judge provides a direct 1-10 Likert score for each response-prompt
# pair, normalized to [0,1]. No rubrics or reference answers are used.
# Section 4.3 of https://arxiv.org/abs/2507.17746

RUBRIC_REWARD_MODE="direct_likert"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="direct_likert"
else
    if [[ "$EXP_NAME_BASE" != *"_direct_likert"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_direct_likert"
    fi
fi

echo "[reward_mode/direct_likert] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
