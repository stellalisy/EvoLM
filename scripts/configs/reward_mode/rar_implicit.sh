#!/bin/bash
# Reward mode config: rar_implicit
# RaR paper method (Gunjal et al., 2025): RaR-Implicit.
# All rubric criteria along with categorical weights are passed to an
# LLM-as-judge, which produces a single holistic 1-10 Likert rating
# normalized to [0,1]. The judge handles rubric aggregation implicitly.
# This is the best-performing RaR variant in the paper.
# Section 2.2 (Implicit Aggregation) and 4.4 of https://arxiv.org/abs/2507.17746
#
# Dataset requirements:
#   - May contain a rubric field (looked up as 'rubric', 'rubrics', or
#     'rubric_items') with instance-specific rubric items.
#   - If rubric data is absent, rubric items are generated on-the-fly from
#     grouped rollout responses using the configured proposer.

RUBRIC_REWARD_MODE="rar_implicit"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rar_implicit"
else
    if [[ "$EXP_NAME_BASE" != *"_rar_implicit"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rar_implicit"
    fi
fi

echo "[reward_mode/rar_implicit] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
