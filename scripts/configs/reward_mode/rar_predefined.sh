#!/bin/bash
# Reward mode config: rar_predefined
# RaR paper method (Gunjal et al., 2025): RaR-Predefined.
# Uses a fixed set of 4 generic rubrics for all prompts with Explicit
# Aggregation (Eq. 1) and uniform weights. Each criterion is independently
# binary-judged and scores are averaged.
# Section 4.4 and Appendix A.5 of https://arxiv.org/abs/2507.17746
#
# No rubric dataset fields are required - rubrics are hardcoded.

RUBRIC_REWARD_MODE="rar_predefined"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rar_predefined"
else
    if [[ "$EXP_NAME_BASE" != *"_rar_predefined"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rar_predefined"
    fi
fi

echo "[reward_mode/rar_predefined] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
