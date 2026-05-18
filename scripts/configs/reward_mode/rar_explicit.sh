#!/bin/bash
# Reward mode config: rar_explicit
# RaR paper method (Gunjal et al., 2025): RaR-Explicit.
# Uses instance-specific rubrics from the dataset with Explicit Aggregation
# (Eq. 1). Each criterion is binary-judged independently, then combined via
# weighted sum using categorical weights:
#   Essential=1.0, Important=0.7, Optional=0.3, Pitfall=0.9
# Section 2.2 and 4.4 of https://arxiv.org/abs/2507.17746
#
# Dataset requirements:
#   - May contain a rubric field (looked up as 'rubric', 'rubrics', or
#     'rubric_items') with instance-specific rubric items.
#   - When rubric data is absent, rubric items are generated on-the-fly from
#     grouped rollout responses using the configured proposer.
#   - Dataset-provided rubric items should have 'description' and 'weight'
#     fields, and descriptions should use category labels
#     (Essential/Important/Optional/Pitfall) for proper weight assignment.

RUBRIC_REWARD_MODE="rar_explicit"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rar_explicit"
else
    if [[ "$EXP_NAME_BASE" != *"_rar_explicit"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_rar_explicit"
    fi
fi

echo "[reward_mode/rar_explicit] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
