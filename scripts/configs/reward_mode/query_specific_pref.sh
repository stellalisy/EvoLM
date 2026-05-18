#!/bin/bash
# Reward mode config: query_specific_pref
# Query-specific rubrics weighted by preferred-vs-dispreferred discrimination.
#
# For each query, rubric items are generated (or read from data), then scored
# per-item on a Likert 1-10 scale (normalized to [0,1]) per Eq. 2 of the paper.
# Weights are derived from preference deltas (pref_i - dispref_i), allowing
# negative weights for error-type criteria. When dataset items include a 'weight'
# field, it is used as a prior multiplied by the preference delta.
# Fallback: dataset weights (if available), then uniform.

RUBRIC_REWARD_MODE="query_specific_pref"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="qspref"
else
    if [[ "$EXP_NAME_BASE" != *"_qspref"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_qspref"
    fi
fi

echo "[reward_mode/query_specific_pref] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
