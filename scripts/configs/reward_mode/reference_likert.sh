#!/bin/bash
# Reward mode config: reference_likert
# RaR paper baseline (Gunjal et al., 2025): Reference-guided Likert scoring.
# An LLM-as-judge compares the generated response against a reference answer
# and assigns a 1-10 Likert score, normalized to [0,1].
# Section 4.3 of https://arxiv.org/abs/2507.17746
#
# Dataset requirements:
#   - Must contain a reference/gold answer field (looked up as 'reference_answer',
#     'ground_truth', or the field specified by GROUND_TRUTHS_KEY).

RUBRIC_REWARD_MODE="reference_likert"

if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="ref_likert"
else
    if [[ "$EXP_NAME_BASE" != *"_ref_likert"* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_ref_likert"
    fi
fi

echo "[reward_mode/reference_likert] RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
