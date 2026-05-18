#!/bin/bash
# Alternating training config: alt20
# Sets ALTERNATING_STEPS_PER_PHASE=20 and appends to experiment name

ALTERNATING_STEPS_PER_PHASE=20

# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"alt${ALTERNATING_STEPS_PER_PHASE}"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_alt${ALTERNATING_STEPS_PER_PHASE}"
fi

echo "[alt/alt20] ALTERNATING_STEPS_PER_PHASE=${ALTERNATING_STEPS_PER_PHASE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
