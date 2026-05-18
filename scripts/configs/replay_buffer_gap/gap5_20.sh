#!/bin/bash

REPLAY_BUFFER_MIN_AGE=5
REPLAY_BUFFER_MAX_AGE=20

# Use short abbreviation: g = gap
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="g${REPLAY_BUFFER_MIN_AGE}-${REPLAY_BUFFER_MAX_AGE}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"g5-20"* ]] && [[ "$EXP_NAME_BASE" != *"_g"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_g${REPLAY_BUFFER_MIN_AGE}-${REPLAY_BUFFER_MAX_AGE}"
fi
