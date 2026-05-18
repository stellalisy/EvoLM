#!/bin/bash

REPLAY_BUFFER_MIN_AGE=100
REPLAY_BUFFER_MAX_AGE=300

# Use short abbreviation: g = gap
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="g${REPLAY_BUFFER_MIN_AGE}-${REPLAY_BUFFER_MAX_AGE}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"g100-300"* ]] && [[ "$EXP_NAME_BASE" != *"_g"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_g${REPLAY_BUFFER_MIN_AGE}-${REPLAY_BUFFER_MAX_AGE}"
fi
