#!/bin/bash
# Data provider config: replay_buffer
# Uses past policy rollouts from replay buffer as rejected answers (default method)

REJECTED_ANSWER_METHOD="replay_buffer"

# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"rb"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_rb"
fi

echo "[data_provider/replay_buffer] REJECTED_ANSWER_METHOD=${REJECTED_ANSWER_METHOD}, EXP_NAME_BASE=${EXP_NAME_BASE}"

