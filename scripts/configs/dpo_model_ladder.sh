#!/bin/bash
# Configuration for scottgeng00/dpo_model_ladder dataset

# Dataset configuration
DATASET_MIXER_LIST="scottgeng00/dpo_model_ladder 1.0"
DATASET_MIXER_LIST_SPLITS="train"
DATASET_MIXER_EVAL_LIST="scottgeng00/dpo_model_ladder 1.0"
DATASET_MIXER_EVAL_LIST_SPLITS="train"

# Augment experiment name with dataset identifier using lazy evaluation
# Append "_dpo" to EXP_NAME_BASE if it doesn't already contain it
# Only modify if EXP_NAME_BASE was NOT provided as an override
if [ -n "$EXP_NAME_BASE" ] && [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]]; then
    if [[ "$EXP_NAME_BASE" != *_dpo* ]] && [[ "$EXP_NAME_BASE" != *dpo* ]]; then
        EXP_NAME_BASE="${EXP_NAME_BASE}_dpo"
    fi
fi

QUESTION_KEY=prompt
ACCEPTED_ANSWER_KEY=qwen-2.5-72b-instruct
REJECTED_ANSWER_KEY=qwen-2.5-1.5b-instruct

# Ensure transform function is set (required for rubric_judge workflow)
# This ensures the eval dataset has the 'prompt' key (RAW_PROMPT_KEY)
DATASET_TRANSFORM_FN="rubric_judge_tokenize_v2"

# Fix for KeyError: 'prompt' - eval dataset was cached without RAW_PROMPT_KEY
# Skip cache to ensure eval dataset is regenerated with the transform function
# Set to false after first successful run to use cached dataset
DATASET_SKIP_CACHE=${DATASET_SKIP_CACHE:-true}

# Unset eval hash to force recomputation (ensures new hash includes transform function)
# This prevents loading from an old cache that was created without the transform
if [ "$DATASET_SKIP_CACHE" = "true" ]; then
    unset DATASET_CONFIG_EVAL_HASH
fi
