#!/bin/bash
# Dataset configuration for RLVR math (verifiable math problems).
# Uses DAPO-Math-17k (Sheng et al.) which is the training dataset for the RLCER paper.
# Compatible with RLCER and other reward modes that require answer correctness signals.

# Dataset configuration - DAPO-Math-17k (processed version with standard columns)
DATASET_MIXER_LIST="open-r1/DAPO-Math-17k-Processed 1.0"
DATASET_MIXER_LIST_SPLITS="train"
DATASET_MIXER_EVAL_LIST="open-r1/DAPO-Math-17k-Processed 1.0"
DATASET_MIXER_EVAL_LIST_SPLITS="train"

# DAPO-Math-17k uses a different column layout than standard rlvr datasets:
#   source_prompt -> list of message dicts
#   reward_model.ground_truth -> answer string
#   data_source -> "math_dapo"
#
# dapo_math_rlvr_tokenize combines column remapping + rlvr tokenization in one step,
# keeping the 2-transform pattern (tokenize + filter) expected by setup_datasets().
QUESTION_KEY=prompt
DATASET_TRANSFORM_FN="dapo_math_rlvr_tokenize rlvr_max_length_filter_v1"
DATASET_SKIP_CACHE=${DATASET_SKIP_CACHE:-true}

echo "[rlvr_math] DATASET_MIXER_LIST=${DATASET_MIXER_LIST}, QUESTION_KEY=${QUESTION_KEY}"
