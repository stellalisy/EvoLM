#!/bin/bash
# Multi-judge configuration: 3 judges
# Uses Qwen3-1.7B, Llama-3-8B, and Gemma-2-9B as judges

multi_judge_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$multi_judge_config_dir/shared_rubric_judge_settings.sh"

# Multi-judge settings
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_MODELS]+x}" ]]; then
    MULTI_JUDGE_MODELS="Qwen/Qwen3-1.7B,meta-llama/Meta-Llama-3-8B-Instruct,google/gemma-2-9b-it"
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_NUM_ENGINES_PER_JUDGE]+x}" ]]; then
    MULTI_JUDGE_NUM_ENGINES_PER_JUDGE=2
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_TENSOR_PARALLEL_SIZE]+x}" ]]; then
    MULTI_JUDGE_TENSOR_PARALLEL_SIZE=1
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_AGGREGATION]+x}" ]]; then
    MULTI_JUDGE_AGGREGATION="majority_vote"
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_TIE_BREAKER]+x}" ]]; then
    MULTI_JUDGE_TIE_BREAKER="mean_score"
fi

# Reward aggregation weights (used only when MULTI_JUDGE_AGGREGATION=agreement_bonus)
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_ALPHA]+x}" ]]; then
    MULTI_JUDGE_ALPHA=0.7  # Pairwise accuracy weight
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_BETA]+x}" ]]; then
    MULTI_JUDGE_BETA=0.3   # Agreement (Kendall's tau) weight
fi

# Note: When using multi-judge, the RUBRIC_JUDGE_MODEL and RUBRIC_JUDGE_NUM_ENGINES
# settings will be ignored. The multi-judge engines replace the single judge.
if [[ " ${POST_CONFIG_FUNCTIONS[*]-} " != *" configure_multi_judge_shared_rubric_settings "* ]]; then
    POST_CONFIG_FUNCTIONS+=("configure_multi_judge_shared_rubric_settings")
fi
if [[ " ${POST_CONFIG_FUNCTIONS[*]-} " != *" configure_multi_judge_request_concurrency "* ]]; then
    POST_CONFIG_FUNCTIONS+=("configure_multi_judge_request_concurrency")
fi
