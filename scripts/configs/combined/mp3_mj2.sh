#!/bin/bash
# Combined configuration: Multi-Policy Frozen (3 models) + Multi-Judge (2 judges)
# This is the most comprehensive configuration for robust rubric training

combined_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$combined_config_dir/../multi_judge/shared_rubric_judge_settings.sh"

# Source multi-policy frozen config (3 policies)
# Multi-policy frozen settings
if [[ -z "${OVERRIDDEN_VALUES[MULTI_POLICY_MODELS]+x}" ]]; then
    MULTI_POLICY_MODELS="Qwen/Qwen3-8B,meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_POLICY_NUM_ENGINES_PER_MODEL]+x}" ]]; then
    MULTI_POLICY_NUM_ENGINES_PER_MODEL=2
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_POLICY_TENSOR_PARALLEL_SIZE]+x}" ]]; then
    MULTI_POLICY_TENSOR_PARALLEL_SIZE=1
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_POLICY_SAMPLING_STRATEGY]+x}" ]]; then
    MULTI_POLICY_SAMPLING_STRATEGY="uniform"
fi
if [[ -z "${OVERRIDDEN_VALUES[FREEZE_POLICY_MODEL]+x}" ]]; then
    FREEZE_POLICY_MODEL=true
fi
if [[ -z "${OVERRIDDEN_VALUES[REJECTED_ANSWER_METHOD]+x}" ]]; then
    REJECTED_ANSWER_METHOD="multi_policy_frozen"
fi

# Source multi-judge config (2 judges)
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_MODELS]+x}" ]]; then
    MULTI_JUDGE_MODELS="Qwen/Qwen3-1.7B,meta-llama/Meta-Llama-3-8B-Instruct"
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
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_ALPHA]+x}" ]]; then
    MULTI_JUDGE_ALPHA=0.7
fi
if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_BETA]+x}" ]]; then
    MULTI_JUDGE_BETA=0.3
fi

# Declare FREEZE_POLICY_MODEL as known (will be used in run_training via base_config.sh)
KNOWN_CONFIG_VARIABLES=(
    FREEZE_POLICY_MODEL
)

# Notes:
# - This configuration uses 3 diverse policies (Qwen, Llama, Mistral) for response generation
# - Responses are judged by 2 judge models (Qwen3-1.7B and Llama-3-8B)
# - Judge winner is resolved by majority vote (ties broken by mean score)
# - MULTI_JUDGE_ALPHA / BETA are only used for agreement_bonus mode
# - Shared rubric_judge engine settings are inferred from the per-model
#   rubric_judge configs unless explicitly overridden
# - GPU allocation:
#   * 3 policies × 2 engines = 6 GPUs
#   * 2 judges × 2 engines = 4 GPUs
#   * Policy/Rubric learners + vLLM = ~12-16 GPUs
#   * Total: ~22-26 GPUs (3-4 nodes)

if [[ " ${POST_CONFIG_FUNCTIONS[*]-} " != *" configure_multi_judge_shared_rubric_settings "* ]]; then
    POST_CONFIG_FUNCTIONS+=("configure_multi_judge_shared_rubric_settings")
fi
if [[ " ${POST_CONFIG_FUNCTIONS[*]-} " != *" configure_multi_judge_request_concurrency "* ]]; then
    POST_CONFIG_FUNCTIONS+=("configure_multi_judge_request_concurrency")
fi
