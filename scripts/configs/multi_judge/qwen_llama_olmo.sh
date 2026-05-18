#!/bin/bash
# Multi-judge configuration: 5 small judges
# Uses Qwen3-1.7B, Llama 3.2 1B Instruct, OLMo 2 1B Instruct, Gemma 3 1B IT, and Qwen3.5 2B.

multi_judge_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$multi_judge_config_dir/three_judges.sh"
source "$multi_judge_config_dir/shared_rubric_judge_settings.sh"

if [[ -z "${OVERRIDDEN_VALUES[MULTI_JUDGE_MODELS]+x}" ]]; then
    MULTI_JUDGE_MODELS="Qwen/Qwen3-1.7B,meta-llama/Llama-3.2-1B-Instruct,allenai/OLMo-2-0425-1B-Instruct,google/gemma-3-1b-it,Qwen/Qwen3-4B"
fi

if [[ " ${POST_CONFIG_FUNCTIONS[*]-} " != *" configure_multi_judge_shared_rubric_settings "* ]]; then
    POST_CONFIG_FUNCTIONS+=("configure_multi_judge_shared_rubric_settings")
fi
if [[ " ${POST_CONFIG_FUNCTIONS[*]-} " != *" configure_multi_judge_request_concurrency "* ]]; then
    POST_CONFIG_FUNCTIONS+=("configure_multi_judge_request_concurrency")
fi
