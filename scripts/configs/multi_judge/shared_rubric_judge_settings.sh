#!/bin/bash

if [ -z "$ROOT_DIR" ]; then
    multi_judge_config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "$multi_judge_config_dir/../../.." && pwd)"
fi

multi_judge_rubric_config_path_for_model() {
    local model_name="$1"

    case "$model_name" in
        "Qwen/Qwen3-0.6B")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/qwen3_0.6b.sh"
            ;;
        "Qwen/Qwen3-1.7B")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/qwen3_1.7b.sh"
            ;;
        "Qwen/Qwen3-4B")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/qwen3_4b.sh"
            ;;
        "Qwen/Qwen3-8B")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/qwen3_8b.sh"
            ;;
        "Qwen/Qwen3-14B")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/qwen3_14b.sh"
            ;;
        "Qwen/Qwen3-32B")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/qwen3_32b.sh"
            ;;
        "meta-llama/Meta-Llama-3-8B-Instruct")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/llama3_8b_instruct.sh"
            ;;
        "meta-llama/Llama-3.2-1B-Instruct")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/llama3_2_1b_instruct.sh"
            ;;
        "google/gemma-3-1b-it")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/gemma3_1b_it.sh"
            ;;
        "allenai/Olmo-3-7B-Think")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/olmo3_7b_think.sh"
            ;;
        "allenai/OLMo-2-0425-1B-Instruct")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/olmo2_0425_1b_instruct.sh"
            ;;
        "Qwen/Qwen3.5-2B")
            echo "$ROOT_DIR/scripts/configs/rubric_judge/qwen3_5_2b.sh"
            ;;
        *)
            return 1
            ;;
    esac
}

multi_judge_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

multi_judge_load_rubric_settings() {
    local model_name="$1"
    local config_path
    config_path="$(multi_judge_rubric_config_path_for_model "$model_name")" || return 1

    (
        declare -A OVERRIDDEN_VALUES=()
        unset EXP_NAME_BASE
        unset RUBRIC_JUDGE_MODEL
        unset RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE
        unset RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION
        unset RUBRIC_JUDGE_MAX_MODEL_LEN
        unset RUBRIC_JUDGE_NUM_ENGINES
        unset VLLM_NUM_ENGINES
        source "$config_path" >/dev/null
        printf '%s\t%s\n' \
            "${RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION:-}" \
            "${RUBRIC_JUDGE_MAX_MODEL_LEN:-}"
    )
}

multi_judge_min_float() {
    local left="$1"
    local right="$2"
    awk -v left="$left" -v right="$right" 'BEGIN { if (left + 0 <= right + 0) print left; else print right }'
}

configure_multi_judge_shared_rubric_settings() {
    if [ -z "${MULTI_JUDGE_MODELS:-}" ]; then
        return 0
    fi

    local shared_gpu_util=""
    local shared_max_len=""
    local raw_model
    local model_name
    local loaded_settings
    local model_gpu_util
    local model_max_len
    local config_path
    local need_gpu_util=0
    local need_max_len=0

    if [[ -z "${OVERRIDDEN_VALUES[RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION]+x}" ]] \
        && [ -z "${RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION:-}" ]; then
        need_gpu_util=1
    fi
    if [[ -z "${OVERRIDDEN_VALUES[RUBRIC_JUDGE_MAX_MODEL_LEN]+x}" ]] \
        && [ -z "${RUBRIC_JUDGE_MAX_MODEL_LEN:-}" ]; then
        need_max_len=1
    fi

    if [ "$need_gpu_util" -eq 0 ] && [ "$need_max_len" -eq 0 ]; then
        return 0
    fi

    IFS=',' read -r -a multi_judge_models_array <<< "$MULTI_JUDGE_MODELS"
    for raw_model in "${multi_judge_models_array[@]}"; do
        model_name="$(multi_judge_trim "$raw_model")"
        config_path="$(multi_judge_rubric_config_path_for_model "$model_name")" || {
            echo "Error: No rubric_judge config mapping found for multi-judge model '$model_name'." >&2
            echo "Add a config under scripts/configs/rubric_judge/ and map it in shared_rubric_judge_settings.sh." >&2
            exit 1
        }
        loaded_settings="$(multi_judge_load_rubric_settings "$model_name")" || {
            echo "Error: Failed to load rubric_judge config for '$model_name' from '$config_path'." >&2
            exit 1
        }
        IFS=$'\t' read -r model_gpu_util model_max_len <<< "$loaded_settings"

        if [ "$need_gpu_util" -eq 1 ]; then
            if [ -z "$model_gpu_util" ]; then
                echo "Error: rubric_judge config '$config_path' did not set RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION." >&2
                exit 1
            fi
            if [ -z "$shared_gpu_util" ]; then
                shared_gpu_util="$model_gpu_util"
            else
                shared_gpu_util="$(multi_judge_min_float "$shared_gpu_util" "$model_gpu_util")"
            fi
        fi

        if [ "$need_max_len" -eq 1 ]; then
            if [ -z "$model_max_len" ]; then
                echo "Error: rubric_judge config '$config_path' did not set RUBRIC_JUDGE_MAX_MODEL_LEN." >&2
                exit 1
            fi
            if [ -z "$shared_max_len" ] || [ "$model_max_len" -lt "$shared_max_len" ]; then
                shared_max_len="$model_max_len"
            fi
        fi
    done

    if [ "$need_gpu_util" -eq 1 ]; then
        RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION="$shared_gpu_util"
    fi
    if [ "$need_max_len" -eq 1 ]; then
        RUBRIC_JUDGE_MAX_MODEL_LEN="$shared_max_len"
    fi

    echo "[multi_judge] shared rubric judge settings: gpu_util=${RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION:-unset}, max_len=${RUBRIC_JUDGE_MAX_MODEL_LEN:-unset}"
}

configure_multi_judge_request_concurrency() {
    if [ -z "${MULTI_JUDGE_MODELS:-}" ]; then
        return 0
    fi

    local raw_model
    local model_name
    local model_count=0
    local engines_per_judge="${MULTI_JUDGE_NUM_ENGINES_PER_JUDGE:-1}"
    local tensor_parallel_size="${MULTI_JUDGE_TENSOR_PARALLEL_SIZE:-1}"

    IFS=',' read -r -a multi_judge_models_array <<< "$MULTI_JUDGE_MODELS"
    for raw_model in "${multi_judge_models_array[@]}"; do
        model_name="$(multi_judge_trim "$raw_model")"
        if [ -n "$model_name" ]; then
            model_count=$((model_count + 1))
        fi
    done

    local judge_pool_gpus=$((engines_per_judge * tensor_parallel_size))
    local total_judge_gpus=$((model_count * judge_pool_gpus))

    # Keep high outer concurrency for efficient GPU batching (vLLM needs
    # enough pending requests to form large batches).  Per-engine depth is
    # controlled by per-judge semaphores in rubric_judge_rewards.py which cap
    # each judge at MAX_CONCURRENT_JUDGE_REQUESTS / num_judges, preventing one
    # slow judge from starving others (convoy effect) and halving the effective
    # per-engine load in the rubric phase (16/engine vs 32 without the cap).
    if [[ -z "${OVERRIDDEN_VALUES[MAX_CONCURRENT_MULTI_JUDGE_REQUESTS]+x}" ]]; then
        MAX_CONCURRENT_MULTI_JUDGE_REQUESTS=$((judge_pool_gpus * 16))
    fi
    if [[ -z "${OVERRIDDEN_VALUES[MAX_CONCURRENT_JUDGE_REQUESTS]+x}" ]]; then
        MAX_CONCURRENT_JUDGE_REQUESTS=$((total_judge_gpus * 16))
    fi
    register_command_env_var MAX_CONCURRENT_MULTI_JUDGE_REQUESTS
    register_command_env_var MAX_CONCURRENT_JUDGE_REQUESTS

    echo "[multi_judge] MAX_CONCURRENT_MULTI_JUDGE_REQUESTS=${MAX_CONCURRENT_MULTI_JUDGE_REQUESTS:-unset} "\
"(judge_pool_gpus=${judge_pool_gpus}, requests_per_gpu=16)"
    echo "[multi_judge] MAX_CONCURRENT_JUDGE_REQUESTS=${MAX_CONCURRENT_JUDGE_REQUESTS:-unset} "\
"(total_judge_gpus=${total_judge_gpus}, requests_per_gpu=16)"
}
