#!/bin/bash
# Base configuration with defaults and training function
# This file defines default values and the run_training() function

# ============================================================================
# Default Values
# ============================================================================

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Model configuration
HF_CHECKPOINT=${HF_CHECKPOINT:-Qwen/Qwen3-8B}
MODEL_ID=${MODEL_ID:-$(basename "$HF_CHECKPOINT")}

# Experiment settings
NUM_QUESTIONS=${NUM_QUESTIONS:-3}
SEED=${SEED:-1}

# Training script selection
TRAIN_SCRIPT=${TRAIN_SCRIPT:-open_instruct/grpo_fast.py}

# Wandb settings
WANDB_PROJECT=${WANDB_PROJECT:-rubric_rl}
USE_WANDB=${USE_WANDB:-true}
WANDB_ENTITY=${WANDB_ENTITY:-}

# Training hyperparameters
LEARNING_RATE=${LEARNING_RATE:-1e-6}
BETA=${BETA:-0.001}
NUM_SAMPLES_PER_PROMPT_ROLLOUT=${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-8}
NUM_UNIQUE_PROMPTS_ROLLOUT=${NUM_UNIQUE_PROMPTS_ROLLOUT:-8}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-1}
NUM_EPOCHS=${NUM_EPOCHS:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
KL_ESTIMATOR=${KL_ESTIMATOR:-kl3}
ASYNC_STEPS=${ASYNC_STEPS:-4}
INFLIGHT_UPDATES=${INFLIGHT_UPDATES:-true}
TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP=${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP:-2.0}

# Dataset settings
DATASET_MIXER_LIST=${DATASET_MIXER_LIST:-"rulins/multi_question_synthetic_single_source_2wiki_3q 1.0"}
DATASET_MIXER_LIST_SPLITS=${DATASET_MIXER_LIST_SPLITS:-train}
DATASET_MIXER_EVAL_LIST=${DATASET_MIXER_EVAL_LIST:-"rulins/multi_question_synthetic_single_source_2wiki_3q 1.0"}
DATASET_MIXER_EVAL_LIST_SPLITS=${DATASET_MIXER_EVAL_LIST_SPLITS:-validation}
DATASET_TRANSFORM_FN=${DATASET_TRANSFORM_FN:-}
SYSTEM_PROMPT_OVERRIDE_FILE=${SYSTEM_PROMPT_OVERRIDE_FILE:-}
RUBRIC_PROMPT_KEY=${RUBRIC_PROMPT_KEY:-rubric_generation}
MAX_PROMPT_TOKEN_LENGTH=${MAX_PROMPT_TOKEN_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-16384}
PACK_LENGTH=${PACK_LENGTH:-18500}
GROUND_TRUTHS_KEY=${GROUND_TRUTHS_KEY:-ground_truth}
SFT_MESSAGES_KEY=${SFT_MESSAGES_KEY:-messages}
TOTAL_EPISODES=${TOTAL_EPISODES:-10000000}

# Model architecture
DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-3}
NUM_LEARNERS_PER_NODE=${NUM_LEARNERS_PER_NODE:-4}
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-true}

# Learning rate scheduler
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
WARM_UP_STEPS=${WARM_UP_STEPS:-0}

# Reward settings
APPLY_VERIFIABLE_REWARD=${APPLY_VERIFIABLE_REWARD:-true}
MASKED_MEAN_AXIS=${MASKED_MEAN_AXIS:-1}
OVERWRITE_REWARD_FN_TAG=${OVERWRITE_REWARD_FN_TAG:-}

# Scalar reward model settings (overrides all verifiers when set)
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-}
REWARD_MODEL_NUM_GPUS=${REWARD_MODEL_NUM_GPUS:-8}

# Rubric judge settings
RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL:-}
RUBRIC_JUDGE_TOKENIZER=${RUBRIC_JUDGE_TOKENIZER:-}
# Ray-based vLLM engine settings (optional - only used when explicitly set)
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-}
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=${RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE:-}
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=${RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION:-}
RUBRIC_JUDGE_MAX_MODEL_LEN=${RUBRIC_JUDGE_MAX_MODEL_LEN:-}
# Rubric judge sampling parameters
RUBRIC_JUDGE_TEMPERATURE=${RUBRIC_JUDGE_TEMPERATURE:-0.6}
RUBRIC_JUDGE_MAX_TOKENS=${RUBRIC_JUDGE_MAX_TOKENS:-16384}
RUBRIC_JUDGE_STOP=${RUBRIC_JUDGE_STOP:-}
RUBRIC_JUDGE_LOGPROBS=${RUBRIC_JUDGE_LOGPROBS:-1}
# Rubric training reward shaping
RUBRIC_CORRECTNESS_FOCUSED=${RUBRIC_CORRECTNESS_FOCUSED:-}
RUBRIC_REWARD_SCORE_SEPARATION=${RUBRIC_REWARD_SCORE_SEPARATION:-}
RUBRIC_REWARD_SCORE_SEPARATION_WEIGHT=${RUBRIC_REWARD_SCORE_SEPARATION_WEIGHT:-}
RUBRIC_REWARD_USE_MARGIN=${RUBRIC_REWARD_USE_MARGIN:-}
RUBRIC_FORMAT_REWARD_WEIGHT=${RUBRIC_FORMAT_REWARD_WEIGHT:-0.0}

# Inference model settings (for question inference when using inferred_question method)
INFERENCE_MODEL=${INFERENCE_MODEL:-}
# Ray-based vLLM engine settings (optional - only used when explicitly set)
INFERENCE_NUM_ENGINES=${INFERENCE_NUM_ENGINES:-}
INFERENCE_TENSOR_PARALLEL_SIZE=${INFERENCE_TENSOR_PARALLEL_SIZE:-}
INFERENCE_GPU_MEMORY_UTILIZATION=${INFERENCE_GPU_MEMORY_UTILIZATION:-}
INFERENCE_MAX_MODEL_LEN=${INFERENCE_MAX_MODEL_LEN:-}

# Generation settings
TEMPERATURE=${TEMPERATURE:-1.0}
NON_STOP_PENALTY=${NON_STOP_PENALTY:-true}
NON_STOP_PENALTY_VALUE=${NON_STOP_PENALTY_VALUE:-0.0}

# Training settings
LOCAL_EVAL_EVERY=${LOCAL_EVAL_EVERY:-10000}
SAVE_FREQ=${SAVE_FREQ:-1000}
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-3}  # -1 to keep all
CHECKPOINT_STATE_FREQ=${CHECKPOINT_STATE_FREQ:--1}
TRY_LAUNCH_BEAKER_EVAL_JOBS_ON_WEKA=${TRY_LAUNCH_BEAKER_EVAL_JOBS_ON_WEKA:-false}

# Feature flags
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
WITH_TRACKING=${WITH_TRACKING:-true}
VERBOSE=${VERBOSE:-false}
ACTIVE_SAMPLING=${ACTIVE_SAMPLING:-true}

# ============================================================================
# Joint Training Parameters (Two-Model Setup)
# ============================================================================
# These parameters are used by train_rubric_policy_joint.py for joint
# rubric and policy training. This setup uses two models (rubric and policy).

# Enable joint training mode (two-model setup)
JOINT_TRAINING=${JOINT_TRAINING:-false}

# Single model mode - when true, only one set of vLLM engines is created
# and shared between rubric and policy actors (reduces GPU memory usage)
SINGLE_MODEL_MODE=${SINGLE_MODEL_MODE:-false}

# Alternating loop parameters (for iterative joint training)
ALTERNATING_CYCLES=${ALTERNATING_CYCLES:-5}
ALTERNATING_STEPS_PER_PHASE=${ALTERNATING_STEPS_PER_PHASE:-10}
USE_BOTH_MODELS=${USE_BOTH_MODELS:-false}
# Whether to use both policy and baseline models when creating rubrics.
RUBRIC_REWARD_MODE=${RUBRIC_REWARD_MODE:-rubric_judge}
# Reward mode for alternating training:
# - rubric_judge: evaluate answers using generated rubrics (default)
# - rrd_uniform: full RRD procedure + uniform weighting
# - rrd_llm: full RRD procedure + LLM-assigned weighting
# - rrd_wu: full RRD procedure + WU weighting
# - query_specific_pref: query-specific rubrics + preference-delta weighting
# - rlcer: RLCER correlation-filtered rubrics + outcome reward
# - rlcer_evolving: RLCER with evolving rubricator
# RaR paper baselines/methods (Gunjal et al., 2025, https://arxiv.org/abs/2507.17746):
# - direct_likert: direct 1-10 Likert scoring (no rubrics)
# - reference_likert: reference-guided 1-10 Likert scoring
# - rar_predefined: fixed generic rubrics + binary explicit aggregation
# - rar_explicit: dataset or generated instance-specific rubrics + weighted binary explicit aggregation
# - rar_implicit: dataset or generated instance-specific rubrics + holistic implicit Likert scoring
# - rubric_arm: Rubric-ARM (Xu et al., 2026) alternating RL for rubric generator + pairwise judge

# Method for generating rejected answers during rubric training
# Options: "replay_buffer" (default, use past policy rollouts) or 
#          "inferred_question" (infer question from accepted answer, then generate rejected)
REJECTED_ANSWER_METHOD=${REJECTED_ANSWER_METHOD:-"replay_buffer"}
# Model to use for question inference when rejected_answer_method='inferred_question'
# Options: 'rubric_judge' (default when available), 'policy'
INFERENCE_MODEL_FOR_QUESTION_INFERENCE=${INFERENCE_MODEL_FOR_QUESTION_INFERENCE:-}
# Weights for combined data provider when rejected_answer_method='combined'
# Format: 'method:weight,method:weight,...' (weights are auto-normalized, don't need to sum to 1)
# Example: 'replay_buffer:1,inferred_question:1,rubric:1' for equal distribution
# Set a method's weight to 0 to disable it (e.g., 'replay_buffer:0,inferred_question:1,rubric:1')
COMBINED_DATA_PROVIDER_WEIGHTS=${COMBINED_DATA_PROVIDER_WEIGHTS:-}

# Replay buffer settings for age-based sampling
REPLAY_BUFFER_SIZE=${REPLAY_BUFFER_SIZE:-2048}
# Maximum size of the replay buffer for storing past policy rollouts
REPLAY_BUFFER_MIN_AGE=${REPLAY_BUFFER_MIN_AGE:-0}
# Minimum step age for sampling from replay buffer (0 = include most recent)
REPLAY_BUFFER_MAX_AGE=${REPLAY_BUFFER_MAX_AGE:-}
# Maximum step age for sampling from replay buffer (empty = no limit)

# Judge size curriculum settings
JUDGE_SIZE_CURRICULUM=${JUDGE_SIZE_CURRICULUM:-}
# Comma-separated list of model names for curriculum (largest to smallest)
# E.g., "Qwen/Qwen3-32B,Qwen/Qwen3-14B,Qwen/Qwen3-8B,Qwen/Qwen3-4B"
JUDGE_CURRICULUM_SCHEDULE=${JUDGE_CURRICULUM_SCHEDULE:-}
# Comma-separated list of cycle indices for model switches (optional)
# E.g., "0,10,20,30" (model[0] for cycles 0-9, model[1] for 10-19, etc.)
# If not set, switches are evenly distributed across total cycles

# Rubric model configuration (defaults to HF_CHECKPOINT - same as policy)
RUBRIC_MODEL=${RUBRIC_MODEL:-$HF_CHECKPOINT}

# API-based rubric generator (for baseline comparison)
# When set, rubrics are generated via litellm API calls instead of the local model
# The rubric model will NOT be trained - only the policy model receives gradients
# Example: API_RUBRIC_GENERATOR="gpt-4.1" or "azure/gpt-4"
API_RUBRIC_GENERATOR=${API_RUBRIC_GENERATOR:-}

# Freeze rubric model (for baseline comparison)
# When true, rubric model is NOT trained but still used for generation
# Rubrics come from the local model (at initialization weights), not an API
FREEZE_RUBRIC_MODEL=${FREEZE_RUBRIC_MODEL:-false}

# Freeze policy model (for rubric-only training)
# When true, policy model is NOT trained but still used for generating responses
# This allows training only the rubric generator using fixed policy outputs
# NOTE: No default set here - individual configs (e.g., multi_policy/frozen_2models.sh)
# set their own defaults. If unset, treated as false by run_training().
# FREEZE_POLICY_MODEL is set by config files that need it

# ============================================================================
# Multi-Judge Training Parameters
# ============================================================================
# These parameters enable training with multiple judge models simultaneously
# Rewards can use average-vote, majority-vote, or agreement-bonus aggregation

# Comma-separated list of judge model names for multi-judge training
# Example: "Qwen/Qwen3-1.7B,meta-llama/Meta-Llama-3-8B-Instruct,google/gemma-2-9b-it"
MULTI_JUDGE_MODELS=${MULTI_JUDGE_MODELS:-}

# Number of vLLM engines to allocate per judge model
MULTI_JUDGE_NUM_ENGINES_PER_JUDGE=${MULTI_JUDGE_NUM_ENGINES_PER_JUDGE:-1}

# Tensor parallel size for each judge engine
MULTI_JUDGE_TENSOR_PARALLEL_SIZE=${MULTI_JUDGE_TENSOR_PARALLEL_SIZE:-1}

# Aggregation strategy for multi-judge comparisons
# Options: majority_vote, average_vote, average_minus_variance, agreement_bonus, margin_kappa_format
MULTI_JUDGE_AGGREGATION=${MULTI_JUDGE_AGGREGATION:-majority_vote}

# Tie-breaker when judges split evenly
# Options: mean_score, first_judge
MULTI_JUDGE_TIE_BREAKER=${MULTI_JUDGE_TIE_BREAKER:-mean_score}

# Weight for pairwise accuracy when MULTI_JUDGE_AGGREGATION=agreement_bonus
MULTI_JUDGE_ALPHA=${MULTI_JUDGE_ALPHA:-0.7}

# Weight for agreement (Kendall's tau) when MULTI_JUDGE_AGGREGATION=agreement_bonus
MULTI_JUDGE_BETA=${MULTI_JUDGE_BETA:-0.3}

# Weights for MULTI_JUDGE_AGGREGATION=margin_kappa_format
# reward = margin_weight * avg_margin + format_weight * format + kappa_weight * fleiss_kappa
MULTI_JUDGE_MARGIN_WEIGHT=${MULTI_JUDGE_MARGIN_WEIGHT:-0.5}
MULTI_JUDGE_FORMAT_WEIGHT=${MULTI_JUDGE_FORMAT_WEIGHT:-0.3}
MULTI_JUDGE_KAPPA_WEIGHT=${MULTI_JUDGE_KAPPA_WEIGHT:-0.2}

# ============================================================================
# Multi-Policy Training Parameters
# ============================================================================
# These parameters enable training with multiple policy models
# Two modes: frozen (inference only) and co-evolution (all policies trained)

# Multi-Policy Frozen: Comma-separated list of policy models for diverse responses
# Used with FREEZE_POLICY_MODEL=true for rubric-only training
# Example: "Qwen/Qwen3-8B,meta-llama/Meta-Llama-3-8B-Instruct,mistralai/Mistral-7B-v0.1"
MULTI_POLICY_MODELS=${MULTI_POLICY_MODELS:-}

# Number of vLLM engines to allocate per policy model (frozen mode)
MULTI_POLICY_NUM_ENGINES_PER_MODEL=${MULTI_POLICY_NUM_ENGINES_PER_MODEL:-1}

# Tensor parallel size for each policy engine (frozen mode)
MULTI_POLICY_TENSOR_PARALLEL_SIZE=${MULTI_POLICY_TENSOR_PARALLEL_SIZE:-1}

# Sampling strategy for multi-policy: 'uniform', 'weighted', or 'round_robin'
MULTI_POLICY_SAMPLING_STRATEGY=${MULTI_POLICY_SAMPLING_STRATEGY:-uniform}

# Multi-Policy Co-Evolution: Comma-separated list of EXTRA policies to train jointly
# All policies alternate training with rubric generator
# The main policy (HF_CHECKPOINT) is always included; list only additional models here
# WARNING: Computationally expensive - requires GPUs for all policy trainers
# Example: "meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"
MULTI_POLICY_COEVOLVE_MODELS=${MULTI_POLICY_COEVOLVE_MODELS:-}

# Number of vLLM engines per extra co-evolving policy model
MULTI_POLICY_COEVOLVE_VLLM_ENGINES=${MULTI_POLICY_COEVOLVE_VLLM_ENGINES:-4}

# Learner GPUs per node for each extra co-evolving policy
# E.g., '1' means 1 learner per node, '1,1,1,1,0,0,0,0' for partial nodes
MULTI_POLICY_COEVOLVE_NUM_LEARNERS=${MULTI_POLICY_COEVOLVE_NUM_LEARNERS:-1}

# Ray-based vLLM engine settings for joint training (optional)
RUBRIC_JUDGE_NUM_ENGINES=${RUBRIC_JUDGE_NUM_ENGINES:-}
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=${RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE:-}
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=${RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION:-}
RUBRIC_JUDGE_MAX_MODEL_LEN=${RUBRIC_JUDGE_MAX_MODEL_LEN:-}

# Policy training parameters (prefixed with POLICY_ARGS_ for routing into GRPO session)
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=${POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT:-8}
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=${POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT:-4}
POLICY_ARGS_RESPONSE_LENGTH=${POLICY_ARGS_RESPONSE_LENGTH:-256}
POLICY_ARGS_ASYNC_STEPS=${POLICY_ARGS_ASYNC_STEPS:-1}
POLICY_ARGS_NUM_EPOCHS=${POLICY_ARGS_NUM_EPOCHS:-1}

# Tokenizer/model configs for policy (prefixed for routing)
TOKENIZER_ARGS_TOKENIZER_NAME_OR_PATH=${TOKENIZER_ARGS_TOKENIZER_NAME_OR_PATH:-$HF_CHECKPOINT}
MODEL_ARGS_MODEL_NAME_OR_PATH=${MODEL_ARGS_MODEL_NAME_OR_PATH:-$HF_CHECKPOINT}

# Temperature settings for alternating training
# RUBRIC_TEMPERATURE: rubric generation (default 0.6, independent of policy TEMPERATURE)
# POLICY/BASELINE_TEMPERATURE: policy rollout diversity (default follows TEMPERATURE)
RUBRIC_TEMPERATURE=${RUBRIC_TEMPERATURE:-0.6}
POLICY_TEMPERATURE=${POLICY_TEMPERATURE:-${TEMPERATURE:-1.0}}
BASELINE_TEMPERATURE=${BASELINE_TEMPERATURE:-${TEMPERATURE:-1.0}}

# ============================================================================
# Argument Arrays (for modular grouping)
# ============================================================================
# All argument arrays are built dynamically in run_training() after configs are loaded
# This ensures that config file overrides are properly applied

# ============================================================================
# Training Function
# ============================================================================

# Helper function to mark variables as used (for unused variable detection)
mark_var_used() {
    local var_name="$1"
    if [[ ! " ${USED_VARIABLES[@]} " =~ " ${var_name} " ]]; then
        USED_VARIABLES+=("$var_name")
    fi
}

run_training() {
    # Build all argument arrays here after config files are loaded
    # This ensures all config file overrides are properly applied
    # If DRY_RUN is set, only build command and track variables without executing
    
    # Core arguments
    mark_var_used "EXP_NAME"
    mark_var_used "SEED"
    local CORE_ARGS=(
        "--exp_name" "${EXP_NAME}"
        "--seed" "${SEED}"
    )
    
    # Dataset arguments
    mark_var_used "DATASET_MIXER_LIST"
    mark_var_used "DATASET_MIXER_LIST_SPLITS"
    mark_var_used "DATASET_MIXER_EVAL_LIST"
    mark_var_used "DATASET_MIXER_EVAL_LIST_SPLITS"
    mark_var_used "MAX_PROMPT_TOKEN_LENGTH"
    mark_var_used "RESPONSE_LENGTH"
    mark_var_used "PACK_LENGTH"
    mark_var_used "GROUND_TRUTHS_KEY"
    mark_var_used "SFT_MESSAGES_KEY"
    mark_var_used "TOTAL_EPISODES"
    local DATASET_ARGS=(
        "--dataset_mixer_list" ${DATASET_MIXER_LIST}
        "--dataset_mixer_list_splits" "${DATASET_MIXER_LIST_SPLITS}"
        "--dataset_mixer_eval_list" ${DATASET_MIXER_EVAL_LIST}
        "--dataset_mixer_eval_list_splits" "${DATASET_MIXER_EVAL_LIST_SPLITS}"
        "--max_prompt_token_length" "${MAX_PROMPT_TOKEN_LENGTH}"
        "--response_length" "${RESPONSE_LENGTH}"
        "--pack_length" "${PACK_LENGTH}"
        "--ground_truths_key" "${GROUND_TRUTHS_KEY}"
        "--sft_messages_key" "${SFT_MESSAGES_KEY}"
        "--total_episodes" "${TOTAL_EPISODES}"
    )
    
    # Add rubric judge dataset keys if provided
    if [ ! -z "$QUESTION_KEY" ]; then
        mark_var_used "QUESTION_KEY"
        DATASET_ARGS+=("--question_key" "${QUESTION_KEY}")
    fi
    if [ ! -z "$ACCEPTED_ANSWER_KEY" ]; then
        mark_var_used "ACCEPTED_ANSWER_KEY"
        DATASET_ARGS+=("--accepted_answer_key" "${ACCEPTED_ANSWER_KEY}")
    fi
    if [ ! -z "$REJECTED_ANSWER_KEY" ]; then
        mark_var_used "REJECTED_ANSWER_KEY"
        DATASET_ARGS+=("--rejected_answer_key" "${REJECTED_ANSWER_KEY}")
    fi
    
    # Add dataset_transform_fn if provided
    if [ ! -z "$DATASET_TRANSFORM_FN" ]; then
        mark_var_used "DATASET_TRANSFORM_FN"
        # Pass all transform function names after a single --dataset_transform_fn flag.
        # HfArgumentParser uses nargs='+', so separate --dataset_transform_fn flags
        # would cause last-wins behavior. All names must follow one flag.
        # shellcheck disable=SC2086
        DATASET_ARGS+=("--dataset_transform_fn" $DATASET_TRANSFORM_FN)
    fi
    
    # Add system_prompt_override_file if provided
    if [ ! -z "$SYSTEM_PROMPT_OVERRIDE_FILE" ]; then
        mark_var_used "SYSTEM_PROMPT_OVERRIDE_FILE"
        DATASET_ARGS+=("--system_prompt_override_file" "${SYSTEM_PROMPT_OVERRIDE_FILE}")
    fi
    
    # Add rubric_prompt_key if non-default (preferred over system_prompt_override_file for rubric prompts)
    if [ -n "${RUBRIC_PROMPT_KEY}" ] && [ "${RUBRIC_PROMPT_KEY}" != "rubric_generation" ]; then
        mark_var_used "RUBRIC_PROMPT_KEY"
        DATASET_ARGS+=("--rubric_prompt_key" "${RUBRIC_PROMPT_KEY}")
    fi
    
    # Add dataset_skip_cache if provided
    if [ ! -z "$DATASET_SKIP_CACHE" ]; then
        mark_var_used "DATASET_SKIP_CACHE"
        DATASET_ARGS+=("--dataset_skip_cache" "${DATASET_SKIP_CACHE}")
    fi
    
    # Training hyperparameters
    mark_var_used "LEARNING_RATE"
    mark_var_used "BETA"
    mark_var_used "NUM_SAMPLES_PER_PROMPT_ROLLOUT"
    mark_var_used "NUM_UNIQUE_PROMPTS_ROLLOUT"
    mark_var_used "NUM_MINI_BATCHES"
    mark_var_used "NUM_EPOCHS"
    mark_var_used "PER_DEVICE_TRAIN_BATCH_SIZE"
    mark_var_used "KL_ESTIMATOR"
    mark_var_used "LR_SCHEDULER_TYPE"
    mark_var_used "WARM_UP_STEPS"
    local TRAINING_ARGS=(
        "--learning_rate" "${LEARNING_RATE}"
        "--beta" "${BETA}"
        "--num_samples_per_prompt_rollout" "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}"
        "--num_unique_prompts_rollout" "${NUM_UNIQUE_PROMPTS_ROLLOUT}"
        "--num_mini_batches" "${NUM_MINI_BATCHES}"
        "--num_epochs" "${NUM_EPOCHS}"
        "--per_device_train_batch_size" "${PER_DEVICE_TRAIN_BATCH_SIZE}"
        "--kl_estimator" "${KL_ESTIMATOR}"
        "--lr_scheduler_type" "${LR_SCHEDULER_TYPE}"
        "--warm_up_steps" "${WARM_UP_STEPS}"
    )
    
    # Add optional training arguments if provided
    if [ ! -z "$INFLIGHT_UPDATES" ]; then
        mark_var_used "INFLIGHT_UPDATES"
        TRAINING_ARGS+=("--inflight_updates" "${INFLIGHT_UPDATES}")
    fi
    if [ ! -z "$ASYNC_STEPS" ]; then
        mark_var_used "ASYNC_STEPS"
        TRAINING_ARGS+=("--async_steps" "${ASYNC_STEPS}")
    fi
    if [ ! -z "$TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP" ]; then
        mark_var_used "TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP"
        TRAINING_ARGS+=("--truncated_importance_sampling_ratio_cap" "${TRUNCATED_IMPORTANCE_SAMPLING_RATIO_CAP}")
    fi
    
    # Model architecture
    mark_var_used "HF_CHECKPOINT"
    mark_var_used "DEEPSPEED_STAGE"
    mark_var_used "NUM_LEARNERS_PER_NODE"
    mark_var_used "VLLM_NUM_ENGINES"
    mark_var_used "VLLM_TENSOR_PARALLEL_SIZE"
    local MODEL_ARGS=(
        "--model_name_or_path" "${HF_CHECKPOINT}"
        "--deepspeed_stage" "${DEEPSPEED_STAGE}"
        "--num_learners_per_node" "${NUM_LEARNERS_PER_NODE}"
        "--vllm_num_engines" "${VLLM_NUM_ENGINES}"
        "--vllm_tensor_parallel_size" "${VLLM_TENSOR_PARALLEL_SIZE}"
    )
    if [ -n "${VLLM_GPU_MEMORY_UTILIZATION+x}" ] && [ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]; then
        mark_var_used "VLLM_GPU_MEMORY_UTILIZATION"
        MODEL_ARGS+=("--vllm_gpu_memory_utilization" "${VLLM_GPU_MEMORY_UTILIZATION}")
    fi

    # DeepSpeed CPU offloading (reduces GPU memory at the cost of speed)
    if [ "${DEEPSPEED_OFFLOAD_OPTIMIZER:-false}" = "true" ]; then
        mark_var_used "DEEPSPEED_OFFLOAD_OPTIMIZER"
        MODEL_ARGS+=("--deepspeed_offload_optimizer" "true")
    fi
    if [ "${DEEPSPEED_OFFLOAD_PARAM:-false}" = "true" ]; then
        mark_var_used "DEEPSPEED_OFFLOAD_PARAM"
        MODEL_ARGS+=("--deepspeed_offload_param" "true")
    fi
    
    # Reward arguments
    mark_var_used "APPLY_VERIFIABLE_REWARD"
    mark_var_used "MASKED_MEAN_AXIS"
    mark_var_used "RUBRIC_JUDGE_MODEL"
    local REWARD_ARGS=(
        "--apply_verifiable_reward" "${APPLY_VERIFIABLE_REWARD}"
        "--masked_mean_axis" "${MASKED_MEAN_AXIS}"
    )

    # Add scalar reward model if provided
    if [ ! -z "$REWARD_MODEL_NAME" ]; then
        mark_var_used "REWARD_MODEL_NAME"
        mark_var_used "REWARD_MODEL_NUM_GPUS"
        REWARD_ARGS+=("--reward_model_name" "${REWARD_MODEL_NAME}")
        REWARD_ARGS+=("--reward_model_num_gpus" "${REWARD_MODEL_NUM_GPUS}")
    fi

    # Add rubric judge model if provided
    if [ ! -z "$RUBRIC_JUDGE_MODEL" ]; then
        mark_var_used "RUBRIC_JUDGE_MODEL"
        REWARD_ARGS+=("--rubric_judge_model" "${RUBRIC_JUDGE_MODEL}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_TOKENIZER" ]; then
        mark_var_used "RUBRIC_JUDGE_TOKENIZER"
        REWARD_ARGS+=("--rubric_judge_tokenizer" "${RUBRIC_JUDGE_TOKENIZER}")
    fi

    # Add Ray-based vLLM engine arguments only if explicitly set
    if [ ! -z "$RUBRIC_JUDGE_NUM_ENGINES" ]; then
        mark_var_used "RUBRIC_JUDGE_NUM_ENGINES"
        REWARD_ARGS+=("--rubric_judge_num_engines" "${RUBRIC_JUDGE_NUM_ENGINES}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE" ]; then
        mark_var_used "RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE"
        REWARD_ARGS+=("--rubric_judge_tensor_parallel_size" "${RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION" ]; then
        mark_var_used "RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION"
        REWARD_ARGS+=("--rubric_judge_gpu_memory_utilization" "${RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_MAX_MODEL_LEN" ]; then
        mark_var_used "RUBRIC_JUDGE_MAX_MODEL_LEN"
        REWARD_ARGS+=("--rubric_judge_max_model_len" "${RUBRIC_JUDGE_MAX_MODEL_LEN}")
    fi

    # Multi-judge and multi-policy args are only understood by
    # train_rubric_policy_joint.py (JOINT_TRAINING), not grpo_fast.py.
    if [ "$JOINT_TRAINING" = "true" ]; then
        # Add multi-judge parameters if provided
        if [ ! -z "$MULTI_JUDGE_MODELS" ]; then
            mark_var_used "MULTI_JUDGE_MODELS"
            REWARD_ARGS+=("--multi_judge_models" "${MULTI_JUDGE_MODELS}")
        fi
        if [ ! -z "$MULTI_JUDGE_NUM_ENGINES_PER_JUDGE" ]; then
            mark_var_used "MULTI_JUDGE_NUM_ENGINES_PER_JUDGE"
            REWARD_ARGS+=("--multi_judge_num_engines_per_judge" "${MULTI_JUDGE_NUM_ENGINES_PER_JUDGE}")
        fi
        if [ ! -z "$MULTI_JUDGE_TENSOR_PARALLEL_SIZE" ]; then
            mark_var_used "MULTI_JUDGE_TENSOR_PARALLEL_SIZE"
            REWARD_ARGS+=("--multi_judge_tensor_parallel_size" "${MULTI_JUDGE_TENSOR_PARALLEL_SIZE}")
        fi
        if [ ! -z "$MULTI_JUDGE_ALPHA" ]; then
            mark_var_used "MULTI_JUDGE_ALPHA"
            REWARD_ARGS+=("--multi_judge_alpha" "${MULTI_JUDGE_ALPHA}")
        fi
        if [ ! -z "$MULTI_JUDGE_BETA" ]; then
            mark_var_used "MULTI_JUDGE_BETA"
            REWARD_ARGS+=("--multi_judge_beta" "${MULTI_JUDGE_BETA}")
        fi
        if [ ! -z "$MULTI_JUDGE_MARGIN_WEIGHT" ]; then
            mark_var_used "MULTI_JUDGE_MARGIN_WEIGHT"
            REWARD_ARGS+=("--multi_judge_margin_weight" "${MULTI_JUDGE_MARGIN_WEIGHT}")
        fi
        if [ ! -z "$MULTI_JUDGE_FORMAT_WEIGHT" ]; then
            mark_var_used "MULTI_JUDGE_FORMAT_WEIGHT"
            REWARD_ARGS+=("--multi_judge_format_weight" "${MULTI_JUDGE_FORMAT_WEIGHT}")
        fi
        if [ ! -z "$MULTI_JUDGE_KAPPA_WEIGHT" ]; then
            mark_var_used "MULTI_JUDGE_KAPPA_WEIGHT"
            REWARD_ARGS+=("--multi_judge_kappa_weight" "${MULTI_JUDGE_KAPPA_WEIGHT}")
        fi

        # Add multi-policy parameters if provided
        if [ ! -z "$MULTI_POLICY_MODELS" ]; then
            mark_var_used "MULTI_POLICY_MODELS"
            REWARD_ARGS+=("--multi_policy_models" "${MULTI_POLICY_MODELS}")
        fi
        if [ ! -z "$MULTI_POLICY_NUM_ENGINES_PER_MODEL" ]; then
            mark_var_used "MULTI_POLICY_NUM_ENGINES_PER_MODEL"
            REWARD_ARGS+=("--multi_policy_num_engines_per_model" "${MULTI_POLICY_NUM_ENGINES_PER_MODEL}")
        fi
        if [ ! -z "$MULTI_POLICY_TENSOR_PARALLEL_SIZE" ]; then
            mark_var_used "MULTI_POLICY_TENSOR_PARALLEL_SIZE"
            REWARD_ARGS+=("--multi_policy_tensor_parallel_size" "${MULTI_POLICY_TENSOR_PARALLEL_SIZE}")
        fi
        if [ ! -z "$MULTI_POLICY_SAMPLING_STRATEGY" ]; then
            mark_var_used "MULTI_POLICY_SAMPLING_STRATEGY"
            REWARD_ARGS+=("--multi_policy_sampling_strategy" "${MULTI_POLICY_SAMPLING_STRATEGY}")
        fi

        # Add multi-policy co-evolution parameters if provided
        if [ ! -z "$MULTI_POLICY_COEVOLVE_MODELS" ]; then
            mark_var_used "MULTI_POLICY_COEVOLVE_MODELS"
            REWARD_ARGS+=("--multi_policy_coevolve_models" "${MULTI_POLICY_COEVOLVE_MODELS}")
        fi
        if [ ! -z "$MULTI_POLICY_COEVOLVE_VLLM_ENGINES" ]; then
            mark_var_used "MULTI_POLICY_COEVOLVE_VLLM_ENGINES"
            REWARD_ARGS+=("--multi_policy_coevolve_vllm_engines_per_model" "${MULTI_POLICY_COEVOLVE_VLLM_ENGINES}")
        fi
        if [ ! -z "$MULTI_POLICY_COEVOLVE_NUM_LEARNERS" ]; then
            mark_var_used "MULTI_POLICY_COEVOLVE_NUM_LEARNERS"
            REWARD_ARGS+=("--multi_policy_coevolve_num_learners_per_node" "${MULTI_POLICY_COEVOLVE_NUM_LEARNERS}")
        fi
    else
        # Mark these as used even when not passing them, to suppress
        # "unused variable" warnings for their non-empty defaults.
        mark_var_used "MULTI_JUDGE_NUM_ENGINES_PER_JUDGE"
        mark_var_used "MULTI_JUDGE_TENSOR_PARALLEL_SIZE"
        REWARD_ARGS+=("--multi_judge_tensor_parallel_size" "${MULTI_JUDGE_TENSOR_PARALLEL_SIZE}")
    fi
    if [ ! -z "$MULTI_JUDGE_AGGREGATION" ]; then
        mark_var_used "MULTI_JUDGE_AGGREGATION"
        REWARD_ARGS+=("--multi_judge_aggregation" "${MULTI_JUDGE_AGGREGATION}")
    fi
    if [ ! -z "$MULTI_JUDGE_TIE_BREAKER" ]; then
        mark_var_used "MULTI_JUDGE_TIE_BREAKER"
        REWARD_ARGS+=("--multi_judge_tie_breaker" "${MULTI_JUDGE_TIE_BREAKER}")
    fi
    if [ ! -z "$MULTI_JUDGE_ALPHA" ]; then
        mark_var_used "MULTI_JUDGE_ALPHA"
        mark_var_used "MULTI_JUDGE_BETA"
        mark_var_used "MULTI_JUDGE_MARGIN_WEIGHT"
        mark_var_used "MULTI_JUDGE_FORMAT_WEIGHT"
        mark_var_used "MULTI_JUDGE_KAPPA_WEIGHT"
        mark_var_used "MULTI_POLICY_NUM_ENGINES_PER_MODEL"
        mark_var_used "MULTI_POLICY_TENSOR_PARALLEL_SIZE"
        mark_var_used "MULTI_POLICY_SAMPLING_STRATEGY"
        mark_var_used "MULTI_POLICY_COEVOLVE_VLLM_ENGINES"
        mark_var_used "MULTI_POLICY_COEVOLVE_NUM_LEARNERS"
    fi

    # Add inference model if provided
    if [ ! -z "$INFERENCE_MODEL" ]; then
        mark_var_used "INFERENCE_MODEL"
        REWARD_ARGS+=("--inference_model" "${INFERENCE_MODEL}")
    fi
    # Add Ray-based vLLM engine arguments for inference model only if explicitly set
    if [ ! -z "$INFERENCE_NUM_ENGINES" ]; then
        mark_var_used "INFERENCE_NUM_ENGINES"
        REWARD_ARGS+=("--inference_num_engines" "${INFERENCE_NUM_ENGINES}")
    fi
    if [ ! -z "$INFERENCE_TENSOR_PARALLEL_SIZE" ]; then
        mark_var_used "INFERENCE_TENSOR_PARALLEL_SIZE"
        REWARD_ARGS+=("--inference_tensor_parallel_size" "${INFERENCE_TENSOR_PARALLEL_SIZE}")
    fi
    if [ ! -z "$INFERENCE_GPU_MEMORY_UTILIZATION" ]; then
        mark_var_used "INFERENCE_GPU_MEMORY_UTILIZATION"
        REWARD_ARGS+=("--inference_gpu_memory_utilization" "${INFERENCE_GPU_MEMORY_UTILIZATION}")
    fi
    if [ ! -z "$INFERENCE_MAX_MODEL_LEN" ]; then
        mark_var_used "INFERENCE_MAX_MODEL_LEN"
        REWARD_ARGS+=("--inference_max_model_len" "${INFERENCE_MAX_MODEL_LEN}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_TEMPERATURE" ]; then
        mark_var_used "RUBRIC_JUDGE_TEMPERATURE"
        REWARD_ARGS+=("--rubric_judge_temperature" "${RUBRIC_JUDGE_TEMPERATURE}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_MAX_TOKENS" ]; then
        mark_var_used "RUBRIC_JUDGE_MAX_TOKENS"
        REWARD_ARGS+=("--rubric_judge_max_tokens" "${RUBRIC_JUDGE_MAX_TOKENS}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_STOP" ]; then
        mark_var_used "RUBRIC_JUDGE_STOP"
        REWARD_ARGS+=("--rubric_judge_stop" "${RUBRIC_JUDGE_STOP}")
    fi
    if [ ! -z "$RUBRIC_JUDGE_LOGPROBS" ]; then
        mark_var_used "RUBRIC_JUDGE_LOGPROBS"
        REWARD_ARGS+=("--rubric_judge_logprobs" "${RUBRIC_JUDGE_LOGPROBS}")
    fi
    if [ "${RUBRIC_CORRECTNESS_FOCUSED}" = "true" ]; then
        mark_var_used "RUBRIC_CORRECTNESS_FOCUSED"
        REWARD_ARGS+=("--rubric_correctness_focused")
    fi
    if [ "${RUBRIC_REWARD_SCORE_SEPARATION}" = "true" ]; then
        mark_var_used "RUBRIC_REWARD_SCORE_SEPARATION"
        REWARD_ARGS+=("--rubric_reward_score_separation")
    fi
    if [ ! -z "$RUBRIC_REWARD_SCORE_SEPARATION_WEIGHT" ]; then
        mark_var_used "RUBRIC_REWARD_SCORE_SEPARATION_WEIGHT"
        REWARD_ARGS+=("--rubric_reward_score_separation_weight" "${RUBRIC_REWARD_SCORE_SEPARATION_WEIGHT}")
    fi
    
    # Add reward function tag if provided
    if [ ! -z "$OVERWRITE_REWARD_FN_TAG" ]; then
        mark_var_used "OVERWRITE_REWARD_FN_TAG"
        REWARD_ARGS+=("--overwrite_reward_fn_tag" "${OVERWRITE_REWARD_FN_TAG}")
    fi
    
    # Add question group mode if provided
    if [ ! -z "$QUESTION_GROUP_MODE" ]; then
        mark_var_used "QUESTION_GROUP_MODE"
        REWARD_ARGS+=("--question_group_mode" "${QUESTION_GROUP_MODE}")
    fi
    
    # Add use format reward if provided
    if [ ! -z "$USE_FORMAT_REWARD" ]; then
        mark_var_used "USE_FORMAT_REWARD"
        if [ "$USE_FORMAT_REWARD" = "true" ]; then
            REWARD_ARGS+=("--use_format_reward")
        fi
    fi
    
    # Generation arguments
    mark_var_used "TEMPERATURE"
    mark_var_used "NON_STOP_PENALTY"
    mark_var_used "NON_STOP_PENALTY_VALUE"
    local GENERATION_ARGS=(
        "--temperature" "${TEMPERATURE}"
        "--non_stop_penalty" "${NON_STOP_PENALTY}"
        "--non_stop_penalty_value" "${NON_STOP_PENALTY_VALUE}"
    )
    
    # Tool arguments - only add all tool args if TOOLS is defined
    local TOOL_ARGS=()
    if [ -n "${TOOLS+x}" ]; then
        mark_var_used "TOOLS"
        mark_var_used "MAX_TOOL_CALLS"
        mark_var_used "ONLY_REWARD_GOOD_OUTPUTS"
        mark_var_used "SEARCH_API_ENDPOINT"
        mark_var_used "NUMBER_DOCUMENTS_TO_SEARCH"
        TOOL_ARGS+=(
            "--tools" "${TOOLS}"
            "--max_tool_calls" "${MAX_TOOL_CALLS}"
            "--only_reward_good_outputs" "${ONLY_REWARD_GOOD_OUTPUTS}"
            "--search_api_endpoint" "${SEARCH_API_ENDPOINT}"
            "--number_documents_to_search" "${NUMBER_DOCUMENTS_TO_SEARCH}"
        )
        
        # Add code tool endpoint if provided
        if [ ! -z "$CODE_TOOL_API_ENDPOINT" ]; then
            mark_var_used "CODE_TOOL_API_ENDPOINT"
            TOOL_ARGS+=("--code_tool_api_endpoint" "${CODE_TOOL_API_ENDPOINT}")
        fi
    fi
    
    # Training settings
    mark_var_used "LOCAL_EVAL_EVERY"
    mark_var_used "SAVE_FREQ"
    mark_var_used "KEEP_LAST_N_CHECKPOINTS"
    mark_var_used "OUTPUT_DIR"
    mark_var_used "CHECKPOINT_STATE_FREQ"
    mark_var_used "TRY_LAUNCH_BEAKER_EVAL_JOBS_ON_WEKA"
    local TRAINING_SETTINGS_ARGS=(
        "--local_eval_every" "${LOCAL_EVAL_EVERY}"
        "--save_freq" "${SAVE_FREQ}"
        "--keep_last_n_checkpoints" "${KEEP_LAST_N_CHECKPOINTS}"
        "--output_dir" "${OUTPUT_DIR}"
        "--checkpoint_state_freq" "${CHECKPOINT_STATE_FREQ}"
        "--try_launch_beaker_eval_jobs_on_weka" "${TRY_LAUNCH_BEAKER_EVAL_JOBS_ON_WEKA}"
    )
    
    # Add allow_world_padding if provided
    if [ ! -z "$ALLOW_WORLD_PADDING" ]; then
        mark_var_used "ALLOW_WORLD_PADDING"
        TRAINING_SETTINGS_ARGS+=("--allow_world_padding" "${ALLOW_WORLD_PADDING}")
    fi
    
    # Add checkpoint_state_dir if needed
    if [ ! -z "$CHECKPOINT_STATE_FREQ" ] && [ "$CHECKPOINT_STATE_FREQ" != "-1" ]; then
        mark_var_used "CHECKPOINT_STATE_DIR"
        if [ -z "$CHECKPOINT_STATE_DIR" ]; then
            CHECKPOINT_STATE_DIR="$CHECKPOINT_DIR"
        fi
        TRAINING_SETTINGS_ARGS+=("--checkpoint_state_dir" "${CHECKPOINT_STATE_DIR}")
    fi

    # Resume from a specific step (for warm-start without DeepSpeed state)
    if [ ! -z "$RESUME_FROM_STEP" ] && [ "$RESUME_FROM_STEP" != "0" ]; then
        mark_var_used "RESUME_FROM_STEP"
        TRAINING_SETTINGS_ARGS+=("--resume_from_step" "${RESUME_FROM_STEP}")
    fi
    # Resume into an existing run directory
    if [ ! -z "$RESUME_RUN_NAME" ]; then
        mark_var_used "RESUME_RUN_NAME"
        TRAINING_SETTINGS_ARGS+=("--resume_run_name" "${RESUME_RUN_NAME}")
    fi
    
    # Wandb arguments
    local WANDB_ARGS=()
    if [ "$WITH_TRACKING" = "true" ]; then
        mark_var_used "WITH_TRACKING"
        WANDB_ARGS+=("--with_tracking")
    fi
    if [ ! -z "$WANDB_PROJECT" ]; then
        mark_var_used "WANDB_PROJECT"
        WANDB_ARGS+=("--wandb_project_name" "${WANDB_PROJECT}")
    fi
    if [ ! -z "$WANDB_ENTITY" ]; then
        mark_var_used "WANDB_ENTITY"
        WANDB_ARGS+=("--wandb_entity" "${WANDB_ENTITY}")
    fi
    
    # Feature flags (boolean flags)
    local FEATURE_ARGS=()
    if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
        mark_var_used "GRADIENT_CHECKPOINTING"
        FEATURE_ARGS+=("--gradient_checkpointing")
    fi
    if [ "$VLLM_ENABLE_PREFIX_CACHING" = "true" ]; then
        mark_var_used "VLLM_ENABLE_PREFIX_CACHING"
        FEATURE_ARGS+=("--vllm_enable_prefix_caching")
    fi
    if [ "$VERBOSE" = "true" ]; then
        mark_var_used "VERBOSE"
        FEATURE_ARGS+=("--verbose")
    fi
    if [ "$ACTIVE_SAMPLING" = "true" ]; then
        mark_var_used "ACTIVE_SAMPLING"
        FEATURE_ARGS+=("--active_sampling")
    fi
    if [ ! -z "$FILTER_ZERO_STD_SAMPLES" ]; then
        mark_var_used "FILTER_ZERO_STD_SAMPLES"
        FEATURE_ARGS+=("--filter_zero_std_samples" "${FILTER_ZERO_STD_SAMPLES}")
    fi
    if [ "$PUSH_TO_HUB" = "false" ]; then
        mark_var_used "PUSH_TO_HUB"
        FEATURE_ARGS+=("--no_push_to_hub")
    fi

    # Joint training arguments (two-model setup - added to standard command)
    local JOINT_ARGS=()
    if [ "$JOINT_TRAINING" = "true" ] ; then
        mark_var_used "JOINT_TRAINING"
        mark_var_used "RUBRIC_TEMPERATURE"
        mark_var_used "POLICY_TEMPERATURE"
        mark_var_used "BASELINE_TEMPERATURE"
        mark_var_used "RUBRIC_MODEL"
        mark_var_used "RUBRIC_JUDGE_NUM_ENGINES"
        mark_var_used "RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE"
        mark_var_used "RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION"
        mark_var_used "RUBRIC_JUDGE_MAX_MODEL_LEN"
        mark_var_used "RUBRIC_JUDGE_TEMPERATURE"
        mark_var_used "RUBRIC_JUDGE_MAX_TOKENS"
        mark_var_used "RUBRIC_JUDGE_STOP"
        mark_var_used "RUBRIC_JUDGE_LOGPROBS"
        mark_var_used "RUBRIC_CORRECTNESS_FOCUSED"
        mark_var_used "RUBRIC_REWARD_SCORE_SEPARATION"
        mark_var_used "RUBRIC_REWARD_SCORE_SEPARATION_WEIGHT"
        mark_var_used "RUBRIC_REWARD_USE_MARGIN"
        mark_var_used "SINGLE_MODEL_MODE"
        mark_var_used "RUBRIC_REWARD_MODE"
        
        # Note: POLICY_ARGS_*, TOKENIZER_ARGS_*, MODEL_ARGS_* variables are defined
        # but not currently used by train_rubric_policy_joint.py. They are reserved
        # for future use when switching to GrpoPolicyTrainingSession.from_flat_config()
        # which routes prefixed variables into GRPO policy session configuration.
        # For now, they are marked as known/internal variables to avoid unused warnings.
        
        # Use RUBRIC_MODEL if set, otherwise fall back to HF_CHECKPOINT
        # Rubric model: Load locally on training GPU (no hosted_vllm prefix - will be loaded by training pipeline)
        local model_for_rubric="${RUBRIC_MODEL:-$HF_CHECKPOINT}"
        
        JOINT_ARGS+=("--rubric-model" "${model_for_rubric}")
        JOINT_ARGS+=("--policy-model" "${HF_CHECKPOINT}")
        if [ -n "${ALTERNATING_CYCLES+x}" ] && [ -n "${ALTERNATING_CYCLES}" ]; then
            mark_var_used "ALTERNATING_CYCLES"
            mark_var_used "ALTERNATING_STEPS_PER_PHASE"
            JOINT_ARGS+=("--cycles" "${ALTERNATING_CYCLES}")
            JOINT_ARGS+=("--steps-per-phase" "${ALTERNATING_STEPS_PER_PHASE}")
        fi
        # Use both models flag
        if [ "${USE_BOTH_MODELS}" = "true" ]; then
            mark_var_used "USE_BOTH_MODELS"
            JOINT_ARGS+=("--use-both-models")
        fi
        # Single model mode - share vLLM engines between rubric and policy actors
        if [ "${SINGLE_MODEL_MODE}" = "true" ]; then
            JOINT_ARGS+=("--single-model-mode")
        fi
        # Use same temperature for all (TEMPERATURE if set, otherwise defaults)
        JOINT_ARGS+=("--rubric-temperature" "${RUBRIC_TEMPERATURE}")
        JOINT_ARGS+=("--policy-temperature" "${POLICY_TEMPERATURE}")
        JOINT_ARGS+=("--baseline-temperature" "${BASELINE_TEMPERATURE}")
        JOINT_ARGS+=("--rubric-reward-mode" "${RUBRIC_REWARD_MODE}")
        if [ "${RUBRIC_REWARD_USE_MARGIN}" = "true" ]; then
            JOINT_ARGS+=("--rubric-reward-use-margin")
        fi
        mark_var_used "RUBRIC_FORMAT_REWARD_WEIGHT"
        if [ "$(echo "${RUBRIC_FORMAT_REWARD_WEIGHT} > 0" | bc -l 2>/dev/null)" = "1" ]; then
            JOINT_ARGS+=("--rubric-format-reward-weight" "${RUBRIC_FORMAT_REWARD_WEIGHT}")
        fi
        JOINT_ARGS+=("--output" "${OUTPUT_DIR}/alternating_training_result.json")
        
        # Generation examples logging (for debugging/analysis)
        if [ -n "${NUM_EXAMPLES_TO_LOG+x}" ] && [ -n "${NUM_EXAMPLES_TO_LOG}" ]; then
            mark_var_used "NUM_EXAMPLES_TO_LOG"
            JOINT_ARGS+=("--num-examples-to-log" "${NUM_EXAMPLES_TO_LOG}")
        fi
        if [ -n "${LOG_EXAMPLES_EVERY_N_STEPS+x}" ] && [ -n "${LOG_EXAMPLES_EVERY_N_STEPS}" ]; then
            mark_var_used "LOG_EXAMPLES_EVERY_N_STEPS"
            JOINT_ARGS+=("--log-examples-every-n-steps" "${LOG_EXAMPLES_EVERY_N_STEPS}")
        fi
        # Rejected answer method for rubric training
        if [ -n "${REJECTED_ANSWER_METHOD+x}" ] && [ -n "${REJECTED_ANSWER_METHOD}" ]; then
            mark_var_used "REJECTED_ANSWER_METHOD"
            JOINT_ARGS+=("--rejected-answer-method" "${REJECTED_ANSWER_METHOD}")
        fi
        # Inference model for question inference (when using inferred_question method)
        if [ -n "${INFERENCE_MODEL_FOR_QUESTION_INFERENCE+x}" ] && [ -n "${INFERENCE_MODEL_FOR_QUESTION_INFERENCE}" ]; then
            mark_var_used "INFERENCE_MODEL_FOR_QUESTION_INFERENCE"
            JOINT_ARGS+=("--inference-model-for-question-inference" "${INFERENCE_MODEL_FOR_QUESTION_INFERENCE}")
        fi
        # Combined data provider weights (when using combined method)
        if [ -n "${COMBINED_DATA_PROVIDER_WEIGHTS+x}" ] && [ -n "${COMBINED_DATA_PROVIDER_WEIGHTS}" ]; then
            mark_var_used "COMBINED_DATA_PROVIDER_WEIGHTS"
            JOINT_ARGS+=("--combined-data-provider-weights" "${COMBINED_DATA_PROVIDER_WEIGHTS}")
        fi
        # Replay buffer settings for age-based sampling
        if [ -n "${REPLAY_BUFFER_SIZE+x}" ] && [ -n "${REPLAY_BUFFER_SIZE}" ]; then
            mark_var_used "REPLAY_BUFFER_SIZE"
            JOINT_ARGS+=("--replay-buffer-size" "${REPLAY_BUFFER_SIZE}")
        fi
        if [ -n "${REPLAY_BUFFER_MIN_AGE+x}" ] && [ -n "${REPLAY_BUFFER_MIN_AGE}" ]; then
            mark_var_used "REPLAY_BUFFER_MIN_AGE"
            JOINT_ARGS+=("--replay-buffer-min-age" "${REPLAY_BUFFER_MIN_AGE}")
        fi
        if [ -n "${REPLAY_BUFFER_MAX_AGE+x}" ] && [ -n "${REPLAY_BUFFER_MAX_AGE}" ]; then
            mark_var_used "REPLAY_BUFFER_MAX_AGE"
            JOINT_ARGS+=("--replay-buffer-max-age" "${REPLAY_BUFFER_MAX_AGE}")
        fi
        # Judge size curriculum settings
        if [ -n "${JUDGE_SIZE_CURRICULUM+x}" ] && [ -n "${JUDGE_SIZE_CURRICULUM}" ]; then
            mark_var_used "JUDGE_SIZE_CURRICULUM"
            JOINT_ARGS+=("--judge-size-curriculum" "${JUDGE_SIZE_CURRICULUM}")
        fi
        if [ -n "${JUDGE_CURRICULUM_SCHEDULE+x}" ] && [ -n "${JUDGE_CURRICULUM_SCHEDULE}" ]; then
            mark_var_used "JUDGE_CURRICULUM_SCHEDULE"
            JOINT_ARGS+=("--judge-curriculum-schedule" "${JUDGE_CURRICULUM_SCHEDULE}")
        fi
        # API-based rubric generator (for baseline comparison)
        if [ -n "${API_RUBRIC_GENERATOR+x}" ] && [ -n "${API_RUBRIC_GENERATOR}" ]; then
            mark_var_used "API_RUBRIC_GENERATOR"
            JOINT_ARGS+=("--api-rubric-generator" "${API_RUBRIC_GENERATOR}")
        fi
        # Freeze rubric model (for baseline comparison)
        if [ "${FREEZE_RUBRIC_MODEL}" = "true" ]; then
            mark_var_used "FREEZE_RUBRIC_MODEL"
            JOINT_ARGS+=("--freeze-rubric-model")
        fi
        # Freeze policy model (for rubric-only training)
        if [ "${FREEZE_POLICY_MODEL}" = "true" ]; then
            mark_var_used "FREEZE_POLICY_MODEL"
            JOINT_ARGS+=("--freeze-policy-model")
        fi
    fi
    
    # ========================================================================
    # Build command prefix with environment variables
    # ========================================================================
    # If any variables were registered via register_command_env_var(), prefix
    # them to the command as VAR=value. This scopes environment variables
    # to only the Python command execution, not the entire shell.
    # ========================================================================
    local cmd_env_prefix=""
    if [ ${#COMMAND_ENV_VARS[@]} -gt 0 ]; then
        for var_name in "${COMMAND_ENV_VARS[@]}"; do
            # Get the value of the variable
            local var_value="${!var_name}"
            # Escape the value properly for shell (handle spaces, special chars)
            # Use printf %q to properly quote/escape the value
            local escaped_value=$(printf '%q' "$var_value")
            cmd_env_prefix+="${var_name}=${escaped_value} "
        done
        cmd_env_prefix="${cmd_env_prefix% }"  # Remove trailing space
    fi
    
    # Mark TRAIN_SCRIPT as used
    mark_var_used "TRAIN_SCRIPT"
    
    # Build python command (without env prefix for easier parsing)
    local python_cmd="python ${ROOT_DIR}/${TRAIN_SCRIPT}"
    python_cmd+=" ${CORE_ARGS[@]}"
    python_cmd+=" ${DATASET_ARGS[@]}"
    python_cmd+=" ${TRAINING_ARGS[@]}"
    python_cmd+=" ${MODEL_ARGS[@]}"
    python_cmd+=" ${REWARD_ARGS[@]}"
    python_cmd+=" ${GENERATION_ARGS[@]}"
    python_cmd+=" ${TOOL_ARGS[@]}"
    python_cmd+=" ${TRAINING_SETTINGS_ARGS[@]}"
    python_cmd+=" ${WANDB_ARGS[@]}"
    python_cmd+=" ${FEATURE_ARGS[@]}"
    # Add joint training arguments if present (two-model setup)
    if [ ${#JOINT_ARGS[@]} -gt 0 ]; then
        python_cmd+=" ${JOINT_ARGS[@]}"
    fi
    
    # Build full command for execution (combine env prefix with python command)
    local cmd=""
    if [ -n "$cmd_env_prefix" ]; then
        cmd="${cmd_env_prefix} ${python_cmd}"
    else
        cmd="$python_cmd"
    fi
    
    # If dry run, just return without executing (don't print command)
    if [ "${DRY_RUN:-false}" = "true" ]; then
        return 0
    fi
    
    echo "=========================================="
    echo "Running training command:"
    
    # Print command env variables on separate lines if any
    if [ ${#COMMAND_ENV_VARS[@]} -gt 0 ]; then
        echo "# Command environment variables: ${COMMAND_ENV_VARS[*]}"
        # Print each environment variable on its own line
        for var_name in "${COMMAND_ENV_VARS[@]}"; do
            local var_value="${!var_name}"
            local escaped_value=$(printf '%q' "$var_value")
            echo "${var_name}=${escaped_value} \\"
        done
    fi
    
    # Parse python_cmd string and print each --flag (with its value) on its own line
    # Split the command string by spaces (bash word splitting)
    local cmd_parts=($python_cmd)
    local arg_count=${#cmd_parts[@]}
    local idx=0
    
    # Find where "python" starts in the command
    local python_idx=0
    for i in "${!cmd_parts[@]}"; do
        if [[ "${cmd_parts[$i]}" == "python" ]]; then
            python_idx=$i
            break
        fi
    done
    
    # Print python and script path on first line (after env vars)
    local first_line="${cmd_parts[$python_idx]} ${cmd_parts[$((python_idx + 1))]}"
    echo "${first_line} \\"
    idx=$((python_idx + 2))
    
    # Process remaining arguments, grouping --flag with its value
    while [ $idx -lt $arg_count ]; do
        local arg="${cmd_parts[$idx]}"
        
        if [[ "$arg" == --* ]]; then
            # This is a flag
            local is_last=$([ $((idx + 1)) -ge $arg_count ] && echo true || echo false)
            
            if [ "$is_last" = "true" ] || [[ "${cmd_parts[$((idx + 1))]}" == --* ]]; then
                # Last arg or next is also a flag (boolean flag, no value)
                if [ "$is_last" = "true" ]; then
                    echo "    $arg"
                else
                    echo "    $arg \\"
                fi
                idx=$((idx + 1))
            else
                # Next is a value - group flag and value together
                local value="${cmd_parts[$((idx + 1))]}"
                if [ $((idx + 2)) -ge $arg_count ]; then
                    # This is the last line
                    echo "    $arg $value"
                else
                    echo "    $arg $value \\"
                fi
                idx=$((idx + 2))
            fi
        else
            # Shouldn't happen if cmd is well-formed, but handle it
            if [ $((idx + 1)) -ge $arg_count ]; then
                echo "    $arg"
            else
                echo "    $arg \\"
            fi
            idx=$((idx + 1))
        fi
    done
    
    echo "=========================================="
    
    # Execute the command
    eval $cmd
}
