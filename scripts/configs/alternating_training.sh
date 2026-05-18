#!/bin/bash
# Configuration for alternating rubric/policy training using the new controller APIs.
# Designed to be used with:
#   ./scripts/launch.sh rubric_judge dpo_model_ladder alternating_training

# Enable joint training mode (two-model setup)
JOINT_TRAINING=true

# Set joint training script
TRAIN_SCRIPT=scripts/train_rubric_policy_joint.py

# Core checkpoint + rubric model defaults
HF_CHECKPOINT=${HF_CHECKPOINT:-"Qwen/Qwen3-8B"}
RUBRIC_MODEL=${RUBRIC_MODEL:-"gpt-4.1-mini"}

# Prefix-based argument routing -------------------------------------------------
# All variables that start with POLICY_ARGS_, TOKENIZER_ARGS_, MODEL_ARGS_ will be
# routed (by alternating_training.py) into the GRPO policy session.

# Policy session defaults
POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT=${POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT:-8}
POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT=${POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT:-4}
POLICY_ARGS_RESPONSE_LENGTH=${POLICY_ARGS_RESPONSE_LENGTH:-256}
POLICY_ARGS_ASYNC_STEPS=${POLICY_ARGS_ASYNC_STEPS:-1}
POLICY_ARGS_NUM_EPOCHS=${POLICY_ARGS_NUM_EPOCHS:-1}

# Tokenizer/model configs (lazy evaluation so other configs can override)
TOKENIZER_ARGS_TOKENIZER_NAME_OR_PATH=${TOKENIZER_ARGS_TOKENIZER_NAME_OR_PATH:-$HF_CHECKPOINT}
MODEL_ARGS_MODEL_NAME_OR_PATH=${MODEL_ARGS_MODEL_NAME_OR_PATH:-$HF_CHECKPOINT}

# Alternating loop parameters ---------------------------------------------------
ALTERNATING_CYCLES=${ALTERNATING_CYCLES:-5}
ALTERNATING_STEPS_PER_PHASE=${ALTERNATING_STEPS_PER_PHASE:-10}

# Generation examples logging (for debugging/analysis) -------------------------
# Log N examples per training step (question, rubric, policy answer, score)
NUM_EXAMPLES_TO_LOG=${NUM_EXAMPLES_TO_LOG:-3}
# Log examples every N steps (1 = every step, 0 = disable)
LOG_EXAMPLES_EVERY_N_STEPS=${LOG_EXAMPLES_EVERY_N_STEPS:-10}

# Experiment naming -------------------------------------------------------------
if [ -z "$EXP_NAME_BASE" ]; then
    # Use short abbreviation: at = alternating training, Q8 = Qwen3-8B
    model_short="${HF_CHECKPOINT##*/}"
    model_short="${model_short//Qwen3-8B/Q8}"
    model_short="${model_short//Qwen\/Qwen3-8B/Q8}"
    EXP_NAME_BASE="at_${model_short}"
fi

# Declare prefixed variables as known (reserved for future GRPO policy session routing)
# These are not currently used by train_rubric_policy_joint.py but reserved for
# future use with GrpoPolicyTrainingSession.from_flat_config()
KNOWN_CONFIG_VARIABLES=(
    POLICY_ARGS_NUM_UNIQUE_PROMPTS_ROLLOUT
    POLICY_ARGS_NUM_SAMPLES_PER_PROMPT_ROLLOUT
    POLICY_ARGS_RESPONSE_LENGTH
    POLICY_ARGS_ASYNC_STEPS
    POLICY_ARGS_NUM_EPOCHS
    TOKENIZER_ARGS_TOKENIZER_NAME_OR_PATH
    MODEL_ARGS_MODEL_NAME_OR_PATH
)

# Announce config summary
echo "[alternating_training] HF_CHECKPOINT=$HF_CHECKPOINT"
echo "[alternating_training] RUBRIC_MODEL=$RUBRIC_MODEL"
echo "[alternating_training] cycles=$ALTERNATING_CYCLES steps_per_phase=$ALTERNATING_STEPS_PER_PHASE"
echo "[alternating_training] log_examples: every ${LOG_EXAMPLES_EVERY_N_STEPS} steps, ${NUM_EXAMPLES_TO_LOG} examples per step"
echo "[alternating_training] rejected_answer_method=$REJECTED_ANSWER_METHOD"
