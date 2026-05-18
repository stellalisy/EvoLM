#!/bin/bash
# Configuration for rubric judge project
# Dataset format: question, accepted_answer, rejected_answer
# Model generates rubric, LLM-as-a-judge evaluates answers

# Default experiment name (will be augmented by dataset config if sourced)
# Use EXP_NAME_BASE so other configs can modify it
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rj"
fi

# Dataset transformation - use rubric_judge_tokenize_v2
# The transform function expects columns: question, accepted_answer, rejected_answer
# If your dataset uses different column names, configure them via:
#   QUESTION_KEY, ACCEPTED_ANSWER_KEY, REJECTED_ANSWER_KEY
DATASET_TRANSFORM_FN=${DATASET_TRANSFORM_FN:-"rubric_judge_tokenize_v2"}

# Dataset column keys (optional - defaults shown)
# QUESTION_KEY=${QUESTION_KEY:-"question"}
# ACCEPTED_ANSWER_KEY=${ACCEPTED_ANSWER_KEY:-"accepted_answer"}
# REJECTED_ANSWER_KEY=${REJECTED_ANSWER_KEY:-"rejected_answer"}

# Rubric judge model (for generating rubrics and judging answers)
RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL:-gpt-4.1-mini}

# Generation settings - rubrics are typically shorter than regular responses
RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}
MAX_PROMPT_TOKEN_LENGTH=${MAX_PROMPT_TOKEN_LENGTH:-512}

# Note: This config is meant to be used with launch.sh:
#   ./scripts/launch.sh rubric_judge
# Or with overrides:
#   ./scripts/launch.sh rubric_judge DATASET_MIXER_LIST="your_dataset 1.0" HF_CHECKPOINT="your_model"

