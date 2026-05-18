#!/bin/bash
# Output directory configuration for paper experiments
# Saves all outputs to OUTPUT_DIR_BASE (default: ./outputs)
#
# This includes:
# - Model checkpoints (every SAVE_FREQ steps)
# - Full training states (every CHECKPOINT_STATE_FREQ steps)
# - Generated rubrics and training logs
# - Evaluation results
#
# Usage: ./scripts/launch.sh ... paper_output ...

# Base output directory on cluster storage
OUTPUT_DIR_BASE=${OVERRIDDEN_VALUES[OUTPUT_DIR_BASE]:-"${OUTPUT_DIR_BASE:-./outputs}"}

# Checkpoint saving settings (override defaults from base_config.sh)
SAVE_FREQ=${OVERRIDDEN_VALUES[SAVE_FREQ]:-25}                          # Save every 25 steps (2 per alternating phase)
KEEP_LAST_N_CHECKPOINTS=${OVERRIDDEN_VALUES[KEEP_LAST_N_CHECKPOINTS]:--1}  # Keep ALL checkpoints (-1 = no deletion)
CHECKPOINT_STATE_FREQ=${OVERRIDDEN_VALUES[CHECKPOINT_STATE_FREQ]:--1}       # Disable full state saves to save disk space

echo "[paper_output] OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE}"
echo "[paper_output] SAVE_FREQ=${SAVE_FREQ}, KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS}"
echo "[paper_output] CHECKPOINT_STATE_FREQ=${CHECKPOINT_STATE_FREQ}"
