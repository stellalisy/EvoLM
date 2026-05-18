#!/bin/bash
# Output directory configuration for V3 prompt experiments
# Saves all outputs to OUTPUT_DIR_BASE (default: ./outputs_v3)

# Base output directory on cluster storage
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-./outputs_v3}"

# Checkpoint saving settings (same as paper_output.sh)
SAVE_FREQ=25                    # Save every 25 steps (2 per alternating phase)
KEEP_LAST_N_CHECKPOINTS=-1      # Keep ALL checkpoints (-1 = no deletion)
CHECKPOINT_STATE_FREQ=-1        # Disable full state saves to save disk space

echo "[paper_output_v3] OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE}"
echo "[paper_output_v3] SAVE_FREQ=${SAVE_FREQ}, KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS}"
echo "[paper_output_v3] CHECKPOINT_STATE_FREQ=${CHECKPOINT_STATE_FREQ}"
