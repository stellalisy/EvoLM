#!/bin/bash
# Override TRAIN_SCRIPT to dump the last N eval prompts instead of training.
# Usage:
#   ./scripts/launch.sh alternating_training dpo_model_ladder dump_eval_prompts <model_config>
#
# The script reuses the exact same dataset loading pipeline (transform, shuffle,
# ShufflingIterator seed) so the output is guaranteed to match what training sees.

TRAIN_SCRIPT=scripts/dump_eval_prompts.py

# Need a value so --vllm_num_engines isn't passed empty
VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES:-1}
