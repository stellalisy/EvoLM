#!/bin/bash
# Configuration for rubric judge server using Ray-based vLLM engines
# This config uses in-process Ray vLLM engines instead of standalone server

# Ensure ROOT_DIR is set (launch.sh already exports it, but check for standalone usage)
if [ -z "$ROOT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    # Don't export - launch.sh handles this
fi

# RUBRIC_JUDGE_MODEL - Default value, can be overridden by rubric_judge/* configs
# When using Ray-based vLLM engines, use the same model as standalone server
RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL:-"Qwen/Qwen3-1.7B"}

# Augment experiment name to indicate Ray-based judge model
# Create compact shorthand from model name (e.g., "Qwen/Qwen3-1.7B" -> "q317b")
# Extract model family and size, remove separators, keep first letter + numbers
JUDGE_MODEL_SHORTHAND=$(echo "$RUBRIC_JUDGE_MODEL" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9]//g' | tr '[:upper:]' '[:lower:]')
# Make it compact: keep first letter and numbers (e.g., "qwen317b" -> "q317b")
if [[ "$JUDGE_MODEL_SHORTHAND" =~ ^([a-z])[a-z]*([0-9].*)$ ]]; then
    JUDGE_MODEL_SHORTHAND="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
elif [[ "$JUDGE_MODEL_SHORTHAND" =~ ^([a-z]{2,3}) ]]; then
    # If no numbers, use first 2-3 chars
    JUDGE_MODEL_SHORTHAND="${BASH_REMATCH[1]}"
else
    # Fallback to "jm" (judge model) if we can't create a meaningful shorthand
    JUDGE_MODEL_SHORTHAND="jm"
fi
# Append "rj" (Ray judge) prefix with judge model shorthand to EXP_NAME_BASE using lazy evaluation
# This will be expanded later in launch.sh when EXP_NAME_BASE is evaluated
# Only modify if EXP_NAME_BASE was NOT provided as an override
LOCAL_JUDGE_TAG="rj${JUDGE_MODEL_SHORTHAND}"
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rj_\${LOCAL_JUDGE_TAG}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"\${LOCAL_JUDGE_TAG}"* ]] && [[ "$EXP_NAME_BASE" != *"rj"* ]] && [[ "$EXP_NAME_BASE" != *"rayjm"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_\${LOCAL_JUDGE_TAG}"
fi

# Ray-based vLLM engine configuration
# These settings control the Ray-based rubric judge vLLM engines
# Set defaults for Ray-based approach (can be overridden by rubric_judge/* configs)
RUBRIC_JUDGE_MODEL=${RUBRIC_JUDGE_MODEL:-"Qwen/Qwen3-1.7B"}
RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE=${RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE:-1}
RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION=${RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION:-0.9}
RUBRIC_JUDGE_MAX_MODEL_LEN=${RUBRIC_JUDGE_MAX_MODEL_LEN:-32768}

# Declare variables that are used internally by this config (not passed to Python)
# These will be excluded from unused variable warnings
KNOWN_CONFIG_VARIABLES=(
    JUDGE_MODEL_SHORTHAND
    LOCAL_JUDGE_TAG
)

# Override function to indicate Ray-based vLLM engines are being used
# This replaces the server startup function from rubric_judge_server.sh
ensure_rubric_judge_server_running() {
    echo "=========================================="
    echo "Rubric Judge Ray vLLM Config"
    echo "=========================================="
    echo "Using Ray-based vLLM engines instead of standalone server"
    echo "Model: $RUBRIC_JUDGE_MODEL"
    echo "Number of engines: $RUBRIC_JUDGE_NUM_ENGINES"
    echo "Tensor parallel size: $RUBRIC_JUDGE_TENSOR_PARALLEL_SIZE"
    echo "GPU memory utilization: $RUBRIC_JUDGE_GPU_MEMORY_UTILIZATION"
    echo "Max model length: $RUBRIC_JUDGE_MAX_MODEL_LEN"
    echo "=========================================="
    echo "Ray vLLM configuration complete."
    echo "=========================================="
}

# Register ensure_rubric_judge_server_running as a post-config function
# This ensures the config is validated before training begins
POST_CONFIG_FUNCTIONS+=("ensure_rubric_judge_server_running")

# Note: Usage with launch.sh:
#   # Use Ray-based rubric judge vLLM engines:
#   ./scripts/launch.sh rubric_judge_server_vllm rubric_judge dpo_model_ladder
#
# This config uses Ray-based vLLM engines instead of starting a separate
# SLURM job for the rubric judge server. The engines run in-process
# within the Ray training job.
