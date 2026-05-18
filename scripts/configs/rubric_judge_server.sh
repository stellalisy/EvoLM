#!/bin/bash
# Configuration for rubric judge server
# This config ensures the server is running and sets environment variables

# Ensure ROOT_DIR is set (launch.sh already exports it, but check for standalone usage)
if [ -z "$ROOT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    # Don't export - launch.sh handles this
fi

# RUBRIC_JUDGE_MODEL - SINGLE SOURCE OF TRUTH
# When using the server config, ALWAYS use Qwen/Qwen3-1.7B (the server model)
# This is the ONLY place where RUBRIC_JUDGE_MODEL should be set when using the server
# It will be transformed to hosted_vllm/Qwen/Qwen3-1.7B after server starts
RUBRIC_JUDGE_MODEL="Qwen/Qwen3-1.7B"

# Augment experiment name to indicate local judge model with shorthand
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
# Append "localjm" (local judge model) prefix with judge model shorthand to EXP_NAME_BASE using lazy evaluation
# This will be expanded later in launch.sh when EXP_NAME_BASE is evaluated
# Only modify if EXP_NAME_BASE was NOT provided as an override
LOCAL_JUDGE_TAG="localjm${JUDGE_MODEL_SHORTHAND}"
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="rj_\${LOCAL_JUDGE_TAG}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"\${LOCAL_JUDGE_TAG}"* ]] && [[ "$EXP_NAME_BASE" != *"localjm"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_\${LOCAL_JUDGE_TAG}"
fi

RUBRIC_JUDGE_SERVER_PORT=${RUBRIC_JUDGE_SERVER_PORT:-8000}
RUBRIC_JUDGE_SERVER_HOST=${RUBRIC_JUDGE_SERVER_HOST:-0.0.0.0}

# Load SLURM template to get SLURM_GPUS before calculating VLLM_DATA_PARALLEL_SIZE
if [ -f "${ROOT_DIR}/env/slurm_judge_server_template.sh" ]; then
    source "${ROOT_DIR}/env/slurm_judge_server_template.sh"
fi

# vLLM server configuration
# Use data parallelism (multiple instances) instead of tensor parallelism
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}

# Automatically calculate VLLM_DATA_PARALLEL_SIZE from SLURM GPU allocation
# if not explicitly set: divide number of GPUs by tensor parallel size
# Only calculate if VLLM_DATA_PARALLEL_SIZE is not explicitly set
if [ -z "$VLLM_DATA_PARALLEL_SIZE" ]; then
    echo "[Data Parallel Config] VLLM_DATA_PARALLEL_SIZE not explicitly set, calculating from SLURM_GPUS in template..." >&2
    
    # Get GPU count from SLURM_GPUS in template file (set explicitly)
    # NUM_GPUS is a temporary variable used only for calculation
    if [ ! -z "$SLURM_GPUS" ]; then
        # Extract number from SLURM_GPUS (might be in format like "4" or "gpu:4")
        NUM_GPUS=$(echo "$SLURM_GPUS" | grep -oE '[0-9]+' | head -n1)
        if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
            echo "[Data Parallel Config] Warning: Could not extract valid GPU count from SLURM_GPUS='$SLURM_GPUS', using default of 1" >&2
            NUM_GPUS=1
        fi
    else
        # Fallback: if SLURM_GPUS not set in template, use default of 1
        echo "[Data Parallel Config] Warning: SLURM_GPUS not set in template, using default of 1" >&2
        NUM_GPUS=1
    fi
    
    echo "[Data Parallel Config] Number of GPUs (from SLURM_GPUS): $NUM_GPUS" >&2
    echo "[Data Parallel Config] Tensor parallel size: $VLLM_TENSOR_PARALLEL_SIZE" >&2
    
    # Calculate data parallel size: GPUs / tensor parallel size
    if [ "$NUM_GPUS" -gt 0 ] && [ "$VLLM_TENSOR_PARALLEL_SIZE" -gt 0 ]; then
        VLLM_DATA_PARALLEL_SIZE=$((NUM_GPUS / VLLM_TENSOR_PARALLEL_SIZE))
        # Ensure at least 1
        if [ "$VLLM_DATA_PARALLEL_SIZE" -lt 1 ]; then
            VLLM_DATA_PARALLEL_SIZE=1
        fi
        echo "[Data Parallel Config] Calculated: $NUM_GPUS / $VLLM_TENSOR_PARALLEL_SIZE = $VLLM_DATA_PARALLEL_SIZE" >&2
    else
        VLLM_DATA_PARALLEL_SIZE=1
        echo "[Data Parallel Config] Using default: VLLM_DATA_PARALLEL_SIZE=1" >&2
    fi
    # Clean up temporary variable
    unset NUM_GPUS
else
    echo "[Data Parallel Config] VLLM_DATA_PARALLEL_SIZE explicitly set to: $VLLM_DATA_PARALLEL_SIZE" >&2
fi
echo "[Data Parallel Config] Final VLLM_DATA_PARALLEL_SIZE: $VLLM_DATA_PARALLEL_SIZE" >&2
VLLM_MAX_MODEL_LEN=30000
VLLM_TRUST_REMOTE_CODE=${VLLM_TRUST_REMOTE_CODE:-false}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
# NOTE: Thinking mode (enable_thinking) is left at the Qwen3 default (True).
# The inline vLLM path (rubric_judge_server_vllm.sh) never disabled thinking,
# and all paper experiments used that path.  Keeping the standalone server
# consistent avoids silent behavior divergence between the two configs.

# Note: SLURM configuration is handled by scripts/start_rubric_judge_server.sh
# which loads env/slurm_judge_server_template.sh
# You can override SLURM settings via environment variables before calling launch.sh

# Register environment variables that will be set by ensure_rubric_judge_server_running
# These need to be available to the Python command
register_command_env_var RUBRIC_JUDGE_SERVER_URL
register_command_env_var HOSTED_VLLM_API_BASE
# RUBRIC_JUDGE_MODEL is already tracked by base_config.sh, but we'll update it

# Declare variables that are used internally by this config (not passed to Python)
# These will be excluded from unused variable warnings
# Note: RUBRIC_JUDGE_SERVER_URL and HOSTED_VLLM_API_BASE are set dynamically inside
# ensure_rubric_judge_server_running (a post-config function), so they're not marked
# as "used" during the dry run. They're registered as command env vars and will be
# available to Python at runtime, so we add them to KNOWN_CONFIG_VARIABLES to prevent
# false unused variable warnings.
KNOWN_CONFIG_VARIABLES=(
    RUBRIC_JUDGE_SERVER_PORT
    RUBRIC_JUDGE_SERVER_HOST
    VLLM_DATA_PARALLEL_SIZE
    VLLM_TENSOR_PARALLEL_SIZE
    VLLM_MAX_MODEL_LEN
    VLLM_TRUST_REMOTE_CODE
    VLLM_ENFORCE_EAGER
    VLLM_GPU_MEMORY_UTILIZATION
    SERVER_URL
    EXIT_CODE
    JUDGE_MODEL_SHORTHAND
    LOCAL_JUDGE_TAG
    NUM_GPUS
    SERVER_OUTPUT
    RUBRIC_JUDGE_SERVER_URL
    HOSTED_VLLM_API_BASE
)

# Function to ensure server is running and get address
ensure_rubric_judge_server_running() {
    echo "=========================================="
    echo "Rubric Judge Server Config"
    echo "=========================================="
    
    # Debug: show what model we're using
    echo "Using model: $RUBRIC_JUDGE_MODEL" >&2
    
    # Call the startup script with variables passed inline (not exported)
    # Output everything directly - don't capture, just label it
    echo "[Server Startup] Starting rubric judge server..." >&2
    echo "[Server Startup] Model: $RUBRIC_JUDGE_MODEL" >&2
    echo "[Server Startup] Output:" >&2
    
    # Run startup script - stderr shows in terminal, stdout contains URL
    SERVER_OUTPUT=$(RUBRIC_JUDGE_MODEL="$RUBRIC_JUDGE_MODEL" \
    RUBRIC_JUDGE_SERVER_PORT="$RUBRIC_JUDGE_SERVER_PORT" \
    RUBRIC_JUDGE_SERVER_HOST="$RUBRIC_JUDGE_SERVER_HOST" \
    VLLM_DATA_PARALLEL_SIZE="$VLLM_DATA_PARALLEL_SIZE" \
    VLLM_TENSOR_PARALLEL_SIZE="$VLLM_TENSOR_PARALLEL_SIZE" \
    VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    VLLM_TRUST_REMOTE_CODE="$VLLM_TRUST_REMOTE_CODE" \
    VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER" \
    VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
    "${ROOT_DIR}/scripts/start_rubric_judge_server.sh")
    EXIT_CODE=$?
    
    # Extract server URL from stdout (startup script outputs URL to stdout)
    SERVER_URL=$(echo "$SERVER_OUTPUT" | grep -E "^http://|^https://" | tail -n1)
    
    if [ $EXIT_CODE -eq 0 ] && [ ! -z "$SERVER_URL" ]; then
        # Set variables (already registered as command env vars above)
        RUBRIC_JUDGE_SERVER_URL="$SERVER_URL"
        
        # Set HOSTED_VLLM_API_BASE for litellm to use
        # litellm expects the API base to include /v1 suffix for OpenAI-compatible endpoints
        HOSTED_VLLM_API_BASE="${SERVER_URL}/v1"
        
        # Transform model name to use hosted_vllm/ prefix for litellm
        # RUBRIC_JUDGE_MODEL is already set to Qwen/Qwen3-1.7B (single source of truth above)
        RUBRIC_JUDGE_MODEL="hosted_vllm/$RUBRIC_JUDGE_MODEL"
        
        echo "Server address: $SERVER_URL"
        echo "=========================================="
        echo "Server configuration complete."
        echo "HOSTED_VLLM_API_BASE=$HOSTED_VLLM_API_BASE"
        echo "RUBRIC_JUDGE_SERVER_URL=$RUBRIC_JUDGE_SERVER_URL"
        echo "RUBRIC_JUDGE_MODEL=$RUBRIC_JUDGE_MODEL"
        echo "=========================================="
    else
        echo "⚠ Warning: Failed to start or get server address" >&2
        echo "Exit code: $EXIT_CODE" >&2
        echo "SERVER_URL: $SERVER_URL" >&2
        echo "" >&2
        echo "This is a fatal error. The training cannot proceed without the server." >&2
        echo "Please check the error messages above and fix the issue." >&2
        exit 1
    fi
}

# Register ensure_rubric_judge_server_running as a post-config function
# launch.sh will call this after all configs are loaded and overrides are applied
# This ensures the server is started before training begins
POST_CONFIG_FUNCTIONS+=("ensure_rubric_judge_server_running")

# Note: Usage with launch.sh:
#   # Start server and run training (server starts first):
#   ./scripts/launch.sh rubric_judge_server rubric_judge dpo_model_ladder
#   
#   # Server will automatically be submitted as SLURM job when launched from within SLURM:
#   ./scripts/launch.sh rubric_judge_server rubric_judge dpo_model_ladder
# 
# Standalone usage (just start server, no training):
#   # Start server directly:
#   source scripts/configs/rubric_judge_server.sh
# 
# Customize SLURM settings by editing:
#   env/slurm_judge_server_template.sh
