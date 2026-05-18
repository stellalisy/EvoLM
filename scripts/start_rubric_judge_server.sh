#!/bin/bash
# Script to start or ensure rubric judge server is running
# Returns the server address via stdout and exit code 0 on success
# Note: When RUN_DIRECTLY=true, the script blocks and does not return (exec replaces process)

# Ensure ROOT_DIR is set
if [ -z "$ROOT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    export ROOT_DIR
fi

# Default values (only set if not already set by config)

# # vLLM server configuration (defaults if not set by config)
# VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
# VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
# VLLM_TRUST_REMOTE_CODE=${VLLM_TRUST_REMOTE_CODE:-false}
# VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}
# VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}

# Load SLURM template configuration - THIS IS THE ONLY SOURCE OF SLURM DEFAULTS
# Note: Template may have already been loaded by the config script, but loading again here
# ensures it's available for job submission. Variables already set will not be overridden.
if [ -f "${ROOT_DIR}/env/slurm_judge_server_template.sh" ]; then
    source "${ROOT_DIR}/env/slurm_judge_server_template.sh"
else
    echo "Error: SLURM template file not found: ${ROOT_DIR}/env/slurm_judge_server_template.sh" >&2
    exit 1
fi

# Generate job name from model and port if not set by template
if [ -z "$SLURM_JOB_NAME" ] || [ "$SLURM_JOB_NAME" = "rubric_judge_server" ]; then
    MODEL_NAME_SANITIZED=$(echo "$RUBRIC_JUDGE_MODEL" | sed 's/[^a-zA-Z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g' | tr '[:upper:]' '[:lower:]')
    # Limit length to 30 chars (SLURM job name limit is 50, leave room for prefix and port)
    MODEL_NAME_SANITIZED=$(echo "$MODEL_NAME_SANITIZED" | cut -c1-30)
    # Add port to job name for identification (e.g., "rubric_judge_qwen317b_p8000")
    SLURM_JOB_NAME="rubric_judge_${MODEL_NAME_SANITIZED}_p${RUBRIC_JUDGE_SERVER_PORT}"
fi

# Always use sbatch for SLURM jobs
SLURM_MODE=sbatch

# Get hostname/IP for the server address
if command -v hostname &> /dev/null; then
    SERVER_HOSTNAME=$(hostname -I | awk '{print $1}' 2>/dev/null || hostname 2>/dev/null || echo "localhost")
else
    SERVER_HOSTNAME="localhost"
fi

# If we're in a cluster environment, try to get the proper address
if [ ! -z "$BEAKER_LEADER_REPLICA_IP" ]; then
    SERVER_HOSTNAME="$BEAKER_LEADER_REPLICA_IP"
elif [ ! -z "$SLURM_NODELIST" ]; then
    if command -v scontrol &> /dev/null; then
        SERVER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
    else
        SERVER_HOSTNAME=$(echo "$SLURM_NODELIST" | sed 's/\[.*\]//' | cut -d',' -f1)
    fi
    # Use hostname directly (e.g., g3080) instead of resolving to IP
elif [ ! -z "$SLURM_JOB_ID" ]; then
    SERVER_HOSTNAME=$(hostname -f 2>/dev/null || hostname)
fi

# Build server address
RUBRIC_JUDGE_SERVER_URL="http://${SERVER_HOSTNAME}:${RUBRIC_JUDGE_SERVER_PORT}"

# Function to start the vLLM server directly
start_server_directly() {
    echo "Starting vLLM server directly..." >&2
    
    # Recalculate hostname in case we're in SLURM (hostname detection happens at script start)
    if [ ! -z "$SLURM_NODELIST" ]; then
        if command -v scontrol &> /dev/null; then
            SERVER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
        else
            SERVER_HOSTNAME=$(echo "$SLURM_NODELIST" | sed 's/\[.*\]//' | cut -d',' -f1)
        fi
        # Use hostname directly (e.g., g3080) instead of resolving to IP
    elif [ ! -z "$SLURM_JOB_ID" ]; then
        # Use hostname (e.g., g3080) instead of IP address
        SERVER_HOSTNAME=$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo "localhost")
    fi
    
    # Rebuild server URL with updated hostname
    RUBRIC_JUDGE_SERVER_URL="http://${SERVER_HOSTNAME}:${RUBRIC_JUDGE_SERVER_PORT}"
    echo "Server will be available at: $RUBRIC_JUDGE_SERVER_URL" >&2
    
    # Ensure Python environment is set up
    if [ -f "${ROOT_DIR}/env/local_config.sh" ]; then
        source "${ROOT_DIR}/env/local_config.sh"
    fi
    
    # Create symlink to SLURM output file if running in SLURM
    if [ ! -z "$SLURM_JOB_ID" ] && [ ! -z "$SLURM_OUTPUT_DIR" ]; then
        TEMP_DIR="${ROOT_DIR}/scratch"
        mkdir -p "$TEMP_DIR"
        SLURM_LOG_FILE="${SLURM_OUTPUT_DIR}/rubric_judge_server_${SLURM_JOB_ID}.out"
        # Use job name for symlink if available, otherwise fall back to generic name
        if [ ! -z "$SLURM_JOB_NAME" ]; then
            LOG_SYMLINK="${TEMP_DIR}/${SLURM_JOB_NAME}.log"
        else
            LOG_SYMLINK="${TEMP_DIR}/rubric_judge_server.log"
        fi
        # Remove existing symlink or file if it exists
        [ -L "$LOG_SYMLINK" ] && rm "$LOG_SYMLINK"
        [ -f "$LOG_SYMLINK" ] && rm "$LOG_SYMLINK"
        # Create symlink to SLURM output file
        ln -s "$SLURM_LOG_FILE" "$LOG_SYMLINK"
        echo "Created symlink: $LOG_SYMLINK -> $SLURM_LOG_FILE" >&2
    fi
    
    local vllm_cmd="vllm serve ${RUBRIC_JUDGE_MODEL}"
    vllm_cmd+=" --host ${RUBRIC_JUDGE_SERVER_HOST}"
    vllm_cmd+=" --port ${RUBRIC_JUDGE_SERVER_PORT}"
    vllm_cmd+=" --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    # Use data parallelism for multiple instances (one per GPU)
    # This creates multiple replicas instead of splitting model across GPUs
    if [ ! -z "${VLLM_DATA_PARALLEL_SIZE}" ] && [ "${VLLM_DATA_PARALLEL_SIZE:-1}" -gt 1 ]; then
        vllm_cmd+=" --data-parallel-size ${VLLM_DATA_PARALLEL_SIZE}"
    fi
    vllm_cmd+=" --max-model-len ${VLLM_MAX_MODEL_LEN}"
    vllm_cmd+=" --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION}"
    
    [ "$VLLM_TRUST_REMOTE_CODE" = "true" ] && vllm_cmd+=" --trust-remote-code"
    [ "$VLLM_ENFORCE_EAGER" = "true" ] && vllm_cmd+=" --enforce-eager"
    
    echo "Starting vLLM server (blocking)..." >&2
    echo "Server will be available at: $RUBRIC_JUDGE_SERVER_URL" >&2
    # Run vLLM server in foreground - this will block until server stops
    # Note: When RUN_DIRECTLY=true, this script does not return - exec replaces the process
    # Output goes to SLURM stdout/stderr (captured in .out and .err files)
    exec bash -c "$vllm_cmd"
}

# Function to check if SLURM job is running
check_slurm_job_running() {
    local job_id="$1"
    if [ -z "$job_id" ]; then
        return 1
    fi
    
    if command -v squeue &> /dev/null; then
        squeue -j "$job_id" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|CONFIGURING"
        return $?
    fi
    return 1
}

# Function to check if SLURM job exists (running, pending, or configuring)
# This prevents submitting duplicate jobs
# COMPLETED and COMPLETING jobs are treated as stopped (not existing)
check_slurm_job_exists() {
    local job_id="$1"
    if [ -z "$job_id" ]; then
        return 1
    fi
    
    if command -v squeue &> /dev/null; then
        # Check if job exists in queue (RUNNING, PENDING, CONFIGURING)
        # COMPLETED and COMPLETING jobs are treated as stopped
        # COMPLETED jobs won't appear in squeue, COMPLETING jobs will but we ignore them
        local state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null | head -n1)
        if [ "$state" = "COMPLETING" ]; then
            return 1  # Treat COMPLETING as stopped
        fi
        echo "$state" | grep -qE "RUNNING|PENDING|CONFIGURING"
        return $?
    fi
    return 1
}

# Function to check if SLURM job is completed (stopped)
check_slurm_job_completed() {
    local job_id="$1"
    if [ -z "$job_id" ]; then
        return 1
    fi
    
    # Check if job is COMPLETED using sacct (more reliable than squeue)
    if command -v sacct &> /dev/null; then
        local state=$(sacct -j "$job_id" -n -o State --noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
        if [ "$state" = "COMPLETED" ]; then
            return 0
        fi
    fi
    
    # Fallback: if job doesn't exist in squeue, it might be completed
    if command -v squeue &> /dev/null; then
        if ! squeue -j "$job_id" -h -o "%T" 2>/dev/null | grep -q .; then
            # Job not in queue - could be completed, cancelled, or failed
            # Check with sacct if available
            if command -v sacct &> /dev/null; then
                local state=$(sacct -j "$job_id" -n -o State --noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
                [ "$state" = "COMPLETED" ] && return 0
            fi
        fi
    fi
    
    return 1
}

# Function to get SLURM job ID by searching squeue for job name pattern
# Uses squeue to find jobs matching the job name (which includes port)
get_slurm_job_id() {
    if [ -z "$SLURM_JOB_NAME" ]; then
        return 1
    fi
    
    if command -v squeue &> /dev/null; then
        # Find job by name pattern - job name includes port so it's unique
        # Format: JOBID NAME USER STATE NODELIST
        JOB_ID=$(squeue -n "$SLURM_JOB_NAME" -h -o "%i" 2>/dev/null | head -n1 | tr -d '[:space:]')
        if [ ! -z "$JOB_ID" ]; then
            echo "$JOB_ID"
            return 0
        fi
    fi
    return 1
}

# Function to get server URL from SLURM job ID
# Returns 0 and outputs URL to stdout if successful, returns 1 if node not available
get_server_url_from_job() {
    local job_id="$1"
    local status_msg="${2:-}"  # Optional status message to print
    
    if [ -z "$job_id" ]; then
        return 1
    fi
    
    if command -v squeue &> /dev/null; then
        NODE=$(squeue -j "$job_id" -h -o "%N" 2>/dev/null | head -n1)
        if [ ! -z "$NODE" ] && [ "$NODE" != "N/A" ]; then
            # Use hostname directly (e.g., g3080) instead of resolving to IP
            SERVER_HOSTNAME="$NODE"
            SERVER_URL="http://${SERVER_HOSTNAME}:${RUBRIC_JUDGE_SERVER_PORT}"
            [ ! -z "$status_msg" ] && echo "$status_msg" >&2
            echo "$SERVER_URL"
            return 0
        fi
    fi
    return 1
}

# Function to submit SLURM job (always uses sbatch)
submit_slurm_job() {
    mkdir -p "$SLURM_OUTPUT_DIR"
    
    # Create temp directory for server files (scratch is already in .gitignore)
    TEMP_DIR="${ROOT_DIR}/scratch"
    mkdir -p "$TEMP_DIR"
    
    # Build SLURM batch script in temp directory
    SLURM_SCRIPT="${TEMP_DIR}/rubric_judge_server_slurm.sh"
    cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --output=${SLURM_OUTPUT_DIR}/rubric_judge_server_%j.out
#SBATCH --error=${SLURM_OUTPUT_DIR}/rubric_judge_server_%j.err
$( [ ! -z "$SLURM_ACCOUNT" ] && echo "#SBATCH --account=${SLURM_ACCOUNT}" )
$( [ ! -z "$SLURM_PARTITION" ] && echo "#SBATCH --partition=${SLURM_PARTITION}" )
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEMORY}
#SBATCH --time=${SLURM_TIME}
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH --exclusive
$( [ ! -z "$SLURM_EXTRA_OPTIONS" ] && echo "#SBATCH ${SLURM_EXTRA_OPTIONS}" )
cd "${ROOT_DIR}"
source "${ROOT_DIR}/env/local_config.sh" 2>/dev/null || true
export ROOT_DIR="${ROOT_DIR}"
export RUBRIC_JUDGE_MODEL="${RUBRIC_JUDGE_MODEL}"
export RUBRIC_JUDGE_SERVER_PORT="${RUBRIC_JUDGE_SERVER_PORT}"
export RUBRIC_JUDGE_SERVER_HOST="${RUBRIC_JUDGE_SERVER_HOST}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}"
export SLURM_OUTPUT_DIR="${SLURM_OUTPUT_DIR}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME}"
export RUN_DIRECTLY=true
# Start server directly - exec replaces process, so this never returns
"${ROOT_DIR}/scripts/start_rubric_judge_server.sh"
EOF
    chmod +x "$SLURM_SCRIPT"
    
    echo "Submitting SLURM job with sbatch..." >&2
    echo "SLURM script: $SLURM_SCRIPT" >&2
    echo "SLURM_ACCOUNT=$SLURM_ACCOUNT, SLURM_PARTITION=$SLURM_PARTITION, SLURM_EXTRA_OPTIONS=$SLURM_EXTRA_OPTIONS" >&2
    
    SBATCH_OUTPUT=$(sbatch "$SLURM_SCRIPT" 2>&1)
    SBATCH_EXIT_CODE=$?
    
    if [ $SBATCH_EXIT_CODE -ne 0 ]; then
        echo "Error: sbatch failed with exit code $SBATCH_EXIT_CODE" >&2
        echo "Output: $SBATCH_OUTPUT" >&2
        echo "SLURM script contents:" >&2
        cat "$SLURM_SCRIPT" >&2
        return 1
    fi
    
    # Try to extract job ID from standard "Submitted batch job XXXXX" format first
    JOB_ID=$(echo "$SBATCH_OUTPUT" | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
    # Fallback: extract first number if standard format not found (some SLURM versions differ)
    [ -z "$JOB_ID" ] && JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oE 'Job [0-9]+' | grep -oE '[0-9]+' | head -n1)
    # Last resort: extract any number (risky but better than nothing)
    [ -z "$JOB_ID" ] && JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oE '[0-9]+' | head -n1)
    
    if [ ! -z "$JOB_ID" ] && [ "$JOB_ID" != "" ]; then
        echo "Job submitted with ID: $JOB_ID" >&2
        echo "$JOB_ID"  # Output job ID to stdout so caller can capture it
        return 0
    else
        echo "⚠ Failed to submit SLURM job. Could not extract job ID." >&2
        echo "sbatch output: $SBATCH_OUTPUT" >&2
        echo "SLURM script contents:" >&2
        cat "$SLURM_SCRIPT" >&2
        # Don't output anything to stdout on failure
        return 1
    fi
}

# Main logic: ensure server is running and return address
main() {
    # If RUN_DIRECTLY=true, skip SLURM job checks and start server directly
    if [ "${RUN_DIRECTLY:-false}" = "true" ]; then
        # Running inside SLURM job - start server directly (blocking)
        # Note: This does not return - exec replaces the process with vLLM server
        start_server_directly
        # This line will never be reached due to exec in start_server_directly
    fi
    
    # Not RUN_DIRECTLY - check if SLURM job exists (running, pending, or configuring)
    JOB_ID=$(get_slurm_job_id)
    if [ ! -z "$JOB_ID" ]; then
        # Get current job state
        JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null | head -n1)
        
        # Check if job is completed or completing (treat as stopped)
        if check_slurm_job_completed "$JOB_ID"; then
            echo "SLURM job $JOB_ID is COMPLETED (treated as stopped). Will start new job." >&2
            # Continue to start a new job
        elif [ "$JOB_STATE" = "COMPLETING" ]; then
            echo "SLURM job $JOB_ID is COMPLETING (treated as stopped). Will start new job." >&2
            # Continue to start a new job
        elif check_slurm_job_exists "$JOB_ID"; then
            echo "Found existing SLURM job $JOB_ID (state: $JOB_STATE)" >&2
            
            # If job is running, get address and return (no health check - if RUNNING, assume server is up)
            if [ "$JOB_STATE" = "RUNNING" ] || [ "$JOB_STATE" = "CONFIGURING" ]; then
                SERVER_URL=$(get_server_url_from_job "$JOB_ID" "SLURM job $JOB_ID running at")
                if [ ! -z "$SERVER_URL" ]; then
                    echo "$SERVER_URL"
                    return 0
                fi
            elif [ "$JOB_STATE" = "PENDING" ]; then
                # Wait for pending job to start
                echo "Job is pending in queue, waiting for it to start..." >&2
                # Wait for pending job to start, then get address
                for i in {1..60}; do
                    sleep 2
                    CURRENT_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null | head -n1)
                    if [ "$CURRENT_STATE" = "RUNNING" ] || [ "$CURRENT_STATE" = "CONFIGURING" ]; then
                        echo "Job started!" >&2
                        SERVER_URL=$(get_server_url_from_job "$JOB_ID")
                        if [ ! -z "$SERVER_URL" ]; then
                            echo "$SERVER_URL"
                            return 0
                        fi
                        break
                    fi
                    echo -n "." >&2
                done
                echo "" >&2
                # Even if not running yet, return the address we'll use
                SERVER_URL=$(get_server_url_from_job "$JOB_ID")
                if [ ! -z "$SERVER_URL" ]; then
                    echo "$SERVER_URL"
                    return 0
                fi
            fi
            # Job exists but we couldn't get address yet - don't submit another
            echo "Using existing job $JOB_ID" >&2
            return 0
        fi
    fi
    
    # Server not running, start it
    
    # # If we're inside a SLURM job but RUN_DIRECTLY is not set, we must submit as a separate job
    # if [ ! -z "$SLURM_JOB_ID" ] || [ ! -z "$SLURM_NODELIST" ]; then
    #     echo "Error: Running inside SLURM environment. Server must be submitted as separate job." >&2
    #     echo "Please submit the server as a separate SLURM job when launching." >&2
    #     return 1
    # fi
    
    # Not in SLURM and RUN_DIRECTLY not set - submit as SLURM job
    if [ "${RUN_DIRECTLY:-false}" != "true" ]; then
        # Submit as separate SLURM job and capture the job ID directly
        # submit_slurm_job outputs messages to stderr and job ID to stdout
        echo "Submitting server as SLURM job..." >&2
        # Capture stdout (job ID) - stderr (messages) will go to stderr naturally
        JOB_ID=$(submit_slurm_job)
        SUBMIT_EXIT=$?
        # Trim whitespace from job ID
        JOB_ID=$(echo "$JOB_ID" | tr -d '[:space:]')
        
        # Validate job ID is non-empty and numeric
        if [ $SUBMIT_EXIT -ne 0 ] || [ -z "$JOB_ID" ] || ! echo "$JOB_ID" | grep -qE '^[0-9]+$'; then
            echo "Error: Failed to submit SLURM job or get valid job ID" >&2
            [ ! -z "$JOB_ID" ] && echo "Captured job ID was: '$JOB_ID'" >&2
            return 1
        fi
        
        echo "Waiting for SLURM job $JOB_ID to start..." >&2
        # Wait for job to start running (up to 5 minutes for resource allocation)
        for i in {1..300}; do
            if check_slurm_job_running "$JOB_ID"; then
                echo "" >&2  # New line after dots
                break
            fi
            echo -n "." >&2  # Print dot without newline
            sleep 1
        done
        
        if ! check_slurm_job_running "$JOB_ID"; then
            echo "Error: SLURM job $JOB_ID failed to start" >&2
            return 1
        fi
        
        # Get server URL from job (job is RUNNING, assume server is up)
        RUBRIC_JUDGE_SERVER_URL=$(get_server_url_from_job "$JOB_ID" "SLURM job $JOB_ID is RUNNING, server should be available at")
        if [ ! -z "$RUBRIC_JUDGE_SERVER_URL" ]; then
            echo "$RUBRIC_JUDGE_SERVER_URL"
            return 0
        else
            echo "Error: Failed to get server URL from job $JOB_ID" >&2
            return 1
        fi
    fi
}

# Run main function and exit with its return code
# Use a subshell or explicit exit to handle return codes properly
EXIT_CODE=0
main "$@" || EXIT_CODE=$?
exit $EXIT_CODE

