#!/bin/bash
# Main entry point for experiment management system
# Usage: ./launch.sh [config1] [config2] ... [override1=value1] [override2=value2] ...

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT_DIR  # Export early so config functions can use it

# Source local config first (environment-specific setup)
if [ -f "$ROOT_DIR/env/local_config.sh" ]; then
    source "$ROOT_DIR/env/local_config.sh"
fi

# Setup Ray cluster
source "$SCRIPT_DIR/slurm_ray_setup.sh"

# ============================================================================
# Command Environment Variable Registration
# ============================================================================
# Config files can register variables that should be prefixed to the command
# as environment variables (VAR=value python script.py ...) instead of using
# 'export' which modifies the shell environment globally.
#
# Usage in config files:
#   LOCAL_DATASET_DIR="/path/to/data"
#   register_command_env_var LOCAL_DATASET_DIR
#
# This will result in the command being executed as:
#   LOCAL_DATASET_DIR=/path/to/data python script.py --args ...
#
# Benefits:
#   - Variables are scoped only to the command, not the entire shell
#   - All environment variables are tracked and visible in output
#   - Prevents hidden configuration that bypasses variable tracking
# ============================================================================
register_command_env_var() {
    local var_name="$1"
    if [[ ! " ${COMMAND_ENV_VARS[@]} " =~ " ${var_name} " ]]; then
        COMMAND_ENV_VARS+=("$var_name")
    fi
}

# Source base config (defaults and run_training function)
if [ -f "$SCRIPT_DIR/base_config.sh" ]; then
    source "$SCRIPT_DIR/base_config.sh"
else
    echo "Error: base_config.sh not found!" >&2
    exit 1
fi

# Parse arguments: separate config files from overrides
CONFIG_FILES=()
OVERRIDES=()

for arg in "$@"; do
    if [[ "$arg" == *"="* ]]; then
        # This is an override (KEY=VALUE format)
        OVERRIDES+=("$arg")
    else
        # This is a config file name
        CONFIG_FILES+=("$arg")
    fi
done

# Apply overrides FIRST (before loading configs)
# This ensures overrides take precedence and configs cannot override them
# Track overridden variables and their values for assertion
declare -A OVERRIDDEN_VALUES
for override in "${OVERRIDES[@]}"; do
    key="${override%%=*}"
    value="${override#*=}"
    echo "Override: $key=$value"
    export "$key"="$value"
    OVERRIDDEN_VALUES["$key"]="$value"
done

# Save original SCRIPT_DIR before sourcing configs (they may modify it)
ORIGINAL_SCRIPT_DIR="$SCRIPT_DIR"

# Initialize array for configs to register post-processing functions
# Configs can add function names to this array, and launch.sh will call them after overrides
POST_CONFIG_FUNCTIONS=()

# Load config files in order and collect variable names
# Configs will see overridden values and cannot change them
CONFIG_VARIABLES=()
ALL_KNOWN_CONFIG_VARIABLES=()
COMMAND_ENV_VARS=()  # Track variables that should be prefixed to command as VAR=value
for config_file in "${CONFIG_FILES[@]}"; do
    # Try with .sh extension if not provided
    if [[ ! "$config_file" == *.sh ]]; then
        config_file="${config_file}.sh"
    fi
    
    # Use ORIGINAL_SCRIPT_DIR to avoid issues if config files modify SCRIPT_DIR
    config_path="$ORIGINAL_SCRIPT_DIR/configs/$config_file"
    
    if [ -f "$config_path" ]; then
        echo "Loading config: $config_path"
        
        # ========================================================================
        # Guardrail: Forbid export statements in config files
        # ========================================================================
        # Export statements are forbidden because they create hidden configuration
        # that isn't tracked by the variable detection system. Instead, configs
        # should use register_command_env_var() to register variables that need
        # to be available as environment variables to the Python command.
        # Those variables will be prefixed to the command as VAR=value python ...
        # ========================================================================
        config_exports=()
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            # Extract variable name from export VAR_NAME pattern
            if [[ "$line" =~ ^[[:space:]]*export[[:space:]]+([A-Z_][A-Z0-9_]*) ]]; then
                exported_var="${BASH_REMATCH[1]}"
                config_exports+=("$exported_var")
            fi
        done < "$config_path"
        
        # Error if export statements found
        if [ ${#config_exports[@]} -gt 0 ]; then
            echo "Error: Config file $config_file contains 'export' statements!" >&2
            echo "Export statements are forbidden. Variables will be prefixed to the command instead." >&2
            echo "Please remove 'export' keywords from these lines:" >&2
            grep -nE '^[[:space:]]*export[[:space:]]+[A-Z_][A-Z0-9_]*' "$config_path" | sed 's|^|  |' >&2
            echo "" >&2
            echo "These variables will be automatically prefixed to the command as: VAR=value python ..." >&2
            exit 1
        fi
        
        # Extract variable names from config file before sourcing
        # Match patterns like VAR=value or VAR=${OTHER_VAR:-default}
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            # Extract variable name from VAR=value pattern
            if [[ "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)= ]]; then
                var_name="${BASH_REMATCH[1]}"
                # Skip if already in list
                if [[ ! " ${CONFIG_VARIABLES[@]} " =~ " ${var_name} " ]]; then
                    CONFIG_VARIABLES+=("$var_name")
                fi
            fi
        done < "$config_path"
        source "$config_path"
        
        # Assert that overridden variables were not changed by this config
        for var_name in "${!OVERRIDDEN_VALUES[@]}"; do
            expected_value="${OVERRIDDEN_VALUES[$var_name]}"
            actual_value="${!var_name}"
            if [ "$actual_value" != "$expected_value" ]; then
                echo "Error: Overridden variable $var_name was changed from '$expected_value' to '$actual_value' by config $config_file" >&2
                exit 1
            fi
        done
        
        # Collect known variables from this config file (if defined)
        if [ -n "${KNOWN_CONFIG_VARIABLES+x}" ]; then
            for known_var in "${KNOWN_CONFIG_VARIABLES[@]}"; do
                if [[ ! " ${ALL_KNOWN_CONFIG_VARIABLES[@]} " =~ " ${known_var} " ]]; then
                    ALL_KNOWN_CONFIG_VARIABLES+=("$known_var")
                fi
            done
            # Clear it so next config can define its own
            unset KNOWN_CONFIG_VARIABLES
        fi
    else
        echo "Error: Config file not found: $config_path" >&2
        echo "Available config files in $ORIGINAL_SCRIPT_DIR/configs/:" >&2
        ls -1 "$ORIGINAL_SCRIPT_DIR/configs/"*.sh 2>/dev/null | sed 's|.*/||' | sed 's|^|  |' >&2 || echo "  (none found)" >&2
        exit 1
    fi
done

# Helper function to assert overridden variables haven't changed
assert_overrides_unchanged() {
    local context="$1"
    for var_name in "${!OVERRIDDEN_VALUES[@]}"; do
        expected_value="${OVERRIDDEN_VALUES[$var_name]}"
        actual_value="${!var_name}"
        if [ "$actual_value" != "$expected_value" ]; then
            echo "Error: Overridden variable $var_name was changed from '$expected_value' to '$actual_value' $context" >&2
            exit 1
        fi
    done
}

# Perform post-processing: call functions registered by configs
# Configs register their post-processing functions in POST_CONFIG_FUNCTIONS array
# launch.sh calls them generically without knowing what they do
for func_name in "${POST_CONFIG_FUNCTIONS[@]}"; do
    if type "$func_name" &>/dev/null; then
        "$func_name"
        assert_overrides_unchanged "by $func_name"
    else
        echo "Warning: Post-config function '$func_name' registered but not found. Skipping." >&2
    fi
done

# Generate experiment directory name (for file organization)
# Get date in MMDD format
DATE=${DATE:-$(date +%m%d)}

# Get git hash (first 7 characters)
# Check if git is available and we're in a git repository
if (cd "$ROOT_DIR" && git rev-parse --git-dir >/dev/null 2>&1); then
    GIT_HASH=$(cd "$ROOT_DIR" && git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
else
    GIT_HASH="nogit"
fi

# Build config string from config file names
CONFIG_STRING=""
if [ ${#CONFIG_FILES[@]} -gt 0 ]; then
    CONFIG_STRING=$(IFS=-; echo "${CONFIG_FILES[*]}")
    # Remove .sh extension if present
    CONFIG_STRING="${CONFIG_STRING//.sh/}"
fi

# Build experiment directory name (for organizing outputs)
if [ -n "$CONFIG_STRING" ]; then
    EXP="${DATE}-${GIT_HASH}-${CONFIG_STRING}"
else
    EXP="${DATE}-${GIT_HASH}-manual"
fi

# Set EXP_NAME for Python script (use EXP_NAME_BASE from config if available, otherwise use EXP)
if [ -z "$EXP_NAME" ]; then
    if [ -n "$EXP_NAME_BASE" ]; then
        # Expand EXP_NAME_BASE (this will expand ${NUM_QUESTIONS} and other variables)
        # Example pattern: multiquestion_fgrpo_finegrained_plus_averaged_on_reward_3q
        EXP_NAME=$(eval echo "$EXP_NAME_BASE")
        echo "Using EXP_NAME_BASE from config: $EXP_NAME"
    else
        EXP_NAME="$EXP"
        echo "Generated experiment name: $EXP_NAME"
    fi
else
    echo "Using provided experiment name: $EXP_NAME"
fi

# Set output directory based on model and experiment
# Note: The Python script will append run_name (exp_name__seed__timestamp) to this
MODEL_ID=${MODEL_ID:-$(basename "${HF_CHECKPOINT:-unknown}")}
# Use a base output directory - Python script will append run_name
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-"${ROOT_DIR}/output"}

# Experiment group for organizing outputs (e.g., "main", "baselines", "ablations/alternating_frequency")
# This mirrors the log directory structure for consistency
EXPERIMENT_GROUP=${EXPERIMENT_GROUP:-}

# Sanitize paths by replacing / with + to avoid nested directories
MODEL_ID_SAFE="${MODEL_ID//\//+}"
EXP_SAFE="${EXP//\//+}"

# Use EXP_NAME for directory if EXP_NAME_BASE was set (shorter names)
# Otherwise fall back to EXP (full config string)
if [ -n "$EXP_NAME_BASE" ]; then
    # EXP_NAME was set from EXP_NAME_BASE, use it for shorter directory names
    DIR_NAME="${DATE}-${GIT_HASH}-${EXP_NAME//\//+}"
else
    # No EXP_NAME_BASE, use full config string
    DIR_NAME="${EXP_SAFE}"
fi

# Build output directory with experiment group if specified
# Skip if OUTPUT_DIR was explicitly overridden (e.g., for resuming into an existing directory)
if [ -n "${OVERRIDDEN_VALUES[OUTPUT_DIR]+x}" ]; then
    echo "Using overridden OUTPUT_DIR: $OUTPUT_DIR"
elif [ -n "$EXPERIMENT_GROUP" ]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${EXPERIMENT_GROUP}/${DIR_NAME}"
else
    # Legacy path: OUTPUT_DIR_BASE/MODEL_ID/DIR_NAME
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_ID_SAFE}/${DIR_NAME}"
fi

# Set checkpoint directory - Python script uses checkpoint_state_dir for full checkpoints
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint"

# Export these for use in base_config.sh
export EXP              # Directory name (for organizing outputs)
export EXP_NAME         # Experiment name (for Python script, wandb, etc.)
export OUTPUT_DIR
export OUTPUT_DIR_BASE
export CHECKPOINT_DIR
# ROOT_DIR already exported earlier
export MODEL_ID

echo "=========================================="
echo "Experiment Configuration:"
echo "  Directory name (EXP): $EXP"
echo "  Python exp_name (EXP_NAME): $EXP_NAME"
echo "  Output directory: $OUTPUT_DIR"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
if [ ${#COMMAND_ENV_VARS[@]} -gt 0 ]; then
    echo "  Command env variables: ${COMMAND_ENV_VARS[*]}"
fi
echo "=========================================="

# Note: COMMAND_ENV_VARS array is available to run_training function
# (arrays can't be exported, but it's in the same shell scope)

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Check for unused config variables BEFORE running training
if [ ${#CONFIG_FILES[@]} -gt 0 ] && [ ${#CONFIG_VARIABLES[@]} -gt 0 ]; then
    echo "Checking for unused configuration variables..."
    USED_VARIABLES=()
    DRY_RUN=true run_training
    
    # Build list of all known variables (internal + config-declared)
    INTERNAL_KNOWN_VARS=(
        CONFIG_STRING DATE GIT_HASH EXP MODEL_ID
        OUTPUT_DIR_BASE CHECKPOINT_DIR SCRIPT_DIR ROOT_DIR ORIGINAL_SCRIPT_DIR
        CONFIG_FILES OVERRIDES CONFIG_VARIABLES USED_VARIABLES UNUSED_VARIABLES
        ALL_KNOWN_CONFIG_VARIABLES KNOWN_CONFIG_VARIABLES EXP_NAME_BASE
    )
    
    UNUSED_VARIABLES=()
    for var in "${CONFIG_VARIABLES[@]}"; do
        # Skip variables that are known to be used for internal purposes
        # Check against internal known vars
        SKIP_VAR=false
        for known_var in "${INTERNAL_KNOWN_VARS[@]}"; do
            if [ "$var" = "$known_var" ]; then
                SKIP_VAR=true
                break
            fi
        done
        
        # Check against config-declared known vars
        if [ "$SKIP_VAR" = false ]; then
            for known_var in "${ALL_KNOWN_CONFIG_VARIABLES[@]}"; do
                if [ "$var" = "$known_var" ]; then
                    SKIP_VAR=true
                    break
                fi
            done
        fi
        
        if [ "$SKIP_VAR" = true ]; then
            continue
        fi
        
        # Check if variable is in used list
        if [[ ! " ${USED_VARIABLES[@]} " =~ " ${var} " ]]; then
            UNUSED_VARIABLES+=("$var")
        fi
    done
    
    if [ ${#UNUSED_VARIABLES[@]} -gt 0 ]; then
        echo "==========================================" >&2
        echo "ERROR: Unused configuration variables detected!" >&2
        echo "The following variables are set in config files but not used in the Python command:" >&2
        for var in "${UNUSED_VARIABLES[@]}"; do
            echo "  - $var" >&2
        done
        echo "" >&2
        echo "This likely indicates a typo or outdated parameter." >&2
        echo "Please remove these variables or ensure they are used in base_config.sh" >&2
        echo "==========================================" >&2
        exit 1
    fi
    echo "✓ All configuration variables are used"
fi

# Run training
# Note: Dataset generation (if needed) is handled by config files when they are sourced
run_training

