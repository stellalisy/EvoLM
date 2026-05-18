#!/bin/bash
# Inference model config: inference_engine
# Uses dedicated inference engines for question inference
# Requires INFERENCE_MODEL and INFERENCE_NUM_ENGINES to be configured

INFERENCE_MODEL_FOR_QUESTION_INFERENCE="inference_engine"

# Only append suffix if EXP_NAME_BASE was NOT provided as an override
if [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"infie"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE:-}_infie"
fi

echo "[inference_model/inference_engine] INFERENCE_MODEL_FOR_QUESTION_INFERENCE=${INFERENCE_MODEL_FOR_QUESTION_INFERENCE}, EXP_NAME_BASE=${EXP_NAME_BASE}"
#!/bin/bash
# Configuration for inference engine using Ray-based vLLM engines
# This config sets up dedicated inference model engines for question inference
# when using the inferred_question rejected answer method

# INFERENCE_MODEL - Model to use for question inference
# When using Ray-based vLLM engines, set the model path/name here
INFERENCE_MODEL=${INFERENCE_MODEL:-"Qwen/Qwen3-8B"}

# Augment experiment name to indicate inference engine model
# Create compact shorthand from model name (e.g., "Qwen/Qwen3-8B" -> "q38b")
# Extract model family and size, remove separators, keep first letter + numbers
INFERENCE_MODEL_SHORTHAND=$(echo "$INFERENCE_MODEL" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9]//g' | tr '[:upper:]' '[:lower:]')
# Make it compact: keep first letter and numbers (e.g., "qwen38b" -> "q38b")
if [[ "$INFERENCE_MODEL_SHORTHAND" =~ ^([a-z])[a-z]*([0-9].*)$ ]]; then
    INFERENCE_MODEL_SHORTHAND="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
elif [[ "$INFERENCE_MODEL_SHORTHAND" =~ ^([a-z]{2,3}) ]]; then
    # If no numbers, use first 2-3 chars
    INFERENCE_MODEL_SHORTHAND="${BASH_REMATCH[1]}"
else
    # Fallback to "inf" (inference) if we can't create a meaningful shorthand
    INFERENCE_MODEL_SHORTHAND="inf"
fi
# Append "rayinf" (Ray inference) prefix with inference model shorthand to EXP_NAME_BASE using lazy evaluation
# This will be expanded later in launch.sh when EXP_NAME_BASE is evaluated
# Only modify if EXP_NAME_BASE was NOT provided as an override
LOCAL_INFERENCE_TAG="rayinf${INFERENCE_MODEL_SHORTHAND}"
if [ -z "$EXP_NAME_BASE" ]; then
    EXP_NAME_BASE="inf_\${LOCAL_INFERENCE_TAG}"
elif [[ -z "${OVERRIDDEN_VALUES[EXP_NAME_BASE]+x}" ]] && [[ "$EXP_NAME_BASE" != *"\${LOCAL_INFERENCE_TAG}"* ]] && [[ "$EXP_NAME_BASE" != *"rayinf"* ]]; then
    EXP_NAME_BASE="${EXP_NAME_BASE}_\${LOCAL_INFERENCE_TAG}"
fi

# Ray-based vLLM engine configuration
# These settings control the Ray-based inference model vLLM engines
# Set defaults for Ray-based approach (only when using this config)
INFERENCE_NUM_ENGINES=${INFERENCE_NUM_ENGINES:-4}
INFERENCE_TENSOR_PARALLEL_SIZE=${INFERENCE_TENSOR_PARALLEL_SIZE:-1}
INFERENCE_GPU_MEMORY_UTILIZATION=${INFERENCE_GPU_MEMORY_UTILIZATION:-0.9}
INFERENCE_MAX_MODEL_LEN=${INFERENCE_MAX_MODEL_LEN:-32768}

# Declare variables that are used internally by this config (not passed to Python)
# These will be excluded from unused variable warnings
KNOWN_CONFIG_VARIABLES=(
    INFERENCE_MODEL_SHORTHAND
    LOCAL_INFERENCE_TAG
)

# Override function to indicate Ray-based vLLM engines are being used
# This validates the inference engine configuration
ensure_inference_engine_configured() {
    echo "=========================================="
    echo "Inference Engine Ray vLLM Config"
    echo "=========================================="
    echo "Using Ray-based vLLM engines for inference model"
    echo "Model: $INFERENCE_MODEL"
    echo "Number of engines: $INFERENCE_NUM_ENGINES"
    echo "Tensor parallel size: $INFERENCE_TENSOR_PARALLEL_SIZE"
    echo "GPU memory utilization: $INFERENCE_GPU_MEMORY_UTILIZATION"
    echo "Max model length: $INFERENCE_MAX_MODEL_LEN"
    echo "=========================================="
    echo "Ray vLLM inference engine configuration complete."
    echo "=========================================="
}

# Register ensure_inference_engine_configured as a post-config function
# This ensures the config is validated before training begins
POST_CONFIG_FUNCTIONS+=("ensure_inference_engine_configured")

# Note: Usage with launch.sh:
#   # Use Ray-based inference model vLLM engines:
#   ./scripts/launch.sh inference_engine_vllm rubric_judge dpo_model_ladder
#
#   # With custom model:
#   ./scripts/launch.sh inference_engine_vllm rubric_judge dpo_model_ladder INFERENCE_MODEL="Qwen/Qwen3-32B" INFERENCE_NUM_ENGINES=2 INFERENCE_TENSOR_PARALLEL_SIZE=2
#
# This config uses Ray-based vLLM engines for the inference model instead of
# using rubric_judge or policy engines. The engines run in-process within
# the Ray training job.

# Set rejected answer method to inferred_question
REJECTED_ANSWER_METHOD="inferred_question"
