#!/bin/bash
# Inference engine config: Qwen3-8B
# Dedicated vLLM engine for question inference (separate from policy/rubric models)
# Used when INFERENCE_MODEL_FOR_QUESTION_INFERENCE="inference_engine"
#
# GPU allocation adjustment:
# Adding inference engine(s) requires reducing other allocations to fit within 64 GPUs.
# We reduce VLLM_NUM_ENGINES to make room for the inference engine.

INFERENCE_MODEL="Qwen/Qwen3-8B"
INFERENCE_NUM_ENGINES=4
INFERENCE_TENSOR_PARALLEL_SIZE=1
INFERENCE_GPU_MEMORY_UTILIZATION=0.85
INFERENCE_MAX_MODEL_LEN=16384

# Reduce policy vLLM engines to make room for inference engine
# Original: 40 (policy) + 16 (rubric_judge) + 8 (training) = 64 GPUs
# With inference: Need to subtract INFERENCE_NUM_ENGINES from VLLM_NUM_ENGINES
# New: 39 (policy) + 16 (rubric_judge) + 8 (training) + 1 (inference) = 64 GPUs
VLLM_NUM_ENGINES=$((${VLLM_NUM_ENGINES:-40} - ${INFERENCE_NUM_ENGINES}))

echo "[inference_engine/qwen3_8b] INFERENCE_MODEL=${INFERENCE_MODEL}, INFERENCE_NUM_ENGINES=${INFERENCE_NUM_ENGINES}, TP=${INFERENCE_TENSOR_PARALLEL_SIZE}"
echo "[inference_engine/qwen3_8b] Reduced VLLM_NUM_ENGINES to ${VLLM_NUM_ENGINES} to make room for inference engine"
