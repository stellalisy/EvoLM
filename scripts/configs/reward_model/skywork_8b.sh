#!/bin/bash
# Scalar reward model: Skywork-Reward-V2-Llama-3.1-8B-40M
# Bradley-Terry reward model that scores (query, response) pairs.
# When set, overrides ALL verifiers so every sample is scored by the RM.
#
# The RM actor pool claims REWARD_MODEL_NUM_GPUS GPUs via Ray, so we
# reduce vLLM engines accordingly to leave room.

REWARD_MODEL_NAME="Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M"
REWARD_MODEL_NUM_GPUS=8

# Reserve GPUs for RM replicas by reducing vLLM engine count
VLLM_NUM_ENGINES=$((${VLLM_NUM_ENGINES:-56} - REWARD_MODEL_NUM_GPUS))

echo "[reward_model/skywork_8b] REWARD_MODEL_NAME=${REWARD_MODEL_NAME}"
echo "[reward_model/skywork_8b] REWARD_MODEL_NUM_GPUS=${REWARD_MODEL_NUM_GPUS}"
echo "[reward_model/skywork_8b] VLLM_NUM_ENGINES=${VLLM_NUM_ENGINES} (reduced for RM replicas)"
