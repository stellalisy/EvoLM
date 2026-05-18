#!/bin/bash
# Configure GPT-4o as the LLM judge
# Usage: ./scripts/launch.sh ... api/gpt4o ...

source "${ROOT_DIR}/scripts/configure_api.sh" gpt-4o

# Override the LLM judge model to use the configured model
LLM_JUDGE_MODEL="${LITELLM_MODEL}"

echo "[api/gpt4o] Configured LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}"
