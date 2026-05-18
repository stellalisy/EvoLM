#!/bin/bash
# Configure GPT-4.1 as the LLM judge
# Usage: ./scripts/launch.sh ... api/gpt4_1 ...

source "${ROOT_DIR}/scripts/configure_api.sh" gpt-4.1

# Override the LLM judge model to use the configured model
LLM_JUDGE_MODEL="${LITELLM_MODEL}"

echo "[api/gpt4_1] Configured LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}"
