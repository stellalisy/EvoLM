#!/bin/bash
# Configure Gemini 2.0 Flash as the LLM judge
# Usage: ./scripts/launch.sh ... api/gemini_flash ...

source "${ROOT_DIR}/scripts/configure_api.sh" gemini-2.0-flash

# Override the LLM judge model to use the configured model
LLM_JUDGE_MODEL="${LITELLM_MODEL}"

echo "[api/gemini_flash] Configured LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}"
