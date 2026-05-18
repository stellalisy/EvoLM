#!/bin/bash
# Configure Claude Sonnet 4 as the LLM judge
# Usage: ./scripts/launch.sh ... api/claude_sonnet ...

source "${ROOT_DIR}/scripts/configure_api.sh" claude-sonnet-4

# Override the LLM judge model to use the configured model
LLM_JUDGE_MODEL="${LITELLM_MODEL}"

echo "[api/claude_sonnet] Configured LLM_JUDGE_MODEL=${LLM_JUDGE_MODEL}"
