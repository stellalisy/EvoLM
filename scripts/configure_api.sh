#!/bin/bash
# Configure API credentials for LLM providers
#
# Usage:
#   source scripts/configure_api.sh <model_name>
#
# Examples:
#   source scripts/configure_api.sh gpt-4o        # Configures Azure OpenAI GPT-4o
#   source scripts/configure_api.sh gemini-2.0-flash  # Configures Google Gemini
#   source scripts/configure_api.sh claude-sonnet-4  # Configures AWS Bedrock Claude
#
# After sourcing, LITELLM_MODEL will be set to the appropriate model string
# and all necessary environment variables will be configured.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

configure_api() {
    local model_name="$1"
    
    if [ -z "$model_name" ]; then
        echo "Usage: source configure_api.sh <model_name>" >&2
        echo "       configure_api <model_name>" >&2
        return 1
    fi
    
    # Use Python to generate and execute export commands
    local exports
    exports=$(python -c "
import sys
sys.path.insert(0, '$ROOT_DIR')
from open_instruct.api_config import print_env_exports
try:
    print_env_exports('$model_name')
except Exception as e:
    print(f'echo \"Error: {e}\" >&2', file=sys.stdout)
    sys.exit(1)
" 2>&1)
    
    if [ $? -ne 0 ]; then
        echo "Failed to configure API for '$model_name'" >&2
        echo "$exports" >&2
        return 1
    fi
    
    # Execute the export commands
    eval "$exports"
    
    echo "Configured API for '$model_name' -> $LITELLM_MODEL"
}

# List available models
list_api_models() {
    python -c "
import sys
sys.path.insert(0, '$ROOT_DIR')
from open_instruct.api_config import list_available_models
models = list_available_models()
print('Available models:')
for key, info in sorted(models.items()):
    print(f'  {key}: {info}')
"
}

# If script is sourced with an argument, configure that model
if [ -n "$1" ]; then
    configure_api "$1"
fi
