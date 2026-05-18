"""
API configuration loader for LLM providers.

Loads credentials from api_info.yaml and configures environment variables
for litellm to use the appropriate provider (Azure, Gemini, Bedrock, etc.).

Usage:
    from open_instruct.api_config import configure_api, get_litellm_model

    # Option 1: Configure env vars and get litellm model string
    litellm_model = configure_api("gpt-4o")
    # Sets AZURE_API_KEY, AZURE_API_BASE, etc. and returns "azure/gpt-4o"

    # Option 2: Just get the litellm model string (for already-configured envs)
    litellm_model = get_litellm_model("gpt-4o")
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to the API info file (sibling to this module, or in env/)
DEFAULT_API_INFO_PATH = Path(__file__).parent / "llm" / "api_info.yaml"


@lru_cache(maxsize=1)
def _load_api_info(api_info_path: str | None = None) -> dict[str, Any]:
    """Load and cache the API info YAML file."""
    path = Path(api_info_path) if api_info_path else DEFAULT_API_INFO_PATH

    if not path.exists():
        logger.warning(f"API info file not found at {path}")
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _find_config(model_name: str, api_info: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Find the config entry for a model name, supporting aliases.

    Tries in order:
    1. Exact match (e.g., "gpt-4o")
    2. Partial match on model_name field

    Returns:
        Tuple of (config_key, config_dict)

    Raises:
        ValueError if no matching config is found
    """
    # 1. Exact match
    if model_name in api_info:
        return model_name, api_info[model_name]

    # 2. Match by model_name field
    for key, config in api_info.items():
        if isinstance(config, dict) and config.get("model_name") == model_name:
            return key, config

    raise ValueError(f"No API config found for model '{model_name}'. Available configs: {list(api_info.keys())}")


def _configure_azure(config: dict[str, Any]) -> str:
    """Configure environment for Azure OpenAI and return litellm model string."""
    model_name = config.get("model_name", "")

    # Set Azure-specific environment variables
    if "api_key" in config:
        os.environ["AZURE_API_KEY"] = config["api_key"]
    if "api_base" in config:
        os.environ["AZURE_API_BASE"] = config["api_base"]
    if "api_version" in config:
        os.environ["AZURE_API_VERSION"] = config["api_version"]

    # litellm format: azure/<deployment-name>
    # Use the model name directly - Azure deployments match the model_name in api_info.yaml
    return f"azure/{model_name}"


def _configure_gemini(config: dict[str, Any]) -> str:
    """Configure environment for Google Gemini and return litellm model string."""
    model_name = config.get("model_name", "")

    if "api_key" in config:
        os.environ["GEMINI_API_KEY"] = config["api_key"]

    # litellm format: gemini/<model-name>
    return f"gemini/{model_name}"


def _configure_bedrock(config: dict[str, Any]) -> str:
    """Configure environment for AWS Bedrock (Claude) and return litellm model string."""
    model_name = config.get("model_name", "")

    # Set AWS credentials
    if "aws_access_key_id" in config:
        os.environ["AWS_ACCESS_KEY_ID"] = config["aws_access_key_id"]
    if "aws_secret_access_key" in config:
        os.environ["AWS_SECRET_ACCESS_KEY"] = config["aws_secret_access_key"]
    if "aws_session_token" in config:
        os.environ["AWS_SESSION_TOKEN"] = config["aws_session_token"]

    # Default region for Bedrock
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")

    # litellm format: bedrock/<model-id>
    return f"bedrock/{model_name}"


def _configure_openai(config: dict[str, Any]) -> str:
    """Configure environment for direct OpenAI API and return litellm model string."""
    model_name = config.get("model_name", "")

    if "api_key" in config:
        os.environ["OPENAI_API_KEY"] = config["api_key"]

    return model_name


def _detect_provider(config: dict[str, Any]) -> str:
    """Detect the provider type from config."""
    api_type = config.get("api_type", "").lower()

    if api_type == "azure" or api_type == "azure_openai":
        return "azure"
    elif api_type == "openai":
        return "openai"
    elif "aws_access_key_id" in config or "anthropic" in config.get("model_name", ""):
        return "bedrock"
    elif (
        "gemini" in config.get("model_name", "").lower()
        or config.get("api_key")
        and "gemini" in str(config.get("model_name", ""))
    ):
        return "gemini"

    # Default to azure if api_key and api_base present
    if "api_key" in config and "api_base" in config:
        return "azure"

    return "unknown"


def configure_api(model_name: str, api_info_path: str | None = None, quiet: bool = False) -> str:
    """
    Configure environment variables for the specified model and return litellm model string.

    Args:
        model_name: Model name or alias (e.g., "gpt-4o", "gemini-2.0-flash")
        api_info_path: Optional path to api_info.yaml (uses default if not specified)
        quiet: If True, suppress info logging

    Returns:
        The litellm-compatible model string (e.g., "azure/gpt-4o-standard")

    Raises:
        ValueError: If no config found for the model

    Example:
        >>> model = configure_api("gpt-4o")
        >>> # Environment is now configured
        >>> response = await litellm.acompletion(model=model, messages=[...])
    """
    api_info = _load_api_info(api_info_path)
    config_key, config = _find_config(model_name, api_info)

    provider = _detect_provider(config)

    if provider == "azure":
        litellm_model = _configure_azure(config)
    elif provider == "gemini":
        litellm_model = _configure_gemini(config)
    elif provider == "bedrock":
        litellm_model = _configure_bedrock(config)
    elif provider == "openai":
        litellm_model = _configure_openai(config)
    else:
        logger.warning(f"Unknown provider for {config_key}, returning raw model name")
        litellm_model = config.get("model_name", model_name)

    if not quiet:
        logger.info(f"Configured API for '{model_name}' -> {litellm_model} (provider: {provider})")

    return litellm_model


def get_litellm_model(model_name: str, api_info_path: str | None = None) -> str:
    """
    Get the litellm model string without configuring environment variables.

    Useful when env vars are already configured (e.g., in a subprocess).

    Args:
        model_name: Model name or alias
        api_info_path: Optional path to api_info.yaml

    Returns:
        The litellm-compatible model string
    """
    api_info = _load_api_info(api_info_path)
    config_key, config = _find_config(model_name, api_info)
    provider = _detect_provider(config)

    if provider == "azure":
        name = config.get("model_name", "")
        return f"azure/{name}"
    elif provider == "gemini":
        return f"gemini/{config.get('model_name', '')}"
    elif provider == "bedrock":
        return f"bedrock/{config.get('model_name', '')}"
    elif provider == "openai":
        return config.get("model_name", model_name)

    return config.get("model_name", model_name)


def list_available_models(api_info_path: str | None = None) -> dict[str, str]:
    """
    List all available models and their providers.

    Returns:
        Dict mapping config key to provider type
    """
    api_info = _load_api_info(api_info_path)
    result = {}

    for key, config in api_info.items():
        if isinstance(config, dict):
            provider = _detect_provider(config)
            model_name = config.get("model_name", "")
            result[key] = f"{provider}: {model_name}"

    return result


# Convenience function for shell scripts
def print_env_exports(model_name: str, api_info_path: str | None = None) -> None:
    """
    Print shell export commands for the specified model.

    Useful for sourcing in shell scripts:
        eval $(python -c "from open_instruct.api_config import print_env_exports; print_env_exports('gpt-4o')")
    """
    api_info = _load_api_info(api_info_path)
    config_key, config = _find_config(model_name, api_info)
    provider = _detect_provider(config)

    exports = []

    if provider == "azure":
        if "api_key" in config:
            exports.append(f'export AZURE_API_KEY="{config["api_key"]}"')
        if "api_base" in config:
            exports.append(f'export AZURE_API_BASE="{config["api_base"]}"')
        if "api_version" in config:
            exports.append(f'export AZURE_API_VERSION="{config["api_version"]}"')
        name = config.get("model_name", "")
        exports.append(f'export LITELLM_MODEL="azure/{name}"')

    elif provider == "gemini":
        if "api_key" in config:
            exports.append(f'export GEMINI_API_KEY="{config["api_key"]}"')
        exports.append(f'export LITELLM_MODEL="gemini/{config.get("model_name", "")}"')

    elif provider == "bedrock":
        if "aws_access_key_id" in config:
            exports.append(f'export AWS_ACCESS_KEY_ID="{config["aws_access_key_id"]}"')
        if "aws_secret_access_key" in config:
            exports.append(f'export AWS_SECRET_ACCESS_KEY="{config["aws_secret_access_key"]}"')
        if "aws_session_token" in config:
            exports.append(f'export AWS_SESSION_TOKEN="{config["aws_session_token"]}"')
        exports.append('export AWS_REGION_NAME="us-east-1"')
        exports.append(f'export LITELLM_MODEL="bedrock/{config.get("model_name", "")}"')

    for line in exports:
        print(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configure API credentials for LLM providers")
    parser.add_argument("model", nargs="?", help="Model name or alias")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--export", action="store_true", help="Print shell export commands")
    parser.add_argument("--config", type=str, help="Path to api_info.yaml")

    args = parser.parse_args()

    if args.list:
        models = list_available_models(args.config)
        print("Available models:")
        for key, info in sorted(models.items()):
            print(f"  {key}: {info}")
    elif args.model:
        if args.export:
            print_env_exports(args.model, args.config)
        else:
            litellm_model = configure_api(args.model, args.config)
            print(f"Configured: {litellm_model}")
    else:
        parser.print_help()
