# Policy Model Configurations

This directory contains configuration files for different policy models used in experiments.

## Available Policy Models

- **`qwen3_8b.sh`**: Qwen3-8B (default model used in main experiments)
- **`llama3_8b.sh`**: Llama-3-8B-Instruct (different architecture for transfer experiments)
- **`mistral_7b.sh`**: Mistral-7B-Instruct-v0.3 (different architecture for transfer experiments)

## Usage

Include the policy config in your launch command to specify which model to use:

```bash
./scripts/launch.sh \
    alternating_training \
    single_model \
    policy/qwen3_8b \
    ...other configs...
```

## What These Configs Do

Each config sets:
- `HF_CHECKPOINT`: The Hugging Face model path
- `EXP_NAME_BASE`: Appends a model identifier suffix (e.g., `_qwen8b`, `_llama8b`)

## Example: Fixed Policy Training with Different Models

Train rubric generator with frozen Llama-3 policy:

```bash
./scripts/launch.sh \
    alternating_training \
    single_model \
    policy/llama3_8b \
    data_provider/inferred_question \
    FREEZE_POLICY_MODEL=true
```

## Example: Single-Policy Transfer

Train Mistral-7B policy with learned rubric from Qwen3-8B:

```bash
./scripts/launch.sh \
    alternating_training \
    single_model \
    policy/mistral_7b \
    LOAD_RUBRIC_CHECKPOINT=/path/to/qwen3_rubric.pt \
    FREEZE_RUBRIC_MODEL=true
```

## Adding New Policy Models

To add a new policy model:

1. Create a new config file: `scripts/configs/policy/your_model.sh`
2. Set `HF_CHECKPOINT` to the Hugging Face model path
3. Add a unique suffix to `EXP_NAME_BASE` for experiment identification
