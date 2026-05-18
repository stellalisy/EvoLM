# EvoLM: Evolving Rubric-Based LLM Evaluation and Training

This repository contains the implementation of EvoLM, a framework for training language models using evolving rubric-based evaluation. EvoLM co-trains a rubric generator and policy model, where rubrics provide structured, interpretable criteria for evaluating model outputs.

## Repository Structure

```
open_instruct/          # Core training framework
  grpo_fast.py          # GRPO trainer implementation
  search_rewards/       # Reward computation (rubric-based, RAR, RRD, RLCER, etc.)
  llm/                  # LLM provider integrations (OpenAI, Gemini, Claude, vLLM)

scripts/                # Training launcher and configuration
  launch.sh             # Config-based training launcher
  base_config.sh        # Default training configuration
  train_rubric_policy_joint.py  # Main training entry point
  rubric_data_provider.py       # Training data provider
  configs/              # Modular config overlays
    reward_mode/        # Baseline reward methods (RAR, RRD, RLCER, random, etc.)
    policy/             # Policy model configs
    rubric_judge/       # Rubric judge model configs
    multi_judge/        # Multi-judge configurations
    api/                # API provider configs
  eval/                 # Evaluation scripts and utilities

env/                    # Local environment config (user-created from templates)
olmes/                  # Evaluation framework (modified from OLMo Eval)
JudgeBench/             # Judge evaluation benchmark (modified)
reward-bench/           # RewardBench (git submodule)
tests/                  # Unit tests
```

## Setup

### Prerequisites

- Python 3.12
- CUDA-compatible GPU(s) (8x H100 recommended for full training)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/stellalisy/EvoLM.git
cd EvoLM

# Install dependencies
uv sync

# Set up olmes (evaluation framework)
cd olmes && pip install -e . && cd ..
```

### Environment Configuration

```bash
# Create local environment config from template
cp env/local_config.sh.template env/local_config.sh
# Edit env/local_config.sh with your paths, HF token, and other settings

# Configure API credentials (if using API-based rubric generation, e.g., GPT-4.1)
cp open_instruct/llm/api_info.yaml.template open_instruct/llm/api_info.yaml
# Edit api_info.yaml with your API keys

# For SLURM-based rubric judge server (optional, multi-node only)
cp env/slurm_judge_server_template.sh.template env/slurm_judge_server_template.sh
# Edit with your SLURM account/partition settings
```

## Training

EvoLM uses a modular config system. Training is launched via `scripts/launch.sh` with composable config overlays:

```bash
# Example: EvoLM training with Qwen3-8B (default policy + rubric co-training)
./scripts/launch.sh \
  qwen3_8b_8nodes_singlemodel \
  paper_output \
  rubric_judge/qwen3_1_7b \
  data_provider/rubric \
  alt/alt50 \
  replay_buffer_gap/gap20_100

# Example: RAR baseline
./scripts/launch.sh \
  qwen3_8b_8nodes_singlemodel \
  paper_output \
  rubric_judge/qwen3_1_7b \
  reward_mode/rar_implicit \
  alt/alt50 \
  replay_buffer_gap/gap20_100

# Example: RRD baseline
./scripts/launch.sh \
  qwen3_8b_8nodes_singlemodel \
  paper_output \
  rubric_judge/qwen3_1_7b \
  reward_mode/rrd_wu \
  alt/alt50 \
  replay_buffer_gap/gap20_100
```

### Config Overlays

Configs are composable shell scripts in `scripts/configs/`. Pass them as arguments to `launch.sh` to override defaults:

- **Cluster size**: `qwen3_8b_8nodes_singlemodel`, `qwen3_8b_4nodes`, etc.
- **Output**: `paper_output` (saves checkpoints every 25 steps)
- **Rubric judge**: `rubric_judge/qwen3_1_7b`, `rubric_judge/llama3_2_1b_instruct`, etc.
- **Data provider**: `data_provider/rubric`, `data_provider/inferred_question`, `data_provider/combined`
- **Alternation schedule**: `alt/alt50` (alternate every 50 steps), `alt/alt20`, etc.
- **Replay buffer**: `replay_buffer_gap/gap20_100`
- **API judge**: `api/gpt4_1`, `api/claude_sonnet`, `api/gemini_flash`

### Reward Modes (Baselines)

The following reward computation methods are implemented in `scripts/configs/reward_mode/`:

| Config | Method | Description |
|--------|--------|-------------|
| (default) | **EvoLM** | Evolving rubric-based rewards with co-training |
| `rar_implicit` | RAR | Reference Answer Rating (implicit aggregation) |
| `rar_explicit` | RAR | Reference Answer Rating (explicit aggregation) |
| `rrd_wu` | RRD | Recursive Rubric Decomposition (whitened-uniform) |
| `rrd_llm` | RRD | Recursive Rubric Decomposition (LLM aggregation) |
| `rlcer` | RLCER | RL with Concept Evolving Rubrics |
| `rlcer_evolving` | RLCER-E | RLCER with evolving rubrics |
| `rubric_arm` | Rubric-ARM | Alternating RL for rubric-based reward modeling |
| `direct_likert` | Direct Likert | Direct quality scoring without rubrics |
| `reference_likert` | Reference Likert | Likert scoring with reference answer |
| `random` | Random | Random reward baseline (control) |

## Evaluation

### OLMES Benchmarks

Run standard benchmarks (GSM8K, MATH, HumanEval+, MBPP+, BBH, MMLU, etc.) on training checkpoints:

```bash
python scripts/eval/run_olmes_on_checkpoints.py \
  --checkpoint-dir /path/to/checkpoints \
  --tasks gsm8k,minerva_math,humaneval_plus,mbppplus,bbh,mmlu \
  --step 500
```

### RewardBench 2

```bash
bash scripts/eval/eval_rewardbench2.sh /path/to/experiment 500
```

### JudgeBench

```bash
bash scripts/eval/eval_judgebench.sh /path/to/experiment 500
```

### Sharded Evaluation

For large benchmarks, evaluation can be sharded across multiple jobs:

```bash
# Run sharded evaluation
bash scripts/eval/launch_sharded_eval.sh /path/to/experiment 500 popqa 40

# Merge shards after completion
python scripts/eval/merge_eval_shards.py \
  --output-root /path/to/experiment/olmes_eval \
  --step 500 --task-name popqa --num-shards 40
```

## Citation

If you use this code, please cite:

```bibtex
@article{evolm2026,
  title={EvoLM: Evolving Rubric-Based LLM Evaluation and Training},
  author={...},
  year={2026}
}
```

## License

This project builds on [open-instruct](https://github.com/allenai/open-instruct) (Apache 2.0), [olmes](https://github.com/allenai/olmes), [JudgeBench](https://github.com/ScalerLab/JudgeBench), and [RewardBench](https://github.com/allenai/reward-bench).
