#!/usr/bin/env python
"""
Driver script for alternating rubric and policy training using Ray actors.

This module combines the alternating training logic and driver script into a single file,
similar to grpo_fast.py structure.

Example usage (single-node):
    python scripts/train_rubric_policy_joint.py \
        --question "Explain quantum entanglement in simple terms." \
        --rubric-model gpt-4o-mini \
        --policy-model hosted_vllm/Qwen/QwQ-32B-Preview \
        --cycles 3 \
        --steps-per-phase 5

Example usage (multi-node Ray cluster):
    python scripts/train_rubric_policy_joint.py \
        --questions-file data/questions.txt \
        --rubric-model gpt-4o-mini \
        --policy-model hosted_vllm/Qwen/QwQ-32B-Preview \
        --baseline-model gpt-4o-mini \
        --cycles 10 \
        --steps-per-phase 5 \
        --ray-address auto \
        --ray-namespace rubric_policy_training
"""

from __future__ import annotations

import asyncio
import collections
import copy
import math
from concurrent import futures
import json
import logging
import os
import random
import threading
import time
import uuid
from concurrent import futures
from collections.abc import Callable
from concurrent import futures
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

import msgspec
import numpy as np
import ray
import torch
import vllm
from datasets import Dataset
from ray.util import queue as ray_queue
from rich.pretty import pprint
from transformers import HfArgumentParser

from open_instruct import grpo_fast
from open_instruct.grpo_fast import Args as GrpoArgs
from open_instruct.grpo_fast import ModelConfig as GrpoModelConfig
from open_instruct.grpo_fast import (
    EnrichedGenerationResult,
    PendingQueriesMap,
    ShufflingIterator,
    create_generation_configs,
    create_model_and_optimizer,
    data_preparation_thread,
    load_data_from_packing_thread,
    one_training_step,
    reward_computation_thread,
    setup_datasets,
    setup_experiment_tracking,
    weight_sync_thread,
)
from open_instruct.grpo_fast import TokenizerConfig as GrpoTokenizerConfig
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.rl_utils import Timer
from open_instruct.search_rewards.rlcer_rubric_utils import (
    RLCERRubricSpec,
    generate_rlcer_rubric_spec,
    precompute_rlcer_evolving_rollout_rubrics,
)
from open_instruct.search_rewards.utils.rubric_chat_templates import format_messages
from open_instruct.search_rewards.utils.run_utils import run_litellm_async
from open_instruct.utils import ray_get_with_progress
from scripts.rubric_data_provider import BaseDataProvider, create_data_provider

LOGGER = logging.getLogger("alternating_training_driver")

# ---------------------------------------------------------------------------
# Dataclasses exchanged between rubric and policy trainers
# ---------------------------------------------------------------------------


@dataclass
class RubricSpec:
    """Container for a rubric that will be shared with the policy trainer."""

    rubric_id: str
    question: str
    rubric_text: str
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationExample:
    """Container for a single generation example to be logged."""

    training_step: int  # Gradient update count (consistent with wandb x-axis)
    actor_type: str  # "policy" or "rubric"
    example_index: int  # Index within the batch (0, 1, 2 for 3 examples)
    question: str
    rubric: str
    policy_answer: str
    score: float
    accepted_reasoning: str = ""
    # Additional fields for rubric training
    rejected_answer: str = ""
    rejected_reasoning: str = ""
    accepted_score: float = 0.0
    rejected_score: float = 0.0
    reward: float = 0.0
    # Field for question inference approach - stores what question the model thought it was answering
    inferred_question: str = ""
    # Fields for replay buffer - track policy step gap for analysis
    rejected_step: int = 0  # Policy step when rejected sample was generated
    current_step: int = 0   # Current policy step
    step_gap: int = 0       # Difference: current_step - rejected_step
    # RRD-specific debug/procedure fields.
    rrd_weighting_method: str = ""
    rrd_iterations: int = 0
    rrd_rejected_count: int = 0
    rrd_rubric_items: list[str] = field(default_factory=list)
    rrd_weights: list[float] = field(default_factory=list)
    rrd_binary_scores: list[float] = field(default_factory=list)
    rrd_trace: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def _write_generation_examples(
    examples: list[GenerationExample],
    output_dir: Path,
    actor_type: str,
    training_step: int,
) -> None:
    """Write generation examples to a JSONL file.

    Args:
        examples: List of GenerationExample objects to write.
        output_dir: Directory to save the examples file.
        actor_type: Type of actor ("policy" or "rubric").
        training_step: Current training step (gradient update count, consistent with wandb).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to a single JSONL file per actor type, appending new examples
    output_file = output_dir / f"{actor_type}_generations.jsonl"

    with output_file.open("a", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    LOGGER.info(
        "Wrote %d %s generation examples for training_step %d to %s",
        len(examples),
        actor_type,
        training_step,
        output_file,
    )


class StreamingDatasetProxy:
    """Mutable wrapper that allows swapping out the underlying dataset without restarting threads."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._lock = threading.Lock()

    def update(self, dataset: Dataset) -> None:
        with self._lock:
            self._dataset = dataset

    def __len__(self) -> int:
        with self._lock:
            return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        with self._lock:
            return self._dataset[idx]


# ---------------------------------------------------------------------------
# Script Arguments
# ---------------------------------------------------------------------------


@dataclass
class ScriptArgs:
    """Arguments specific to the alternating training script."""

    rubric_model: str = ""
    """Model name to use for rubric generation/updates."""
    policy_model: str = ""
    """Policy model name (LiteLLM compatible)."""
    baseline_model: str | None = None
    """Optional baseline model for rejected answers (defaults to policy model)."""
    # Note: rubric_judge_* fields are defined in GrpoArgs to avoid duplication
    # Access them via args.grpo_args.rubric_judge_num_engines, etc.
    policy_temperature: float = 1.0
    """Sampling temperature for the policy model (1.0 for GRPO rollout diversity)."""
    baseline_temperature: float = 1.0
    """Sampling temperature for the baseline model (1.0 for GRPO rollout diversity)."""
    rubric_temperature: float = 0.6
    """Sampling temperature for rubric generation."""
    api_rubric_generator: str | None = None
    """Optional API model for rubric generation (e.g., 'gpt-4.1', 'gpt-4o').
    When set, rubrics are generated via litellm API calls instead of the local rubric model.
    The rubric model will NOT be trained - only the policy model receives gradient updates.
    This is useful for baselines comparing learned vs. prompted rubric generation."""
    freeze_rubric_model: bool = False
    """When True, skip rubric model training (freeze at initialization).
    Rubrics are still generated using the local rubric model, but no gradient updates are applied.
    This is useful for baselines testing prompted (non-learned) rubric generation with local models."""
    freeze_policy_model: bool = False
    """When True, skip policy model training (freeze at initialization).
    The policy model still generates responses, but no gradient updates are applied.
    This is useful for training only the rubric generator using fixed policy outputs,
    proving that rubric training doesn't require co-evolution with the policy."""
    training_mode: str = "policy_rubric"
    """Training mode for the alternating trainer.
    Options:
    - 'policy_rubric': Train both policy and rubric generator (default).
    - 'rubric_judge': Train the rubric judge model directly on preference pairs."""
    rubric_reward_mode: str = "rubric_judge"
    """Reward mode used by the alternating trainer.
    Options:
    - 'rubric_judge': Judge answers conditioned on generated rubrics (default).
    - 'rrd_uniform': Full RRD + uniform rubric weights.
    - 'rrd_llm': Full RRD + LLM-assigned rubric weights.
    - 'rrd_wu': Full RRD + whitened-uniform rubric weights (paper method).
    - 'query_specific_pref': Query-specific rubric weighting from preferred vs. dispreferred answers.
    - 'rlcer': RLCER without evolving (Sheng et al., 2026) - correlation-filtered rubric rewards + outcome reward.
    - 'rlcer_evolving': RLCER with evolving - also trains the rubricator with K_valid/K + r_format reward.
    - 'rubric_arm': Rubric-ARM (Xu et al., 2026) - alternating RL for rubric generator + pairwise judge."""
    rubric_reward_use_margin: bool = False
    """Use margin-based reward for rubric training instead of binary pairwise accuracy.
    When False (default): reward = 1.0 if accepted_score > rejected_score, else 0.0.
    When True: reward = accepted_score - rejected_score (continuous, can be negative).
    This incentivizes rubrics that produce well-separated judge scores rather than
    rubrics that just barely rank the accepted answer above the rejected."""
    rubric_format_reward_weight: float = 0.0
    """Weight for rubric format reward (0.0 to 1.0). When > 0, blends the base reward
    (pairwise or margin) with a binary format reward that checks whether the generated
    rubric is valid JSON with criteria and weights:
        final_reward = (1 - w) * base_reward + w * format_reward
    Set to 0.0 (default) to disable format reward."""
    cycles: int = 10
    """Number of alternating cycles to run."""
    steps_per_phase: int = 5
    """Number of policy steps between rubric updates."""
    use_both_models: bool = False
    """Whether to use both policy and baseline models when creating rubrics."""
    single_model_mode: bool = False
    """When True, initialize only one set of vLLM engines and share them between rubric and policy actors.
    This reduces GPU memory usage by using a single model for both rubric generation and policy training."""
    ray_address: str | None = None
    """Ray cluster address (e.g. 'auto' to connect to an existing cluster)."""
    ray_namespace: str = "rubric_policy_training"
    """Ray namespace to use for the training session."""
    output: str = "outputs/alternating_training_result.json"
    """Path to save the resulting rubrics and baselines."""
    generation_examples_dir: str | None = None
    """Directory to save generation examples for inspection. Defaults to output dir + '/generation_examples'."""
    num_examples_to_log: int = 3
    """Number of generation examples to log per training step."""
    log_examples_every_n_steps: int = 1
    """Log generation examples every N training steps. Set to 0 to disable logging. Default: 1 (every step)."""
    rejected_answer_method: str = "replay_buffer"
    """Method for generating rejected answers during rubric training. Options:
    - 'replay_buffer': Use past policy rollouts from replay buffer (default, original method)
    - 'inferred_question': Generate rejected by inferring what question the model thinks it's 
       answering from the accepted answer, then generating a new rollout to that inferred question
    - 'rubric': Generate a rubric first, then use rubric-conditioned answer as chosen and
       non-rubric answer as rejected
    - 'combined': Combine multiple methods with configurable weights. Requires combined_data_provider_weights.
    """
    replay_buffer_size: int = 2048
    """Maximum size of the replay buffer for storing past policy rollouts."""
    replay_buffer_min_age: int = 0
    """Minimum step age for sampling from replay buffer (0 = include most recent)."""
    replay_buffer_max_age: int | None = None
    """Maximum step age for sampling from replay buffer (None = no limit)."""
    inference_model_for_question_inference: str | None = None
    """Model to use for question inference when rejected_answer_method='inferred_question'. 
    Options:
    - 'inference_engine': Use dedicated inference model engines (requires INFERENCE_MODEL and INFERENCE_NUM_ENGINES)
    - 'rubric_judge': Use rubric judge model
    - 'policy': Use policy model (requires single_model_mode=True)
    Must be explicitly set. Raises ValueError if not configured or if required resources are unavailable."""
    combined_data_provider_weights: str | None = None
    """Weights for combined data provider when rejected_answer_method='combined'.
    Format: 'method:weight,method:weight,...' where method is one of replay_buffer, inferred_question, rubric.
    Weights are auto-normalized (don't need to sum to 1). Example: 'replay_buffer:1,inferred_question:1,rubric:1'
    Set a method's weight to 0 to disable it (e.g., 'replay_buffer:0,inferred_question:1,rubric:1')."""
    # Judge size curriculum arguments
    judge_size_curriculum: str | None = None
    """Comma-separated list of model names for judge size curriculum, from largest to smallest.
    E.g., 'Qwen/Qwen3-32B,Qwen/Qwen3-14B,Qwen/Qwen3-8B,Qwen/Qwen3-4B'
    The training starts with the first (largest) model and progressively switches to smaller ones."""
    judge_curriculum_schedule: str | None = None
    """Comma-separated list of cycle indices at which to switch to the next judge model.
    E.g., '0,5,10,15' means: use model[0] for cycles 0-4, model[1] for cycles 5-9, etc.
    Must have same length as judge_size_curriculum. If not specified, switches are evenly distributed."""
    # Multi-judge training arguments
    multi_judge_models: str | None = None
    """Comma-separated list of judge model names for multi-judge training.
    E.g., 'Qwen/Qwen3-1.7B,meta-llama/Llama-3-8B-Instruct,google/gemma-2-9b-it'
    When set, multiple judges will evaluate each rubric+answer pair and rewards will be aggregated.
    Each judge model requires its own set of vLLM engines (configure via multi_judge_num_engines_per_judge)."""
    multi_judge_num_engines_per_judge: int = 1
    """Number of vLLM engines to allocate per judge model in multi-judge training.
    Total judge engines = len(multi_judge_models) * multi_judge_num_engines_per_judge.
    Default: 1 engine per judge."""
    multi_judge_tensor_parallel_size: int = 1
    """Tensor parallel size for each judge engine in multi-judge training. Default: 1."""
    multi_judge_aggregation: str = "majority_vote"
    """How to aggregate multiple judges.
    Options:
    - 'majority_vote': use the hard winner after tallying per-judge votes.
    - 'average_vote': use the fraction of judges preferring the accepted answer.
    - 'average_minus_variance': use average_vote - (1 - clipped Fleiss's kappa).
    - 'agreement_bonus': use alpha * pairwise_accuracy + beta * agreement.
    - 'margin_kappa_format': margin_weight * avg_margin + format_weight * format + kappa_weight * fleiss_kappa."""
    multi_judge_tie_breaker: str = "mean_score"
    """Tie-breaker when judges split evenly.
    Options:
    - 'mean_score': break ties using the mean judge scores, then first non-tie vote.
    - 'first_judge': use the first judge's vote, then fall back to mean score."""
    multi_judge_alpha: float = 0.7
    """Weight for pairwise accuracy when multi_judge_aggregation='agreement_bonus'.
    Alpha and beta will be normalized to sum to 1.0. Default: 0.7."""
    multi_judge_beta: float = 0.3
    """Weight for agreement (Kendall's tau) when multi_judge_aggregation='agreement_bonus'.
    Alpha and beta will be normalized to sum to 1.0. Default: 0.3."""
    multi_judge_margin_weight: float = 0.5
    """Weight for avg score margin when multi_judge_aggregation='margin_kappa_format'. Default: 0.5."""
    multi_judge_format_weight: float = 0.3
    """Weight for rubric format validity when multi_judge_aggregation='margin_kappa_format'. Default: 0.3."""
    multi_judge_kappa_weight: float = 0.2
    """Weight for Fleiss's kappa when multi_judge_aggregation='margin_kappa_format'. Default: 0.2."""
    # Multi-policy training arguments
    multi_policy_models: str | None = None
    """Comma-separated list of policy model names for multi-policy frozen training.
    E.g., 'Qwen/Qwen3-8B,meta-llama/Llama-3-8B-Instruct,mistralai/Mistral-7B-v0.1'
    When set with freeze_policy_model=True, multiple frozen policies will generate diverse responses
    for rubric training. Each policy requires its own set of vLLM engines.
    NOTE: For co-evolving multi-policy training, use multi_policy_coevolve_models instead."""
    multi_policy_num_engines_per_model: int = 1
    """Number of vLLM engines to allocate per policy model in multi-policy frozen training.
    Total policy engines = len(multi_policy_models) * multi_policy_num_engines_per_model.
    Default: 1 engine per policy."""
    multi_policy_tensor_parallel_size: int = 1
    """Tensor parallel size for each policy engine in multi-policy frozen training. Default: 1."""
    multi_policy_sampling_strategy: str = "uniform"
    """Strategy for sampling from multiple policies when creating preference pairs.
    Options:
    - 'uniform': Sample uniformly from all policies (default)
    - 'weighted': Sample proportionally to policy quality (requires policy_weights)
    - 'round_robin': Cycle through policies in order"""
    multi_policy_coevolve_models: str | None = None
    """Comma-separated list of policy model names for multi-policy co-evolution training.
    E.g., 'meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3'
    When set, multiple policies will be trained jointly, alternating with rubric training.
    Each model gets its own PolicyTrainerActor with dedicated GPU resources.
    The main policy (--policy-model) is always included; these are ADDITIONAL policies.
    WARNING: Computationally expensive - requires GPUs for all policy trainers."""
    multi_policy_coevolve_vllm_engines_per_model: int = 4
    """Number of vLLM engines per co-evolving policy model.
    Each extra policy creates its own set of vLLM engines for inference.
    Default: 4 engines per model."""
    multi_policy_coevolve_num_learners_per_node: str = "1"
    """Learner GPUs per node for each co-evolving policy model (comma-separated list or single int).
    Each extra policy creates its own DeepSpeed model group with this many learner GPUs per node.
    E.g., '1' means 1 learner per node = 8 total with 8 nodes.
    E.g., '1,1,1,1,0,0,0,0' means learners only on first 4 nodes.
    Default: '1' (1 learner per node)."""
    # GRPO configs (will be populated by parse_args)
    grpo_args: GrpoArgs | None = None
    tokenizer_config: GrpoTokenizerConfig | None = None
    model_config: GrpoModelConfig | None = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _current_timestamp() -> float:
    return time.time()


@dataclass
class JudgeCurriculumState:
    """Tracks state for judge size curriculum."""
    models: list[str]
    """List of model names in curriculum order (largest to smallest)."""
    schedule: list[int]
    """List of cycle indices at which to switch to each model."""
    current_model_idx: int = 0
    """Index of currently active model in the models list."""
    
    def get_model_for_cycle(self, cycle_idx: int) -> tuple[str, bool]:
        """Get the model to use for a given cycle and whether a switch is needed.
        
        Args:
            cycle_idx: The current cycle index.
            
        Returns:
            Tuple of (model_name, needs_switch) where needs_switch is True if
            the model changed from the previous cycle.
        """
        # Find which model should be active for this cycle
        target_model_idx = 0
        for i, switch_cycle in enumerate(self.schedule):
            if cycle_idx >= switch_cycle:
                target_model_idx = i
        
        needs_switch = target_model_idx != self.current_model_idx
        self.current_model_idx = target_model_idx
        return self.models[target_model_idx], needs_switch


def parse_judge_curriculum(
    judge_size_curriculum: str | None,
    judge_curriculum_schedule: str | None,
    total_cycles: int,
) -> JudgeCurriculumState | None:
    """Parse curriculum arguments into a JudgeCurriculumState.
    
    Args:
        judge_size_curriculum: Comma-separated model names.
        judge_curriculum_schedule: Comma-separated cycle indices for switches.
        total_cycles: Total number of training cycles.
        
    Returns:
        JudgeCurriculumState if curriculum is specified, None otherwise.
    """
    if not judge_size_curriculum:
        return None
        
    models = [m.strip() for m in judge_size_curriculum.split(",")]
    if len(models) < 2:
        LOGGER.warning("Judge curriculum requires at least 2 models, got %d. Disabling curriculum.", len(models))
        return None
    
    if judge_curriculum_schedule:
        schedule = [int(c.strip()) for c in judge_curriculum_schedule.split(",")]
        if len(schedule) != len(models):
            raise ValueError(
                f"judge_curriculum_schedule must have same length as judge_size_curriculum. "
                f"Got {len(schedule)} schedule entries for {len(models)} models."
            )
        # Validate schedule is monotonically increasing and starts at 0
        if schedule[0] != 0:
            raise ValueError(f"judge_curriculum_schedule must start at 0, got {schedule[0]}")
        for i in range(1, len(schedule)):
            if schedule[i] <= schedule[i-1]:
                raise ValueError(f"judge_curriculum_schedule must be strictly increasing. Got {schedule}")
    else:
        # Evenly distribute switches across cycles
        schedule = [int(i * total_cycles / len(models)) for i in range(len(models))]
    
    LOGGER.info("Judge size curriculum: %s", models)
    LOGGER.info("Curriculum schedule (switch at cycles): %s", schedule)
    
    return JudgeCurriculumState(models=models, schedule=schedule)


@dataclass
class AuxiliaryModelConfig:
    """Configuration for an auxiliary model (e.g., rubric judge, inference model)."""
    model_name: str
    """Model name/path for the auxiliary model."""
    num_engines: int
    """Number of vLLM engines to create."""
    tensor_parallel_size: int = 1
    """Tensor parallel size for multi-GPU inference."""
    gpu_memory_utilization: float = 0.9
    """GPU memory utilization."""
    max_model_len: int | None = 32768
    """Maximum model length. None = auto-detect from model config."""
    is_rubric_judge_engine: bool = False
    """Whether this is a rubric judge engine (affects engine behavior)."""
    name: str = ""
    """Name identifier for this auxiliary model (used in logging and actor names)."""
    tokenizer_name: str | None = None
    """Tokenizer name/path. If None, uses model_name."""
    generate_text_timeout: float = 7200.0
    """Timeout in seconds for generate_text requests (default: 7200.0 = 2 hours)."""


@dataclass
class AuxiliaryModelEngines:
    """Container for auxiliary model vLLM engines and related components."""
    engines: list[Any]
    """List of vLLM engine actors."""
    generate_text_actor: Any
    """GenerateTextActor for load-balanced generation."""
    tokenizer: Any
    """Tokenizer for the model."""
    prompt_queue: ray_queue.Queue
    """Prompt queue for the engines."""
    results_queue: ray_queue.Queue
    """Results queue for the engines."""
    generate_text_results_queue: ray_queue.Queue
    """Generate text results queue."""
    actor_manager: Any
    """ActorManager for monitoring queues."""


def create_auxiliary_model_engines(
    config: AuxiliaryModelConfig,
    grpo_args: GrpoArgs,
    tokenizer_config: GrpoTokenizerConfig | None = None,
) -> AuxiliaryModelEngines | None:
    """Create vLLM engines for an auxiliary model (e.g., rubric judge, inference model).
    
    Args:
        config: Configuration for the auxiliary model (must be validated before calling).
        grpo_args: GRPO arguments containing engine settings.
        tokenizer_config: Optional tokenizer config (for tokenizer name resolution).
        
    Returns:
        AuxiliaryModelEngines if engines were created, None otherwise.
    """
    if config.num_engines <= 0:
        return None
    
    LOGGER.info(
        "Creating %d dedicated %s vLLM engines",
        config.num_engines,
        config.name or "auxiliary model",
    )
    from open_instruct import vllm_utils
    
    # Create queues
    prompt_queue = ray_queue.Queue()
    results_queue = ray_queue.Queue()
    generate_text_results_queue = ray_queue.Queue()
    
    # Create ActorManager
    from open_instruct.actor_manager import ActorManager
    queues_to_monitor = {
        f"{config.name} Prompt Queue": prompt_queue,
        f"{config.name} Results Queue": results_queue,
        f"{config.name} Generate Text Results Queue": generate_text_results_queue,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, grpo_args)
    
    # Resolve tokenizer name
    # For auxiliary models (judges, inference engines), always use the model's own tokenizer.
    # The tokenizer_config.tokenizer_name_or_path is for the main policy model and should NOT
    # be used for auxiliary models as they may have completely different tokenizers
    # (e.g., Mistral policy with Qwen judge would have different vocab sizes).
    if config.tokenizer_name:
        tokenizer_name = config.tokenizer_name
    elif config.model_name:
        # Always use the model's own tokenizer for auxiliary models
        tokenizer_name = config.model_name
    else:
        # This should not happen if validate_args was called
        raise RuntimeError(
            f"Cannot determine tokenizer name for {config.name or 'auxiliary model'}. "
            "This indicates a bug - validate_args should have caught this."
        )
    
    # Use config.model_name directly (already validated in validate_args)
    model_name = config.model_name
    
    # Create vLLM engines
    engines = vllm_utils.create_vllm_engines(
        config.num_engines,
        config.tensor_parallel_size,
        grpo_args.vllm_enforce_eager,
        tokenizer_name,
        model_name,
        None,  # revision
        grpo_args.seed,
        grpo_args.vllm_enable_prefix_caching,
        config.max_model_len,
        config.gpu_memory_utilization,
        grpo_args.single_gpu_mode,
        pg=None,
        tools={},
        max_tool_calls=None,
        prompt_queue=prompt_queue,
        results_queue=results_queue,
        eval_results_queue=None,
        actor_manager=actor_manager,
        inference_batch_size=None,
        use_fp8_kv_cache=grpo_args.use_fp8_kv_cache,
        inflight_updates=grpo_args.inflight_updates,
        is_rubric_judge_engine=config.is_rubric_judge_engine,
        generate_text_results_queue=generate_text_results_queue,
        name=config.name,
    )
    
    # Create tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create GenerateTextActor
    generate_text_actor = vllm_utils.GenerateTextActor.remote(
        prompt_queue=prompt_queue,
        generate_text_results_queue=generate_text_results_queue,
        tokenizer=tokenizer,
        name=f"{config.name}_generate_text",
        generate_text_timeout=config.generate_text_timeout,
    )
    
    LOGGER.info("======== ✅ %s GenerateTextActor created ========", config.name)

    # Register engine handles so the GenerateTextActor can abort orphaned
    # requests directly in the vLLM engines when a timeout occurs.
    ray.get(generate_text_actor.set_engines.remote(engines))
    
    return AuxiliaryModelEngines(
        engines=engines,
        generate_text_actor=generate_text_actor,
        tokenizer=tokenizer,
        prompt_queue=prompt_queue,
        results_queue=results_queue,
        generate_text_results_queue=generate_text_results_queue,
        actor_manager=actor_manager,
    )




# ---------------------------------------------------------------------------
# Multi-Judge Support
# ---------------------------------------------------------------------------


@dataclass
class MultiJudgeEngines:
    """Container for multiple judge engine sets for multi-judge training."""
    judge_engines: list[AuxiliaryModelEngines]
    """List of AuxiliaryModelEngines, one per judge model."""
    judge_models: list[str]
    """List of judge model names (parallel to judge_engines)."""
    aggregation_mode: str
    """How to aggregate multiple judges."""
    tie_breaker: str
    """Tie-breaker when judges split evenly."""
    alpha: float
    """Weight for pairwise accuracy in reward aggregation."""
    beta: float
    """Weight for agreement (Kendall's tau) in reward aggregation."""
    margin_weight: float = 0.5
    """Weight for avg margin in margin_kappa_format mode."""
    format_weight: float = 0.3
    """Weight for rubric format in margin_kappa_format mode."""
    kappa_weight: float = 0.2
    """Weight for Fleiss's kappa in margin_kappa_format mode."""

    def get_judge_actors_with_sampling_params(
        self,
        sampling_params_template: Any,
    ) -> list[tuple[Any, Any]]:
        """Get list of (generate_text_actor, sampling_params) tuples for each judge.

        Args:
            sampling_params_template: Template sampling params (will be cloned for each judge).

        Returns:
            List of tuples, one per judge.
        """
        actors = []
        for engines_obj in self.judge_engines:
            # Clone sampling params for this judge
            sampling_params = sampling_params_template.clone() if sampling_params_template else None
            actors.append((engines_obj.generate_text_actor, sampling_params))
        return actors


def create_multi_judge_engines(
    script_args: ScriptArgs,
    grpo_args: GrpoArgs,
    tokenizer_config: GrpoTokenizerConfig | None = None,
) -> MultiJudgeEngines | None:
    """Create vLLM engines for multiple judge models in multi-judge training.

    Args:
        script_args: Script arguments containing multi-judge configuration.
        grpo_args: GRPO arguments containing engine settings.
        tokenizer_config: Optional tokenizer config.

    Returns:
        MultiJudgeEngines if multi-judge is enabled, None otherwise.
    """
    if not script_args.multi_judge_models:
        return None

    judge_models = [m.strip() for m in script_args.multi_judge_models.split(",")]
    if len(judge_models) < 2:
        LOGGER.warning(
            "Multi-judge training requires at least 2 judge models, got %d. "
            "Disabling multi-judge mode.",
            len(judge_models)
        )
        return None

    LOGGER.info(
        "Creating multi-judge engines for %d judges: %s",
        len(judge_models),
        judge_models
    )

    judge_engines = []
    for idx, model_name in enumerate(judge_models):
        # Create config for this judge
        # Each judge MUST use its own tokenizer (vocab sizes differ across models)
        config = AuxiliaryModelConfig(
            model_name=model_name,
            tokenizer_name=model_name,  # Each judge must use its own tokenizer
            num_engines=script_args.multi_judge_num_engines_per_judge,
            tensor_parallel_size=script_args.multi_judge_tensor_parallel_size,
            # Multi-judge configs can resolve shared rubric_judge_* engine knobs
            # from per-model configs. Keep the single-judge GPU utilization default
            # as a defensive fallback for ad hoc launches, but allow max_model_len
            # to stay unset when not provided so each judge can auto-detect safely.
            gpu_memory_utilization=grpo_args.rubric_judge_gpu_memory_utilization or 0.9,
            max_model_len=grpo_args.rubric_judge_max_model_len,
            # Actor-level timeout must be ≤ the per-judge timeout so that the
            # engine aborts orphaned requests *before* the judge layer gives up.
            # Without this, timed-out requests continue consuming engine compute
            # and progressively starve new requests (queue buildup).
            generate_text_timeout=float(os.environ.get("MULTI_JUDGE_TIMEOUT_S", "300")) * 0.9,
            name=f"Multi-Judge-{idx+1}",
            is_rubric_judge_engine=True,
        )

        # Create engines for this judge
        engines_obj = create_auxiliary_model_engines(config, grpo_args, tokenizer_config)
        if engines_obj is None:
            raise RuntimeError(
                f"Failed to create engines for multi-judge model {idx+1}: {model_name}"
            )

        judge_engines.append(engines_obj)
        LOGGER.info(
            "Created judge %d/%d: %s (%d engines)",
            idx + 1,
            len(judge_models),
            model_name,
            script_args.multi_judge_num_engines_per_judge
        )

    return MultiJudgeEngines(
        judge_engines=judge_engines,
        judge_models=judge_models,
        aggregation_mode=script_args.multi_judge_aggregation,
        tie_breaker=script_args.multi_judge_tie_breaker,
        alpha=script_args.multi_judge_alpha,
        beta=script_args.multi_judge_beta,
        margin_weight=script_args.multi_judge_margin_weight,
        format_weight=script_args.multi_judge_format_weight,
        kappa_weight=script_args.multi_judge_kappa_weight,
    )


def shutdown_multi_judge_engines(multi_judge_obj: MultiJudgeEngines | None) -> None:
    """Shutdown and cleanup multi-judge engines.

    Args:
        multi_judge_obj: The MultiJudgeEngines to shutdown, or None (no-op).
    """
    if multi_judge_obj is None:
        return

    LOGGER.info("Shutting down multi-judge engines...")
    for idx, engines_obj in enumerate(multi_judge_obj.judge_engines):
        LOGGER.info("Shutting down judge %d/%d", idx + 1, len(multi_judge_obj.judge_engines))
        shutdown_auxiliary_model_engines(engines_obj)


# ---------------------------------------------------------------------------
# Multi-Policy Frozen Support
# ---------------------------------------------------------------------------


@dataclass
class MultiPolicyFrozenEngines:
    """Container for multiple frozen policy engine sets for multi-policy training."""

    policy_engines: dict[str, AuxiliaryModelEngines]
    """Dict mapping policy model names to their AuxiliaryModelEngines."""
    policy_models: list[str]
    """List of policy model names."""
    sampling_strategy: str
    """Sampling strategy for policy selection ('uniform', 'round_robin')."""

    def get_policy_engines_dict(self) -> dict[str, tuple[Any, Any]]:
        """Get dict of policy engines for MultiPolicyFrozenDataProvider.

        Returns:
            Dict mapping model names to (generate_text_actor, sampling_params) tuples.
        """
        result = {}
        for model_name, engines_obj in self.policy_engines.items():
            # Create sampling params for this policy
            sampling_params = vllm.SamplingParams(temperature=1.0, top_p=0.95, max_tokens=16384, n=1, logprobs=1, stop=None)
            result[model_name] = (engines_obj.generate_text_actor, sampling_params)
        return result


def create_multi_policy_frozen_engines(
    script_args: ScriptArgs,
    grpo_args: GrpoArgs,
    tokenizer_config: GrpoTokenizerConfig | None = None,
) -> MultiPolicyFrozenEngines | None:
    """Create vLLM engines for multiple frozen policy models.

    Args:
        script_args: Script arguments containing multi-policy configuration.
        grpo_args: GRPO arguments containing engine settings.
        tokenizer_config: Optional tokenizer config.

    Returns:
        MultiPolicyFrozenEngines if multi-policy is enabled, None otherwise.
    """
    if not script_args.multi_policy_models:
        return None

    policy_models = [m.strip() for m in script_args.multi_policy_models.split(",")]
    if len(policy_models) < 2:
        LOGGER.warning(
            "Multi-policy frozen training requires at least 2 policy models, got %d. " "Disabling multi-policy mode.",
            len(policy_models),
        )
        return None

    LOGGER.info(
        "Creating multi-policy frozen engines for %d policies: %s (strategy=%s)",
        len(policy_models),
        policy_models,
        script_args.multi_policy_sampling_strategy,
    )

    policy_engines = {}
    for idx, model_name in enumerate(policy_models):
        # Create config for this policy
        # Each policy MUST use its own tokenizer (vocab sizes differ across models,
        # e.g. Qwen3-8B has 151936 tokens vs Llama-3-8B has 128256).
        # Setting tokenizer_name=model_name ensures the model's own tokenizer is used
        # instead of falling through to the shared tokenizer_config.
        config = AuxiliaryModelConfig(
            model_name=model_name,
            tokenizer_name=model_name,  # Each policy must use its own tokenizer
            num_engines=script_args.multi_policy_num_engines_per_model,
            tensor_parallel_size=script_args.multi_policy_tensor_parallel_size,
            gpu_memory_utilization=grpo_args.vllm_gpu_memory_utilization,
            max_model_len=None,  # Auto-detect from model config (models may have different max lengths)
            generate_text_timeout=1200.0,
            name=f"Multi-Policy-{idx+1}",
            is_rubric_judge_engine=False,
        )

        # Create engines for this policy
        engines_obj = create_auxiliary_model_engines(config, grpo_args, tokenizer_config)
        if engines_obj is None:
            raise RuntimeError(f"Failed to create engines for multi-policy model {idx+1}: {model_name}")

        policy_engines[model_name] = engines_obj
        LOGGER.info(
            "Created policy %d/%d: %s (%d engines)",
            idx + 1,
            len(policy_models),
            model_name,
            script_args.multi_policy_num_engines_per_model,
        )

    return MultiPolicyFrozenEngines(
        policy_engines=policy_engines,
        policy_models=policy_models,
        sampling_strategy=script_args.multi_policy_sampling_strategy,
    )


def shutdown_multi_policy_frozen_engines(multi_policy_obj: MultiPolicyFrozenEngines | None) -> None:
    """Shutdown and cleanup multi-policy frozen engines.

    Args:
        multi_policy_obj: The MultiPolicyFrozenEngines to shutdown, or None (no-op).
    """
    if multi_policy_obj is None:
        return

    LOGGER.info("Shutting down multi-policy frozen engines...")
    for idx, (model_name, engines_obj) in enumerate(multi_policy_obj.policy_engines.items()):
        LOGGER.info(
            "Shutting down policy %d/%d: %s", idx + 1, len(multi_policy_obj.policy_engines), model_name
        )
        shutdown_auxiliary_model_engines(engines_obj)


def shutdown_auxiliary_model_engines(engines_obj: AuxiliaryModelEngines | None) -> None:
    """Shutdown and cleanup auxiliary model engines.

    Args:
        engines_obj: The AuxiliaryModelEngines to shutdown, or None (no-op).
    """
    if engines_obj is None:
        return

    LOGGER.info("Shutting down auxiliary model engines...")

    # Kill all engine actors
    for engine in engines_obj.engines:
        try:
            ray.kill(engine)
        except Exception as e:
            LOGGER.warning("Failed to kill engine: %s", e)

    # Kill the generate_text_actor
    if engines_obj.generate_text_actor is not None:
        try:
            ray.kill(engines_obj.generate_text_actor)
        except Exception as e:
            LOGGER.warning("Failed to kill generate_text_actor: %s", e)
    
    # Kill the actor_manager
    if engines_obj.actor_manager is not None:
        try:
            ray.kill(engines_obj.actor_manager)
        except Exception as e:
            LOGGER.warning("Failed to kill actor_manager: %s", e)
    
    LOGGER.info("Auxiliary model engines shutdown complete")


def swap_rubric_judge_model(
    current_engines: AuxiliaryModelEngines | None,
    new_model_name: str,
    grpo_args: GrpoArgs,
    tokenizer_config: GrpoTokenizerConfig | None,
    num_engines: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 32768,
    generate_text_timeout: float = 7200.0,
) -> AuxiliaryModelEngines:
    """Swap the rubric judge model by shutting down old engines and creating new ones.
    
    Args:
        current_engines: Current engines to shutdown (can be None).
        new_model_name: Name of the new model to load.
        grpo_args: GRPO arguments.
        tokenizer_config: Tokenizer configuration.
        num_engines: Number of engines to create.
        tensor_parallel_size: Tensor parallel size.
        gpu_memory_utilization: GPU memory utilization.
        max_model_len: Maximum model length.
        generate_text_timeout: Timeout for generate_text requests.
        
    Returns:
        New AuxiliaryModelEngines with the new model loaded.
    """
    LOGGER.info("Swapping rubric judge model to: %s", new_model_name)
    
    # Shutdown existing engines
    shutdown_auxiliary_model_engines(current_engines)
    
    # Wait a bit for resources to be freed
    import time
    time.sleep(5)
    
    # Create new engines
    config = AuxiliaryModelConfig(
        model_name=new_model_name,
        num_engines=num_engines,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        is_rubric_judge_engine=True,
        name="rubric_judge",
        # Each judge model must use its own tokenizer
        tokenizer_name=None,  # Will default to model_name in create_auxiliary_model_engines
        generate_text_timeout=generate_text_timeout,
    )
    
    new_engines = create_auxiliary_model_engines(config, grpo_args, tokenizer_config)
    if new_engines is None:
        raise RuntimeError(f"Failed to create engines for new model: {new_model_name}")
    
    # Wait for engines to be ready
    ray_get_with_progress(
        [engine.ready.remote() for engine in new_engines.engines],
        f"Initializing rubric judge vLLM engines for {new_model_name}",
        timeout=1200
    )
    LOGGER.info("======== ✅ Swapped to rubric judge model: %s ========", new_model_name)
    
    return new_engines


# ---------------------------------------------------------------------------
# Dynamic Datasets
# ---------------------------------------------------------------------------

class RubricDynamicDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        return self._dataset[idx % len(self._dataset)]


class PolicyDynamicDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        return self._dataset[idx % len(self._dataset)]


# ---------------------------------------------------------------------------
# Ray actors
# ---------------------------------------------------------------------------


class BaseTrainerActor:
    """Base class for trainer actors that use create_model_and_optimizer."""

    def __init__(self):
        """Initialize base attributes."""
        self.rubric_judge_engines: list[Any] | None = None
        self.rubric_judge_generate_text_actor: Any | None = None
        self.policy_generate_text_actor: Any | None = None
        self.train_dataset_proxy: StreamingDatasetProxy | None = None
        self.generation_configs: dict[str, Any] | None = None
        # Shared engines support for single_model_mode
        self._shared_vllm_engines: list[Any] | None = None
        self._shared_policy_group: Any | None = None
        self._shared_actor_manager: Any | None = None
        self._use_shared_engines: bool = False
        LOGGER.info("BaseTrainerActor initialized")
        LOGGER.debug(f"{self.__class__.__name__} initialized")

    def add_prompt_to_generator(
        self,
        example: dict[str, Any],
        example_index: int,
        epoch_number: int,
        training_step: int,
        pending_queries_map: PendingQueriesMap,
        param_prompt_Q: ray_queue.Queue,
        generation_config: vllm.SamplingParams,
        is_eval: bool,
    ) -> None:
        """Add a prompt to the generator queue.
        
        Subclasses must override this method to provide custom prompt handling logic.
        This method is called by both initial prompt sending and replenishment logic.
        """
        raise NotImplementedError("Subclasses must implement add_prompt_to_generator")

    def set_shared_engines(
        self,
        vllm_engines: list[Any],
        policy_group: Any,
        actor_manager: Any,
        args: GrpoArgs,
        tokenizer_config: GrpoTokenizerConfig,
        model_config: GrpoModelConfig,
        tokenizer: Any,
        inference_results_Q: ray_queue.Queue,
        param_prompt_Q: ray_queue.Queue,
        evaluation_inference_results_Q: ray_queue.Queue,
    ) -> None:
        """Set shared vLLM engines and components for single_model_mode.
        
        This allows multiple actors to share the same vLLM engines and model,
        reducing GPU memory usage.
        
        Args:
            vllm_engines: Pre-created vLLM engine actors
            policy_group: Pre-created ModelGroup with trainable model
            actor_manager: Pre-created ActorManager
            args: GRPO arguments
            tokenizer_config: Tokenizer configuration
            model_config: Model configuration
            tokenizer: Pre-created tokenizer
            inference_results_Q: Shared inference results queue
            param_prompt_Q: Shared parameter/prompt queue
            evaluation_inference_results_Q: Shared evaluation queue
        """
        self._shared_vllm_engines = vllm_engines
        self._shared_policy_group = policy_group
        self._shared_actor_manager = actor_manager
        self._use_shared_engines = True
        
        # Store args and configs
        self.args = args
        self.tokenizer_config = tokenizer_config
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.inference_results_Q = inference_results_Q
        self.param_prompt_Q = param_prompt_Q
        self.evaluation_inference_results_Q = evaluation_inference_results_Q
        
        # Use shared components
        self.vllm_engines = vllm_engines
        self.policy_group = policy_group
        self.actor_manager = actor_manager
        
        # Get model dims from first engine
        if self.vllm_engines:
            self.model_dims = ray.get(self.vllm_engines[0].get_model_dims.remote())
        else:
            self.model_dims = None
            
        # Create generation configs
        self.generation_configs = create_generation_configs(args)
        # Initialize training step and episode
        self.resume_training_step = 1
        self.episode = 0
        self.tool_objects = {}
        
        LOGGER.info(f"{self.__class__.__name__}: Using shared engines (single_model_mode)")
        LOGGER.debug(f"{self.__class__.__name__}: {len(vllm_engines)} shared vLLM engines attached")

    def update_rubric_judge_engines(
        self,
        engines: list[Any],
        generate_text_actor: Any,
        tokenizer: Any,
    ) -> None:
        """Update rubric judge engines during curriculum switching.
        
        Args:
            engines: New vLLM engine actors for the rubric judge.
            generate_text_actor: New GenerateTextActor for the rubric judge.
            tokenizer: New tokenizer for the rubric judge.
        """
        LOGGER.info(f"{self.__class__.__name__}: Updating rubric judge engines")
        self.rubric_judge_engines = engines
        self.rubric_judge_generate_text_actor = generate_text_actor
        # Store the tokenizer if needed by subclasses
        if hasattr(self, 'rubric_judge_tokenizer'):
            self.rubric_judge_tokenizer = tokenizer
        LOGGER.info(f"{self.__class__.__name__}: Rubric judge engines updated")

    @staticmethod
    def _safe_queue_size(queue_obj: Any | None) -> int | None:
        """Best-effort queue size lookup for debug logging."""
        if queue_obj is None:
            return None
        qsize_fn = getattr(queue_obj, "qsize", None)
        if qsize_fn is None:
            return None
        try:
            return int(qsize_fn())
        except Exception:
            return None

    @staticmethod
    def _safe_map_size(map_obj: Any | None) -> int | None:
        """Best-effort map size lookup for debug logging."""
        if map_obj is None:
            return None
        try:
            return int(len(map_obj))
        except Exception:
            return None

    def _collect_transition_debug_state(self, include_engine_health: bool = False) -> dict[str, Any]:
        """Collect a compact snapshot for phase-transition debugging."""
        args = getattr(self, "args", None)
        state: dict[str, Any] = {
            "actor_class": self.__class__.__name__,
            "actor_id": self._get_actor_id() if hasattr(self, "_get_actor_id") else None,
            "model_name_or_path": getattr(getattr(self, "model_config", None), "model_name_or_path", None),
            "paused": bool(self.pause_event.is_set()) if hasattr(self, "pause_event") else None,
            "use_shared_engines": bool(getattr(self, "_use_shared_engines", False)),
            "resume_training_step": getattr(self, "resume_training_step", None),
            "current_training_step": getattr(self, "_current_training_step", None),
            "episode": getattr(self, "episode", None),
            "async_steps": getattr(args, "async_steps", None),
            "num_unique_prompts_rollout": getattr(args, "num_unique_prompts_rollout", None),
            "num_samples_per_prompt_rollout": getattr(args, "num_samples_per_prompt_rollout", None),
            "queues": {
                "param_prompt_Q": self._safe_queue_size(getattr(self, "param_prompt_Q", None)),
                "packed_sequences_Q": self._safe_queue_size(getattr(self, "packed_sequences_Q", None)),
                "inference_results_Q": self._safe_queue_size(getattr(self, "inference_results_Q", None)),
                "evaluation_inference_results_Q": self._safe_queue_size(
                    getattr(self, "evaluation_inference_results_Q", None)
                ),
                "enriched_results_Q": self._safe_queue_size(getattr(self, "enriched_results_Q", None)),
                "generate_metrics_Q": self._safe_queue_size(getattr(self, "generate_metrics_Q", None)),
                "weight_sync_metrics_Q": self._safe_queue_size(getattr(self, "weight_sync_metrics_Q", None)),
            },
            "pending_maps": {
                "pending_queries_map": self._safe_map_size(getattr(self, "pending_queries_map", None)),
                "eval_pending_queries_map": self._safe_map_size(getattr(self, "eval_pending_queries_map", None)),
            },
        }

        actor_manager = getattr(self, "actor_manager", None)
        if actor_manager is not None:
            try:
                state["actor_manager_queue_stats"] = ray.get(actor_manager.get_queue_stats.remote(), timeout=5.0)
            except Exception as e:
                state["actor_manager_queue_stats_error"] = str(e)

            try:
                actor_stats = ray.get(actor_manager.get_vllm_actor_stats.remote(), timeout=5.0)
                state["actor_manager_vllm_stats"] = {
                    "total_actors": actor_stats.get("total_actors", 0),
                    "total_active_tasks": actor_stats.get("total_active_tasks", 0),
                    "policy_active_tasks": actor_stats.get("policy_active_tasks", 0),
                    "rubric_judge_active_tasks": actor_stats.get("rubric_judge_active_tasks", 0),
                }
            except Exception as e:
                state["actor_manager_vllm_stats_error"] = str(e)

        if include_engine_health:
            for engine_attr in ("vllm_engines", "rubric_judge_engines"):
                engines = getattr(self, engine_attr, None) or []
                state[f"{engine_attr}_count"] = len(engines)
                if not engines:
                    continue
                try:
                    health_results = ray.get([engine.health_check.remote() for engine in engines], timeout=15.0)
                    state[f"{engine_attr}_health"] = [
                        {
                            "engine_idx": idx,
                            "prefetch_alive": result.get("prefetch_alive"),
                            "process_alive": result.get("process_alive"),
                            "loop_alive": result.get("loop_alive"),
                            "active_tasks": result.get("active_tasks"),
                        }
                        for idx, result in enumerate(health_results)
                    ]
                except Exception as e:
                    state[f"{engine_attr}_health_error"] = str(e)

        return state

    def get_transition_debug_state(self, include_engine_health: bool = False) -> dict[str, Any]:
        """Expose transition debug state to the orchestration loop."""
        return self._collect_transition_debug_state(include_engine_health=include_engine_health)

    def _log_transition_debug_state(self, phase: str, *, include_engine_health: bool = False) -> None:
        """Log a compact snapshot around pause/resume boundaries."""
        state = self._collect_transition_debug_state(include_engine_health=include_engine_health)
        LOGGER.info(
            "%s transition[%s]: %s",
            self.__class__.__name__,
            phase,
            json.dumps(state, sort_keys=True, default=str),
        )

    def _try_restore_data_iterator_state(self, steps_per_phase: int | None = None) -> None:
        """Restore ShufflingIterator state from checkpoint, or fast-forward.

        On resume, both policy and rubric actors create fresh iterators that
        start from index 0, causing the model to re-visit training examples
        it already saw.  This method first tries to load a saved
        ``data_iterator_state.pt`` from the checkpoint directory.  If none
        exists (e.g. checkpoints from before this feature was added), it
        deterministically fast-forwards the iterator by the estimated number
        of consumed data points so the model skips already-seen examples.
        """
        resume_step = getattr(self.args, "resume_from_step", 0)
        if resume_step <= 0 or not hasattr(self, "iter_dataloader") or self.iter_dataloader is None:
            return

        actor_type = self.__class__.__name__.replace("TrainerActor", "").lower()
        checkpoint_base = os.path.join(f"{self.args.output_dir}_checkpoints", actor_type)

        # --- attempt 1: restore from saved state file ---
        best_state_path = None
        best_step = -1
        if os.path.isdir(checkpoint_base):
            try:
                for dirname in os.listdir(checkpoint_base):
                    if not dirname.startswith("step_"):
                        continue
                    try:
                        step_num = int(dirname.split("_", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if step_num <= resume_step:
                        candidate = os.path.join(checkpoint_base, dirname, "data_iterator_state.pt")
                        if os.path.exists(candidate) and step_num > best_step:
                            best_step = step_num
                            best_state_path = candidate
            except OSError as e:
                LOGGER.warning(
                    "%s: Error scanning checkpoint dir %s: %s",
                    self.__class__.__name__, checkpoint_base, e,
                )

        if best_state_path is not None:
            try:
                state = torch.load(best_state_path, weights_only=False)
                self.iter_dataloader.set_state(state)
                LOGGER.info(
                    "%s: Restored data iterator state from %s "
                    "(step %d, resume_from_step=%d, iterator index=%d, epoch=%d)",
                    self.__class__.__name__, best_state_path,
                    best_step, resume_step,
                    state.get("index", -1), state.get("epoch_number", -1),
                )
                return
            except Exception as e:
                LOGGER.warning(
                    "%s: Failed to restore data iterator state from %s: %s — will fast-forward instead",
                    self.__class__.__name__, best_state_path, e,
                )

        # --- attempt 2: deterministic fast-forward ---
        self._fast_forward_data_iterator(resume_step, steps_per_phase)

    def _fast_forward_data_iterator(
        self, resume_step: int, steps_per_phase: int | None = None
    ) -> None:
        """Advance the data iterator past already-consumed examples.

        Each training step consumes ``num_unique_prompts_rollout`` data points
        (via the replenish mechanism).  Each actor-resume refills the queue
        with ``async_steps * num_unique_prompts_rollout`` fresh prompts.

        With symmetric alternation (policy then rubric, equal phase lengths),
        each actor completes ``resume_step / 2`` training steps and is resumed
        once per cycle.
        """
        num_unique = getattr(self.args, "num_unique_prompts_rollout", 0)
        async_steps = getattr(self.args, "async_steps", 0)

        if num_unique <= 0:
            LOGGER.warning(
                "%s: Cannot fast-forward — num_unique_prompts_rollout=%s",
                self.__class__.__name__, num_unique,
            )
            return

        actor_steps = resume_step // 2
        training_consumption = actor_steps * num_unique

        resume_overhead = 0
        if steps_per_phase is not None and steps_per_phase > 0:
            num_resumes = resume_step // (2 * steps_per_phase)
            resume_overhead = num_resumes * async_steps * num_unique

        total = training_consumption + resume_overhead

        LOGGER.info(
            "%s: No saved data_iterator_state found. Fast-forwarding iterator by %d "
            "next() calls (actor_steps=%d × num_unique=%d + %d resume refills × "
            "async_steps=%d × num_unique=%d = %d + %d)",
            self.__class__.__name__, total,
            actor_steps, num_unique,
            resume_step // (2 * steps_per_phase) if steps_per_phase else 0,
            async_steps, num_unique,
            training_consumption, resume_overhead,
        )
        for _ in range(total):
            next(self.iter_dataloader)

        LOGGER.info(
            "%s: Fast-forward complete (iterator index=%d, epoch=%d)",
            self.__class__.__name__,
            self.iter_dataloader.index,
            self.iter_dataloader.epoch_number,
        )

    @staticmethod
    def _calculate_resources(grpo_args: GrpoArgs) -> dict[str, float]:
        """Calculate GPU and CPU resources needed for policy training.

        Args:
            grpo_args: GRPO arguments containing:
                - num_learners_per_node: list[int] - GPUs per node for learners
                - vllm_num_engines: int - number of vLLM engines
                - vllm_tensor_parallel_size: int - GPUs per vLLM engine
                - single_gpu_mode: bool - whether in single GPU mode

        Returns:
            Dictionary with 'num_gpus' and 'num_cpus' keys
        """
        num_learners_per_node = grpo_args.num_learners_per_node or [1]
        if isinstance(num_learners_per_node, int):
            num_learners_per_node = [num_learners_per_node]

        vllm_num_engines = grpo_args.vllm_num_engines or 1
        vllm_tensor_parallel_size = grpo_args.vllm_tensor_parallel_size or 1
        single_gpu_mode = grpo_args.single_gpu_mode or False

        # Calculate total GPUs: learners + vLLM engines
        total_learner_gpus = sum(num_learners_per_node)
        total_vllm_gpus = vllm_num_engines * vllm_tensor_parallel_size

        # In single GPU mode, vLLM uses 0.5 GPU per engine when tensor_parallel_size=1
        if single_gpu_mode and vllm_tensor_parallel_size == 1:
            total_vllm_gpus = vllm_num_engines * 0.5

        total_gpus = total_learner_gpus + total_vllm_gpus

        # CPUs: 4 per learner GPU (as per ModelGroup.num_cpus_per_actor)
        # Plus CPUs for vLLM (typically 1 per GPU)
        total_cpus = total_learner_gpus * 4 + total_vllm_gpus

        return {"num_gpus": total_gpus, "num_cpus": max(1, int(total_cpus))}

    def _initialize_grpo_session(
        self,
        grpo_args: GrpoArgs,
        tokenizer_config: GrpoTokenizerConfig,
        model_config: GrpoModelConfig,
        *,
        model_override: str | None = None,
        log_prefix: str = "",
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        skip_engine_creation: bool = False,
        placement_group_name: str | None = None,
    ) -> bool:
        """Initialize GRPO components.

        Args:
            grpo_args: GRPO arguments
            tokenizer_config: Tokenizer configuration
            model_config: Model configuration
            model_override: Optional model name to override in config (e.g., for rubric model)
            log_prefix: Optional prefix for log messages
            train_dataset: Optional training dataset (if None, will be loaded)
            eval_dataset: Optional eval dataset (if None, will be loaded)
            skip_engine_creation: If True, skip creating vLLM engines (use when shared engines will be set later)

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            LOGGER.info("%sInitializing GRPO components (skip_engine_creation=%s)", log_prefix, skip_engine_creation)

            # Override model if specified
            if model_override:
                model_config.model_name_or_path = model_override
                # Also override tokenizer if not explicitly set
                if tokenizer_config.tokenizer_name_or_path is None:
                    tokenizer_config.tokenizer_name_or_path = model_override

            # Setup runtime variables
            self.args = grpo_fast.setup_runtime_variables(grpo_args)
            self.tokenizer_config = tokenizer_config
            self.model_config = model_config

            # Create tokenizer
            self.tokenizer = grpo_fast.make_tokenizer(tokenizer_config, model_config)

            # Store datasets (use provided or load if None)
            if train_dataset is not None and eval_dataset is not None:
                self.train_dataset = train_dataset
                self.eval_dataset = eval_dataset
            else:
                (self.train_dataset, self.eval_dataset) = grpo_fast.setup_datasets(
                    self.args, tokenizer_config, self.tokenizer
                )

            # Skip engine creation if shared engines will be provided later
            if skip_engine_creation:
                LOGGER.info("%sSkipping engine creation (will use shared engines)", log_prefix)
                # Initialize queues (will be replaced when shared engines are set)
                queue_size = (self.args.async_steps + 1) * self.args.num_unique_prompts_rollout
                self.inference_results_Q = ray_queue.Queue(maxsize=queue_size)
                self.param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
                self.evaluation_inference_results_Q = ray_queue.Queue()
                # Create generation configs
                self.generation_configs = create_generation_configs(self.args)
                return True

            # Initialize GRPO components
            queue_size = (self.args.async_steps + 1) * self.args.num_unique_prompts_rollout
            self.inference_results_Q = ray_queue.Queue(maxsize=queue_size)
            self.param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
            self.evaluation_inference_results_Q = ray_queue.Queue()

            # Use provided placement group name if given, otherwise generate one
            # from actor class name + model name to ensure uniqueness.
            # This ensures each actor instance gets its own placement group (important for multi-policy co-evolution)
            if placement_group_name is None:
                model_suffix = ""
                if hasattr(self, "policy_model") and self.policy_model:
                    # Sanitize model name for use in placement group name (e.g. "meta-llama/Meta-Llama-3-8B" -> "meta_llama_3_8b")
                    model_suffix = "_" + self.policy_model.split("/")[-1].lower().replace("-", "_")
                placement_group_name = f"{self.__class__.__name__}{model_suffix}_pg"

            # No logging utilities in actors - all logging handled in main thread
            beaker_config = None
            wandb_url = None

            self.setup_grpo(
                self.args,
                self.tokenizer_config,
                self.model_config,
                beaker_config,
                wandb_url,
                self.tokenizer,
                self.inference_results_Q,
                self.param_prompt_Q,
                self.evaluation_inference_results_Q,
                placement_group_name=placement_group_name,
            )

            LOGGER.info("%sGRPO components initialized (queues, engines)", log_prefix)
            return True

        except Exception as e:
            raise e

    def setup_grpo(
        self,
        args: GrpoArgs,
        tokenizer_config: GrpoTokenizerConfig,
        model_config: GrpoModelConfig,
        beaker_config: Any | None,
        wandb_url: str | None,
        tokenizer: Any,
        inference_results_Q: ray_queue.Queue,
        param_prompt_Q: ray_queue.Queue,
        evaluation_inference_results_Q: ray_queue.Queue,
        placement_group_name: str | None = None,
    ) -> None:
        """Initialize GRPO components and store them as attributes.

        Args:
            placement_group_name: Optional name for the placement group. Useful for ensuring
                different GPUs when called multiple times (e.g., "RubricTrainerActor_pg").
        """
        self.args = args
        self.tokenizer_config = tokenizer_config
        self.model_config = model_config
        # Note: beaker_config and wandb_url are passed to create_model_and_optimizer but not stored
        # as logging is handled in the main thread
        self.tokenizer = tokenizer
        self.inference_results_Q = inference_results_Q
        self.param_prompt_Q = param_prompt_Q
        self.evaluation_inference_results_Q = evaluation_inference_results_Q

        (
            self.policy_group,
            self.vllm_engines,
            self.tool_objects,
            self.resume_training_step,
            self.episode,
            self.actor_manager,
            _,  # rubric_judge_engines (suppressed)
            self.policy_generate_text_actor,
            _,  # rubric_judge_generate_text_actor (suppressed)
        ) = create_model_and_optimizer(
            args,
            tokenizer_config,
            model_config,
            beaker_config,
            wandb_url,
            tokenizer,
            inference_results_Q,
            param_prompt_Q,
            evaluation_inference_results_Q,
            suppress_judge_engine_initialization=True, # Don't initialize rubric judge engines here, we'll do it in the main function
            placement_group_name=placement_group_name,
        )

        # Print placement group information if available
        if hasattr(self.policy_group, "pg") and self.policy_group.pg:
            try:
                pg_info = ray.util.placement_group_table(self.policy_group.pg)
                LOGGER.info(f"Placement group name: {placement_group_name or 'None'}")
                LOGGER.info(f"Placement group info: {pg_info}")
                LOGGER.info(f"Placement group bundles_to_node_id: {pg_info.get('bundles_to_node_id', {})}")
            except Exception as e:
                LOGGER.warning(f"Could not print placement group info: {e}")

        # Cache generation configs for unified sampling params
        self.generation_configs = create_generation_configs(args)

        # Store first engine for direct access if needed
        if self.vllm_engines:
            self.model_dims = ray.get(self.vllm_engines[0].get_model_dims.remote())
        else:
            self.model_dims = None

    def _build_sampling_params(self, config_key: str = "train", **overrides: Any):
        """Clone a generation config and apply overrides."""
        if not self.generation_configs:
            raise RuntimeError("Generation configs not initialized. Ensure GRPO session setup completed successfully.")
        try:
            sampling_params = self.generation_configs[config_key].clone()
        except KeyError:
            raise ValueError(f"Unknown generation config '{config_key}'") from None

        for key, value in overrides.items():
            if value is not None and hasattr(sampling_params, key):
                setattr(sampling_params, key, value)
        return sampling_params

    def _build_judge_sampling_params(self):
        """Build sampling params for the rubric judge, using rubric_judge_* config fields."""
        return self._build_sampling_params(
            config_key="train",
            n=1,
            temperature=getattr(self.args, "rubric_judge_temperature", 0.6),
            max_tokens=getattr(self.args, "rubric_judge_max_tokens", 16384),
        )

    def start_training_threads(
        self,
        train_dataset: Dataset,
        iter_dataloader: ShufflingIterator | None = None,
        steps_per_phase: int | None = None,
    ) -> None:
        """Start long-lived GRPO training threads (data preparation, weight sync).

        Uses _make_reward_fn() to create the reward function from the subclass hook.
        Uses _make_replenish_prompt_fn() to create the replenish prompt function from the subclass hook.
        """
        LOGGER.info(f"{self.__class__.__name__}: start_training_threads called (pause_event exists: {hasattr(self, 'pause_event')})")

        if not hasattr(self, "args") or not self.args:
            raise RuntimeError("setup_grpo must be called before start_training_threads")

        # Create reward function from subclass hook
        reward_fn = self._make_reward_fn()

        # Initialize pause event early (before wrapping replenish_prompt_fn)
        self.pause_event = threading.Event()  # Separate event for pause/resume
        self.pause_event.set()  # Start paused by default

        # Create replenish prompt function from subclass hook (optional)
        original_replenish_prompt_fn = self._make_replenish_prompt_fn()
        assert original_replenish_prompt_fn is not None, "replenish_prompt_fn must be provided"

        # Wrap replenish_prompt_fn to check pause state
        def wrapped_replenish_prompt_fn(*args, **kwargs):
            if self.pause_event.is_set():
                LOGGER.debug(f"{self.__class__.__name__}: replenish_prompt_fn blocked (actor is paused)")
                return  # Block replenishment when paused
            return original_replenish_prompt_fn(*args, **kwargs)

        replenish_prompt_fn = wrapped_replenish_prompt_fn

        # Create queues for thread coordination
        self.packed_sequences_Q = Queue(maxsize=self.args.async_steps)
        self.generate_metrics_Q = Queue(maxsize=self.args.async_steps)
        self.weight_sync_metrics_Q = Queue(maxsize=self.args.async_steps)
        self.pending_queries_map = PendingQueriesMap()
        self.eval_pending_queries_map = PendingQueriesMap()
        # Create enriched results queue for reward computation thread output
        self.enriched_results_Q = Queue()

        # Wrap dataset in a proxy so we can refresh contents without restarting threads
        self.train_dataset_proxy = StreamingDatasetProxy(train_dataset)
        dataset_for_threads: StreamingDatasetProxy = self.train_dataset_proxy

        # Store references for resume functionality
        self._original_replenish_prompt_fn = original_replenish_prompt_fn
        self._dataset_for_threads = dataset_for_threads

        # Create thread executor (4 workers: reward computation, data preparation, weight sync, and one spare)
        self.executor = futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix=f"{self.__class__.__name__}_train"
        )
        self.stop_event = threading.Event()
        self.weight_sync_trigger_event = threading.Event()

        # Initialize iter_dataloader if not provided
        if iter_dataloader is None:
            train_dataset_idxs = np.arange(len(dataset_for_threads))
            iter_dataloader = ShufflingIterator(train_dataset_idxs, batch_size=1, seed=self.args.seed)
        self.iter_dataloader = iter_dataloader

        # Try to restore data iterator state from a previous checkpoint
        self._try_restore_data_iterator_state(steps_per_phase=steps_per_phase)

        # Start reward computation thread
        # Determine actor_id for routing in single_model_mode
        actor_id = self._get_actor_id() if hasattr(self, '_get_actor_id') else None
        LOGGER.info(f"======== ✅ {self.__class__.__name__} reward computation thread starts (actor_id={actor_id}) =========")
        self.reward_thread_future = self.executor.submit(
            reward_computation_thread,
            reward_fn,
            self.inference_results_Q,
            self.enriched_results_Q,
            self.pending_queries_map,
            self.tokenizer,
            self.generation_configs["train"],
            self.stop_event,
            None,  # timeout
            actor_id,  # For routing in single_model_mode
            self.pause_event,  # Suppress warnings when paused
        )

        # Start weight sync thread
        self.weight_sync_future = self.executor.submit(
            weight_sync_thread,
            self.args,
            self.stop_event,
            self.weight_sync_trigger_event,
            self.policy_group,
            self.actor_manager,
            self.weight_sync_metrics_Q,
            self.resume_training_step,
        )

        packing_args = self._build_data_preparation_args()

        # Start data preparation thread (uses enriched_results_Q instead of inference_results_Q)
        LOGGER.info(f"======== ✅ {self.__class__.__name__} data preparation thread starts =========")
        self.packing_future = self.executor.submit(
            data_preparation_thread,
            self.enriched_results_Q,  # Use enriched_results_Q instead of inference_results_Q
            self.param_prompt_Q,
            self.packed_sequences_Q,
            self.pending_queries_map,
            packing_args,
            self.tokenizer,
            self.args.num_training_steps,
            self.generation_configs["train"],
            self.resume_training_step,
            self.iter_dataloader,
            dataset_for_threads,
            self.actor_manager,
            self.model_dims,
            replenish_prompt_fn=replenish_prompt_fn,
        )

        # Skip sending initial prompts when paused (they'll be sent on resume)
        # Initial prompts will be sent when resume_training_threads() is called
        # This ensures actors start in a paused state and need explicit resume
        # Pause is enforced by:
        # 1. pause_event.set() blocks replenish_prompt_fn
        # 2. reward_fn checks pause_event before computing rewards

        self.num_total_tokens = 0
        self.training_start_time = time.perf_counter()
        self.episode = 0

        LOGGER.info(f"{self.__class__.__name__} training threads started (paused by default)")

    def _build_data_preparation_args(self) -> GrpoArgs:
        """Return thread-local args for the data preparation / packing path.

        rlcer_evolving rubric training consumes cached rollout groups from the
        policy actor rather than actively resampling prompts until a non-zero
        reward std appears. Keep the global policy-side filtering semantics
        intact, but disable zero-std prompt filtering for rubric-side packing so
        cached batches cannot be filtered away before they reach
        ``packed_sequences_Q``.
        """
        actor_id = self._get_actor_id() if hasattr(self, "_get_actor_id") else None
        reward_mode = getattr(self, "_reward_mode", None) or getattr(self.args, "rubric_reward_mode", None)
        if actor_id == "rubric" and reward_mode == "rlcer_evolving":
            if getattr(self.args, "filter_zero_std_samples", False):
                LOGGER.info(
                    "%s: disabling active_sampling and filter_zero_std_samples for rlcer_evolving rubric packing",
                    self.__class__.__name__,
                )
                return replace(self.args, active_sampling=False, filter_zero_std_samples=False)
        return self.args

    def pause_training_threads(self) -> None:
        """Pause training threads by setting the pause event.

        This will:
        - Block replenish_prompt_fn calls (no new prompts added)
        - Drain packed_sequences_Q to clear any pending work
        - Signal actor manager to pause
        Threads remain alive and can be resumed with resume_training_threads().
        """
        if not hasattr(self, "pause_event"):
            LOGGER.warning(f"{self.__class__.__name__}: Training threads not started, nothing to pause")
            return

        if self.pause_event.is_set():
            LOGGER.debug(f"{self.__class__.__name__}: Training threads already paused")
            return

        LOGGER.info(f"{self.__class__.__name__}: Pausing training threads...")
        self._log_transition_debug_state("pause_start")

        # Set pause event to block replenish_prompt_fn (no new prompts added)
        self.pause_event.set()

        start_wait = time.time()

        # These are long-lived worker threads. They should remain alive across
        # phase transitions, so we only surface failures here instead of waiting
        # for the futures themselves to complete.
        for future_attr, label in [
            ("reward_thread_future", "reward computation"),
            ("packing_future", "packing"),
            ("weight_sync_future", "weight sync"),
        ]:
            future = getattr(self, future_attr, None)
            if future is None or not future.done():
                continue
            try:
                future.result()
            except Exception as e:
                LOGGER.warning(f"{self.__class__.__name__}: {label} thread raised during pause: {e}")

        if hasattr(self, "weight_sync_trigger_event") and self.weight_sync_trigger_event is not None:
            self.weight_sync_trigger_event.clear()

        # NOTE: We intentionally do NOT call pause_vllm_engines() here.
        # The engine-level set_should_stop flag has a 0.1s TTL and is
        # overridden by the actor_manager's weight-sync state, making it
        # ineffective for phase transitions.  Instead we rely on:
        #   (a) pause_event blocking new prompt submission,
        #   (b) engine backlog clearing below, and
        #   (c) the reward_fn wait-while-paused loop.

        # Give in-flight requests a short grace period to complete naturally,
        # but do not block the phase transition on old work.
        if hasattr(self, "vllm_engines") and self.vllm_engines:
            LOGGER.info(f"{self.__class__.__name__}: Waiting for {len(self.vllm_engines)} vLLM engine(s) to finish active tasks...")

            max_wait_time = 5.0
            poll_interval = 0.5  # 500ms
            poll_deadline = time.time() + max_wait_time

            while time.time() < poll_deadline:
                try:
                    health_futures = [engine.health_check.remote() for engine in self.vllm_engines]
                    health_results = ray.get(health_futures, timeout=5.0)

                    total_active = sum(result.get('active_tasks', 0) for result in health_results)
                    if total_active == 0:
                        LOGGER.info(f"{self.__class__.__name__}: All vLLM engines idle (0 active tasks)")
                        break

                    LOGGER.debug(f"{self.__class__.__name__}: vLLM engines have {total_active} active tasks, waiting...")
                    time.sleep(poll_interval)
                except Exception as e:
                    LOGGER.warning(f"{self.__class__.__name__}: Failed to check vLLM engine status: {e}")
                    break
            else:
                LOGGER.warning(f"{self.__class__.__name__}: vLLM engines did not finish all tasks within grace period")

        # 5. Clear ALL engine backlogs to prevent request saturation across
        # phase transitions. In-flight requests left over from the previous
        # phase compete for GPU resources with the next actor's requests,
        # causing progressive throughput collapse and eventual pipeline stalls.
        for engine_attr, label in [
            ("vllm_engines", "shared vLLM"),
            ("rubric_judge_engines", "rubric judge"),
        ]:
            engines = getattr(self, engine_attr, None)
            if not engines:
                continue
            try:
                clear_futures = [engine.clear_pending_requests.remote() for engine in engines]
                clear_results = ray.get(clear_futures, timeout=30.0)
                total_cleared = sum(r.get("cleared_active", 0) for r in clear_results)
                if total_cleared > 0:
                    LOGGER.info(
                        f"{self.__class__.__name__}: Cleared {total_cleared} stale request(s) "
                        f"from {len(engines)} {label} engine(s)"
                    )
            except Exception as e:
                LOGGER.warning(f"{self.__class__.__name__}: Failed to clear {label} engines: {e}")

        # 5b. Drain the shared prompt queue so queued-but-unfetched requests
        # from this actor's phase don't get picked up by engines during the
        # next actor's phase. Only need to drain from one engine per group
        # since all engines in a group share the same queue.
        for engine_attr, label in [
            ("vllm_engines", "shared vLLM"),
            ("rubric_judge_engines", "rubric judge"),
        ]:
            engines = getattr(self, engine_attr, None)
            if not engines:
                continue
            try:
                drain_result = ray.get(engines[0].drain_prompt_queue.remote(), timeout=30.0)
                queue_drained = drain_result.get("drained", 0)
                if queue_drained > 0:
                    LOGGER.info(
                        f"{self.__class__.__name__}: Drained {queue_drained} queued request(s) "
                        f"from {label} prompt queue"
                    )
            except Exception as e:
                LOGGER.warning(f"{self.__class__.__name__}: Failed to drain {label} prompt queue: {e}")

        # Drain only pipeline-control queues. Do NOT drain inference_results_Q,
        # enriched_results_Q, or clear pending_queries_map here — the reward
        # thread's wait-while-paused loop will hold those results until resume,
        # avoiding the cold-start penalty of discarding valid in-flight work.
        drained_total = 0
        for queue_name, queue_obj in [
            ("param_prompt_Q", getattr(self, "param_prompt_Q", None)),
            ("packed_sequences_Q", getattr(self, "packed_sequences_Q", None)),
            ("generate_metrics_Q", getattr(self, "generate_metrics_Q", None)),
            ("weight_sync_metrics_Q", getattr(self, "weight_sync_metrics_Q", None)),
        ]:
            if queue_obj is not None:
                drained = 0
                try:
                    while True:
                        queue_obj.get_nowait()
                        drained += 1
                except Empty:
                    pass
                if drained > 0:
                    LOGGER.info(f"{self.__class__.__name__}: Drained {drained} item(s) from {queue_name}")
                    drained_total += drained

        elapsed = time.time() - start_wait
        LOGGER.info(
            f"{self.__class__.__name__}: ✅ Training threads paused "
            f"(waited {elapsed:.1f}s for in-flight work, drained {drained_total} queued items)"
        )
        self._log_transition_debug_state("pause_complete", include_engine_health=True)

    def resume_training_threads(self) -> None:
        """Resume training threads by clearing the pause event.

        This will:
        - Allow replenish_prompt_fn calls to proceed
        - Refill the queue with initial prompts (similar to start_training_threads)
        - Signal actor manager to resume
        """
        if not hasattr(self, "pause_event"):
            LOGGER.warning(f"{self.__class__.__name__}: Training threads not started, nothing to resume")
            return

        if not self.pause_event.is_set():
            LOGGER.debug(f"{self.__class__.__name__}: Training threads already running")
            return

        LOGGER.info(f"{self.__class__.__name__}: Resuming training threads...")
        self._log_transition_debug_state("resume_start")

        # Health-check vLLM engines before resuming (catches silent thread deaths from long idle)
        # Check both direct vllm_engines and policy_generate_text_actor (which has engines)
        engines_to_check = []

        if hasattr(self, "vllm_engines") and self.vllm_engines:
            engines_to_check.extend(self.vllm_engines)

        # For actors using shared engines via generate_text_actor, get engines from there
        if (hasattr(self, "policy_generate_text_actor") and
            self.policy_generate_text_actor and
            not engines_to_check):
            try:
                # Try to get vllm_engines from the generate_text_actor
                actor_engines = ray.get(self.policy_generate_text_actor.get_vllm_engines.remote(), timeout=5.0)
                if actor_engines:
                    engines_to_check.extend(actor_engines)
            except Exception as e:
                LOGGER.warning(f"{self.__class__.__name__}: Could not get engines from policy_generate_text_actor: {e}")

        if engines_to_check:
            LOGGER.info(f"{self.__class__.__name__}: Running health check on {len(engines_to_check)} vLLM engine(s)...")
            health_futures = [engine.health_check.remote() for engine in engines_to_check]
            try:
                health_results = ray.get(health_futures, timeout=60.0)
                for i, result in enumerate(health_results):
                    LOGGER.info(
                        f"{self.__class__.__name__}: Engine {i} health: "
                        f"prefetch={'OK' if result.get('prefetch_alive') else 'DEAD'}, "
                        f"process={'OK' if result.get('process_alive') else 'DEAD'}, "
                        f"loop={'OK' if result.get('loop_alive') else 'DEAD'}, "
                        f"active_tasks={result.get('active_tasks', '?')}"
                    )
            except Exception as e:
                LOGGER.error(f"{self.__class__.__name__}: Engine health check failed: {e}", exc_info=True)
                raise RuntimeError(f"vLLM engine health check failed during resume: {e}") from e

            # Warmup engines with a short inference to prevent cold-start hangs after long idle.
            # In multi-policy co-evolve mode, extra policy actors can sit idle for 30-60+ minutes
            # while the main policy trains. Their vLLM engines may become unresponsive (stale CUDA
            # context, hung engine loops, etc). A warmup forces actual CUDA execution and catches
            # hangs early (with a 120s timeout) instead of stalling the pipeline for hours.
            LOGGER.info(f"{self.__class__.__name__}: Warming up {len(engines_to_check)} vLLM engine(s)...")
            warmup_futures = [engine.warmup.remote() for engine in engines_to_check]
            try:
                warmup_results = ray.get(warmup_futures, timeout=180.0)
                ok_count = sum(1 for r in warmup_results if r.get("success"))
                LOGGER.info(
                    f"{self.__class__.__name__}: Warmup complete: {ok_count}/{len(engines_to_check)} engines OK"
                )
            except Exception as e:
                LOGGER.error(f"{self.__class__.__name__}: Engine warmup failed: {e}", exc_info=True)
                raise RuntimeError(f"vLLM engine warmup failed during resume: {e}") from e
        else:
            LOGGER.warning(f"{self.__class__.__name__}: No vLLM engines found for health check during resume")

        self._log_transition_debug_state("resume_post_warmup", include_engine_health=True)

        # NOTE: We intentionally do NOT call resume_vllm_engines() here.
        # See the symmetric comment in pause_training_threads — the
        # engine-level set_should_stop flag is only effective for the
        # weight-sync path where actor_manager coordinates it.

        # Drain only packed_sequences_Q — any packed batches left over
        # from the previous phase are stale (wrong model weights).
        # Do NOT drain inference_results_Q, enriched_results_Q, or clear
        # pending_queries_map: the reward thread holds valid in-flight
        # results that will be processed once we clear pause_event below.
        for queue_name, queue_obj in [
            ("packed_sequences_Q", getattr(self, "packed_sequences_Q", None)),
        ]:
            if queue_obj is None:
                continue
            drained_count = 0
            while True:
                try:
                    queue_obj.get_nowait()
                    drained_count += 1
                except Empty:
                    break
            if drained_count > 0:
                LOGGER.info(f"{self.__class__.__name__}: Drained {drained_count} item(s) from {queue_name}")

        # Clear pause event to allow replenish_prompt_fn
        self.pause_event.clear()

        # Refill queue with initial prompts (similar to start_training_threads).
        # Count only calls where the replenish function actually enqueued a
        # prompt (returns non-None).  In rlcer_evolving mode the replenish
        # function is a no-op that returns None, so refilled stays 0.
        refilled = 0
        if hasattr(self, "_original_replenish_prompt_fn") and self._original_replenish_prompt_fn is not None:
            num_initial_prompts = max(
                self.args.async_steps * self.args.num_unique_prompts_rollout,
                self.args.num_unique_prompts_rollout,
            )
            for _ in range(num_initial_prompts):
                ret = self._original_replenish_prompt_fn(
                    result=None,
                    iter_dataloader=self.iter_dataloader,
                    prompt_dataset=self._dataset_for_threads,
                    pending_queries_map=self.pending_queries_map,
                    param_prompt_Q=self.param_prompt_Q,
                    generation_config=self.generation_configs["train"],
                    training_step=self.resume_training_step,
                )
                if ret is not None:
                    refilled += 1
        if refilled > 0:
            LOGGER.info(f"{self.__class__.__name__}: ✅ Queue refilled with {refilled} prompt(s)")

        self._log_transition_debug_state("resume_complete")
        LOGGER.info(f"{self.__class__.__name__}: ✅ Training threads resumed")

    def pause_vllm_engines(self) -> None:
        """Pause vLLM engines by calling set_should_stop(True) on all engines.

        Only effective when called from the weight-sync path, where the
        actor_manager's should_stop flag is also set (see grpo_fast.py
        weight_sync_thread).  Do NOT call this during phase transitions —
        the engine-level flag has a 0.1s TTL and is overridden by the
        actor_manager's state, making it a no-op outside weight sync.
        """
        engines = []
        if hasattr(self, 'vllm_engines') and self.vllm_engines:
            engines.extend(self.vllm_engines)

        if engines:
            LOGGER.info(f"[{self.__class__.__name__}] Pausing {len(engines)} vLLM engines")
            ray.get([engine.set_should_stop.remote(True) for engine in engines])
            LOGGER.debug(f"[{self.__class__.__name__}] vLLM engines paused")

    def resume_vllm_engines(self) -> None:
        """Resume vLLM engines by calling set_should_stop(False) on all engines.

        Only effective when called from the weight-sync path.  See the
        symmetric note on pause_vllm_engines.
        """
        engines = []
        if hasattr(self, 'vllm_engines') and self.vllm_engines:
            engines.extend(self.vllm_engines)

        if engines:
            LOGGER.info(f"[{self.__class__.__name__}] Resuming {len(engines)} vLLM engines")
            ray.get([engine.set_should_stop.remote(False) for engine in engines])
            LOGGER.debug(f"[{self.__class__.__name__}] vLLM engines resumed")

    async def _compute_rewards_and_prompts(
        self, decoded_responses: list[str], ground_truths: list[Any], queries: list[str] | None = None
    ) -> tuple[list[float], dict[str, Any]]:
        """Subclass hook for custom reward computation.

        Subclasses should override this to:
        - Query the counterpart actor's vLLM for prompts/rubrics
        - Compute rewards based on the responses
        - Return (rewards_list, metrics_dict)
        """
        raise NotImplementedError("Subclasses must implement _compute_rewards_and_prompts")

    def _make_reward_fn(self) -> Any:
        """Create a reward function that wraps _compute_rewards_and_prompts.

        Returns an async function matching the signature expected by reward_computation_thread.
        The signature matches grpo_fast.py's make_reward_fn pattern.

        The returned function checks pause_event before computing rewards, ensuring
        paused actors don't make vLLM requests (via rubric generation, etc.).
        """

        async def reward_fn(
            responses: list[Any],
            decoded_responses: list[str],
            ground_truths: list[Any],
            datasets: list[str],
            finish_reasons: list[str],
            infos: Any,  # RequestInfo from GenerationResult
            queries: list[str] | None = None,
        ) -> tuple[list[float], dict[str, Any]]:
            # Wait while the actor is paused (phase transition in progress).
            # This preserves in-flight results instead of dropping them, so the
            # pipeline doesn't have to cold-start from an empty queue on resume.
            if hasattr(self, "pause_event") and self.pause_event is not None:
                pause_wait_start = time.time()
                while self.pause_event.is_set():
                    elapsed = time.time() - pause_wait_start
                    if elapsed > 120.0:
                        return [], {"__drop_result__": "paused_timeout_120s"}
                    await asyncio.sleep(1.0)

            with Timer(
                "[Reward Computation Thread] Computing rewards and prompts -- 🧮 Computing rewards and prompts", noop=True
            ) as timer:
                rewards, metrics = await self._compute_rewards_and_prompts(decoded_responses, ground_truths, queries)
            metrics["time/reward"] = timer.duration
            return rewards, metrics

        return reward_fn

    def _send_next_prompt_to_generator(
        self,
        iter_dataloader: ShufflingIterator,
        prompt_dataset: Dataset,
        pending_queries_map: PendingQueriesMap,
        param_prompt_Q: ray_queue.Queue,
        generation_config: vllm.SamplingParams,
        training_step: int,
    ) -> None:
        """Send the next prompt to the generator using add_prompt_to_generator.

        This is a helper method used by both initial prompt sending and replenish logic.
        Subclasses override add_prompt_to_generator to customize behavior.
        """
        dataset_index = next(iter_dataloader)
        self.add_prompt_to_generator(
            prompt_dataset[dataset_index],
            dataset_index,
            iter_dataloader.epoch_number,
            training_step,
            pending_queries_map,
            param_prompt_Q,
            generation_config,
            is_eval=False,
        )

    def _make_replenish_prompt_fn(self) -> Callable:
        """Create a replenish prompt function that uses add_prompt_to_generator.

        Returns a function that calls _send_next_prompt_to_generator, which in turn
        calls add_prompt_to_generator. Subclasses override add_prompt_to_generator
        to provide custom behavior, ensuring both initial and replenished prompts
        use the same logic.

        Returns:
            A callable function that uses _send_next_prompt_to_generator.
        """
        def replenish_prompt_fn(
            result,
            iter_dataloader,
            prompt_dataset,
            pending_queries_map,
            param_prompt_Q,
            generation_config,
            training_step,
        ) -> None:
            LOGGER.debug(f"[BaseTrainerActor] replenish_prompt_fn: iter_dataloader={iter_dataloader}, prompt_dataset={prompt_dataset}, pending_queries_map={pending_queries_map}, param_prompt_Q={param_prompt_Q}, generation_config={generation_config}, training_step={training_step}")
            """Replenish prompt using _send_next_prompt_to_generator."""
            self._send_next_prompt_to_generator(
                iter_dataloader,
                prompt_dataset,
                pending_queries_map,
                param_prompt_Q,
                generation_config,
                training_step,
            )

        return replenish_prompt_fn

    def train_one_step(self, training_step: int, train_dataset: Dataset) -> dict[str, Any]:
        """Execute one training step using GRPO infrastructure.

        This method:
        1. Gets packed sequences from the data preparation thread
        2. Calls the subclass's reward computation hook
        3. Executes one_training_step with the collated data
        4. Triggers weight sync
        """
        if not hasattr(self, "packed_sequences_Q"):
            raise RuntimeError("start_training_threads must be called before train_one_step")

        # Update current training step for use by _compute_rewards_and_prompts
        # This is used by replay buffer for age-based sampling (min_age/max_age are in training steps)
        self._current_training_step = training_step

        start_time = time.perf_counter()

        # Health check
        def health_check_fn():
            if hasattr(self, "reward_thread_future") and self.reward_thread_future.done():
                LOGGER.debug(f"[{self.__class__.__name__}] Health check: reward_thread_future is done, checking result...")
                try:
                    self.reward_thread_future.result()
                except Exception as e:
                    LOGGER.error(f"[{self.__class__.__name__}] Health check: reward_thread_future raised exception: {e}", exc_info=True)
                    raise
            if self.packing_future.done():
                LOGGER.debug(f"[{self.__class__.__name__}] Health check: packing_future is done, checking result...")
                try:
                    self.packing_future.result()
                except Exception as e:
                    LOGGER.error(f"[{self.__class__.__name__}] Health check: packing_future raised exception: {e}", exc_info=True)
                    raise
            if self.weight_sync_future.done():
                LOGGER.debug(f"[{self.__class__.__name__}] Health check: weight_sync_future is done, checking result...")
                try:
                    self.weight_sync_future.result()
                except Exception as e:
                    LOGGER.error(f"[{self.__class__.__name__}] Health check: weight_sync_future raised exception: {e}", exc_info=True)
                    raise

        health_check_fn()

        def stall_recovery_fn():
            actor_name = self.__class__.__name__

            # 1. Clear policy engine pending requests
            engines = self.vllm_engines if hasattr(self, "vllm_engines") and self.vllm_engines else []
            if engines:
                LOGGER.warning("[%s Stall Recovery] Clearing pending requests on %d policy engines", actor_name, len(engines))
                clear_futures = [engine.clear_pending_requests.remote() for engine in engines]
                try:
                    results = ray.get(clear_futures, timeout=120.0)
                    total_cleared = sum(r.get("cleared_active", 0) if isinstance(r, dict) else 0 for r in results)
                    total_aborted = sum(r.get("aborted", 0) if isinstance(r, dict) else 0 for r in results)
                    LOGGER.warning(
                        "[%s Stall Recovery] Policy engines: cleared %d active tasks, aborted %d engine requests",
                        actor_name, total_cleared, total_aborted,
                    )
                except Exception as e:
                    LOGGER.error("[%s Stall Recovery] Failed to clear policy engines: %s", actor_name, e)

            # 2. Clear Multi-Judge engine pending requests AND drain their
            #    prompt queues.  A hung judge engine leaves queued requests
            #    that will never be consumed; draining prevents them from
            #    blocking new work submitted after recovery.
            mj = getattr(self, "_multi_judge_engines", None)
            if mj is not None and hasattr(mj, "judge_engines"):
                judge_clear_futures = []
                judge_drain_futures = []
                total_judge_engines = 0
                seen_queues = set()
                for engines_obj in mj.judge_engines:
                    for eng in engines_obj.engines:
                        judge_clear_futures.append(eng.clear_pending_requests.remote())
                        total_judge_engines += 1
                    # Drain the shared prompt queue once per judge group
                    if engines_obj.engines:
                        q_id = id(engines_obj)
                        if q_id not in seen_queues:
                            seen_queues.add(q_id)
                            judge_drain_futures.append(engines_obj.engines[0].drain_prompt_queue.remote())
                if judge_clear_futures:
                    LOGGER.warning("[%s Stall Recovery] Clearing %d Multi-Judge engines", actor_name, total_judge_engines)
                    try:
                        jresults = ray.get(judge_clear_futures, timeout=120.0)
                        jc = sum(r.get("cleared_active", 0) if isinstance(r, dict) else 0 for r in jresults)
                        ja = sum(r.get("aborted", 0) if isinstance(r, dict) else 0 for r in jresults)
                        LOGGER.warning(
                            "[%s Stall Recovery] Judge engines: cleared %d active tasks, aborted %d engine requests",
                            actor_name, jc, ja,
                        )
                    except Exception as e:
                        LOGGER.error("[%s Stall Recovery] Failed to clear judge engines: %s", actor_name, e)
                if judge_drain_futures:
                    try:
                        drain_results = ray.get(judge_drain_futures, timeout=30.0)
                        total_drained = sum(r.get("drained", 0) if isinstance(r, dict) else 0 for r in drain_results)
                        if total_drained > 0:
                            LOGGER.warning(
                                "[%s Stall Recovery] Drained %d queued judge requests from %d judge group(s)",
                                actor_name, total_drained, len(drain_results),
                            )
                    except Exception as e:
                        LOGGER.error("[%s Stall Recovery] Failed to drain judge queues: %s", actor_name, e)

            # 3. Inject fresh prompts so the pipeline has work to do
            if (
                hasattr(self, "iter_dataloader") and self.iter_dataloader is not None
                and hasattr(self, "train_dataset") and self.train_dataset is not None
                and hasattr(self, "param_prompt_Q") and self.param_prompt_Q is not None
                and hasattr(self, "pending_queries_map")
                and hasattr(self, "generation_configs") and self.generation_configs
            ):
                num_inject = self.args.num_unique_prompts_rollout
                LOGGER.warning("[%s Stall Recovery] Injecting %d fresh prompts", actor_name, num_inject)
                for _ in range(num_inject):
                    try:
                        dataset_index = next(self.iter_dataloader)
                        self.add_prompt_to_generator(
                            self.train_dataset[dataset_index],
                            dataset_index,
                            self.iter_dataloader.epoch_number,
                            training_step,
                            self.pending_queries_map,
                            self.param_prompt_Q,
                            self.generation_configs["train"],
                            is_eval=False,
                        )
                    except Exception as e:
                        LOGGER.error("[%s Stall Recovery] Failed to inject prompt: %s", actor_name, e)
                        break
                LOGGER.warning("[%s Stall Recovery] Recovery complete", actor_name)

        # Get packed sequences from thread
        LOGGER.debug(f"[{self.__class__.__name__}] train_one_step: Waiting for packed sequences from queue (queue_size={self.packed_sequences_Q.qsize() if hasattr(self.packed_sequences_Q, 'qsize') else 'N/A'}, training_step={training_step})")
        (
            collated_data,
            data_thread_metrics,
            self.num_total_tokens,
            num_step_tokens,
            prompt_lengths,
            response_lengths,
            num_filtered_prompts,
        ) = load_data_from_packing_thread(
            self.packed_sequences_Q, self.num_total_tokens, self.stop_event, health_check_fn, stall_recovery_fn
        )
        LOGGER.debug(f"[{self.__class__.__name__}] train_one_step: Received packed sequences (collated_data is None: {collated_data is None})")

        if collated_data is None:
            return {"skipped": True}

        # Merge metrics from queues
        for metrics_Q in [self.generate_metrics_Q, self.weight_sync_metrics_Q]:
            try:
                data_thread_metrics.update(metrics_Q.get_nowait())
            except Empty:
                pass

        # Update episode counter
        self.episode += self.args.num_unique_prompts_rollout * self.args.num_samples_per_prompt_rollout

        # Execute training step
        actor_type = self.__class__.__name__.replace("TrainerActor", "").lower()
        metric_prefix = actor_type if actor_type else None

        wandb_url = ""

        original_with_tracking = self.args.with_tracking
        self.args.with_tracking = False

        checkpoint_subdir = actor_type if actor_type else None

        metrics = one_training_step(
            self.args,
            self.policy_group,
            collated_data,
            self.tokenizer,
            data_thread_metrics,
            self.episode,
            training_step,
            self.num_total_tokens,
            num_step_tokens,
            start_time,
            train_dataset,
            self.training_start_time,
            wandb_url,
            self.tokenizer_config.chat_template_name if hasattr(self.tokenizer_config, "chat_template_name") else None,
            self.model_dims,
            prompt_lengths,
            response_lengths,
            self.actor_manager,
            self.iter_dataloader,
            metric_prefix=metric_prefix,
            checkpoint_subdir=checkpoint_subdir,
        )

        # Restore original with_tracking value
        self.args.with_tracking = original_with_tracking

        # Trigger weight sync (fire-and-forget pattern from working 3a349fb)
        self.weight_sync_trigger_event.set()

        return {
            "completed": True,
            "training_step": training_step,
            "episode": self.episode,
            "metrics": metrics,
        }

    def get_actor_manager(self) -> ray.actor.ActorHandle | None:
        """Get the actor_manager handle for this actor."""
        return getattr(self, "actor_manager", None)



@ray.remote
class RubricTrainerActor(BaseTrainerActor):
    """Ray actor responsible for creating and updating rubrics."""

    def __init__(
        self,
        rubric_model: str,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        rubric_judge_generate_text_actor: Any | None = None,
        policy_generate_text_actor: Any | None = None,
        grpo_args: GrpoArgs | None = None,
        tokenizer_config: GrpoTokenizerConfig | None = None,
        model_config: GrpoModelConfig | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        single_model_mode: bool = False,
        generation_examples_dir: str | None = None,
        num_examples_to_log: int = 3,
        log_examples_every_n_steps: int = 1,
        inference_model_engines_obj: Any | None = None,
        rubric_judge_tokenizer: Any | None = None,
        script_args: ScriptArgs | None = None,
        api_rubric_generator: str | None = None,
        multi_judge_engines: MultiJudgeEngines | None = None,
        reward_mode: str = "rubric_judge",
        rubric_reward_use_margin: bool = False,
        rubric_format_reward_weight: float = 0.0,
        rubric_prompt_key: str = "rubric_generation",
    ) -> None:
        super().__init__()
        self.generation_kwargs = generation_kwargs or {}
        self.rubric_model = rubric_model
        self.rubric_prompt_key = rubric_prompt_key
        self.rubric_judge_generate_text_actor = rubric_judge_generate_text_actor
        self.policy_generate_text_actor = policy_generate_text_actor
        self.policy_actor: ray.actor.ActorHandle | None = None
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self.iter_dataloader: grpo_fast.ShufflingIterator | None = None
        self._single_model_mode = single_model_mode
        self._generation_examples_dir = Path(generation_examples_dir) if generation_examples_dir else None
        self._num_examples_to_log = num_examples_to_log
        self._log_examples_every_n_steps = log_examples_every_n_steps
        self._current_episode = 0  # Track episode for wandb-consistent logging
        self._current_training_step = 0  # Track training step (gradient updates) for replay buffer age
        self._reward_mode = reward_mode
        self._rubric_reward_use_margin = rubric_reward_use_margin
        self._rubric_format_reward_weight = rubric_format_reward_weight
        # Store auxiliary objects for data provider (passed during initialization)
        self._inference_model_engines_obj = inference_model_engines_obj
        self._rubric_judge_tokenizer = rubric_judge_tokenizer
        # Store script_args for access by data provider (contains inference_model_for_question_inference)
        self.script_args = script_args
        # API-based rubric generator (for baseline comparison)
        self._api_rubric_generator = api_rubric_generator
        # Multi-judge engines (for multi-judge training)
        self._multi_judge_engines = multi_judge_engines
        if multi_judge_engines:
            LOGGER.info(
                "Using multi-judge training with %d judges: %s (mode=%s, tie_breaker=%s, alpha=%.2f, beta=%.2f)",
                len(multi_judge_engines.judge_models),
                multi_judge_engines.judge_models,
                multi_judge_engines.aggregation_mode,
                multi_judge_engines.tie_breaker,
                multi_judge_engines.alpha,
                multi_judge_engines.beta,
            )
        if api_rubric_generator:
            LOGGER.info("Using API-based rubric generator: %s (rubric model will NOT be trained)", api_rubric_generator)

        # RLCER "with evolving": buffer for storing policy rollout data from the
        # policy phase so the rubric phase can compute K_valid/K rewards.
        # Keyed by question text → {"answers": list[str], "correctness": list[float]}.
        self._rlcer_rollout_buffer: dict[str, dict[str, Any]] = {}
        self._rlcer_evolving_cached_generations: list[dict[str, Any]] = []

        # Initialize data provider (will be set up properly after policy_actor is set)
        self._data_provider: BaseDataProvider | None = None

        # Initialize GRPO session if config is provided
        if grpo_args and tokenizer_config and model_config:
            # Create unique placement group name for this rubric actor to avoid collisions
            model_suffix = rubric_model.split("/")[-1].lower().replace("-", "_")
            pg_name = f"RubricTrainerActor_{model_suffix}_pg"

            if self._initialize_grpo_session(
                grpo_args, tokenizer_config, model_config, model_override=rubric_model, log_prefix="Rubric model: ",
                train_dataset=train_dataset, eval_dataset=eval_dataset,
                skip_engine_creation=single_model_mode,
                placement_group_name=pg_name,
            ):

                # Use dataset from initialization (already loaded by _initialize_grpo_session)
                # self.train_dataset is already set by _initialize_grpo_session

                # Create iter_dataloader similar to grpo_fast.py line 3318-3319
                train_dataset_idxs = np.arange(len(self.train_dataset))
                self.iter_dataloader = grpo_fast.ShufflingIterator(train_dataset_idxs, 1, seed=self.args.seed)

                LOGGER.info(
                    "Loaded dataset with %d examples and created iter_dataloader",
                    len(self.train_dataset),
                )

                if self._is_rlcer_evolving_mode():
                    LOGGER.info(
                        "RubricTrainerActor: rlcer_evolving reuses grouped cached rubric generations; "
                        "preserving num_samples_per_prompt_rollout=%d",
                        self.args.num_samples_per_prompt_rollout,
                    )

        LOGGER.info(
            "RubricTrainerActor initialised with model %s (grpo=%s, dataset=%s, single_model_mode=%s)",
            self.rubric_model,
            "enabled" if hasattr(self, "args") and self.args else "disabled",
            "loaded" if self.train_dataset is not None else "none",
            single_model_mode,
        )

    def get_generate_text_actor(self) -> Any:
        """Return the rubric model's GenerateTextActor handle.

        This is used by the data provider to generate rubrics from the evolving
        rubric model without going through the RubricTrainerActor (which would
        deadlock, since the trainer's main thread is blocked in the training loop).
        """
        return self.policy_generate_text_actor

    def set_policy_actor(
        self,
        policy_actor: ray.actor.ActorHandle,
    ) -> None:
        """Set reference to policy actor for querying rollouts."""
        self.policy_actor = policy_actor
    
    def set_data_provider(
        self,
        data_provider: ray.actor.ActorHandle,
    ) -> None:
        """Set the data provider Ray actor.
        
        Args:
            data_provider: Ray actor handle to the data provider
        """
        self._data_provider = data_provider

    def _get_actor_id(self) -> str:
        """Return the actor ID for routing in single_model_mode."""
        return "rubric"

    # ---- public API -------------------------------------------------------

    async def create_rubric(
        self, question: str, *, rubric_id: str | None = None, metadata: dict[str, Any] | None = None, use_both_models: bool = False
    ) -> RubricSpec:
        """Generate a rubric for ``question`` and return its spec."""

        # Generate example answers if using both models
        policy_answer = None
        baseline_answer = None
        assert not use_both_models, "use_both_models is not supported yet"
        if use_both_models and self.policy_actor:
            # Generate answers from both policy and baseline models for better rubric creation
            policy_answer = await self.policy_actor._generate_policy_answer.remote(question, "")  # Empty rubric for initial generation
            baseline_answer = await self.policy_actor._generate_baseline_answer.remote(question)

        # Use API model for rubric generation if configured (baseline mode)
        if self._api_rubric_generator:
            rubric_text = await self._generate_rubric_via_api(question)
            model_name = self._api_rubric_generator
        else:
            # Use local vLLM engines for rubric generation (normal training mode)
            # Apply rubric-specific generation_kwargs (e.g. temperature) on top of base config
            kwargs = dict(self.generation_kwargs)
            kwargs.setdefault("n", 1)
            sampling_params = self._build_sampling_params(config_key="train", **kwargs)

            # Use rubric chat templates for message formatting
            if use_both_models and policy_answer and baseline_answer:
                # Use template with examples
                prompt_token_ids = format_messages(
                    "rubric_generation_with_examples",
                    {"question": question, "policy_answer": policy_answer, "baseline_answer": baseline_answer},
                    tokenizer=self.tokenizer,
                    add_generation_prompt=True,
                )
            else:
                prompt_token_ids = format_messages(
                    self.rubric_prompt_key,
                    {"question": question},
                    tokenizer=self.tokenizer,
                    add_generation_prompt=True,
                )

            # Use policy_generate_text_actor for load-balanced generation
            if not self.policy_generate_text_actor:
                raise RuntimeError("policy_generate_text_actor is not initialized")
            sp_dict = msgspec.structs.asdict(sampling_params)
            rubric_text = await self.policy_generate_text_actor.generate_text_from_token_ids.remote(prompt_token_ids, sp_dict)
            model_name = self.rubric_model
            
        rubric_spec = RubricSpec(
            rubric_id=rubric_id or uuid.uuid4().hex,
            question=question,
            rubric_text=rubric_text,
            model_name=model_name,
            metadata={"created_at": _current_timestamp(), "history_index": 0, **(metadata or {})},
        )
        LOGGER.debug("Created rubric %s for question hash %s", rubric_spec.rubric_id, hash(question))
        return rubric_spec

    async def create_rlcer_rubric(
        self,
        question: str,
        reference_response: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RLCERRubricSpec:
        """Generate an RL-CER-format rubric with per-item importance scores."""
        return await generate_rlcer_rubric_spec(
            question,
            reference_response,
            api_rubric_generator=self._api_rubric_generator,
            rubric_model=self.rubric_model,
            tokenizer=self.tokenizer,
            generation_kwargs=self.generation_kwargs,
            build_sampling_params=self._build_sampling_params,
            policy_generate_text_actor=self.policy_generate_text_actor,
            metadata={
                "created_at": _current_timestamp(),
                **(metadata or {}),
            },
        )
    
    async def _generate_rubric_via_api(self, question: str) -> str:
        """Generate a rubric using an API model (e.g., GPT-4).
        
        Args:
            question: The question to generate a rubric for.
            
        Returns:
            The generated rubric text.
        """
        from open_instruct.search_rewards.utils.run_utils import run_litellm_async
        from open_instruct.search_rewards.utils.rubric_chat_templates import get_rubric_system_prompt
        
        try:
            response = await run_litellm_async(
                model_name=self._api_rubric_generator,
                system_prompt=get_rubric_system_prompt(self.rubric_prompt_key),
                user_prompt=question,
            )
            return response
        except Exception as e:
            LOGGER.error("Error generating rubric via API: %s", e)
            return f"Error generating rubric: {str(e)}"

    def _is_rlcer_evolving_mode(self) -> bool:
        """Check if the rubric actor is in RLCER 'with evolving' mode."""
        return isinstance(self._reward_mode, str) and self._reward_mode == "rlcer_evolving"

    def _is_rubric_arm_mode(self) -> bool:
        return isinstance(self._reward_mode, str) and self._reward_mode == "rubric_arm"

    def add_experience(self, question: str, answer: str, step: int = 0) -> None:
        """Add a policy rollout to the replay buffer with step number for age-based sampling.
        
        Args:
            question: The question that was answered
            answer: The policy's answer
            step: The policy training step when this experience was generated
        """
        if self._data_provider:
            ray.get(self._data_provider.add_experience.remote(question, answer, step))

    def add_rlcer_rollout_data(
        self,
        question: str,
        answers: list[str],
        correctness_vector: list[float],
    ) -> None:
        """Store policy rollout data for RLCER 'with evolving' rubricator reward.

        Called by the policy phase after computing RLCER rewards.  During the
        subsequent rubric training phase, this data is retrieved to compute
        K_valid/K for the rubricator's newly generated rubrics.

        Args:
            question: The question text (used as lookup key).
            answers: N policy rollout answers for this question.
            correctness_vector: Binary correctness labels (len = N).
        """
        self._rlcer_rollout_buffer[question] = {
            "answers": list(answers),
            "correctness": list(correctness_vector),
        }
        LOGGER.debug(
            "[RubricTrainerActor] Stored RLCER rollout data for question hash %s: "
            "%d answers, %.1f%% correct",
            hash(question),
            len(answers),
            100.0 * sum(correctness_vector) / max(len(correctness_vector), 1),
        )

    def set_rlcer_evolving_cached_generations(self, cached_generations: list[dict[str, Any]]) -> None:
        """Install the next rubric step's cached policy-step rubric generations."""
        self._rlcer_evolving_cached_generations = [dict(item) for item in cached_generations]
        LOGGER.info(
            "[RubricTrainerActor] Loaded %d cached RL-CER evolving generation(s) for the next rubric step",
            len(self._rlcer_evolving_cached_generations),
        )

    def _make_replenish_prompt_fn(self) -> Callable:
        if self._is_rlcer_evolving_mode():
            def replenish_prompt_fn(
                result,
                iter_dataloader,
                prompt_dataset,
                pending_queries_map,
                param_prompt_Q,
                generation_config,
                training_step,
            ) -> None:
                # RL-CER evolving prompts are staged explicitly from the latest
                # policy step so the rubric actor does not free-run on stale data.
                return None

            return replenish_prompt_fn

        return super()._make_replenish_prompt_fn()

    def start_training_threads(
        self,
        train_dataset: Dataset,
        iter_dataloader: ShufflingIterator | None = None,
        steps_per_phase: int | None = None,
    ) -> None:
        dynamic_dataset = RubricDynamicDataset(train_dataset)
        super().start_training_threads(dynamic_dataset, iter_dataloader, steps_per_phase=steps_per_phase)

    def _group_rlcer_evolving_cached_generations(
        self,
        cached_generations: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        expected_group_size = max(int(getattr(self.args, "num_samples_per_prompt_rollout", 1)), 1)
        
        def _sort_group(group_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(
                group_items,
                key=lambda entry: int(entry.get("sample_index_within_prompt", 0)),
            )

        def _group_by_key(key_fn: Callable[[dict[str, Any]], Any]) -> list[list[dict[str, Any]]]:
            grouped: collections.OrderedDict[Any, list[dict[str, Any]]] = collections.OrderedDict()
            for item in cached_generations:
                grouped.setdefault(key_fn(item), []).append(item)
            return [_sort_group(group_items) for group_items in grouped.values()]

        grouped_items: list[list[dict[str, Any]]]
        if all("prompt_group_index" in item for item in cached_generations):
            grouped_items = _group_by_key(
                lambda item: int(item.get("prompt_group_index", 0))
            )
        else:
            grouped_items = [
                cached_generations[start : start + expected_group_size]
                for start in range(0, len(cached_generations), expected_group_size)
            ]

        if all(len(group) == expected_group_size for group in grouped_items):
            return grouped_items

        # Policy-side RL-CER reward computation buffers one question-group at a time,
        # so prompt_group_index is local to that callback and can reset across a step.
        # When that happens, regroup using the question text, which is stable across the
        # cached step batch and preserves the original rollout grouping.
        if all(str(item.get("question", "")) for item in cached_generations):
            question_grouped_items = _group_by_key(lambda item: str(item.get("question", "")))
            if all(len(group) == expected_group_size for group in question_grouped_items):
                return question_grouped_items

        raise ValueError(
            "rlcer_evolving cached generations must preserve the full policy rollout group "
            f"size ({expected_group_size}); got {[len(group) for group in grouped_items]}"
        )

    @staticmethod
    def _combine_rlcer_cached_group_result(
        cached_group: list[dict[str, Any]],
        *,
        dataset_index: int,
        epoch_number: int,
        training_step: int,
        actor_id: str | None,
    ) -> GenerationResult:
        responses: list[list[int]] = []
        finish_reasons: list[str] = []
        masks: list[list[int]] = []
        logprobs: list[list[float]] = []
        num_calls: list[int] = []
        timeouts: list[int] = []
        tool_errors: list[str] = []
        tool_outputs: list[str] = []
        tool_runtimes: list[float] = []
        tool_calleds: list[bool] = []
        total_prompt_tokens = 0
        total_response_tokens = 0
        max_generation_time = 0.0
        earliest_start_time: float | None = None
        start_time: float | None = None

        for cached_item in cached_group:
            generation_result = cached_item.get("generation_result")
            if generation_result is None:
                raise ValueError("rlcer_evolving cached generation is missing generation_result")
            if len(generation_result.responses) != 1:
                raise ValueError(
                    "rlcer_evolving cached generation_result must contain exactly one response; "
                    f"got {len(generation_result.responses)}"
                )

            responses.extend(copy.deepcopy(generation_result.responses))
            finish_reasons.extend(list(generation_result.finish_reasons))
            masks.extend(copy.deepcopy(generation_result.masks))

            sample_logprobs = generation_result.logprobs or [
                [float("nan")] * len(response)
                for response in generation_result.responses
            ]
            logprobs.extend(copy.deepcopy(sample_logprobs))

            request_info = generation_result.request_info
            num_calls.extend(list(request_info.num_calls))
            timeouts.extend(list(request_info.timeouts))
            tool_errors.extend(list(request_info.tool_errors))
            tool_outputs.extend(list(request_info.tool_outputs))
            tool_runtimes.extend(list(request_info.tool_runtimes))
            tool_calleds.extend(list(request_info.tool_calleds))

            if generation_result.start_time is not None:
                start_time = (
                    generation_result.start_time
                    if start_time is None
                    else min(start_time, generation_result.start_time)
                )

            if generation_result.token_statistics is not None:
                stats = generation_result.token_statistics
                total_prompt_tokens += stats.num_prompt_tokens
                total_response_tokens += stats.num_response_tokens
                max_generation_time = max(max_generation_time, float(stats.generation_time))
                if stats.earliest_start_time is not None:
                    earliest_start_time = (
                        stats.earliest_start_time
                        if earliest_start_time is None
                        else min(earliest_start_time, stats.earliest_start_time)
                    )

        token_statistics = (
            TokenStatistics(
                num_prompt_tokens=total_prompt_tokens,
                num_response_tokens=total_response_tokens,
                generation_time=max_generation_time,
                earliest_start_time=earliest_start_time,
            )
            if total_prompt_tokens or total_response_tokens or earliest_start_time is not None
            else None
        )

        return GenerationResult(
            responses=responses,
            finish_reasons=finish_reasons,
            masks=masks,
            request_info=RequestInfo(
                num_calls=num_calls,
                timeouts=timeouts,
                tool_errors=tool_errors,
                tool_outputs=tool_outputs,
                tool_runtimes=tool_runtimes,
                tool_calleds=tool_calleds,
            ),
            dataset_index=dataset_index,
            epoch_number=epoch_number,
            training_step=training_step,
            token_statistics=token_statistics,
            start_time=start_time,
            logprobs=logprobs,
            actor_id=actor_id,
        )

    async def _build_rlcer_evolving_enriched_results(
        self,
        cached_generations: list[dict[str, Any]],
        *,
        training_step: int,
        epoch_number: int,
        actor_id: str | None,
    ) -> list[EnrichedGenerationResult]:
        from open_instruct.search_rewards.rubric_judge_rewards import (
            compute_rlcer_rubricator_reward,
            parse_rlcer_rubric_items_with_scores,
        )

        if not cached_generations:
            return []

        grouped_cached_generations = self._group_rlcer_evolving_cached_generations(cached_generations)
        flat_cached_generations = [item for group in grouped_cached_generations for item in group]

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )
        correlation_threshold = float(os.environ.get("RLCER_CORRELATION_THRESHOLD", "0.2"))

        async def _compute_reward_detail(cached_item: dict[str, Any]) -> dict[str, Any]:
            cached_reward_detail = cached_item.get("rubricator_reward_detail")
            if isinstance(cached_reward_detail, dict):
                return dict(cached_reward_detail)

            question = str(cached_item.get("question", ""))
            rubric_text = str(cached_item.get("rubric_text", ""))
            rollout_data = self._rlcer_rollout_buffer.get(question)
            if rollout_data and len(rollout_data.get("answers", [])) > 1:
                return await compute_rlcer_rubricator_reward(
                    question=question,
                    rubric_text=rubric_text,
                    rollout_answers=list(rollout_data["answers"]),
                    correctness_vector=list(rollout_data["correctness"]),
                    correlation_threshold=correlation_threshold,
                    rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                )

            parsed_items, _ = parse_rlcer_rubric_items_with_scores(rubric_text)
            r_format = 1.0 if parsed_items else 0.0
            return {
                "reward": r_format,
                "k_valid": 0,
                "k_total": len(parsed_items),
                "validity_fraction": 0.0,
                "r_format": r_format,
                "rubric_items": parsed_items,
                "valid_indices": [],
                "skipped_no_rollouts": 1,
            }

        with Timer("[RubricTrainerActor] Building cached RL-CER evolving enriched results", noop=True) as timer:
            reward_details = await asyncio.gather(*[_compute_reward_detail(item) for item in flat_cached_generations])

        enriched_results: list[EnrichedGenerationResult] = []
        reward_cursor = 0
        for group_index, cached_group in enumerate(grouped_cached_generations):
            group_reward_details = reward_details[reward_cursor : reward_cursor + len(cached_group)]
            reward_cursor += len(cached_group)

            if any(not list(cached_item.get("prompt_token_ids") or []) for cached_item in cached_group):
                raise ValueError("rlcer_evolving cached generation group is missing prompt_token_ids")

            grouped_result = self._combine_rlcer_cached_group_result(
                cached_group,
                dataset_index=group_index,
                epoch_number=epoch_number,
                training_step=training_step,
                actor_id=actor_id,
            )
            group_scores = [float(detail["reward"]) for detail in group_reward_details]
            reward_metrics = {
                "time/reward": timer.duration,
                "objective/rlcer_evolving_reward": float(np.mean(group_scores)) if group_scores else 0.0,
                "objective/rlcer_evolving_reward_std": float(np.std(group_scores)) if group_scores else 0.0,
                "objective/rlcer_evolving_k_valid": float(np.mean([detail["k_valid"] for detail in group_reward_details])) if group_reward_details else 0.0,
                "objective/rlcer_evolving_k_total": float(np.mean([detail["k_total"] for detail in group_reward_details])) if group_reward_details else 0.0,
                "objective/rlcer_evolving_validity_fraction": float(
                    np.mean([detail["validity_fraction"] for detail in group_reward_details])
                ) if group_reward_details else 0.0,
                "objective/rlcer_evolving_skipped_no_rollouts": int(
                    sum(int(detail.get("skipped_no_rollouts", 0)) for detail in group_reward_details)
                ),
            }
            enriched_results.append(
                EnrichedGenerationResult(
                    result=grouped_result,
                    scores=group_scores,
                    reward_metrics=reward_metrics,
                    k_queries=[list(item.get("prompt_token_ids") or []) for item in cached_group],
                    k_ground_truths=[
                        {
                            "question": str(item.get("question", "")),
                            "ground_truth_answer": str(item.get("ground_truth_answer", "")),
                            "verifier_type": str(item.get("verifier_type", "math")),
                        }
                        for item in cached_group
                    ],
                    k_datasets=["rubric_training"] * len(cached_group),
                    k_raw_queries=[str(item.get("prompt_text", "")) for item in cached_group],
                    decoded_responses=[str(item.get("rubric_text", "")) for item in cached_group],
                )
            )

        return enriched_results

    async def enqueue_rlcer_evolving_cached_generations(self, training_step: int) -> int:
        """Enqueue cached policy-step rubric generations for the next rubric update."""
        if not self._is_rlcer_evolving_mode():
            return 0

        cached_generations = list(self._rlcer_evolving_cached_generations)
        if not cached_generations:
            return 0

        epoch_number = self.iter_dataloader.epoch_number if self.iter_dataloader is not None else 0
        actor_id = self._get_actor_id()
        enriched_results = await self._build_rlcer_evolving_enriched_results(
            cached_generations,
            training_step=training_step,
            epoch_number=epoch_number,
            actor_id=actor_id,
        )
        expected_prompt_groups = max(int(getattr(self.args, "num_unique_prompts_rollout", 1)), 1)
        if len(enriched_results) < expected_prompt_groups:
            LOGGER.warning(
                "%s: Incomplete cached RL-CER evolving batch for rubric step %d: got %d/%d prompt groups; "
                "skipping enqueue to avoid partial-batch hangs",
                self.__class__.__name__,
                training_step,
                len(enriched_results),
                expected_prompt_groups,
            )
            return 0
        for enriched_result in enriched_results:
            self.enriched_results_Q.put(enriched_result)

        self._rlcer_evolving_cached_generations = []

        LOGGER.info(
            "%s: Enqueued %d cached RL-CER evolving enriched result(s) from the latest policy step "
            "(enriched_results_Q id=%s qsize=%s)",
            self.__class__.__name__,
            len(enriched_results),
            id(self.enriched_results_Q),
            self.enriched_results_Q.qsize() if hasattr(self.enriched_results_Q, "qsize") else "N/A",
        )
        return len(enriched_results)


    def add_prompt_to_generator(
        self,
        example: dict[str, Any],
        example_index: int,
        epoch_number: int,
        training_step: int,
        pending_queries_map: PendingQueriesMap,
        param_prompt_Q: ray_queue.Queue,
        generation_config: vllm.SamplingParams,
        is_eval: bool,
    ) -> None:
        # RLCER "with evolving" mode: training only consumes cached policy-step
        # rubric generations, so there is no prompt to enqueue here.
        if self._is_rlcer_evolving_mode():
            return

        # Rubric-ARM mode: generate rubric from prompt only, evaluate via
        # pairwise judging on dataset preference pairs.
        if self._is_rubric_arm_mode():
            self._add_prompt_rubric_arm(
                example, example_index, epoch_number, training_step,
                pending_queries_map, param_prompt_Q, generation_config, is_eval,
            )
            return

        if not self._data_provider:
            raise RuntimeError("Data provider not initialized. Call set_policy_actor first.")
        
        LOGGER.debug(f"[RubricTrainerActor] add_prompt_to_generator: example_index={example_index}, epoch={epoch_number}, training_step={training_step}, is_eval={is_eval}")
        
        # Get pair context immediately (e.g., buffer question for replay buffer provider)
        pair_context = ray.get(self._data_provider.get_pair_context.remote(example))
        
        # Start answer pair creation asynchronously - returns ObjectRef immediately
        answer_pair_future = self._data_provider.create_answer_pair.remote(example, pair_context)
        
        # Use context question if provided, otherwise use example question
        question_key = getattr(self.args, "question_key", "question")
        
        # Handle different pair_context structures:
        # - None: use example question
        # - Dict with "provider_context": nested structure from CombinedMethodDataProvider
        # - Dict with "question": direct structure from ReplayBufferDataProvider
        if not pair_context:
            question = example[question_key]
        elif "provider_context" in pair_context:
            # CombinedMethodDataProvider wraps context in provider_context
            provider_context = pair_context["provider_context"]
            question = provider_context["question"] if provider_context else example[question_key]
        else:
            # Direct context from single provider (e.g., ReplayBufferDataProvider)
            question = pair_context.get("question", example[question_key])

        # Use rubric chat templates for message formatting
        prompt_token_ids, messages = format_messages(
            "rubric_generation",
            {"question": question},
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            return_messages=True,
        )

        # Create prompt_text for logging (format as readable string)
        system_prompt = messages[0]["content"]
        prompt_text = f"{system_prompt}\n\nQuestion:\n{question}"

        # Store ground truth with answer pair future and overridden question
        ground_truth = {
            "question": question,
            "answer_pair_future": answer_pair_future,
        }

        pending_queries_map.insert(
            example_index,
            prompt_token_ids,
            ground_truth,
            "rubric_training",
            prompt_text
        )

        LOGGER.debug(f"[RubricTrainerActor] Putting prompt into param_prompt_Q: example_index={example_index}, prompt_length={len(prompt_token_ids)}, queue_size={param_prompt_Q.qsize() if hasattr(param_prompt_Q, 'qsize') else 'N/A'}")
        param_prompt_Q.put(
            PromptRequest(
                prompt=prompt_token_ids,
                generation_config=generation_config,
                epoch_number=epoch_number,
                training_step=training_step,
                dataset_index=example_index,
                is_eval=is_eval,
                actor_id="rubric",  # For routing in single_model_mode
            )
        )
        LOGGER.debug(f"[RubricTrainerActor] add_prompt_to_generator completed: example_index={example_index}, prompt_length={len(prompt_token_ids)}")

    def _get_rubric_arm_reference_map(self) -> dict[str, tuple[str, str]]:
        """Build a question -> (preferred, dispreferred) cache from local JSONL dataset.

        The dataset transformation pops accepted/rejected keys, so we re-read
        the original JSONL file to recover them.
        """
        cached = getattr(self, "_rubric_arm_reference_map", None)
        if cached is not None:
            return cached

        import json

        reference_map: dict[str, tuple[str, str]] = {}
        dataset_mixer_list = getattr(self.args, "dataset_mixer_list", None) or []
        question_key = getattr(self.args, "question_key", "question")
        accepted_key = getattr(self.args, "accepted_answer_key", "")
        rejected_key = getattr(self.args, "rejected_answer_key", "")

        for i in range(0, len(dataset_mixer_list), 2):
            path = dataset_mixer_list[i]
            if not isinstance(path, str) or not path.endswith(".jsonl"):
                continue
            try:
                with open(path) as f:
                    for line in f:
                        row = json.loads(line)
                        q = row.get(question_key, "")
                        pref = row.get(accepted_key, "")
                        dispref = row.get(rejected_key, "")
                        while isinstance(pref, list):
                            pref = pref[0] if pref else ""
                        while isinstance(dispref, list):
                            dispref = dispref[0] if dispref else ""
                        pref = str(pref or "")
                        dispref = str(dispref or "")
                        if q and pref and dispref:
                            reference_map[q] = (pref, dispref)
            except Exception as e:
                LOGGER.warning("Failed to load Rubric-ARM reference data from %s: %s", path, e)

        LOGGER.info("Rubric-ARM reference map: loaded %d entries", len(reference_map))
        self._rubric_arm_reference_map = reference_map
        return reference_map

    def _add_prompt_rubric_arm(
        self,
        example: dict[str, Any],
        example_index: int,
        epoch_number: int,
        training_step: int,
        pending_queries_map: PendingQueriesMap,
        param_prompt_Q: ray_queue.Queue,
        generation_config: vllm.SamplingParams,
        is_eval: bool,
    ) -> None:
        """Rubric-ARM: generate rubric from prompt only, no data provider needed.

        Ground truth includes preference pair for computing R_r = I[judge correct].
        Uses the Rubric-ARM rubric generation template ([Hard Rule] / [Principle]).
        """
        question_key = getattr(self.args, "question_key", "question")
        question = example[question_key]

        prompt_token_ids, messages = format_messages(
            "rubric_arm_rubric_generation",
            {"question": question},
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            return_messages=True,
        )
        prompt_text = messages[0]["content"] if messages else question

        accepted_key = getattr(self.args, "accepted_answer_key", "accepted")
        rejected_key = getattr(self.args, "rejected_answer_key", "rejected")

        pref = example.get(accepted_key, example.get("accepted_answer", ""))
        dispref = example.get(rejected_key, example.get("rejected_answer", ""))

        if not pref or not dispref:
            ref_map = self._get_rubric_arm_reference_map()
            if question in ref_map:
                cached_pref, cached_dispref = ref_map[question]
                pref = pref or cached_pref
                dispref = dispref or cached_dispref

        ground_truth: dict[str, Any] = {
            "question": question,
            "preferred_answer": pref,
            "dispreferred_answer": dispref,
        }

        pending_queries_map.insert(
            example_index,
            prompt_token_ids,
            ground_truth,
            "rubric_training",
            prompt_text,
        )
        param_prompt_Q.put(
            PromptRequest(
                prompt=prompt_token_ids,
                generation_config=generation_config,
                epoch_number=epoch_number,
                training_step=training_step,
                dataset_index=example_index,
                is_eval=is_eval,
                actor_id="rubric",
            )
        )

    async def _compute_rewards_and_prompts(
        self,
        decoded_responses: list[str],  # Generated rubrics
        ground_truths: list[Any],  # Dict with question
        queries: list[str] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        # RLCER "with evolving": compute K_valid/K rubricator reward
        if self._is_rlcer_evolving_mode():
            return await self._compute_rlcer_evolving_rewards(decoded_responses, ground_truths)

        # Rubric-ARM: rubricator reward = I[judge predicts correct preference]
        if self._is_rubric_arm_mode():
            return await self._compute_rubric_arm_rubricator_rewards(decoded_responses, ground_truths)

        # Update episode counter (batch_size samples processed per call)
        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            f"[RubricTrainerActor] Computing rewards for {batch_size} rubrics (episode {self._current_episode})"
        )
        rewards = []
        metrics = {}

        if not self.policy_actor:
            raise RuntimeError("Policy actor not set. Call set_policy_actor first.")
        
        if not self._data_provider:
            raise RuntimeError("Data provider not initialized. Call set_policy_actor first.")

        # Resolve all answer pair futures in parallel
        with Timer(
            "[Data Preparation Thread] Computing rewards and prompts -- 🔄 Resolving answer pair futures", noop=True
        ) as resolve_timer:
            # Gather all answer pair futures (Ray ObjectRefs from data provider actor)
            answer_pair_futures = [gt["answer_pair_future"] for gt in ground_truths]
            
            # Await all futures - they resolve to AnswerPair objects
            answer_pairs = await asyncio.gather(*answer_pair_futures)
            
            # Extract answers and extra fields from pairs
            accepted_answers = [pair.accepted_answer for pair in answer_pairs]
            rejected_answers = [pair.rejected_answer for pair in answer_pairs]
            extra_fields_list = [pair.get_extra_fields() for pair in answer_pairs]
        
        metrics["time/resolve_answer_pair_futures"] = resolve_timer.duration
        LOGGER.debug(
            f"[RubricTrainerActor] Resolved {len(accepted_answers)} answer pair futures in {resolve_timer.duration:.3f}s"
        )

        accepted_scores = []
        rejected_scores = []

        # Step 3: Compute reward using compute_rubric_judge_reward_async (single judge)
        # or compute_rubric_judge_reward_multi_judge_async (multiple judges)
        # accepted_answer = current policy output on original question
        # rejected_answer = either from replay buffer (past rollout) or generated via inferred question

        # Determine if using multi-judge mode
        use_multi_judge = self._multi_judge_engines is not None
        max_concurrent_judge = int(os.environ.get("MAX_CONCURRENT_JUDGE_REQUESTS", "128"))
        max_concurrent_multi_judge = int(
            os.environ.get("MAX_CONCURRENT_MULTI_JUDGE_REQUESTS", str(max_concurrent_judge))
        )

        if use_multi_judge:
            from open_instruct.search_rewards.rubric_judge_rewards import compute_rubric_judge_reward_multi_judge_async
            # Get sampling params template for judges
            sampling_params_template = self._build_judge_sampling_params()
            # Get list of judge actors
            judge_actors = self._multi_judge_engines.get_judge_actors_with_sampling_params(sampling_params_template)
            LOGGER.info(
                f"[RubricTrainerActor] Using multi-judge mode with {len(judge_actors)} judges "
                f"(mode={self._multi_judge_engines.aggregation_mode}, "
                f"tie_breaker={self._multi_judge_engines.tie_breaker}, "
                f"alpha={self._multi_judge_engines.alpha:.2f}, beta={self._multi_judge_engines.beta:.2f}, "
                f"max_concurrent_multi_judge_requests={max_concurrent_multi_judge}, "
                f"max_concurrent_judge_requests={max_concurrent_judge})"
            )
        else:
            from open_instruct.search_rewards.rubric_judge_rewards import compute_rubric_judge_reward_async
            # Create sampling params for judge if using rubric_judge_generate_text_actor
            # Override n=1 for judge since we only need one judgment per answer
            sampling_params = self._build_judge_sampling_params() if self.rubric_judge_generate_text_actor else None
            LOGGER.debug(f"[RubricTrainerActor] Using single-judge mode")

        with Timer(
            "[Data Preparation Thread] Computing rewards and prompts -- 🏆 Computing rubric judge rewards", noop=True
        ) as reward_timer:
            # Bound the number of in-flight reward computations so multi-judge
            # scoring cannot fan out unbounded coroutines before the underlying
            # per-actor semaphore gets a chance to apply backpressure.
            judge_semaphore = asyncio.Semaphore(max_concurrent_multi_judge if use_multi_judge else max_concurrent_judge)

            async def _bounded_reward(coro):
                async with judge_semaphore:
                    return await coro

            # Create all reward computation tasks upfront for parallel execution
            reward_tasks = []
            for i, (rubric_text, gt, accepted_answer, rejected_answer) in enumerate(
                zip(decoded_responses, ground_truths, accepted_answers, rejected_answers)
            ):
                question = gt["question"]

                # Use data provider to format log message with extra fields from future resolution
                log_message = ray.get(self._data_provider.format_log_message.remote(
                    index=i + 1,
                    total=len(decoded_responses),
                    question=question,
                    rubric_length=len(rubric_text),
                    accepted_answer_length=len(accepted_answer),
                    rejected_answer_length=len(rejected_answer),
                    extra_fields=extra_fields_list[i],
                ))
                LOGGER.debug(log_message)

                # Create coroutine (don't await yet)
                if use_multi_judge:
                    # Multi-judge: Pass judge_actors list
                    reward_tasks.append(
                        _bounded_reward(
                            compute_rubric_judge_reward_multi_judge_async(
                                question=question,
                                accepted_answer=accepted_answer,
                                rejected_answer=rejected_answer,
                                generated_rubric=rubric_text,
                                judge_actors=judge_actors,
                                aggregation_mode=self._multi_judge_engines.aggregation_mode,
                                alpha=self._multi_judge_engines.alpha,
                                beta=self._multi_judge_engines.beta,
                                tie_breaker=self._multi_judge_engines.tie_breaker,
                                margin_weight=self._multi_judge_engines.margin_weight,
                                format_weight=self._multi_judge_engines.format_weight,
                                kappa_weight=self._multi_judge_engines.kappa_weight,
                            )
                        )
                    )
                else:
                    # Single judge: Use existing rubric_judge_generate_text_actor
                    reward_tasks.append(
                        _bounded_reward(
                            compute_rubric_judge_reward_async(
                                question=question,
                                accepted_answer=accepted_answer,
                                rejected_answer=rejected_answer,
                                generated_rubric=rubric_text,
                                rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                                sampling_params=sampling_params,
                                margin_reward=self._rubric_reward_use_margin,
                            )
                        )
                    )

            # Execute all reward tasks in parallel
            reward_dicts = await asyncio.gather(*reward_tasks)

            # Compute format rewards if weight > 0
            format_reward_weight = self._rubric_format_reward_weight
            format_rewards: list[float] = []
            if format_reward_weight > 0:
                from open_instruct.search_rewards.rubric_judge_rewards import is_valid_rubric_format
                format_rewards = [is_valid_rubric_format(rubric) for rubric in decoded_responses]
                valid_count = sum(1 for r in format_rewards if r > 0)
                LOGGER.info(
                    f"[RubricTrainerActor] Format reward: {valid_count}/{len(format_rewards)} rubrics "
                    f"have valid JSON format (weight={format_reward_weight:.2f})"
                )

            # Process results in order and collect examples for logging
            generation_examples = []
            for i, reward_dict in enumerate(reward_dicts):
                base_reward = reward_dict["reward"]
                if format_reward_weight > 0 and not math.isnan(base_reward):
                    final_reward = (1.0 - format_reward_weight) * base_reward + format_reward_weight * format_rewards[i]
                else:
                    final_reward = base_reward
                rewards.append(final_reward)
                # Multi-judge returns "mean_accepted_score"/"mean_rejected_score",
                # single-judge returns "accepted_score"/"rejected_score"
                acc_score = reward_dict.get("accepted_score", reward_dict.get("mean_accepted_score", 0.0))
                rej_score = reward_dict.get("rejected_score", reward_dict.get("mean_rejected_score", 0.0))
                accepted_scores.append(acc_score)
                rejected_scores.append(rej_score)

                LOGGER.debug(
                    f"[RubricTrainerActor] Reward {i+1}: base_reward={base_reward:.4f}, "
                    f"format_reward={format_rewards[i] if format_rewards else 'N/A'}, "
                    f"final_reward={final_reward:.4f}, "
                    f"accepted_score={acc_score:.4f}, "
                    f"rejected_score={rej_score:.4f}"
                )

                # Collect examples for logging (up to num_examples_to_log)
                if i < self._num_examples_to_log:
                    # Get provider-specific fields from future resolution (e.g., inferred_question)
                    # Only log about inferred_question when InferredQuestionDataProvider is actually used
                    if "inferred_question" in extra_fields_list[i]:
                        inferred_question = extra_fields_list[i]["inferred_question"]
                        if inferred_question:
                            LOGGER.info(
                                "[RubricTrainerActor] Inferred question for example %d: original='%s...', inferred='%s...'",
                                i,
                                ground_truths[i]["question"][:100] if len(ground_truths[i]["question"]) > 100 else ground_truths[i]["question"],
                                inferred_question[:100] if len(inferred_question) > 100 else inferred_question,
                            )
                        else:
                            # Only warn when InferredQuestionDataProvider was selected but inference failed
                            LOGGER.warning(
                                "[RubricTrainerActor] Inferred question is EMPTY for example %d: original='%s...'",
                                i,
                                ground_truths[i]["question"][:100] if len(ground_truths[i]["question"]) > 100 else ground_truths[i]["question"],
                            )
                    generation_examples.append(GenerationExample(
                        training_step=self._current_training_step,
                        actor_type="rubric",
                        example_index=i,
                        question=ground_truths[i]["question"],
                        rubric=decoded_responses[i],
                        policy_answer=accepted_answers[i],
                        score=acc_score,
                        accepted_reasoning=reward_dict.get("accepted_reasoning", ""),
                        rejected_answer=rejected_answers[i],
                        rejected_reasoning=reward_dict.get("rejected_reasoning", ""),
                        accepted_score=acc_score,
                        rejected_score=rej_score,
                        reward=final_reward,
                        **extra_fields_list[i],  # Unpack provider-specific fields from future
                    ))

        metrics["time/compute_rewards"] = reward_timer.duration

        # Log generation examples at configured interval (every N training steps)
        # Debug logging to diagnose why examples aren't being written
        LOGGER.info(
            f"[RubricTrainerActor] Generation examples check: "
            f"dir={bool(self._generation_examples_dir)}, "
            f"interval={self._log_examples_every_n_steps}, "
            f"step={self._current_training_step}, "
            f"modulo={self._current_training_step % self._log_examples_every_n_steps if self._log_examples_every_n_steps > 0 else 'N/A'}, "
            f"examples_count={len(generation_examples)}"
        )
        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            LOGGER.info(
                f"[RubricTrainerActor] Writing {len(generation_examples)} generation examples for step {self._current_training_step}"
            )
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "rubric",
                self._current_training_step,
            )

        metrics["objective/differential_reward"] = np.nanmean(rewards)
        metrics["objective/accepted_score"] = np.nanmean(accepted_scores)
        metrics["objective/rejected_score"] = np.nanmean(rejected_scores)
        if format_rewards:
            metrics["objective/rubric_format_reward"] = np.mean(format_rewards)
            metrics["objective/rubric_format_valid_pct"] = np.mean([1.0 if r > 0 else 0.0 for r in format_rewards]) * 100

        # Add multi-judge specific metrics if using multi-judge mode
        if use_multi_judge:
            # Extract pairwise_accuracy and agreement from reward_dicts
            pairwise_accuracies = [rd.get("pairwise_accuracy", 0.0) for rd in reward_dicts]
            agreements = [rd.get("agreement", 0.0) for rd in reward_dicts]
            fleiss_kappas = [rd.get("fleiss_kappa", 0.0) for rd in reward_dicts]
            winner_is_accepted = [1.0 if rd.get("winner") == "accepted" else 0.0 for rd in reward_dicts]
            vote_ties = [1.0 if rd.get("is_vote_tie") else 0.0 for rd in reward_dicts]
            metrics["objective/multi_judge_pairwise_accuracy"] = np.mean(pairwise_accuracies)
            metrics["objective/multi_judge_agreement"] = np.mean(agreements)
            metrics["objective/multi_judge_fleiss_kappa"] = np.mean(fleiss_kappas)
            metrics["objective/multi_judge_pairwise_accuracy_std"] = np.std(pairwise_accuracies)
            metrics["objective/multi_judge_agreement_std"] = np.std(agreements)
            metrics["objective/multi_judge_fleiss_kappa_std"] = np.std(fleiss_kappas)
            metrics["objective/multi_judge_winner_is_accepted"] = np.mean(winner_is_accepted)
            metrics["objective/multi_judge_vote_tie_rate"] = np.mean(vote_ties)
            avg_margins = [rd.get("avg_margin", 0.0) for rd in reward_dicts]
            mkf_rewards = [rd.get("margin_kappa_format_reward", 0.0) for rd in reward_dicts]
            rubric_format_scores = [rd.get("rubric_format_score", 0.0) for rd in reward_dicts]
            metrics["objective/multi_judge_avg_margin"] = np.mean(avg_margins)
            metrics["objective/multi_judge_margin_kappa_format_reward"] = np.mean(mkf_rewards)
            metrics["objective/multi_judge_rubric_format_score"] = np.mean(rubric_format_scores)
            LOGGER.info(
                f"[RubricTrainerActor] Multi-judge metrics: "
                f"pairwise_accuracy={np.mean(pairwise_accuracies):.4f}, "
                f"agreement={np.mean(agreements):.4f}, "
                f"fleiss_kappa={np.mean(fleiss_kappas):.4f}, "
                f"winner_is_accepted={np.mean(winner_is_accepted):.4f}, "
                f"vote_tie_rate={np.mean(vote_ties):.4f}, "
                f"avg_margin={np.mean(avg_margins):.4f}, "
                f"rubric_format_score={np.mean(rubric_format_scores):.4f}"
            )

        LOGGER.info(
            f"[RubricTrainerActor] Computed rewards: mean={np.nanmean(rewards):.4f}, "
            f"std={np.nanstd(rewards):.4f}, mean_accepted={np.nanmean(accepted_scores):.4f}, "
            f"mean_rejected={np.nanmean(rejected_scores):.4f}, total_time={reward_timer.duration:.3f}s"
        )

        return rewards, metrics

    async def _compute_rlcer_evolving_rewards(
        self,
        decoded_responses: list[str],  # Generated rubric texts
        ground_truths: list[Any],      # Dicts with question + ground_truth_answer + verifier_type
    ) -> tuple[list[float], dict[str, Any]]:
        """Compute RLCER 'with evolving' rubricator reward: K_valid/K + r_format.

        For each generated rubric:
          1. Parse rubric items from the generated text.
          2. Retrieve stored policy rollout data for the question (answers + correctness).
          3. Score all rollouts against rubric items (binary satisfaction matrix).
          4. Filter valid rubrics by correlation with correctness.
          5. Reward = K_valid / K + r_format.
        """
        from open_instruct.search_rewards.rubric_judge_rewards import (
            compute_rlcer_rubricator_reward,
            parse_rlcer_rubric_items_with_scores,
        )

        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            f"[RubricTrainerActor/RLCER-evolving] Computing rewards for {batch_size} rubrics "
            f"(episode {self._current_episode})"
        )
        rewards: list[float] = []
        metrics: dict[str, Any] = {}

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )

        correlation_threshold = float(os.environ.get("RLCER_CORRELATION_THRESHOLD", "0.2"))

        with Timer(
            "[Data Preparation Thread] Computing rewards -- 🏆 RLCER rubricator K_valid/K",
            noop=True,
        ) as reward_timer:
            reward_tasks = []
            skipped_indices: list[int] = []

            for i, (rubric_text, gt) in enumerate(zip(decoded_responses, ground_truths)):
                question = gt["question"]
                rollout_data = self._rlcer_rollout_buffer.get(question)

                if not rollout_data or len(rollout_data["answers"]) <= 1:
                    # No stored rollout data for this question — give a baseline reward.
                    # This can happen at the start of training before the policy phase
                    # has populated the buffer.
                    skipped_indices.append(i)
                    continue

                reward_tasks.append(
                    (
                        i,
                        compute_rlcer_rubricator_reward(
                            question=question,
                            rubric_text=rubric_text,
                            rollout_answers=rollout_data["answers"],
                            correctness_vector=rollout_data["correctness"],
                            correlation_threshold=correlation_threshold,
                            rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                            sampling_params=sampling_params,
                        ),
                    )
                )

            # Resolve all async reward tasks
            if reward_tasks:
                task_results = await asyncio.gather(*[t for _, t in reward_tasks])
                task_map = {idx: result for (idx, _), result in zip(reward_tasks, task_results)}
            else:
                task_map = {}

            # Assemble rewards in order
            generation_examples: list[GenerationExample] = []
            all_k_valid: list[int] = []
            all_k_total: list[int] = []
            all_validity_fractions: list[float] = []

            for i in range(batch_size):
                if i in task_map:
                    result = task_map[i]
                    reward = float(result["reward"])
                    all_k_valid.append(result["k_valid"])
                    all_k_total.append(result["k_total"])
                    all_validity_fractions.append(result["validity_fraction"])
                else:
                    # Skipped — no rollout data available. Give r_format only.
                    parsed_items, _ = parse_rlcer_rubric_items_with_scores(decoded_responses[i])
                    reward = 1.0 if parsed_items else 0.0
                    all_k_valid.append(0)
                    all_k_total.append(len(parsed_items))
                    all_validity_fractions.append(0.0)

                rewards.append(reward)
                LOGGER.debug(
                    f"[RubricTrainerActor/RLCER-evolving] Reward {i+1}: {reward:.4f}"
                )

                if i < self._num_examples_to_log:
                    result_info = task_map.get(i, {})
                    generation_examples.append(
                        GenerationExample(
                            training_step=self._current_training_step,
                            actor_type="rubric",
                            example_index=i,
                            question=ground_truths[i]["question"],
                            rubric=decoded_responses[i],
                            policy_answer="",
                            score=reward,
                            accepted_score=reward,
                            reward=reward,
                            rrd_weighting_method="rlcer_evolving",
                            rrd_rubric_items=result_info.get("rubric_items", []),
                            rrd_weights=[],
                            rrd_binary_scores=[],
                            rrd_trace={
                                "k_valid": result_info.get("k_valid", 0),
                                "k_total": result_info.get("k_total", 0),
                                "validity_fraction": result_info.get("validity_fraction", 0.0),
                                "r_format": result_info.get("r_format", 0.0),
                                "valid_indices": result_info.get("valid_indices", []),
                            },
                        )
                    )

        metrics["time/compute_rewards"] = reward_timer.duration

        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "rubric",
                self._current_training_step,
            )

        metrics["objective/rlcer_evolving_reward"] = np.mean(rewards) if rewards else 0.0
        metrics["objective/rlcer_evolving_reward_std"] = np.std(rewards) if rewards else 0.0
        metrics["objective/rlcer_evolving_k_valid"] = np.mean(all_k_valid) if all_k_valid else 0.0
        metrics["objective/rlcer_evolving_k_total"] = np.mean(all_k_total) if all_k_total else 0.0
        metrics["objective/rlcer_evolving_validity_fraction"] = (
            np.mean(all_validity_fractions) if all_validity_fractions else 0.0
        )
        metrics["objective/rlcer_evolving_skipped_no_rollouts"] = len(skipped_indices)

        LOGGER.info(
            f"[RubricTrainerActor/RLCER-evolving] Computed rewards: mean={np.mean(rewards):.4f}, "
            f"std={np.std(rewards):.4f}, mean_validity={np.mean(all_validity_fractions):.3f}, "
            f"mean_k_valid={np.mean(all_k_valid):.1f}/{np.mean(all_k_total):.1f}, "
            f"skipped={len(skipped_indices)}/{batch_size}, time={reward_timer.duration:.3f}s"
        )

        return rewards, metrics

    async def _compute_rubric_arm_rubricator_rewards(
        self,
        decoded_responses: list[str],
        ground_truths: list[Any],
    ) -> tuple[list[float], dict[str, Any]]:
        """Compute Rubric-ARM rubricator reward: R_r = I[judge predicts correct pref] + r_format.

        Uses stored preference pairs from the policy data buffer. The (frozen)
        judge evaluates whether the preferred answer beats the dispreferred one
        when using the generated rubric.
        """
        from open_instruct.search_rewards.rubric_judge_rewards import (
            compute_rubric_arm_rubricator_reward,
        )

        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            f"[RubricTrainerActor/RubricARM] Computing rubricator rewards for {batch_size} rubrics "
            f"(episode {self._current_episode})"
        )
        rewards: list[float] = []
        metrics: dict[str, Any] = {}

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )

        with Timer(
            "[Data Preparation Thread] Computing rewards -- RubricARM rubricator",
            noop=True,
        ) as reward_timer:
            reward_tasks = []
            skipped_indices: list[int] = []

            for i, (rubric_text, gt) in enumerate(zip(decoded_responses, ground_truths)):
                question = gt["question"]
                preferred = gt.get("preferred_answer", "")
                dispreferred = gt.get("dispreferred_answer", "")

                if not preferred or not dispreferred:
                    skipped_indices.append(i)
                    continue

                reward_tasks.append(
                    (
                        i,
                        compute_rubric_arm_rubricator_reward(
                            question=question,
                            rubric_text=rubric_text,
                            preferred_answer=preferred,
                            dispreferred_answer=dispreferred,
                            rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                            sampling_params=sampling_params,
                        ),
                    )
                )

            if reward_tasks:
                task_results = await asyncio.gather(*[t for _, t in reward_tasks])
                task_map = {idx: result for (idx, _), result in zip(reward_tasks, task_results)}
            else:
                task_map = {}

            generation_examples: list[GenerationExample] = []
            all_judge_correct: list[bool] = []
            all_r_format: list[float] = []

            for i in range(batch_size):
                if i in task_map:
                    result = task_map[i]
                    reward = float(result["score"])
                    all_judge_correct.append(result.get("judge_correct", False))
                    all_r_format.append(result.get("r_format", 0.0))
                else:
                    reward = 0.0
                    all_judge_correct.append(False)
                    all_r_format.append(0.0)

                rewards.append(reward)
                LOGGER.debug(
                    f"[RubricTrainerActor/RubricARM] Reward {i+1}: {reward:.4f}"
                )

                if i < self._num_examples_to_log:
                    result_info = task_map.get(i, {})
                    generation_examples.append(
                        GenerationExample(
                            training_step=self._current_training_step,
                            actor_type="rubric",
                            example_index=i,
                            question=ground_truths[i]["question"],
                            rubric=decoded_responses[i],
                            policy_answer="",
                            score=reward,
                            accepted_score=reward,
                            reward=reward,
                            rrd_weighting_method="rubric_arm",
                            rrd_rubric_items=result_info.get("rubric_items", []),
                            rrd_weights=[],
                            rrd_binary_scores=[],
                            rrd_trace={
                                "judge_correct": result_info.get("judge_correct", False),
                                "r_format": result_info.get("r_format", 0.0),
                                "r_acc": result_info.get("r_acc", 0.0),
                                "judge_winner": result_info.get("judge_winner"),
                            },
                        )
                    )

        metrics["time/compute_rewards"] = reward_timer.duration

        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "rubric",
                self._current_training_step,
            )

        judge_correct_rate = np.mean(all_judge_correct) if all_judge_correct else 0.0
        metrics["objective/rubric_arm_rubricator_reward"] = np.mean(rewards) if rewards else 0.0
        metrics["objective/rubric_arm_rubricator_reward_std"] = np.std(rewards) if rewards else 0.0
        metrics["objective/rubric_arm_judge_correct_rate"] = judge_correct_rate
        metrics["objective/rubric_arm_r_format"] = np.mean(all_r_format) if all_r_format else 0.0
        metrics["objective/rubric_arm_skipped_no_pref"] = len(skipped_indices)

        LOGGER.info(
            f"[RubricTrainerActor/RubricARM] Computed rewards: mean={np.mean(rewards):.4f}, "
            f"judge_correct_rate={judge_correct_rate:.3f}, "
            f"skipped={len(skipped_indices)}/{batch_size}, time={reward_timer.duration:.3f}s"
        )

        return rewards, metrics


class PolicyTrainerActor(BaseTrainerActor):
    """Ray actor coordinating policy rollouts using the latest rubric."""

    def __init__(
        self,
        policy_model: str,
        *,
        baseline_model: str | None = None,
        policy_generation_kwargs: dict[str, Any] | None = None,
        baseline_generation_kwargs: dict[str, Any] | None = None,
        rubric_judge_generate_text_actor: Any | None = None,
        policy_generate_text_actor: Any | None = None,
        grpo_args: GrpoArgs | None = None,
        tokenizer_config: GrpoTokenizerConfig | None = None,
        model_config: GrpoModelConfig | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        single_model_mode: bool = False,
        multi_judge_engines: MultiJudgeEngines | None = None,
        generation_examples_dir: str | None = None,
        num_examples_to_log: int = 3,
        log_examples_every_n_steps: int = 1,
        reward_mode: str = "rubric_judge",
        rubric_prompt_key: str = "rubric_generation",
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.baseline_model = baseline_model or policy_model
        self.rubric_judge_generate_text_actor = rubric_judge_generate_text_actor
        self.policy_generate_text_actor = policy_generate_text_actor
        self.grpo_args = grpo_args  # Store for accessing rubric judge config
        self.policy_generation_kwargs = policy_generation_kwargs or {}
        self.baseline_generation_kwargs = baseline_generation_kwargs or {}
        self.rubric_prompt_key = rubric_prompt_key
        self.rubric_actor: ray.actor.ActorHandle | None = None
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None
        self.iter_dataloader: grpo_fast.ShufflingIterator | None = None
        self._single_model_mode = single_model_mode
        self._multi_judge_engines = multi_judge_engines
        self._generation_examples_dir = Path(generation_examples_dir) if generation_examples_dir else None
        self._num_examples_to_log = num_examples_to_log
        self._log_examples_every_n_steps = log_examples_every_n_steps
        self._current_episode = 0  # Track episode for wandb-consistent logging
        self._current_training_step = 0  # Track training step (gradient updates) for replay buffer age
        self._reward_mode = reward_mode
        self._rlcer_cache_deque: collections.deque[dict[str, Any]] = collections.deque()
        self._rlcer_cache_condition = threading.Condition()

        # Initialize GRPO session if config is provided
        if grpo_args and tokenizer_config and model_config:
            # Create unique placement group name for this policy actor to avoid collisions
            # in multi-policy co-evolution mode where multiple PolicyTrainerActors exist
            model_suffix = policy_model.split("/")[-1].lower().replace("-", "_")
            pg_name = f"PolicyTrainerActor_{model_suffix}_pg"

            if self._initialize_grpo_session(
                grpo_args, tokenizer_config, model_config, model_override=policy_model, log_prefix="Policy model: ",
                train_dataset=train_dataset, eval_dataset=eval_dataset,
                skip_engine_creation=single_model_mode,
                placement_group_name=pg_name,
            ):
                # Use dataset from initialization (already loaded by _initialize_grpo_session)
                # self.train_dataset is already set by _initialize_grpo_session

                # Create iter_dataloader similar to grpo_fast.py line 3318-3319
                train_dataset_idxs = np.arange(len(self.train_dataset))
                self.iter_dataloader = grpo_fast.ShufflingIterator(train_dataset_idxs, 1, seed=self.args.seed)

                LOGGER.info(
                    "Loaded dataset with %d examples and created iter_dataloader",
                    len(self.train_dataset),
                )

        LOGGER.info(
            "PolicyTrainerActor initialised with policy %s (baseline=%s, dataset=%s, single_model_mode=%s)",
            policy_model,
            self.baseline_model,
            "loaded" if self.train_dataset is not None else "none",
            single_model_mode,
        )
        if self._multi_judge_engines:
            LOGGER.info(
                "PolicyTrainerActor using multi-judge scoring with %d judges: %s",
                len(self._multi_judge_engines.judge_models),
                self._multi_judge_engines.judge_models,
            )

    # ---- internal helpers -------------------------------------------------

    async def _generate_policy_answer(self, question: str, rubric_text: str) -> str:
        # Use rubric chat templates for message formatting
        # Tokenize if tokenizer is available, otherwise use messages format
        prompt_token_ids = format_messages(
            "policy",
            {"question": question},
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )
        kwargs = dict(self.policy_generation_kwargs)
        sampling_params = self._build_sampling_params(**kwargs)
        # Use policy_generate_text_actor for load-balanced generation
        if not self.policy_generate_text_actor:
            raise RuntimeError("policy_generate_text_actor is not initialized")
        sp_dict = msgspec.structs.asdict(sampling_params)
        result = await self.policy_generate_text_actor.generate_text_from_token_ids.remote(prompt_token_ids, sp_dict)
        return result

    async def _generate_rubric(self, question: str) -> str:
        """Generate a rubric for the given question using the policy model.
        
        Args:
            question: The question to generate a rubric for
            
        Returns:
            The generated rubric text
        """
        prompt_token_ids = format_messages(
            self.rubric_prompt_key,
            {"question": question},
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )
        kwargs = dict(self.policy_generation_kwargs)
        sampling_params = self._build_sampling_params(**kwargs)
        # Use policy_generate_text_actor for load-balanced generation
        if not self.policy_generate_text_actor:
            raise RuntimeError("policy_generate_text_actor is not initialized")
        sp_dict = msgspec.structs.asdict(sampling_params)
        result = await self.policy_generate_text_actor.generate_text_from_token_ids.remote(prompt_token_ids, sp_dict)
        return result

    async def _generate_policy_answer_with_rubric(self, question: str, rubric_text: str) -> str:
        """Generate a policy answer conditioned on both the question and rubric.
        
        Args:
            question: The question to answer
            rubric_text: The rubric to condition the answer on
            
        Returns:
            The generated answer text
        """
        prompt_token_ids = format_messages(
            "policy_with_rubric",
            {"question": question, "rubric": rubric_text},
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )
        kwargs = dict(self.policy_generation_kwargs)
        sampling_params = self._build_sampling_params(**kwargs)
        # Use policy_generate_text_actor for load-balanced generation
        if not self.policy_generate_text_actor:
            raise RuntimeError("policy_generate_text_actor is not initialized")
        sp_dict = msgspec.structs.asdict(sampling_params)
        result = await self.policy_generate_text_actor.generate_text_from_token_ids.remote(prompt_token_ids, sp_dict)
        return result

    def _generate_baseline_answer(self, question: str) -> str:
        messages = format_messages(
            "baseline",
            {"question": question},
            tokenize=False,
        )
        return asyncio.run(
            run_litellm_async(
                model_name=self.baseline_model,
                messages=messages,
                **self.baseline_generation_kwargs,
            )
        )

    async def _infer_question_from_answer(self, answer: str) -> str:
        """Infer what question the model thinks an answer is responding to.
        
        IMPORTANT: Strips thinking tokens (<think>...</think>) before inference.
        The inference model should only see the final answer, not internal reasoning.
        
        Args:
            answer: The policy model's answer (may contain thinking tokens)
            
        Returns:
            The inferred question string
        """
        # CRITICAL: Strip thinking tokens before question inference!
        # The inference model should only see the final answer, not the thinking process.
        from open_instruct.ground_truth_utils import remove_thinking_section
        answer_for_inference = remove_thinking_section(answer)
        
        prompt_token_ids = format_messages(
            "question_inference",
            {"answer": answer_for_inference},  # Use cleaned answer
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
        )
        # Use lower temperature for more deterministic question inference
        kwargs = dict(self.policy_generation_kwargs)
        kwargs["temperature"] = 0.3
        sampling_params = self._build_sampling_params(**kwargs)
        # Use policy_generate_text_actor for load-balanced generation
        if not self.policy_generate_text_actor:
            raise RuntimeError("policy_generate_text_actor is not initialized")
        sp_dict = msgspec.structs.asdict(sampling_params)
        result = await self.policy_generate_text_actor.generate_text_from_token_ids.remote(prompt_token_ids, sp_dict)
        return result.strip()

    async def _generate_rejected_with_inferred_question(self, original_question: str, accepted_answer: str) -> tuple[str, str]:
        """Generate a rejected answer by first inferring the question, then generating an answer to it.
        
        This creates a rejected sample by:
        1. Asking the model what question the accepted_answer was trying to answer
        2. Generating a new policy answer to that inferred question
        
        The intuition is that if the rubric is bad, the policy might be answering
        a different question than intended, and we want to capture that as a "rejected" sample.
        
        Args:
            original_question: The original question (for reference/logging)
            accepted_answer: The current policy's answer to the original question
            
        Returns:
            Tuple of (inferred_question, rejected_answer)
        """
        # Step 1: Infer what question the model thinks it was answering
        inferred_question = await self._infer_question_from_answer(accepted_answer)
        LOGGER.debug(
            f"[PolicyTrainerActor] Inferred question from answer: original='{original_question[:50]}...', "
            f"inferred='{inferred_question[:50]}...'"
        )
        
        # Step 2: Generate a policy answer to the inferred question
        rejected_answer = await self._generate_policy_answer(inferred_question, "")
        
        return inferred_question, rejected_answer

    def set_rubric_actor(self, rubric_actor: ray.actor.ActorHandle) -> None:
        """Set reference to rubric actor for querying rubrics."""
        self.rubric_actor = rubric_actor

    def _get_actor_id(self) -> str:
        """Return the actor ID for routing in single_model_mode."""
        return "policy"

    def start_training_threads(
        self,
        train_dataset: Dataset,
        iter_dataloader: ShufflingIterator | None = None,
        steps_per_phase: int | None = None,
    ) -> None:
        dynamic_dataset = PolicyDynamicDataset(train_dataset)
        super().start_training_threads(dynamic_dataset, iter_dataloader, steps_per_phase=steps_per_phase)

    def add_prompt_to_generator(
        self,
        example: dict[str, Any],
        example_index: int,
        epoch_number: int,
        training_step: int,
        pending_queries_map: PendingQueriesMap,
        param_prompt_Q: ray_queue.Queue,
        generation_config: vllm.SamplingParams,
        is_eval: bool,
    ) -> None:
        LOGGER.debug(f"[PolicyTrainerActor] add_prompt_to_generator: example_index={example_index}, epoch={epoch_number}, training_step={training_step}, is_eval={is_eval}")
        question_key = getattr(self.args, "question_key", "question")
        question = example[question_key]

        # Generate Rubric in parallel by storing the future
        # We assume RubricTrainerActor has create_rubric method which returns a future or we call it remotely
        rubric_future = self.rubric_actor.create_rubric.remote(question)

        # Policy prompt no longer depends on rubric text immediately
        # Use rubric chat templates for message templates
        prompt_token_ids, messages = format_messages(
            "policy",
            {"question": question},
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            return_messages=True,
        )

        # Create prompt_text for logging (format as readable string)
        system_prompt = messages[0]["content"]
        prompt_text = f"{system_prompt}\n\nQuestion:\n{question}\n\nProvide your best possible answer."

        ground_truth = {
            "question": question,
            "rubric_future": rubric_future  # Store future to resolve later
        }

        pending_queries_map.insert(
            example_index,
            prompt_token_ids,
            ground_truth,
            "policy_training",
            prompt_text
        )

        LOGGER.debug(f"[PolicyTrainerActor] Putting prompt into param_prompt_Q: example_index={example_index}, prompt_length={len(prompt_token_ids)}, queue_size={param_prompt_Q.qsize() if hasattr(param_prompt_Q, 'qsize') else 'N/A'}")
        param_prompt_Q.put(
            PromptRequest(
                prompt=prompt_token_ids,
                generation_config=generation_config,
                epoch_number=epoch_number,
                training_step=training_step,
                dataset_index=example_index,
                is_eval=is_eval,
                actor_id="policy",  # For routing in single_model_mode
            )
        )
        LOGGER.debug(f"[PolicyTrainerActor] add_prompt_to_generator completed: example_index={example_index}, prompt_length={len(prompt_token_ids)}")

    async def _compute_rewards_and_prompts(
        self,
        decoded_responses: list[str],  # Policy answers
        ground_truths: list[Any],  # Dict with question, rubric
        queries: list[str] | None = None,  # Full prompts (with rubric embedded)
    ) -> tuple[list[float], dict[str, Any]]:
        """Compute rewards for policy answers using rubrics from rubric actor.

        For each policy answer, we:
        1. Extract question and rubric from ground_truths
        2. Score the answer using the rubric via judge_answer_with_rubric
        3. Return the score as reward
        """
        # Update episode counter (batch_size samples processed per call)
        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            f"[PolicyTrainerActor] Computing rewards for {batch_size} policy answers (episode {self._current_episode})"
        )
        rewards = []
        metrics = {}

        # Resolve all rubric futures in parallel
        with Timer(
            "[Data Preparation Thread] Computing rewards and prompts -- 🔄 Resolving rubric futures", noop=True
        ) as resolve_timer:
            rubric_futures = [gt["rubric_future"] for gt in ground_truths]
            rubric_specs = await asyncio.gather(*rubric_futures)
        metrics["time/resolve_futures"] = resolve_timer.duration
        LOGGER.debug(
            f"[PolicyTrainerActor] Resolved {len(rubric_specs)} rubrics in {resolve_timer.duration:.3f}s"
        )

        # Push rollout to rubric actor's replay buffer with the training step from when this prompt was submitted
        # The _training_step is injected into ground_truths by reward_computation_thread in grpo_fast.py
        # This ensures accurate age tracking even with async processing (ASYNC_STEPS > 1)
        request_training_step = ground_truths[0].get("_training_step", self._current_training_step) or self._current_training_step
        self.rubric_actor.add_experience.remote(ground_truths[0]["question"], decoded_responses[0], request_training_step)

        from open_instruct.search_rewards.rubric_judge_rewards import (
            judge_answer_with_multiple_judges,
            judge_answer_with_rubric,
        )

        use_multi_judge = self._multi_judge_engines is not None
        max_concurrent_judge = int(os.environ.get("MAX_CONCURRENT_JUDGE_REQUESTS", "128"))
        max_concurrent_multi_judge = int(
            os.environ.get("MAX_CONCURRENT_MULTI_JUDGE_REQUESTS", str(max_concurrent_judge))
        )
        if use_multi_judge:
            sampling_params_template = self._build_judge_sampling_params()
            judge_actors = self._multi_judge_engines.get_judge_actors_with_sampling_params(sampling_params_template)
            LOGGER.info(
                "[PolicyTrainerActor] Using multi-judge scoring with %d judges "
                "(max_concurrent_multi_judge_requests=%d, max_concurrent_judge_requests=%d)",
                len(judge_actors),
                max_concurrent_multi_judge,
                max_concurrent_judge,
            )
        else:
            # Create sampling params for judge if using rubric_judge_generate_text_actor
            # Override n=1 for judge since we only need one judgment per answer
            sampling_params = self._build_judge_sampling_params() if self.rubric_judge_generate_text_actor else None

        # scoring_mode is passed through to judge_answer_with_rubric which
        # handles all template / aggregation differences internally.
        scoring_mode = getattr(self, "_reward_mode", "rubric_judge") or "rubric_judge"

        with Timer(
            "[Data Preparation Thread] Computing rewards and prompts -- 🏆 Computing policy rubric scores", noop=True
        ) as reward_timer:
            # Create all judge tasks with concurrency limiting to prevent
            # overwhelming the judge actor (fixes >10K pending task buildup
            # seen with rar_implicit and other heavy scoring modes).
            judge_semaphore = asyncio.Semaphore(max_concurrent_multi_judge if use_multi_judge else max_concurrent_judge)

            async def _bounded_judge(question, rubric_text, answer):
                async with judge_semaphore:
                    if use_multi_judge:
                        judge_results = await judge_answer_with_multiple_judges(
                            question=question,
                            rubric=rubric_text,
                            answer=answer,
                            judge_actors=judge_actors,
                            answer_type="policy",
                        )
                        judge_scores = [float(result.get("score", 0.0)) for result in judge_results]
                        valid_scores = [s for s in judge_scores if not math.isnan(s)]
                        mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")
                        reasoning = "; ".join(
                            f"judge{idx + 1}={score:.3f} {result.get('reasoning', '')}".strip()
                            for idx, (score, result) in enumerate(zip(judge_scores, judge_results))
                        )
                        return {
                            "score": mean_score,
                            "reasoning": reasoning,
                            "judge_scores": judge_scores,
                        }

                    return await judge_answer_with_rubric(
                        question=question,
                        rubric=rubric_text,
                        answer=answer,
                        rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                        sampling_params=sampling_params,
                        scoring_mode=scoring_mode,
                    )

            judge_tasks = []
            for i, (answer, gt, rubric_spec) in enumerate(zip(decoded_responses, ground_truths, rubric_specs)):
                question = gt["question"]
                rubric_text = rubric_spec.rubric_text

                LOGGER.debug(
                    f"[PolicyTrainerActor] Creating judge task {i+1}/{len(decoded_responses)}: "
                    f"question={question[:50]}..., rubric_length={len(rubric_text)}, "
                    f"answer_length={len(answer)}"
                )

                judge_tasks.append(_bounded_judge(question, rubric_text, answer))

            # Execute all judge tasks in parallel (bounded by semaphore)
            eval_results = await asyncio.gather(*judge_tasks)

            # Process results in order and collect examples for logging
            generation_examples = []
            for i, eval_result in enumerate(eval_results):
                # Use the score directly as reward
                score = eval_result.get("score", 0.0)
                rewards.append(score)

                LOGGER.debug(
                    f"[PolicyTrainerActor] Reward {i+1}: score={score:.4f}"
                )

                # Collect examples for logging (up to num_examples_to_log)
                if i < self._num_examples_to_log:
                    # For policy training, we evaluate ONE answer (no accepted/rejected comparison)
                    # Set accepted_score = score for consistency (the policy's answer IS the accepted)
                    # rejected_* fields remain empty as there's no rejected answer in policy eval
                    generation_examples.append(GenerationExample(
                        training_step=self._current_training_step,
                        actor_type="policy",
                        example_index=i,
                        question=ground_truths[i]["question"],
                        rubric=rubric_specs[i].rubric_text,
                        policy_answer=decoded_responses[i],
                        score=score,
                        accepted_score=score,  # Policy's answer is the "accepted" answer being scored
                        accepted_reasoning=eval_result.get("reasoning", ""),
                        reward=score,  # For policy, reward = score directly
                        # rejected_* fields intentionally left empty (no rejected answer in policy training)
                    ))

        metrics["time/compute_rewards"] = reward_timer.duration
        if use_multi_judge:
            per_answer_stds = [
                float(np.std(eval_result["judge_scores"]))
                for eval_result in eval_results
                if eval_result.get("judge_scores")
            ]
            if per_answer_stds:
                metrics["objective/multi_judge_policy_score_std"] = float(np.mean(per_answer_stds))

        # Log generation examples at configured interval (every N training steps)
        # Debug logging to diagnose why examples aren't being written
        LOGGER.info(
            f"[PolicyTrainerActor] Generation examples check: "
            f"dir={bool(self._generation_examples_dir)}, "
            f"interval={self._log_examples_every_n_steps}, "
            f"step={self._current_training_step}, "
            f"modulo={self._current_training_step % self._log_examples_every_n_steps if self._log_examples_every_n_steps > 0 else 'N/A'}, "
            f"examples_count={len(generation_examples)}"
        )
        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            LOGGER.info(
                f"[PolicyTrainerActor] Writing {len(generation_examples)} generation examples for step {self._current_training_step}"
            )
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "policy",
                self._current_training_step,
            )

        metrics["objective/rubric_score"] = np.nanmean(rewards)
        metrics["objective/rubric_score_std"] = np.nanstd(rewards)

        LOGGER.info(
            f"[PolicyTrainerActor] Computed rewards: mean={np.nanmean(rewards):.4f}, "
            f"std={np.nanstd(rewards):.4f}, total_time={reward_timer.duration:.3f}s"
        )

        return rewards, metrics


class PolicyTrainerNoRubricModelActor(PolicyTrainerActor):
    """Policy trainer variant for modes that skip rubric model updates."""

    @cached_property
    def _api_rubric_proposer(self) -> str | None:
        """API model for rubric proposing (e.g. GPT-4.1).

        When set, rubric generation calls go through litellm instead of the
        local rubric-judge vLLM actor, while judging/scoring still uses the
        local actor.
        """
        model = os.environ.get("API_RUBRIC_GENERATOR")
        if model:
            LOGGER.info(
                "[PolicyTrainerNoRubricModelActor] Using API rubric proposer: %s "
                "(judge remains local vLLM actor)",
                model,
            )
        return model

    def _is_rrd_reward_mode(self) -> bool:
        return isinstance(self._reward_mode, str) and self._reward_mode.startswith("rrd_")

    def _is_rlcer_reward_mode(self) -> bool:
        return isinstance(self._reward_mode, str) and self._reward_mode.startswith("rlcer")

    def _is_query_specific_reward_mode(self) -> bool:
        return isinstance(self._reward_mode, str) and self._reward_mode == "query_specific_pref"

    def _is_rubric_arm_reward_mode(self) -> bool:
        return isinstance(self._reward_mode, str) and self._reward_mode == "rubric_arm"

    def _is_random_reward_mode(self) -> bool:
        return isinstance(self._reward_mode, str) and self._reward_mode == "random"

    def _is_rar_reward_mode(self) -> bool:
        """Check if using a RaR paper mode that needs standalone reward computation.

        All RaR baselines/methods are scored directly in the policy actor so
        they can consume dataset-provided references/rubrics or trigger the
        dedicated RaR rubric-generation helpers when rubric_data is absent.
        """
        return isinstance(self._reward_mode, str) and self._reward_mode in (
            "direct_likert", "reference_likert", "rar_predefined", "rar_implicit", "rar_explicit",
        )

    def add_prompt_to_generator(
        self,
        example: dict[str, Any],
        example_index: int,
        epoch_number: int,
        training_step: int,
        pending_queries_map: PendingQueriesMap,
        param_prompt_Q: ray_queue.Queue,
        generation_config: vllm.SamplingParams,
        is_eval: bool,
    ) -> None:
        if (
            not self._is_rrd_reward_mode()
            and not self._is_rlcer_reward_mode()
            and not self._is_query_specific_reward_mode()
            and not self._is_rar_reward_mode()
            and not self._is_rubric_arm_reward_mode()
            and not self._is_random_reward_mode()
        ):
            super().add_prompt_to_generator(
                example=example,
                example_index=example_index,
                epoch_number=epoch_number,
                training_step=training_step,
                pending_queries_map=pending_queries_map,
                param_prompt_Q=param_prompt_Q,
                generation_config=generation_config,
                is_eval=is_eval,
            )
            return

        LOGGER.debug(
            f"[PolicyTrainerNoRubricModelActor] add_prompt_to_generator: example_index={example_index}, "
            f"epoch={epoch_number}, training_step={training_step}, is_eval={is_eval}"
        )
        question_key = getattr(self.args, "question_key", "question")
        question = example[question_key]

        prompt_token_ids, messages = format_messages(
            "policy",
            {"question": question},
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            return_messages=True,
        )
        system_prompt = messages[0]["content"]
        prompt_text = f"{system_prompt}\n\nQuestion:\n{question}\n\nProvide your best possible answer."
        ground_truth = {"question": question}

        # For RLCER mode, include ground truth answer and verifier type for correctness checking
        if self._is_rlcer_reward_mode():
            ground_truth["ground_truth_answer"] = example.get("ground_truth", "")
            ground_truth["verifier_type"] = example.get("dataset", "math")

        if self._is_query_specific_reward_mode() or self._is_rubric_arm_reward_mode():
            accepted_key = getattr(self.args, "accepted_answer_key", "accepted")
            rejected_key = getattr(self.args, "rejected_answer_key", "rejected")
            ground_truth["preferred_answer"] = example.get(
                accepted_key, example.get("accepted_answer", example.get("ground_truth", ""))
            )
            ground_truth["dispreferred_answer"] = example.get(
                rejected_key, example.get("rejected_answer", "")
            )
            ground_truth["rubric_data"] = example.get(
                "rubric", example.get("rubrics", example.get("rubric_items", ""))
            )

        # For RaR paper modes, include reference answers and rubric data from dataset
        if self._is_rar_reward_mode():
            # Reference answer (used by reference_likert; may also be in rar_implicit/explicit datasets)
            ground_truth["reference_answer"] = example.get(
                "reference_answer", example.get("ground_truth", "")
            )
            # Rubric data (used by rar_implicit, rar_explicit; JSON list or text)
            ground_truth["rubric_data"] = example.get(
                "rubric", example.get("rubrics", example.get("rubric_items", ""))
            )

        pending_queries_map.insert(
            example_index,
            prompt_token_ids,
            ground_truth,
            "policy_training",
            prompt_text,
        )
        param_prompt_Q.put(
            PromptRequest(
                prompt=prompt_token_ids,
                generation_config=generation_config,
                epoch_number=epoch_number,
                training_step=training_step,
                dataset_index=example_index,
                is_eval=is_eval,
                actor_id="policy",
            )
        )

    async def _compute_rewards_and_prompts(
        self,
        decoded_responses: list[str],
        ground_truths: list[Any],
        queries: list[str] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        if self._is_random_reward_mode():
            return await self._compute_random_rewards(decoded_responses, ground_truths)
        if self._is_rlcer_reward_mode():
            return await self._compute_rlcer_rewards(decoded_responses, ground_truths)
        if self._is_query_specific_reward_mode():
            return await self._compute_query_specific_pref_rewards(decoded_responses, ground_truths)
        if self._is_rubric_arm_reward_mode():
            return await self._compute_rubric_arm_rewards(decoded_responses, ground_truths)
        if self._is_rar_reward_mode():
            return await self._compute_rar_rewards(decoded_responses, ground_truths)
        if not self._is_rrd_reward_mode():
            return await super()._compute_rewards_and_prompts(decoded_responses, ground_truths, queries)

        from open_instruct.search_rewards.rubric_judge_rewards import score_policy_rollouts_with_rrd_samples

        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            f"[PolicyTrainerNoRubricModelActor] Computing rewards for {batch_size} policy answers (episode {self._current_episode})"
        )
        rewards: list[float] = []
        metrics: dict[str, Any] = {"time/resolve_futures": 0.0}

        request_training_step = (
            ground_truths[0].get("_training_step", self._current_training_step) or self._current_training_step
        )
        self.rubric_actor.add_experience.remote(
            ground_truths[0]["question"],
            decoded_responses[0],
            request_training_step,
        )

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )

        with Timer(
            "[Data Preparation Thread] Computing rewards and prompts -- 🏆 Computing policy RRD scores",
            noop=True,
        ) as reward_timer:
            questions = [gt["question"] for gt in ground_truths]
            api_proposer = self._api_rubric_proposer
            eval_results = await score_policy_rollouts_with_rrd_samples(
                questions=questions,
                answers=decoded_responses,
                weighting_method=self._reward_mode.replace("rrd_", ""),
                decomposition_trigger=int(os.environ.get("RRD_DECOMPOSITION_TRIGGER", "2")),
                termination_threshold=int(os.environ.get("RRD_TERMINATION_THRESHOLD", "15")),
                max_rounds=int(os.environ.get("RRD_MAX_ROUNDS", "6")),
                proposer_model=api_proposer,
                proposer_generate_text_actor=None if api_proposer else self.rubric_judge_generate_text_actor,
                judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )

            generation_examples: list[GenerationExample] = []
            for i, eval_result in enumerate(eval_results):
                score = float(eval_result.get("score", 0.0))
                rewards.append(score)
                LOGGER.debug(f"[PolicyTrainerNoRubricModelActor] Reward {i+1}: score={score:.4f}")

                if i < self._num_examples_to_log:
                    rrd_items = list(eval_result.get("rubric_items", []))
                    generation_examples.append(
                        GenerationExample(
                            training_step=self._current_training_step,
                            actor_type="policy",
                            example_index=i,
                            question=ground_truths[i]["question"],
                            rubric="\n".join(f"{j+1}. {item}" for j, item in enumerate(rrd_items)),
                            policy_answer=decoded_responses[i],
                            score=score,
                            accepted_score=score,
                            accepted_reasoning=eval_result.get("reasoning", ""),
                            reward=score,
                            rrd_weighting_method=eval_result.get("weighting_method", ""),
                            rrd_iterations=int(eval_result.get("rrd_iterations", 0)),
                            rrd_rejected_count=int(eval_result.get("rrd_rejected_count", 0)),
                            rrd_rubric_items=rrd_items,
                            rrd_weights=[float(weight) for weight in (eval_result.get("weights", []) or [])],
                            rrd_binary_scores=[
                                float(binary_score)
                                for binary_score in (eval_result.get("binary_scores", []) or [])
                            ],
                            rrd_trace=eval_result.get("rrd_trace", {}),
                        )
                    )

        metrics["time/compute_rewards"] = reward_timer.duration

        # Debug logging to diagnose why examples aren't being written
        LOGGER.info(
            f"[PolicyTrainerNoRubricModelActor] Generation examples check: "
            f"dir={bool(self._generation_examples_dir)}, "
            f"interval={self._log_examples_every_n_steps}, "
            f"step={self._current_training_step}, "
            f"modulo={self._current_training_step % self._log_examples_every_n_steps if self._log_examples_every_n_steps > 0 else 'N/A'}, "
            f"examples_count={len(generation_examples)}"
        )
        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            LOGGER.info(
                f"[PolicyTrainerNoRubricModelActor] Writing {len(generation_examples)} generation examples for step {self._current_training_step}"
            )
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "policy",
                self._current_training_step,
            )

        metric_key = self._reward_mode.replace("rrd_", "")
        metrics[f"objective/rrd_{metric_key}_score"] = np.nanmean(rewards)
        metrics[f"objective/rrd_{metric_key}_score_std"] = np.nanstd(rewards)

        LOGGER.info(
            f"[PolicyTrainerNoRubricModelActor] Computed rewards: mean={np.nanmean(rewards):.4f}, "
            f"std={np.nanstd(rewards):.4f}, total_time={reward_timer.duration:.3f}s"
        )

        return rewards, metrics

    async def _compute_query_specific_pref_rewards(
        self,
        decoded_responses: list[str],
        ground_truths: list[Any],
    ) -> tuple[list[float], dict[str, Any]]:
        """Compute rewards using query-specific rubrics weighted by preference discrimination."""
        from open_instruct.search_rewards.rubric_judge_rewards import (
            score_policy_rollouts_with_query_specific_pref,
        )

        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            "[PolicyTrainerNoRubricModelActor/QuerySpecific] Computing rewards for %d policy answers (episode %d)",
            batch_size,
            self._current_episode,
        )
        rewards: list[float] = []
        metrics: dict[str, Any] = {"time/resolve_futures": 0.0}

        request_training_step = (
            ground_truths[0].get("_training_step", self._current_training_step) or self._current_training_step
        )
        self.rubric_actor.add_experience.remote(
            ground_truths[0]["question"],
            decoded_responses[0],
            request_training_step,
        )

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )

        with Timer(
            "[Data Preparation Thread] Computing rewards -- QuerySpecific",
            noop=True,
        ) as reward_timer:
            reference_map = self._get_query_specific_reference_map()
            questions = [gt["question"] for gt in ground_truths]
            preferred_answers: list[str] = []
            dispreferred_answers: list[str] = []
            for gt in ground_truths:
                q = gt.get("question", "")
                pref = gt.get("preferred_answer", "")
                dispref = gt.get("dispreferred_answer", "")
                if (not pref or not dispref) and q in reference_map:
                    cached_pref, cached_dispref = reference_map[q]
                    pref = pref or cached_pref
                    dispref = dispref or cached_dispref
                preferred_answers.append(pref)
                dispreferred_answers.append(dispref)
            rubric_data_list = [gt.get("rubric_data", None) for gt in ground_truths]

            api_proposer = self._api_rubric_proposer
            eval_results = await score_policy_rollouts_with_query_specific_pref(
                questions=questions,
                answers=decoded_responses,
                preferred_answers=preferred_answers,
                dispreferred_answers=dispreferred_answers,
                rubric_data_list=rubric_data_list,
                proposer_model=api_proposer,
                proposer_generate_text_actor=None if api_proposer else self.rubric_judge_generate_text_actor,
                rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )

            generation_examples: list[GenerationExample] = []
            for i, eval_result in enumerate(eval_results):
                score = float(eval_result.get("score", 0.0))
                rewards.append(score)
                LOGGER.debug(
                    "[PolicyTrainerNoRubricModelActor/QuerySpecific] Reward %d: score=%.4f",
                    i + 1,
                    score,
                )

                if i < self._num_examples_to_log:
                    rubric_items = list(eval_result.get("rubric_items", []))
                    generation_examples.append(
                        GenerationExample(
                            training_step=self._current_training_step,
                            actor_type="policy",
                            example_index=i,
                            question=ground_truths[i]["question"],
                            rubric="\n".join(f"{j+1}. {item}" for j, item in enumerate(rubric_items)),
                            policy_answer=decoded_responses[i],
                            score=score,
                            accepted_score=score,
                            accepted_reasoning=eval_result.get("reasoning", ""),
                            reward=score,
                            rrd_weighting_method=eval_result.get("weighting_method", "query_specific_pref"),
                            rrd_iterations=0,
                            rrd_rejected_count=0,
                            rrd_rubric_items=rubric_items,
                            rrd_weights=[float(weight) for weight in (eval_result.get("weights", []) or [])],
                            rrd_binary_scores=[
                                float(s) for s in (
                                    eval_result.get("item_scores")
                                    or eval_result.get("binary_scores")
                                    or []
                                )
                            ],
                            rrd_trace=eval_result.get("trace", {}),
                        )
                    )

        metrics["time/compute_rewards"] = reward_timer.duration

        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "policy",
                self._current_training_step,
            )

        metrics["objective/query_specific_pref_score"] = np.mean(rewards)
        metrics["objective/query_specific_pref_score_std"] = np.std(rewards)

        LOGGER.info(
            "[PolicyTrainerNoRubricModelActor/QuerySpecific] Computed rewards: mean=%.4f, std=%.4f, total_time=%.3fs",
            np.mean(rewards),
            np.std(rewards),
            reward_timer.duration,
        )

        return rewards, metrics

    async def _compute_rubric_arm_rewards(
        self,
        decoded_responses: list[str],
        ground_truths: list[Any],
    ) -> tuple[list[float], dict[str, Any]]:
        """Compute rewards using Rubric-ARM debiased pairwise evaluation (Eq. 16)."""
        from open_instruct.search_rewards.rubric_judge_rewards import (
            score_policy_rollouts_with_rubric_arm,
        )

        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            "[PolicyTrainerNoRubricModelActor/RubricARM] Computing rewards for %d policy answers (episode %d)",
            batch_size,
            self._current_episode,
        )
        rewards: list[float] = []
        metrics: dict[str, Any] = {"time/resolve_futures": 0.0}

        request_training_step = (
            ground_truths[0].get("_training_step", self._current_training_step) or self._current_training_step
        )
        self.rubric_actor.add_experience.remote(
            ground_truths[0]["question"],
            decoded_responses[0],
            request_training_step,
        )

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )

        with Timer(
            "[Data Preparation Thread] Computing rewards -- RubricARM",
            noop=True,
        ) as reward_timer:
            reference_map = self._get_query_specific_reference_map()
            questions = [gt["question"] for gt in ground_truths]
            preferred_answers: list[str] = []
            dispreferred_answers: list[str] = []
            for gt in ground_truths:
                q = gt.get("question", "")
                pref = gt.get("preferred_answer", "")
                dispref = gt.get("dispreferred_answer", "")
                if (not pref or not dispref) and q in reference_map:
                    cached_pref, cached_dispref = reference_map[q]
                    pref = pref or cached_pref
                    dispref = dispref or cached_dispref
                preferred_answers.append(pref)
                dispreferred_answers.append(dispref)

            eval_results = await score_policy_rollouts_with_rubric_arm(
                questions=questions,
                answers=decoded_responses,
                preferred_answers=preferred_answers,
                dispreferred_answers=dispreferred_answers,
                proposer_generate_text_actor=self.rubric_judge_generate_text_actor,
                rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )

            generation_examples: list[GenerationExample] = []
            for i, eval_result in enumerate(eval_results):
                score = float(eval_result.get("score", 0.0))
                rewards.append(score)
                LOGGER.debug(
                    "[PolicyTrainerNoRubricModelActor/RubricARM] Reward %d: score=%.4f",
                    i + 1,
                    score,
                )

                if i < self._num_examples_to_log:
                    rubric_items = list(eval_result.get("rubric_items", []))
                    generation_examples.append(
                        GenerationExample(
                            training_step=self._current_training_step,
                            actor_type="policy",
                            example_index=i,
                            question=ground_truths[i]["question"],
                            rubric="\n".join(f"{j+1}. {item}" for j, item in enumerate(rubric_items)),
                            policy_answer=decoded_responses[i],
                            score=score,
                            accepted_score=score,
                            accepted_reasoning=eval_result.get("reasoning", ""),
                            reward=score,
                            rrd_weighting_method="rubric_arm",
                            rrd_iterations=0,
                            rrd_rejected_count=0,
                            rrd_rubric_items=rubric_items,
                            rrd_weights=[],
                            rrd_binary_scores=[],
                            rrd_trace={
                                "winner_forward": eval_result.get("winner_forward"),
                                "winner_reverse": eval_result.get("winner_reverse"),
                                "r_fmt_forward": eval_result.get("r_fmt_forward", 0.0),
                                "r_fmt_reverse": eval_result.get("r_fmt_reverse", 0.0),
                            },
                        )
                    )

        metrics["time/compute_rewards"] = reward_timer.duration

        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "policy",
                self._current_training_step,
            )

        metrics["objective/rubric_arm_score"] = np.mean(rewards)
        metrics["objective/rubric_arm_score_std"] = np.std(rewards)

        LOGGER.info(
            "[PolicyTrainerNoRubricModelActor/RubricARM] Computed rewards: mean=%.4f, std=%.4f, total_time=%.3fs",
            np.mean(rewards),
            np.std(rewards),
            reward_timer.duration,
        )

        return rewards, metrics

    def _get_query_specific_reference_map(self) -> dict[str, tuple[str, str]]:
        """Build a question -> (preferred, dispreferred) cache from local JSONL mixers."""
        cached = getattr(self, "_query_specific_reference_map", None)
        if cached is not None:
            return cached

        reference_map: dict[str, tuple[str, str]] = {}
        dataset_mixer_list = getattr(self.args, "dataset_mixer_list", None) or []
        question_key = getattr(self.args, "question_key", "question")
        accepted_key = getattr(self.args, "accepted_answer_key", "")
        rejected_key = getattr(self.args, "rejected_answer_key", "")

        # dataset_mixer_list format: [path_or_hf_id, weight, path_or_hf_id, weight, ...]
        for i in range(0, max(len(dataset_mixer_list) - 1, 0), 2):
            dataset_ref = str(dataset_mixer_list[i])
            if not dataset_ref.endswith(".jsonl") or not os.path.exists(dataset_ref):
                continue
            try:
                with open(dataset_ref, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        q = str(row.get(question_key, ""))
                        pref = str(row.get(accepted_key, "")) if accepted_key else ""
                        dispref = str(row.get(rejected_key, "")) if rejected_key else ""
                        if q and pref and dispref and q not in reference_map:
                            reference_map[q] = (pref, dispref)
            except OSError as e:
                LOGGER.warning(
                    "[PolicyTrainerNoRubricModelActor/QuerySpecific] Failed loading preference references from %s: %s",
                    dataset_ref,
                    e,
                )

        self._query_specific_reference_map = reference_map
        LOGGER.info(
            "[PolicyTrainerNoRubricModelActor/QuerySpecific] Loaded %d question-level preference references",
            len(reference_map),
        )
        return reference_map

    async def _precompute_rlcer_evolving_rollout_rubrics(
        self,
        *,
        questions: list[str],
        answers: list[str],
    ) -> list[dict[str, Any]]:
        """Use the evolving rubric actor to propose one rubric per policy rollout."""
        if not self.rubric_actor:
            raise RuntimeError("rubric_actor is not initialized")
        return await precompute_rlcer_evolving_rollout_rubrics(
            questions=questions,
            answers=answers,
            rubric_actor=self.rubric_actor,
        )

    def _buffer_rlcer_evolving_cached_generations(
        self,
        *,
        cached_generations: list[dict[str, Any]],
    ) -> None:
        """Append cached rubric generations to the drain queue."""
        if not cached_generations:
            return
        with self._rlcer_cache_condition:
            self._rlcer_cache_deque.extend(dict(item) for item in cached_generations)
            LOGGER.info(
                "[PolicyTrainerNoRubricModelActor/RLCER] Buffered %d cached RL-CER evolving generation(s) (queue size: %d)",
                len(cached_generations),
                len(self._rlcer_cache_deque),
            )
            self._rlcer_cache_condition.notify_all()

    def drain_rlcer_evolving_cached_generations(
        self,
        expected_count: int,
        timeout_s: float = 1800.0,
    ) -> list[dict[str, Any]]:
        """Block until *expected_count* cached generations are available, then drain them.

        Returns at most *expected_count* items (extras stay queued for the next
        drain).  Falls back to a partial drain after *timeout_s* seconds so the
        caller is never stuck forever.
        """
        target = max(int(expected_count), 0)
        if target <= 0:
            return []

        with self._rlcer_cache_condition:
            ok = self._rlcer_cache_condition.wait_for(
                lambda: len(self._rlcer_cache_deque) >= target,
                timeout=max(float(timeout_s), 0.0),
            )
            available = len(self._rlcer_cache_deque)
            drain_count = min(target, available)
            items = [self._rlcer_cache_deque.popleft() for _ in range(drain_count)]

        if not ok:
            LOGGER.warning(
                "[PolicyTrainerNoRubricModelActor/RLCER] Timed out waiting for cached RL-CER evolving "
                "generations: got %d/%d after %.0fs",
                drain_count,
                target,
                timeout_s,
            )
        else:
            LOGGER.info(
                "[PolicyTrainerNoRubricModelActor/RLCER] Drained %d cached RL-CER evolving generation(s)",
                drain_count,
            )
        return items

    async def _compute_rlcer_rewards(
        self,
        decoded_responses: list[str],
        ground_truths: list[Any],
    ) -> tuple[list[float], dict[str, Any]]:
        """Compute rewards using the RLCER method (Sheng et al., 2026).

        Uses correlation-filtered rubrics to reward CoT quality alongside outcome rewards.
        """
        from open_instruct.search_rewards.rubric_judge_rewards import score_policy_rollouts_with_rlcer

        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            f"[PolicyTrainerNoRubricModelActor/RLCER] Computing rewards for {batch_size} policy answers "
            f"(episode {self._current_episode})"
        )
        rewards: list[float] = []
        metrics: dict[str, Any] = {"time/resolve_futures": 0.0}

        request_training_step = (
            ground_truths[0].get("_training_step", self._current_training_step) or self._current_training_step
        )
        self.rubric_actor.add_experience.remote(
            ground_truths[0]["question"],
            decoded_responses[0],
            request_training_step,
        )

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )

        with Timer(
            "[Data Preparation Thread] Computing rewards and prompts -- 🏆 Computing RLCER scores",
            noop=True,
        ) as reward_timer:
            questions = [gt["question"] for gt in ground_truths]
            gt_answers = [gt.get("ground_truth_answer", "") for gt in ground_truths]
            verifier_types = [gt.get("verifier_type", "math") for gt in ground_truths]

            api_proposer = self._api_rubric_proposer
            precomputed_rollout_rubrics = None
            if self._reward_mode == "rlcer_evolving" and self.rubric_actor and not api_proposer:
                precomputed_rollout_rubrics = await self._precompute_rlcer_evolving_rollout_rubrics(
                    questions=questions,
                    answers=decoded_responses,
                )

            eval_results = await score_policy_rollouts_with_rlcer(
                questions=questions,
                answers=decoded_responses,
                ground_truths=gt_answers,
                verifier_types=verifier_types,
                correlation_threshold=float(os.environ.get("RLCER_CORRELATION_THRESHOLD", "0.2")),
                outcome_reward_weight=float(os.environ.get("RLCER_OUTCOME_WEIGHT", "1.0")),
                cot_reward_weight=float(os.environ.get("RLCER_COT_WEIGHT", "1.0")),
                proposer_model=api_proposer,
                proposer_generate_text_actor=None if (api_proposer or precomputed_rollout_rubrics is not None) else self.rubric_judge_generate_text_actor,
                judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
                precomputed_rollout_rubrics=precomputed_rollout_rubrics,
            )

            # For "rlcer_evolving": each reward callback receives one prompt's
            # rollout group (the same question repeated n times), so store that
            # single-question rollout data plus the full cached rubric rollouts
            # from the current policy step.
            if self._reward_mode == "rlcer_evolving" and self.rubric_actor:
                unique_questions = {str(question) for question in questions}
                if len(unique_questions) != 1:
                    raise ValueError(
                        "rlcer_evolving policy reward callback expects one prompt-group per call; "
                        f"got {len(unique_questions)} distinct questions in the same reward batch"
                    )

                group_correctness = [
                    float(eval_result.get("correctness", 0.0)) for eval_result in eval_results
                ]
                self.rubric_actor.add_rlcer_rollout_data.remote(
                    questions[0],
                    list(decoded_responses),
                    group_correctness,
                )

                cached_rubric_generations: list[dict[str, Any]] = []
                if precomputed_rollout_rubrics is not None:
                    policy_samples_per_prompt = max(
                        int(getattr(self.args, "num_samples_per_prompt_rollout", 1)),
                        1,
                    )
                    for rollout_idx, cached_rubric in enumerate(precomputed_rollout_rubrics):
                        if rollout_idx >= len(questions):
                            continue
                        eval_result = eval_results[rollout_idx]
                        rubric_items = list(eval_result.get("rubric_items", []))
                        valid_indices = [
                            int(index) for index in eval_result.get("valid_rubric_indices", [])
                        ]
                        k_total = int(eval_result.get("num_total_rubrics", len(rubric_items)))
                        k_valid = int(eval_result.get("num_valid_rubrics", len(valid_indices)))
                        r_format = 1.0 if k_total > 0 else 0.0
                        validity_fraction = float(k_valid / k_total) if k_total > 0 else 0.0
                        cached_rubric_generations.append(
                            {
                                "question": questions[rollout_idx],
                                "ground_truth_answer": str(gt_answers[rollout_idx]),
                                "verifier_type": str(verifier_types[rollout_idx]),
                                "rubric_text": str(cached_rubric.get("rubric_text", "")),
                                "prompt_token_ids": list(cached_rubric.get("prompt_token_ids") or []),
                                "prompt_text": str(cached_rubric.get("prompt_text", "")),
                                "generation_result": cached_rubric.get("generation_result"),
                                "prompt_group_index": rollout_idx // policy_samples_per_prompt,
                                "sample_index_within_prompt": rollout_idx % policy_samples_per_prompt,
                                "rubricator_reward_detail": {
                                    "reward": validity_fraction + r_format,
                                    "k_valid": k_valid,
                                    "k_total": k_total,
                                    "validity_fraction": validity_fraction,
                                    "r_format": r_format,
                                    "rubric_items": rubric_items,
                                    "valid_indices": valid_indices,
                                },
                            }
                        )
                self._buffer_rlcer_evolving_cached_generations(
                    cached_generations=cached_rubric_generations,
                )

            generation_examples: list[GenerationExample] = []
            for i, eval_result in enumerate(eval_results):
                score = float(eval_result.get("score", 0.0))
                rewards.append(score)
                LOGGER.debug(f"[PolicyTrainerNoRubricModelActor/RLCER] Reward {i+1}: score={score:.4f}")

                if i < self._num_examples_to_log:
                    rlcer_items = list(eval_result.get("rubric_items", []))
                    valid_indices = eval_result.get("valid_rubric_indices", [])
                    generation_examples.append(
                        GenerationExample(
                            training_step=self._current_training_step,
                            actor_type="policy",
                            example_index=i,
                            question=ground_truths[i]["question"],
                            rubric="\n".join(
                                f"{j+1}. {'[VALID] ' if j in valid_indices else ''}{item}"
                                for j, item in enumerate(rlcer_items)
                            ),
                            policy_answer=decoded_responses[i],
                            score=score,
                            accepted_score=float(eval_result.get("outcome_reward", 0.0)),
                            accepted_reasoning=eval_result.get("reasoning", ""),
                            reward=score,
                            rrd_weighting_method="rlcer",
                            rrd_iterations=0,
                            rrd_rejected_count=0,
                            rrd_rubric_items=rlcer_items,
                            rrd_weights=[],
                            rrd_binary_scores=[
                                float(bs) for bs in (eval_result.get("binary_scores", []) or [])
                            ],
                            rrd_trace={
                                "valid_rubric_indices": valid_indices,
                                "correctness": eval_result.get("correctness", 0.0),
                                "outcome_reward": eval_result.get("outcome_reward", 0.0),
                                "cot_reward": eval_result.get("cot_reward", 0.0),
                                "num_valid_rubrics": eval_result.get("num_valid_rubrics", 0),
                                "num_total_rubrics": eval_result.get("num_total_rubrics", 0),
                                "rubric_scores": eval_result.get("rubric_scores", []),
                            },
                        )
                    )

        metrics["time/compute_rewards"] = reward_timer.duration

        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "policy",
                self._current_training_step,
            )

        # RLCER-specific metrics
        all_correctness = [float(r.get("correctness", 0.0)) for r in eval_results if r]
        all_cot = [float(r.get("cot_reward", 0.0)) for r in eval_results if r]
        all_outcome = [float(r.get("outcome_reward", 0.0)) for r in eval_results if r]
        all_valid = [float(r.get("num_valid_rubrics", 0)) for r in eval_results if r]
        all_total = [float(r.get("num_total_rubrics", 0)) for r in eval_results if r]

        metrics["objective/rlcer_score"] = np.mean(rewards)
        metrics["objective/rlcer_score_std"] = np.std(rewards)
        metrics["objective/rlcer_correctness"] = np.mean(all_correctness)
        metrics["objective/rlcer_cot_reward"] = np.mean(all_cot)
        metrics["objective/rlcer_outcome_reward"] = np.mean(all_outcome)
        metrics["objective/rlcer_valid_rubrics"] = np.mean(all_valid)
        metrics["objective/rlcer_total_rubrics"] = np.mean(all_total)
        if np.mean(all_total) > 0:
            metrics["objective/rlcer_valid_ratio"] = np.mean(all_valid) / np.mean(all_total)

        LOGGER.info(
            f"[PolicyTrainerNoRubricModelActor/RLCER] Computed rewards: mean={np.mean(rewards):.4f}, "
            f"std={np.std(rewards):.4f}, correctness={np.mean(all_correctness):.4f}, "
            f"cot={np.mean(all_cot):.4f}, valid_rubrics={np.mean(all_valid):.1f}/{np.mean(all_total):.1f}, "
            f"total_time={reward_timer.duration:.3f}s"
        )

        return rewards, metrics

    async def _compute_random_rewards(
        self,
        decoded_responses: list[str],
        ground_truths: list[Any],
    ) -> tuple[list[float], dict[str, Any]]:
        """Assign random 0/1 rewards to each response (no judge calls)."""
        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        rewards = [random.choice([0.0, 1.0]) for _ in range(batch_size)]
        mean_reward = sum(rewards) / len(rewards)
        LOGGER.info(
            f"[PolicyTrainerNoRubricModelActor/Random] Assigned random rewards to "
            f"{batch_size} responses (mean={mean_reward:.3f}, episode={self._current_episode})"
        )
        metrics: dict[str, Any] = {
            "time/resolve_futures": 0.0,
            "reward/random_mean": mean_reward,
        }
        return rewards, metrics

    async def _compute_rar_rewards(
        self,
        decoded_responses: list[str],
        ground_truths: list[Any],
    ) -> tuple[list[float], dict[str, Any]]:
        """Compute rewards using RaR paper methods (Gunjal et al., 2025).

        Dispatches to the appropriate scoring function based on reward mode:
        - direct_likert: Direct Likert scoring (no rubrics)
        - reference_likert: Reference-based Likert scoring
        - rar_predefined: Fixed generic rubrics + binary judging
        - rar_implicit: Holistic Likert judging with dataset or generated RaR rubrics
        - rar_explicit: Weighted binary judging with dataset or generated RaR rubrics
        """
        from open_instruct.search_rewards.rubric_judge_rewards import (
            score_policy_rollouts_with_direct_likert,
            score_policy_rollouts_with_rar_explicit,
            score_policy_rollouts_with_rar_implicit,
            score_policy_rollouts_with_rar_predefined,
            score_policy_rollouts_with_reference_likert,
        )

        batch_size = len(decoded_responses)
        self._current_episode += batch_size
        LOGGER.debug(
            f"[PolicyTrainerNoRubricModelActor/RaR] Computing {self._reward_mode} rewards "
            f"for {batch_size} policy answers (episode {self._current_episode})"
        )
        rewards: list[float] = []
        metrics: dict[str, Any] = {"time/resolve_futures": 0.0}

        request_training_step = (
            ground_truths[0].get("_training_step", self._current_training_step) or self._current_training_step
        )
        self.rubric_actor.add_experience.remote(
            ground_truths[0]["question"],
            decoded_responses[0],
            request_training_step,
        )

        sampling_params = (
            self._build_judge_sampling_params()
            if self.rubric_judge_generate_text_actor
            else None
        )

        with Timer(
            f"[Data Preparation Thread] Computing rewards -- RaR {self._reward_mode}",
            noop=True,
        ) as reward_timer:
            questions = [gt["question"] for gt in ground_truths]
            mode = self._reward_mode
            api_proposer = self._api_rubric_proposer

            if mode == "direct_likert":
                eval_results = await score_policy_rollouts_with_direct_likert(
                    questions=questions,
                    answers=decoded_responses,
                    rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                )
            elif mode == "reference_likert":
                reference_answers = [gt.get("reference_answer", "") for gt in ground_truths]
                eval_results = await score_policy_rollouts_with_reference_likert(
                    questions=questions,
                    answers=decoded_responses,
                    reference_answers=reference_answers,
                    rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                )
            elif mode == "rar_implicit":
                rubric_data_list = [gt.get("rubric_data", None) for gt in ground_truths]
                eval_results = await score_policy_rollouts_with_rar_implicit(
                    questions=questions,
                    answers=decoded_responses,
                    rubric_data_list=rubric_data_list,
                    proposer_model=api_proposer,
                    proposer_generate_text_actor=None if api_proposer else self.rubric_judge_generate_text_actor,
                    rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                )
            elif mode == "rar_explicit":
                rubric_data_list = [gt.get("rubric_data", None) for gt in ground_truths]
                eval_results = await score_policy_rollouts_with_rar_explicit(
                    questions=questions,
                    answers=decoded_responses,
                    rubric_data_list=rubric_data_list,
                    proposer_model=api_proposer,
                    proposer_generate_text_actor=None if api_proposer else self.rubric_judge_generate_text_actor,
                    rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                )
            elif mode == "rar_predefined":
                eval_results = await score_policy_rollouts_with_rar_predefined(
                    questions=questions,
                    answers=decoded_responses,
                    rubric_judge_generate_text_actor=self.rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                )
            else:
                raise ValueError(f"Unknown RaR reward mode: {mode}")

            generation_examples: list[GenerationExample] = []
            for i, eval_result in enumerate(eval_results):
                score = float(eval_result.get("score", 0.0))
                rewards.append(score)
                LOGGER.debug(f"[PolicyTrainerNoRubricModelActor/RaR] Reward {i+1}: score={score:.4f}")

                if i < self._num_examples_to_log:
                    rubric_items = list(eval_result.get("rubric_items", []))
                    generation_examples.append(
                        GenerationExample(
                            training_step=self._current_training_step,
                            actor_type="policy",
                            example_index=i,
                            question=ground_truths[i]["question"],
                            rubric="\n".join(
                                f"{j+1}. {item}" for j, item in enumerate(rubric_items)
                            ) if rubric_items else f"({mode} — no explicit rubric items)",
                            policy_answer=decoded_responses[i],
                            score=score,
                            accepted_score=score,
                            accepted_reasoning=eval_result.get("reasoning", ""),
                            reward=score,
                            rrd_weighting_method=mode,
                            rrd_iterations=0,
                            rrd_rejected_count=0,
                            rrd_rubric_items=rubric_items,
                            rrd_weights=[float(w) for w in (eval_result.get("weights", []) or [])],
                            rrd_binary_scores=[
                                float(bs) for bs in (eval_result.get("binary_scores", []) or [])
                            ],
                            rrd_trace={},
                        )
                    )

        metrics["time/compute_rewards"] = reward_timer.duration

        if (
            self._generation_examples_dir
            and self._log_examples_every_n_steps > 0
            and self._current_training_step % self._log_examples_every_n_steps == 0
            and generation_examples
        ):
            _write_generation_examples(
                generation_examples,
                self._generation_examples_dir,
                "policy",
                self._current_training_step,
            )

        metrics[f"objective/{mode}_score"] = np.mean(rewards)
        metrics[f"objective/{mode}_score_std"] = np.std(rewards)

        LOGGER.info(
            f"[PolicyTrainerNoRubricModelActor/RaR] Computed {mode} rewards: mean={np.mean(rewards):.4f}, "
            f"std={np.std(rewards):.4f}, total_time={reward_timer.duration:.3f}s"
        )

        return rewards, metrics


PolicyTrainerActor = ray.remote(PolicyTrainerActor)
PolicyTrainerNoRubricModelActor = ray.remote(PolicyTrainerNoRubricModelActor)


# ---------------------------------------------------------------------------
# Alternating orchestration
# ---------------------------------------------------------------------------


def _collect_actor_queue_metrics(actor: ray.actor.ActorHandle, actor_name: str) -> dict[str, Any]:
    """Collect queue metrics from an actor's actor_manager.

    Args:
        actor: Ray actor handle
        actor_name: Name prefix for metrics (e.g., "policy", "rubric")

    Returns:
        Dictionary of queue metrics with actor_name prefix
    """
    metrics = {}
    try:
        # Get actor_manager from actor
        actor_manager = ray.get(actor.get_actor_manager.remote())
        if actor_manager is None:
            return metrics

        # Get queue stats
        queue_stats = ray.get(actor_manager.get_queue_stats.remote())
        for key, value in queue_stats.items():
            metrics[f"{actor_name}/{key}"] = value

        # Get active tasks stats with history (includes current and average values)
        active_tasks_stats = ray.get(actor_manager.get_active_tasks_stats.remote())
        for key, value in active_tasks_stats.items():
            metrics[f"{actor_name}/{key}"] = value

        # Get detailed actor stats for individual actor breakdown
        actor_stats = ray.get(actor_manager.get_vllm_actor_stats.remote())
        metrics[f"{actor_name}/actors/total_actors"] = actor_stats.get("total_actors", 0)
        metrics[f"{actor_name}/actors/total_active_tasks"] = actor_stats.get("total_active_tasks", 0)
        metrics[f"{actor_name}/actors/policy_active_tasks"] = actor_stats.get("policy_active_tasks", 0)
        metrics[f"{actor_name}/actors/rubric_judge_active_tasks"] = actor_stats.get("rubric_judge_active_tasks", 0)

        # Log individual actor active tasks
        for i, actor_info in enumerate(actor_stats.get("actors", [])):
            actor_type = "rubric_judge" if actor_info.get("is_rubric_judge", False) else "policy"
            metrics[f"{actor_name}/actors/{actor_type}_{i}/active_tasks"] = actor_info.get("active_tasks", 0)
    except Exception as e:
        LOGGER.debug(f"Failed to collect queue metrics from {actor_name} actor: {e}")
    return metrics


def _collect_rubric_judge_queue_metrics(
    rubric_judge_actor_manager: ray.actor.ActorHandle | None,
) -> dict[str, Any]:
    """Collect queue metrics from rubric judge engines via their actor_manager.

    Args:
        rubric_judge_actor_manager: ActorManager handle for rubric judge engines

    Returns:
        Dictionary of queue metrics
    """
    metrics = {}
    if rubric_judge_actor_manager is None:
        return metrics

    try:
        # Get queue stats from actor_manager
        queue_stats = ray.get(rubric_judge_actor_manager.get_queue_stats.remote())
        for key, value in queue_stats.items():
            metrics[f"rubric_judge/{key}"] = value

        # Get active tasks stats with history
        active_tasks_stats = ray.get(rubric_judge_actor_manager.get_active_tasks_stats.remote())
        for key, value in active_tasks_stats.items():
            metrics[f"rubric_judge/{key}"] = value

        # Get detailed actor stats
        actor_stats = ray.get(rubric_judge_actor_manager.get_vllm_actor_stats.remote())
        metrics["rubric_judge/actors/total_actors"] = actor_stats.get("total_actors", 0)
        metrics["rubric_judge/actors/total_active_tasks"] = actor_stats.get("total_active_tasks", 0)
        metrics["rubric_judge/actors/policy_active_tasks"] = actor_stats.get("policy_active_tasks", 0)
        metrics["rubric_judge/actors/rubric_judge_active_tasks"] = actor_stats.get("rubric_judge_active_tasks", 0)

        # Log individual actor active tasks
        for i, actor_info in enumerate(actor_stats.get("actors", [])):
            actor_type = "rubric_judge" if actor_info.get("is_rubric_judge", False) else "policy"
            metrics[f"rubric_judge/actors/{actor_type}_{i}/active_tasks"] = actor_info.get("active_tasks", 0)
    except Exception as e:
        LOGGER.debug(f"Failed to collect rubric judge queue metrics: {e}")
    return metrics


def _log_metrics_to_wandb(metrics: dict[str, Any], training_step: int) -> None:
    """Log metrics to wandb from the main thread.

    Args:
        metrics: Dictionary of metrics to log
        training_step: Training step (gradient update count) for step tracking
    """
    if not metrics:
        return

    try:
        import wandb
        if wandb.run is None:
            LOGGER.warning("Wandb not initialized, skipping metric logging")
            return

        # Convert array/list metrics to wandb histograms for logging
        wandb_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                wandb_metrics[key] = wandb.Histogram(value)
            elif isinstance(value, (float, int, str, bool)):
                wandb_metrics[key] = value
            # Skip other types (e.g., torch tensors) as they're not serializable

        if wandb_metrics:
            wandb.log(wandb_metrics, step=training_step)
    except Exception as e:
        LOGGER.error(f"Failed to log metrics to wandb: {e}", exc_info=True)


def run_alternating_training(
    policy_actor: ray.actor.ActorHandle,
    rubric_actor: ray.actor.ActorHandle,
    train_dataset: Dataset,
    *,
    cycles: int,
    steps_per_phase: int = 5,
    use_both_models: bool = False,
    rubric_judge_actor_manager: ray.actor.ActorHandle | None = None,
    curriculum_state: JudgeCurriculumState | None = None,
    curriculum_swap_callback: Callable[[str], AuxiliaryModelEngines] | None = None,
    skip_rubric_training: bool = False,
    skip_policy_training: bool = False,
    extra_policy_actors: list[ray.actor.ActorHandle] | None = None,
    extra_policy_names: list[str] | None = None,
    data_provider: ray.actor.ActorHandle | None = None,
    resume_from_step: int = 0,
    force_single_step_alternation: bool = False,
    enqueue_rlcer_evolving_cached_results: bool = False,
    rlcer_expected_cached_rollouts: int | None = None,
) -> dict[str, Any]:
    """Coordinate alternating training between policy and rubric actors.

    Args:
        policy_actor: Ray actor handle for the main policy trainer.
        rubric_actor: Ray actor handle for the rubric trainer.
        train_dataset: Dataset to use for training.
        cycles: Number of alternating cycles to run.
        steps_per_phase: Number of policy steps to run before handing feedback to the rubric trainer.
        use_both_models: Whether to use both policy and baseline models when creating rubrics.
        rubric_judge_actor_manager: Optional actor manager for rubric judge engines.
        curriculum_state: Optional curriculum state for judge size curriculum.
        curriculum_swap_callback: Callback to swap judge model, takes new model name and returns new engines.
        skip_rubric_training: If True, skip rubric model training (for API-based rubric generator baseline).
        skip_policy_training: If True, skip policy model training (for rubric-only training with fixed policy).
        extra_policy_actors: Additional policy actors for multi-policy co-evolution.
            Each gets its own training phase per cycle.
        extra_policy_names: Human-readable names for extra policies (for logging).
        data_provider: Optional data provider Ray actor handle. When set with extra_policy_actors,
            the active policy actor is rotated during rubric training for diverse responses.
        force_single_step_alternation: If True, override ``steps_per_phase`` to 1.
        enqueue_rlcer_evolving_cached_results: If True, the rubric actor explicitly
            enqueues cached policy-step RL-CER generations before each rubric step.
        rlcer_expected_cached_rollouts: Expected number of cached policy rollouts per
            RL-CER evolving policy step.

    Returns:
        Dictionary containing metadata.
    """

    if steps_per_phase <= 0:
        raise ValueError("steps_per_phase must be positive.")
    if cycles <= 0:
        raise ValueError("cycles must be positive.")
    if skip_policy_training and skip_rubric_training:
        raise ValueError(
            "Cannot skip both policy and rubric training. "
            "At least one model must be trained. "
            "Set skip_policy_training=True for rubric-only training, "
            "or skip_rubric_training=True for policy-only training."
        )
    if enqueue_rlcer_evolving_cached_results and not force_single_step_alternation:
        raise ValueError(
            "rlcer_evolving cached-result handoff requires exactly one update per phase."
        )

    effective_steps_per_phase = 1 if force_single_step_alternation else steps_per_phase

    extra_policy_actors = extra_policy_actors or []
    extra_policy_names = extra_policy_names or [f"policy_{i+2}" for i in range(len(extra_policy_actors))]
    all_policy_actors = [policy_actor] + extra_policy_actors
    all_policy_names = ["policy"] + extra_policy_names

    # Set cross-references between actors
    # All policies reference the same rubric actor; rubric references the main policy
    ray.get(policy_actor.set_rubric_actor.remote(rubric_actor))
    ray.get(rubric_actor.set_policy_actor.remote(policy_actor))

    # Set cross-references for extra policy actors
    for i, extra_actor in enumerate(extra_policy_actors):
        LOGGER.info("Setting cross-references for extra policy actor: %s", extra_policy_names[i])
        ray.get(extra_actor.set_rubric_actor.remote(rubric_actor))

    LOGGER.info("======== ✅ All cross-references set ========")

    # NOW start training threads on all actors (after all initialization is complete)
    # Extra policy actors are already fully initialized at this point since they were
    # created with .remote() and their __init__ completed before control returned to main
    LOGGER.info("Starting training threads on all actors...")
    ray.get(policy_actor.start_training_threads.remote(
        train_dataset=train_dataset, steps_per_phase=effective_steps_per_phase,
    ))
    ray.get(rubric_actor.start_training_threads.remote(
        train_dataset=train_dataset, steps_per_phase=effective_steps_per_phase,
    ))

    for i, extra_actor in enumerate(extra_policy_actors):
        LOGGER.info("Starting training threads for extra policy actor: %s", extra_policy_names[i])
        ray.get(extra_actor.start_training_threads.remote(
            train_dataset=train_dataset, steps_per_phase=effective_steps_per_phase,
        ))

    LOGGER.info("======== ✅ All training threads started ========")

    # All actors start paused by default
    LOGGER.info(
        "Initial state: all actors paused. %d policy actors (%s) + 1 rubric actor",
        len(all_policy_actors), ", ".join(all_policy_names),
    )

    # Global training step counter for wandb logging - uses gradient update count
    # This increments for each gradient update from EITHER actor (policy or rubric)
    # This is consistent with grpo_fast.py which uses training_step as the wandb x-axis
    global_training_step = resume_from_step
    if resume_from_step > 0:
        LOGGER.info(f"Resuming alternating training from step {resume_from_step}")
    
    # Track current rubric judge actor manager (can change during curriculum)
    current_rubric_judge_actor_manager = rubric_judge_actor_manager

    def _log_actor_transition_snapshot(
        actor: ray.actor.ActorHandle,
        actor_name: str,
        label: str,
        *,
        include_engine_health: bool = False,
    ) -> None:
        """Fetch and log a transition snapshot from a trainer actor."""
        try:
            snapshot = ray.get(
                actor.get_transition_debug_state.remote(include_engine_health=include_engine_health),
                timeout=60.0,
            )
            LOGGER.info(
                "Transition snapshot [%s] %s: %s",
                label,
                actor_name,
                json.dumps(snapshot, sort_keys=True, default=str),
            )
        except Exception as e:
            LOGGER.warning(
                "Failed to capture transition snapshot [%s] for %s: %s",
                label,
                actor_name,
                e,
            )

    # Helper to pause all actors except one
    def _pause_all_except(active_actor: ray.actor.ActorHandle | None = None) -> None:
        """Pause all policy and rubric actors, except the specified active one.

        For rlcer_evolving cached-result training, policy actors stay live during
        the rubric phase so they can continue preparing the next policy batch while
        the rubric actor consumes the just-finished cached batch. The rubric actor
        still does not free-run prompts in that mode because its replenish hook is
        a no-op.
        """
        futures = []
        keep_policy_running = (
            enqueue_rlcer_evolving_cached_results
            and active_actor is rubric_actor
        )

        for actor in all_policy_actors:
            if actor is not active_actor:
                if keep_policy_running:
                    continue
                futures.append(actor.pause_training_threads.remote())

        # Always pause rubric actor when it's not the active actor.
        # Pausing training threads only stops the training loop; the actor can still
        # respond to create_rubric() calls from policy actors.
        if rubric_actor is not active_actor:
            futures.append(rubric_actor.pause_training_threads.remote())

        if futures:
            ray.get(futures)

    # Helper to train one actor for steps_per_phase steps
    def _train_actor_phase(
        actor: ray.actor.ActorHandle,
        actor_name: str,
        metric_prefix: str,
    ) -> None:
        """Train a single actor for steps_per_phase steps."""
        nonlocal global_training_step

        # In rlcer_evolving mode the rubric actor consumes pre-computed
        # cached generations — it never sends prompts to vLLM engines.
        # We skip _pause_all_except so policy actors stay live (they can
        # prepare the next batch while the rubric actor trains).  We still
        # call resume_training_threads to clear pause_event — without it
        # the data pipeline stalls because pause-gated code paths block.
        is_rlcer_rubric = enqueue_rlcer_evolving_cached_results and actor is rubric_actor

        LOGGER.info("=== Phase transition: activating %s ===", actor_name)
        _log_actor_transition_snapshot(actor, actor_name, "pre_transition")
        if is_rlcer_rubric:
            LOGGER.info(
                "rlcer_evolving rubric phase: keeping policy actors live "
                "(rubric actor consumes cached data only)"
            )
        else:
            _pause_all_except(actor)
            LOGGER.info("Paused other actors. Resuming %s...", actor_name)
        ray.get(actor.resume_training_threads.remote())
        _log_actor_transition_snapshot(actor, actor_name, "post_resume", include_engine_health=True)
        LOGGER.info("Training %s for %d steps...", actor_name, effective_steps_per_phase)

        for step in range(1, effective_steps_per_phase + 1):
            if is_rlcer_rubric:
                queued = ray.get(actor.enqueue_rlcer_evolving_cached_generations.remote(global_training_step + 1))
                if queued <= 0:
                    LOGGER.warning("Rubric step %d skipped: no cached RL-CER evolving generations", step)
                    break
            global_training_step += 1
            result = ray.get(actor.train_one_step.remote(global_training_step, train_dataset))
            if result.get("skipped"):
                LOGGER.warning("%s training step %d skipped", actor_name, step)
                break

            if enqueue_rlcer_evolving_cached_results and actor is not rubric_actor:
                expected_cached_rollouts = max(int(rlcer_expected_cached_rollouts or 0), 0)
                cached_rubric_generations = ray.get(
                    actor.drain_rlcer_evolving_cached_generations.remote(
                        expected_cached_rollouts,
                    )
                )
                if cached_rubric_generations:
                    ray.get(rubric_actor.set_rlcer_evolving_cached_generations.remote(cached_rubric_generations))

            # Collect metrics from paused actors and rubric judge
            additional_metrics = {}
            rubric_judge_metrics = _collect_rubric_judge_queue_metrics(current_rubric_judge_actor_manager)
            additional_metrics.update(rubric_judge_metrics)

            # Collect metrics from ALL paused actors
            for i, other_actor in enumerate(all_policy_actors):
                if other_actor is not actor:
                    other_metrics = _collect_actor_queue_metrics(other_actor, all_policy_names[i])
                    additional_metrics.update(other_metrics)
            if actor is not rubric_actor:
                rubric_metrics = _collect_actor_queue_metrics(rubric_actor, "rubric")
                additional_metrics.update(rubric_metrics)

            if result.get("metrics") and result.get("completed"):
                all_metrics = {**result["metrics"], **additional_metrics}
                all_metrics["training_step"] = global_training_step
                all_metrics[f"{metric_prefix}/episode"] = result.get("episode", 0)
                # Track which actor is training (useful for multi-policy)
                all_metrics["active_actor"] = actor_name
                if curriculum_state is not None:
                    all_metrics["curriculum/current_model_idx"] = curriculum_state.current_model_idx
                    all_metrics["curriculum/current_model"] = curriculum_state.models[curriculum_state.current_model_idx]
                _log_metrics_to_wandb(all_metrics, global_training_step)

                LOGGER.info(
                    f"{actor_name} training_step {global_training_step}: Active tasks - "
                    f"Rubric Judge: {additional_metrics.get('rubric_judge/actors/total_active_tasks', 0)}"
                )

    # Counter for round-robin data provider policy rotation
    _rubric_step_policy_idx = 0

    for cycle_idx in range(cycles):
        # Check if we need to swap judge model (curriculum)
        if curriculum_state is not None and curriculum_swap_callback is not None:
            new_model, needs_swap = curriculum_state.get_model_for_cycle(cycle_idx)
            if needs_swap:
                LOGGER.info(
                    "Curriculum: Switching to judge model %s at cycle %d/%d",
                    new_model, cycle_idx + 1, cycles
                )
                _pause_all_except(None)
                
                new_engines = curriculum_swap_callback(new_model)
                current_rubric_judge_actor_manager = new_engines.actor_manager
                
                # Update ALL policy actors and rubric actor with new judge engines
                update_futures = []
                for actor in all_policy_actors:
                    update_futures.append(actor.update_rubric_judge_engines.remote(
                        new_engines.engines,
                        new_engines.generate_text_actor,
                        new_engines.tokenizer,
                    ))
                update_futures.append(rubric_actor.update_rubric_judge_engines.remote(
                    new_engines.engines,
                    new_engines.generate_text_actor,
                    new_engines.tokenizer,
                ))
                ray.get(update_futures)
                LOGGER.info("Curriculum: Judge model swap complete")
        
        LOGGER.info("Starting alternating cycle %d/%d", cycle_idx + 1, cycles)

        # ---- Policy training phases ----
        if skip_policy_training:
            LOGGER.info("Skipping policy training phase (using fixed policy for rubric-only training)")
        else:
            # Train each policy actor in sequence
            for actor_idx, (actor, name) in enumerate(zip(all_policy_actors, all_policy_names)):
                _train_actor_phase(actor, name, name)

        # ---- Rubric training phase ----
        if skip_rubric_training:
            LOGGER.info("Skipping rubric training phase (using API-based rubric generator)")
            continue
        
        # Before rubric training, rotate the data provider's active policy actor
        # so the rubric sees responses from different policies across steps
        if data_provider is not None and len(all_policy_actors) > 1:
            active_policy = all_policy_actors[_rubric_step_policy_idx % len(all_policy_actors)]
            active_name = all_policy_names[_rubric_step_policy_idx % len(all_policy_actors)]
            LOGGER.info(
                "Rotating data provider to %s for rubric training (cycle %d)",
                active_name, cycle_idx + 1,
            )
            _log_actor_transition_snapshot(active_policy, active_name, "rubric_route_target")
            ray.get(data_provider.set_active_policy_actor.remote(active_policy))
            LOGGER.info(
                "Rubric data provider active policy set to %s (cycle %d)",
                active_name,
                cycle_idx + 1,
            )
            _rubric_step_policy_idx += 1
        
        _train_actor_phase(rubric_actor, "rubric", "rubric")

    metadata = {
        "cycles": cycles,
        "steps_per_phase": effective_steps_per_phase,
        "requested_steps_per_phase": steps_per_phase,
        "completed_at": _current_timestamp(),
        "skip_rubric_training": skip_rubric_training,
        "num_policy_actors": len(all_policy_actors),
        "policy_actor_names": all_policy_names,
        "force_single_step_alternation": force_single_step_alternation,
        "enqueue_rlcer_evolving_cached_results": enqueue_rlcer_evolving_cached_results,
    }
    
    # Add curriculum info to metadata
    if curriculum_state is not None:
        metadata["curriculum"] = {
            "models": curriculum_state.models,
            "schedule": curriculum_state.schedule,
            "final_model_idx": curriculum_state.current_model_idx,
        }

    return {
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Driver script functions
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> ray.LoggingConfig:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)

    if verbose:
        # Set root logger level and handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)



def _maybe_save_output(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    LOGGER.info("Saved training artefacts to %s", output_path)


def _serialise_result(result: dict[str, Any]) -> dict[str, Any]:
    """Convert dataclasses within the result to JSON-serialisable dicts."""

    serialised: dict[str, Any] = {"metadata": result["metadata"]}
    return serialised


def validate_args(args: ScriptArgs) -> None:
    """Validate parsed arguments and raise ValueError if configuration is invalid.
    
    Args:
        args: Parsed script arguments
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    grpo_args = args.grpo_args
    if not grpo_args:
        return

    effective_policy_model = args.policy_model or args.model_config.model_name_or_path
    effective_rubric_model = args.rubric_model or effective_policy_model

    valid_multi_judge_aggregations = {"average_vote", "majority_vote", "average_minus_variance", "agreement_bonus", "margin_kappa_format"}
    if args.multi_judge_aggregation not in valid_multi_judge_aggregations:
        raise ValueError(
            f"Unknown multi_judge_aggregation='{args.multi_judge_aggregation}'. "
            f"Valid options: {sorted(valid_multi_judge_aggregations)}"
        )

    valid_multi_judge_tie_breakers = {"mean_score", "first_judge"}
    if args.multi_judge_tie_breaker not in valid_multi_judge_tie_breakers:
        raise ValueError(
            f"Unknown multi_judge_tie_breaker='{args.multi_judge_tie_breaker}'. "
            f"Valid options: {sorted(valid_multi_judge_tie_breakers)}"
        )
    valid_training_modes = {"policy_rubric", "rubric_judge"}
    if args.training_mode not in valid_training_modes:
        raise ValueError(
            f"Unknown training_mode='{args.training_mode}'. "
            f"Valid options: {sorted(valid_training_modes)}"
        )

    if args.training_mode == "rubric_judge":
        if args.rejected_answer_method != "dataset_pair":
            raise ValueError(
                "training_mode='rubric_judge' requires rejected_answer_method='dataset_pair' "
                "so both judge and rubric phases train on fixed preference pairs."
            )
        if args.rubric_reward_mode != "rubric_judge":
            raise ValueError(
                "training_mode='rubric_judge' only supports rubric_reward_mode='rubric_judge'."
            )
        if args.freeze_policy_model:
            raise ValueError(
                "freeze_policy_model is not supported when training_mode='rubric_judge'."
            )
        if args.multi_policy_models:
            raise ValueError(
                "multi_policy_models is not supported when training_mode='rubric_judge'."
            )
        if args.multi_policy_coevolve_models:
            raise ValueError(
                "multi_policy_coevolve_models is not supported when training_mode='rubric_judge'."
            )

    if args.single_model_mode and args.freeze_policy_model:
        raise ValueError(
            "single_model_mode with freeze_policy_model is invalid. In single-model mode, "
            "rubric training updates the shared main model, so the policy does not stay frozen. "
            "Use a two-model topology for fixed-policy baselines."
        )
        if args.multi_judge_models:
            raise ValueError(
                "multi_judge_models is not supported when training_mode='rubric_judge'."
            )
        if args.judge_size_curriculum:
            raise ValueError(
                "judge_size_curriculum is not supported when training_mode='rubric_judge'."
            )
        if grpo_args.rubric_judge_num_engines is not None and grpo_args.rubric_judge_num_engines > 0:
            raise ValueError(
                "training_mode='rubric_judge' uses the trainable judge model directly. "
                "Do not also allocate auxiliary RUBRIC_JUDGE engines."
            )

    if args.single_model_mode:
        if args.freeze_rubric_model and not args.api_rubric_generator:
            raise ValueError(
                "single_model_mode + freeze_rubric_model is invalid. "
                "In single_model_mode, rubric generation uses the shared policy model stack, "
                "so a local rubric generator cannot remain frozen while the policy trains. "
                "Use a two-model config instead."
            )
        if effective_rubric_model != effective_policy_model:
            raise ValueError(
                "single_model_mode does not support a separate rubric model. "
                f"Got rubric_model={effective_rubric_model!r} and policy_model={effective_policy_model!r}, "
                "but rubric generation uses the shared policy model stack in single_model_mode. "
                "Use a two-model config instead."
            )
    if args.rubric_reward_mode not in [
        "rubric_judge", "rrd_uniform", "rrd_llm", "rrd_wu", "query_specific_pref", "rlcer", "rlcer_evolving",
        "direct_likert", "reference_likert", "rar_predefined", "rar_explicit", "rar_implicit", "rubric_arm",
        "random",
    ]:
        raise ValueError(
            f"Unknown rubric_reward_mode='{args.rubric_reward_mode}'. "
            "Valid options: 'rubric_judge', 'rrd_uniform', 'rrd_llm', 'rrd_wu', 'query_specific_pref', "
            "'rlcer', 'rlcer_evolving', 'direct_likert', 'reference_likert', "
            "'rar_predefined', 'rar_explicit', 'rar_implicit', 'random'"
        )
    
    # Validate inference_model_for_question_inference when using inferred_question method
    if args.rejected_answer_method == "inferred_question":
        if args.inference_model_for_question_inference is None:
            raise ValueError(
                "inference_model_for_question_inference must be explicitly set when "
                "rejected_answer_method='inferred_question'. "
                "Valid options: 'inference_engine', 'rubric_judge', 'policy'"
            )
        
        if args.inference_model_for_question_inference not in ["inference_engine", "rubric_judge", "policy"]:
            raise ValueError(
                f"Unknown inference_model_for_question_inference='{args.inference_model_for_question_inference}'. "
                f"Valid options: 'inference_engine', 'rubric_judge', 'policy'"
            )
        
        if args.inference_model_for_question_inference == "inference_engine":
            # Validate that dedicated inference engines are configured
            if not grpo_args:
                raise ValueError(
                    "inference_model_for_question_inference='inference_engine' requires grpo_args to be set."
                )
            if not (hasattr(grpo_args, "inference_num_engines") and grpo_args.inference_num_engines is not None and grpo_args.inference_num_engines > 0):
                raise ValueError(
                    "inference_model_for_question_inference='inference_engine' requires "
                    "INFERENCE_NUM_ENGINES > 0 to be set."
                )
            if not (hasattr(grpo_args, "inference_model") and grpo_args.inference_model is not None):
                raise ValueError(
                    "inference_model_for_question_inference='inference_engine' requires "
                    "INFERENCE_MODEL to be set."
                )
        
        if args.inference_model_for_question_inference == "policy":
            if not args.single_model_mode:
                raise ValueError(
                    "inference_model_for_question_inference='policy' requires single_model_mode=True."
                )
            
            # Validate tokenizer resolution for policy inference
            tokenizer_available = (
                args.tokenizer_config
                and args.tokenizer_config.tokenizer_name_or_path
            ) or (
                args.model_config
                and args.model_config.model_name_or_path
            )
            if not tokenizer_available:
                raise ValueError(
                    "Cannot determine policy tokenizer name for question inference. "
                    "Either tokenizer_config.tokenizer_name_or_path or model_config.model_name_or_path must be set."
                )
    
    # Validate rubric judge configuration
    if grpo_args.rubric_judge_num_engines is not None and grpo_args.rubric_judge_num_engines > 0:
        if not grpo_args.rubric_judge_model:
            raise ValueError(
                "rubric_judge_num_engines > 0 requires rubric_judge_model to be explicitly set."
            )
        if grpo_args.rubric_judge_tensor_parallel_size is None:
            raise ValueError(
                "rubric_judge_tensor_parallel_size must be explicitly set when rubric_judge_num_engines > 0."
            )
        if grpo_args.rubric_judge_gpu_memory_utilization is None:
            raise ValueError(
                "rubric_judge_gpu_memory_utilization must be explicitly set when rubric_judge_num_engines > 0."
            )
        if grpo_args.rubric_judge_max_model_len is None:
            raise ValueError(
                "rubric_judge_max_model_len must be explicitly set when rubric_judge_num_engines > 0."
            )
    
    # Validate inference model configuration
    if (
        grpo_args.inference_num_engines is not None
        and grpo_args.inference_num_engines > 0
        and grpo_args.inference_model is not None
    ):
        if grpo_args.inference_tensor_parallel_size is None:
            raise ValueError(
                "inference_tensor_parallel_size must be explicitly set when inference_num_engines > 0."
            )
        if grpo_args.inference_gpu_memory_utilization is None:
            raise ValueError(
                "inference_gpu_memory_utilization must be explicitly set when inference_num_engines > 0."
            )
        if grpo_args.inference_max_model_len is None:
            raise ValueError(
                "inference_max_model_len must be explicitly set when inference_num_engines > 0."
            )
    
    # Validate AuxiliaryModelConfig requirements
    # This will be checked when creating configs, but we validate here for early failure
    if (
        grpo_args.rubric_judge_num_engines is not None
        and grpo_args.rubric_judge_num_engines > 0
        and grpo_args.rubric_judge_model
    ):
        # Check tokenizer resolution
        tokenizer_available = (
            args.tokenizer_config
            and args.tokenizer_config.tokenizer_name_or_path
        ) or grpo_args.rubric_judge_model
        if not tokenizer_available:
            raise ValueError(
                "Cannot determine tokenizer name for rubric_judge. "
                "Either tokenizer_config.tokenizer_name_or_path or rubric_judge_model must be set."
            )
    
    if (
        grpo_args.inference_num_engines is not None
        and grpo_args.inference_num_engines > 0
        and grpo_args.inference_model
    ):
        # Check tokenizer resolution
        tokenizer_available = (
            args.tokenizer_config
            and args.tokenizer_config.tokenizer_name_or_path
        ) or grpo_args.inference_model
        if not tokenizer_available:
            raise ValueError(
                "Cannot determine tokenizer name for inference_model. "
                "Either tokenizer_config.tokenizer_name_or_path or inference_model must be set."
            )
    
    # Validate combined data provider configuration
    if args.rejected_answer_method == "combined":
        if args.combined_data_provider_weights is None:
            raise ValueError(
                "combined_data_provider_weights must be set when rejected_answer_method='combined'. "
                "Example: --combined-data-provider-weights 'replay_buffer:0.5,inferred_question:0.25,rubric:0.25'"
            )
        
        # Parse weights to validate format
        from scripts.rubric_data_provider import CombinedWeights
        try:
            weights = CombinedWeights.from_string(args.combined_data_provider_weights)
        except ValueError as e:
            raise ValueError(f"Invalid combined_data_provider_weights: {e}")
        
        # Validate that at least one method has positive weight
        if weights.replay_buffer <= 0 and weights.inferred_question <= 0 and weights.rubric <= 0:
            raise ValueError(
                "At least one data provider method must have positive weight in combined_data_provider_weights."
            )
        
        # Validate inferred_question requirements if it has positive weight
        if weights.inferred_question > 0:
            if args.inference_model_for_question_inference is None:
                raise ValueError(
                    "inference_model_for_question_inference must be set when using inferred_question "
                    "in combined data provider (inferred_question weight > 0). "
                    "Valid options: 'inference_engine', 'rubric_judge', 'policy'"
                )
            if args.inference_model_for_question_inference not in ["inference_engine", "rubric_judge", "policy"]:
                raise ValueError(
                    f"Unknown inference_model_for_question_inference='{args.inference_model_for_question_inference}'. "
                    f"Valid options: 'inference_engine', 'rubric_judge', 'policy'"
                )
            if args.inference_model_for_question_inference == "policy" and not args.single_model_mode:
                raise ValueError(
                    "inference_model_for_question_inference='policy' in combined mode requires "
                    "single_model_mode=True. Use 'rubric_judge' or 'inference_engine' for "
                    "separated-engine multi-policy runs."
                )
    
    # Validate multi-policy co-evolution configuration
    if args.multi_policy_coevolve_models:
        coevolve_models = [m.strip() for m in args.multi_policy_coevolve_models.split(",")]
        if len(coevolve_models) == 0:
            raise ValueError("multi_policy_coevolve_models must contain at least one model name.")
        if args.freeze_policy_model:
            raise ValueError(
                "multi_policy_coevolve_models requires freeze_policy_model=False. "
                "All policies must be trainable in co-evolution mode."
            )
        LOGGER.info(
            "Multi-policy co-evolution: main policy + %d extra policies (%s). "
            "single_model_mode=%s (main policy + rubric share weights if True)",
            len(coevolve_models), coevolve_models, args.single_model_mode,
        )


def parse_args() -> ScriptArgs:
    """Parse all arguments using HfArgumentParser (script args + GRPO args)."""
    # Parse all dataclasses together (same as grpo_fast.py)
    parser = HfArgumentParser((ScriptArgs, GrpoArgs, GrpoTokenizerConfig, GrpoModelConfig))
    script_args, grpo_args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()

    # Attach GRPO configs to script_args
    script_args.grpo_args = grpo_args  # type: ignore
    script_args.tokenizer_config = tokenizer_config  # type: ignore
    script_args.model_config = model_config  # type: ignore
    
    # Validate arguments
    validate_args(script_args)
    
    return script_args


def main() -> None:
    args = parse_args()
    _setup_logging(args.grpo_args.verbose)

    # Derive generation_examples_dir from output path if not explicitly set
    if args.generation_examples_dir is None:
        output_dir = Path(args.output).parent
        args.generation_examples_dir = str(output_dir / "generation_examples")
        LOGGER.info("Generation examples will be saved to: %s", args.generation_examples_dir)

    args.grpo_args = grpo_fast.setup_runtime_variables(args.grpo_args)

    # rubric_prompt_key lives in GrpoArgs (shared with grpo_fast.py dataset setup).
    # Log a warning if both rubric_prompt_key and system_prompt_override_file are set.
    if args.grpo_args.rubric_prompt_key != "rubric_generation":
        if args.grpo_args.system_prompt_override_file is not None:
            LOGGER.warning(
                "Both --rubric_prompt_key (%s) and --system_prompt_override_file (%s) are set. "
                "system_prompt_override_file takes precedence for dataset tokenization.",
                args.grpo_args.rubric_prompt_key,
                args.grpo_args.system_prompt_override_file,
            )

    tokenizer = grpo_fast.make_tokenizer(args.tokenizer_config, args.model_config)

    # Initialize wandb in main process
    # All actors will join this same run using the run_id
    beaker_config, _ = setup_experiment_tracking(args.grpo_args, args.tokenizer_config, args.model_config)

    # Get wandb run ID to pass to actors so they join the same run
    wandb_run_id = None
    if args.grpo_args.with_tracking:
        import wandb
        if wandb.run is not None:
            wandb_run_id = wandb.run.id
            LOGGER.info("Main process wandb run ID: %s", wandb_run_id)

    # Add explicit model configs to wandb after initialization
    if args.grpo_args.with_tracking:
        import wandb
        if wandb.run is not None:
            # Add model-specific configs that aren't in the standard GrpoArgs/ModelConfig
            additional_config = {
                "rubric_model": args.rubric_model,
                "policy_model": args.policy_model,
                "baseline_model": args.baseline_model or args.policy_model,
                "rubric_temperature": args.rubric_temperature,
                "policy_temperature": args.policy_temperature,
                "baseline_temperature": args.baseline_temperature,
                "single_model_mode": args.single_model_mode,
                "rubric_reward_mode": args.rubric_reward_mode,
                "rubric_reward_use_margin": args.rubric_reward_use_margin,
                "rubric_format_reward_weight": args.rubric_format_reward_weight,
                "multi_judge_aggregation": args.multi_judge_aggregation,
                "multi_judge_tie_breaker": args.multi_judge_tie_breaker,
            }
            # Add rubric judge (reward model) config if available
            if args.grpo_args.rubric_judge_model:
                additional_config["rubric_judge_model"] = args.grpo_args.rubric_judge_model
            if args.grpo_args.rubric_judge_num_engines is not None:
                additional_config["rubric_judge_num_engines"] = args.grpo_args.rubric_judge_num_engines
                if args.grpo_args.rubric_judge_tensor_parallel_size is not None:
                    additional_config["rubric_judge_tensor_parallel_size"] = args.grpo_args.rubric_judge_tensor_parallel_size
                if args.grpo_args.rubric_judge_max_model_len is not None:
                    additional_config["rubric_judge_max_model_len"] = args.grpo_args.rubric_judge_max_model_len
                if args.grpo_args.rubric_judge_gpu_memory_utilization is not None:
                    additional_config["rubric_judge_gpu_memory_utilization"] = args.grpo_args.rubric_judge_gpu_memory_utilization

            wandb.config.update(additional_config)

    train_dataset, eval_dataset = setup_datasets(args.grpo_args, args.tokenizer_config, tokenizer)

    if len(train_dataset) < (needed := max(args.grpo_args.async_steps, 1) * args.grpo_args.num_unique_prompts_rollout):
        raise ValueError(
            f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed to have enough prompts for bsz and prefill. Try reducing async_steps or num_unique_prompts_rollout, or increasing the dataset size."
        )

    if args.grpo_args.cache_dataset_only:
        return

    pprint([args.grpo_args, args.model_config])

    ray.init(address=args.ray_address, namespace=args.ray_namespace, runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})

    # CRITICAL: Create main training placement group FIRST (before auxiliary engines)
    # In multi-node setups, creating auxiliary engines first can fragment GPU allocation,
    # preventing the main placement group from finding contiguous resources on one node.
    # By creating the main placement group first, we reserve the required resources upfront,
    # and auxiliary engines use the remaining GPUs.

    # Shared engines for single_model_mode (created early to reserve placement group)
    shared_vllm_engines = None
    shared_policy_group = None
    shared_actor_manager = None
    shared_inference_results_Q = None
    shared_param_prompt_Q = None
    shared_evaluation_inference_results_Q = None
    shared_policy_generate_text_actor = None

    # Actor-specific result queues for single_model_mode routing
    actor_results_queues = None
    rubric_inference_results_Q = None
    policy_inference_results_Q = None

    if args.single_model_mode:
        LOGGER.info("======== Single model mode: creating main placement group FIRST (before auxiliary engines) ========")

        # Create shared queues
        base_queue_size = (args.grpo_args.async_steps + 1) * args.grpo_args.num_unique_prompts_rollout
        queue_size = base_queue_size * 2
        LOGGER.info(f"Using 2x queue size for single_model_mode: {queue_size} (base: {base_queue_size})")
        shared_inference_results_Q = ray_queue.Queue(maxsize=queue_size)
        shared_param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
        shared_evaluation_inference_results_Q = ray_queue.Queue()

        # Create actor-specific result queues for proper routing
        rubric_inference_results_Q = ray_queue.Queue(maxsize=queue_size)
        policy_inference_results_Q = ray_queue.Queue(maxsize=queue_size)
        actor_results_queues = {
            "rubric": rubric_inference_results_Q,
            "policy": policy_inference_results_Q,
        }
        LOGGER.info("Created actor-specific result queues for routing: %s", list(actor_results_queues.keys()))

        # Create main model and placement group (suppress judge engine creation here)
        (
            shared_policy_group,
            shared_vllm_engines,
            _,  # tool_objects
            _,  # resume_training_step
            _,  # episode
            shared_actor_manager,
            _,  # rubric_judge_engines (suppressed)
            shared_policy_generate_text_actor,
            _,  # rubric_judge_generate_text_actor (suppressed)
        ) = create_model_and_optimizer(
            args.grpo_args,
            args.tokenizer_config,
            args.model_config,
            beaker_config,
            None,  # wandb_url
            tokenizer,
            shared_inference_results_Q,
            shared_param_prompt_Q,
            shared_evaluation_inference_results_Q,
            suppress_judge_engine_initialization=True,
            placement_group_name="shared_model_pg",
            actor_results_queues=actor_results_queues,
        )
        LOGGER.info("======== ✅ Main placement group created successfully ========")

    # Now create auxiliary engines (judges, multi-policy, etc.) using remaining resources
    # Create rubric judge engines if specified
    rubric_judge_engines_obj: AuxiliaryModelEngines | None = None
    if (
        args.grpo_args
        and args.grpo_args.rubric_judge_num_engines is not None
        and args.grpo_args.rubric_judge_num_engines > 0
    ):
        rubric_judge_config = AuxiliaryModelConfig(
            model_name=args.grpo_args.rubric_judge_model,
            num_engines=args.grpo_args.rubric_judge_num_engines,
            tensor_parallel_size=args.grpo_args.rubric_judge_tensor_parallel_size,
            gpu_memory_utilization=args.grpo_args.rubric_judge_gpu_memory_utilization,
            max_model_len=args.grpo_args.rubric_judge_max_model_len,
            is_rubric_judge_engine=True,
            name="rubric_judge",
            tokenizer_name=args.grpo_args.rubric_judge_tokenizer,
        )
        rubric_judge_engines_obj = create_auxiliary_model_engines(
            rubric_judge_config,
            args.grpo_args,
            args.tokenizer_config,
        )
    
    # Extract components for backward compatibility
    rubric_judge_engines = rubric_judge_engines_obj.engines if rubric_judge_engines_obj else None
    rubric_judge_generate_text_actor = rubric_judge_engines_obj.generate_text_actor if rubric_judge_engines_obj else None
    rubric_judge_tokenizer = rubric_judge_engines_obj.tokenizer if rubric_judge_engines_obj else None
    rubric_judge_actor_manager = rubric_judge_engines_obj.actor_manager if rubric_judge_engines_obj else None

    # Create multi-judge engines if specified (Milestone 1)
    multi_judge_engines: MultiJudgeEngines | None = None
    if args.multi_judge_models:
        LOGGER.info("======== Creating multi-judge engines ========")
        multi_judge_engines = create_multi_judge_engines(
            script_args=args,
            grpo_args=args.grpo_args,
            tokenizer_config=args.tokenizer_config,
        )
        if multi_judge_engines:
            LOGGER.info(
                "✅ Multi-judge engines created: %d judges (%s), mode=%s, tie_breaker=%s",
                len(multi_judge_engines.judge_models),
                ", ".join(multi_judge_engines.judge_models),
                multi_judge_engines.aggregation_mode,
                multi_judge_engines.tie_breaker,
            )
            # Log to wandb
            if args.grpo_args.with_tracking:
                import wandb
                if wandb.run is not None:
                    wandb.config.update({
                        "multi_judge_models": args.multi_judge_models,
                        "multi_judge_num_engines_per_judge": args.multi_judge_num_engines_per_judge,
                        "multi_judge_aggregation": args.multi_judge_aggregation,
                        "multi_judge_tie_breaker": args.multi_judge_tie_breaker,
                        "multi_judge_alpha": args.multi_judge_alpha,
                        "multi_judge_beta": args.multi_judge_beta,
                        "multi_judge_margin_weight": args.multi_judge_margin_weight,
                        "multi_judge_format_weight": args.multi_judge_format_weight,
                        "multi_judge_kappa_weight": args.multi_judge_kappa_weight,
                    })

    # Create multi-policy frozen engines if specified (Milestone 2)
    # NOTE: Frozen engines are created whenever multi_policy_models is set, regardless of
    # freeze_policy_model. This enables co-evolve + multi-policy mode where the main policy
    # trains normally while frozen engines from other models provide diverse responses for
    # rubric training (via multi_policy_frozen data provider).
    multi_policy_frozen_engines: MultiPolicyFrozenEngines | None = None
    if args.multi_policy_models:
        LOGGER.info("======== Creating multi-policy frozen engines ========")
        multi_policy_frozen_engines = create_multi_policy_frozen_engines(
            script_args=args,
            grpo_args=args.grpo_args,
            tokenizer_config=args.tokenizer_config,
        )
        if multi_policy_frozen_engines:
            LOGGER.info(
                "✅ Multi-policy frozen engines created: %d policies (%s)",
                len(multi_policy_frozen_engines.policy_models),
                ", ".join(multi_policy_frozen_engines.policy_models),
            )
            # Log to wandb
            if args.grpo_args.with_tracking:
                import wandb

                if wandb.run is not None:
                    wandb.config.update(
                        {
                            "multi_policy_models": args.multi_policy_models,
                            "multi_policy_num_engines_per_model": args.multi_policy_num_engines_per_model,
                            "multi_policy_sampling_strategy": args.multi_policy_sampling_strategy,
                        }
                    )

    # Create inference model engines if specified
    inference_model_engines_obj: AuxiliaryModelEngines | None = None
    if (
        args.grpo_args
        and args.grpo_args.inference_num_engines is not None
        and args.grpo_args.inference_num_engines > 0
        and args.grpo_args.inference_model is not None
    ):
        inference_model_config = AuxiliaryModelConfig(
            model_name=args.grpo_args.inference_model,
            num_engines=args.grpo_args.inference_num_engines,
            tensor_parallel_size=args.grpo_args.inference_tensor_parallel_size,
            gpu_memory_utilization=args.grpo_args.inference_gpu_memory_utilization,
            max_model_len=args.grpo_args.inference_max_model_len,
            is_rubric_judge_engine=False,
            name="inference_model",
            # Auxiliary models should use their own tokenizer, not the policy tokenizer
            tokenizer_name=None,
        )
        inference_model_engines_obj = create_auxiliary_model_engines(
            inference_model_config,
            args.grpo_args,
            args.tokenizer_config,
        )

    try:
        # Calculate resources for rubric trainer
        rubric_resources = BaseTrainerActor._calculate_resources(args.grpo_args)
        LOGGER.info(
            "RubricTrainerActor resources: %.1f GPUs, %d CPUs",
            rubric_resources["num_gpus"],
            rubric_resources["num_cpus"],
        )

        LOGGER.info("======== Creating RubricTrainerActor ========")
        rubric_actor = RubricTrainerActor.options(
            name="rubric_trainer",
            num_gpus=0,
            num_cpus=1,
            max_concurrency=1000,  # Enable concurrent execution for async methods like create_rubric
        ).remote(
            rubric_model=args.rubric_model,
            generation_kwargs={"temperature": args.rubric_temperature, "n": 1},
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            policy_generate_text_actor=shared_policy_generate_text_actor if args.single_model_mode else None,
            grpo_args=args.grpo_args,
            tokenizer_config=args.tokenizer_config,
            model_config=args.model_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            single_model_mode=args.single_model_mode,
            generation_examples_dir=args.generation_examples_dir,
            num_examples_to_log=args.num_examples_to_log,
            log_examples_every_n_steps=args.log_examples_every_n_steps,
            inference_model_engines_obj=inference_model_engines_obj,
            rubric_judge_tokenizer=rubric_judge_tokenizer,
            script_args=args,  # Pass ScriptArgs so data provider can access inference_model_for_question_inference
            api_rubric_generator=args.api_rubric_generator,
            multi_judge_engines=multi_judge_engines,  # Multi-judge support (Milestone 1)
            reward_mode=args.rubric_reward_mode,
            rubric_reward_use_margin=args.rubric_reward_use_margin,
            rubric_format_reward_weight=args.rubric_format_reward_weight,
            rubric_prompt_key=args.grpo_args.rubric_prompt_key,
        )
        # Wait for rubric actor to finish __init__ before creating policy actor
        # This prevents CPU memory exhaustion from simultaneous model loading
        LOGGER.info("Waiting for RubricTrainerActor initialization to complete...")
        ray.get(rubric_actor.__ray_ready__.remote())
        LOGGER.info("======== ✅ RubricTrainerActor initialized ========")

        # Whether to skip rubric model gradient updates in the alternating loop.
        # Plain "rlcer" can still train the rubric model, but it uses the
        # existing dataset answer-pair pipeline rather than rollout-buffer data.
        # "rlcer_evolving" keeps its separate rollout-buffer rubric reward.
        #
        # RRD and RaR paper baselines compute rewards directly in the policy
        # actor and do not update rubric-model weights here.
        skip_rubric_training = (
            bool(args.api_rubric_generator)
            or args.freeze_rubric_model
            or args.rubric_reward_mode.startswith("rrd_")
            or args.rubric_reward_mode == "query_specific_pref"
            or args.rubric_reward_mode in (
                "direct_likert",
                "reference_likert",
                "rar_predefined",
                "rar_implicit",
                "rar_explicit",
                "random",
            )
            # Notes:
            # - "rlcer" is intentionally absent — rubric trains on existing data.
            # - "rlcer_evolving" is intentionally absent — rubric trains on rollout data.
        )

        # Whether to use the standalone policy actor (computes its own rewards
        # independently of the rubric→judge flow used by PolicyTrainerActor).
        _no_rubric_model_modes = {
            "direct_likert",
            "reference_likert",
            "rar_predefined",
            "rar_implicit",
            "rar_explicit",
            "query_specific_pref",
            "random",
        }
        use_standalone_policy_actor = (
            (skip_rubric_training and args.rubric_reward_mode in _no_rubric_model_modes)
            or args.rubric_reward_mode in _no_rubric_model_modes
            or args.rubric_reward_mode.startswith("rrd_")
            or args.rubric_reward_mode.startswith("rlcer")
            or args.rubric_reward_mode == "rubric_arm"
        )

        policy_resources = BaseTrainerActor._calculate_resources(args.grpo_args)
        LOGGER.info(
            "PolicyTrainerActor requiring cluster resources: %.1f GPUs, %d CPUs",
            policy_resources["num_gpus"],
            policy_resources["num_cpus"],
        )

        policy_actor_cls = PolicyTrainerNoRubricModelActor if use_standalone_policy_actor else PolicyTrainerActor
        policy_actor_cls_name = "PolicyTrainerNoRubricModelActor" if use_standalone_policy_actor else "PolicyTrainerActor"
        LOGGER.info(
            "Using policy actor class: %s (skip_rubric_training=%s)",
            policy_actor_cls_name,
            skip_rubric_training,
        )

        LOGGER.info("======== Creating PolicyTrainerActor ========")
        policy_actor = policy_actor_cls.options(
            name="policy_trainer",
            num_gpus=0,
            num_cpus=1,
            max_concurrency=1000,  # Enable concurrent execution for async methods
        ).remote(
            policy_model=args.policy_model,
            baseline_model=args.baseline_model,
            policy_generation_kwargs={"temperature": args.policy_temperature, "n": 1},
            baseline_generation_kwargs={"temperature": args.baseline_temperature, "n": 1},
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            policy_generate_text_actor=shared_policy_generate_text_actor if args.single_model_mode else None,
            grpo_args=args.grpo_args,
            tokenizer_config=args.tokenizer_config,
            model_config=args.model_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            single_model_mode=args.single_model_mode,
            multi_judge_engines=multi_judge_engines,
            generation_examples_dir=args.generation_examples_dir,
            num_examples_to_log=args.num_examples_to_log,
            log_examples_every_n_steps=args.log_examples_every_n_steps,
            reward_mode=args.rubric_reward_mode,
            rubric_prompt_key=args.grpo_args.rubric_prompt_key,
        )
        # Wait for policy actor to finish __init__ before creating extra policy actors
        LOGGER.info("Waiting for PolicyTrainerActor initialization to complete...")
        ray.get(policy_actor.__ray_ready__.remote())
        LOGGER.info("======== ✅ PolicyTrainerActor initialized ========")

        # ---- Create extra co-evolving policy actors ----
        extra_policy_actors = []
        extra_policy_names = []
        if args.multi_policy_coevolve_models:
            coevolve_models = [m.strip() for m in args.multi_policy_coevolve_models.split(",")]
            LOGGER.info(
                "======== Creating %d extra co-evolving policy actors: %s ========",
                len(coevolve_models), coevolve_models,
            )

            # Parse per-model learner config (must be list[int] for placement group bundles)
            coevolve_learners_str = args.multi_policy_coevolve_num_learners_per_node
            if "," in coevolve_learners_str:
                coevolve_learners = [int(x) for x in coevolve_learners_str.split(",")]
            else:
                # Single int: replicate across the same number of nodes as the main policy
                # so that STRICT_SPREAD places one bundle per node
                n_nodes = len(args.grpo_args.num_learners_per_node)
                coevolve_learners = [int(coevolve_learners_str)] * n_nodes

            for model_idx, model_name in enumerate(coevolve_models):
                # Create a deep copy of GrpoArgs with modified resources
                extra_grpo_args = copy.deepcopy(args.grpo_args)
                extra_grpo_args.vllm_num_engines = args.multi_policy_coevolve_vllm_engines_per_model
                extra_grpo_args.num_learners_per_node = coevolve_learners

                # Create a deep copy of model config with the extra model
                extra_model_config = copy.deepcopy(args.model_config)
                extra_model_config.model_name_or_path = model_name

                # Create a deep copy of tokenizer config for the extra model
                extra_tokenizer_config = copy.deepcopy(args.tokenizer_config)
                extra_tokenizer_config.tokenizer_name_or_path = model_name

                # CRITICAL: Clear the cached tokenizer property to force reloading with the new model
                # The @cached_property is copied by deepcopy, causing all actors to share the same tokenizer!
                if hasattr(extra_tokenizer_config, "_tokenizer"):
                    delattr(extra_tokenizer_config, "_tokenizer")
                if hasattr(extra_tokenizer_config, "__dict__") and "tokenizer" in extra_tokenizer_config.__dict__:
                    del extra_tokenizer_config.__dict__["tokenizer"]

                # Use a unique output dir for checkpoints
                base_output = args.grpo_args.output_dir
                model_short_name = model_name.split("/")[-1].lower().replace("-", "_")
                extra_grpo_args.output_dir = f"{base_output}_{model_short_name}"

                # Human-readable name for logging
                policy_name = f"policy_{model_short_name}"

                LOGGER.info(
                    "Creating extra policy actor %d: model=%s, vllm_engines=%d, learners=%s, output=%s",
                    model_idx + 1, model_name,
                    extra_grpo_args.vllm_num_engines,
                    coevolve_learners,
                    extra_grpo_args.output_dir,
                )

                extra_actor = PolicyTrainerActor.options(
                    name=f"policy_trainer_{model_short_name}",
                    num_gpus=0,
                    num_cpus=1,
                    max_concurrency=1000,
                ).remote(
                    policy_model=model_name,
                    baseline_model=model_name,  # Each policy is its own baseline
                    policy_generation_kwargs={"temperature": args.policy_temperature, "n": 1},
                    baseline_generation_kwargs={"temperature": args.baseline_temperature, "n": 1},
                    rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                    policy_generate_text_actor=None,  # Extra policies don't share engines
                    grpo_args=extra_grpo_args,
                    tokenizer_config=extra_tokenizer_config,
                    model_config=extra_model_config,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    single_model_mode=False,  # Extra policies always have independent models
                    multi_judge_engines=multi_judge_engines,
                    generation_examples_dir=args.generation_examples_dir,
                    num_examples_to_log=args.num_examples_to_log,
                    log_examples_every_n_steps=args.log_examples_every_n_steps,
                    rubric_prompt_key=args.grpo_args.rubric_prompt_key,
                )

                # Wait for this extra actor to finish __init__ before creating the next one
                # This prevents CPU memory exhaustion from simultaneous model loading
                LOGGER.info("Waiting for extra policy actor %d (%s) initialization...", model_idx + 1, policy_name)
                ray.get(extra_actor.__ray_ready__.remote())

                extra_policy_actors.append(extra_actor)
                extra_policy_names.append(policy_name)
                LOGGER.info("======== ✅ Extra policy actor %d (%s) initialized ========", model_idx + 1, policy_name)

            LOGGER.info(
                "======== ✅ Created %d extra co-evolving policy actors ========",
                len(extra_policy_actors),
            )

        # Set shared engines if in single_model_mode
        if args.single_model_mode and shared_vllm_engines is not None:
            LOGGER.info("Setting shared engines on both actors...")
            # Each actor gets its own result queue for proper routing (no busy-wait)
            ray.get([
                rubric_actor.set_shared_engines.remote(
                    shared_vllm_engines,
                    shared_policy_group,
                    shared_actor_manager,
                    args.grpo_args,
                    args.tokenizer_config,
                    args.model_config,
                    tokenizer,
                    rubric_inference_results_Q,  # Actor-specific queue
                    shared_param_prompt_Q,
                    shared_evaluation_inference_results_Q,
                ),
                policy_actor.set_shared_engines.remote(
                    shared_vllm_engines,
                    shared_policy_group,
                    shared_actor_manager,
                    args.grpo_args,
                    args.tokenizer_config,
                    args.model_config,
                    tokenizer,
                    policy_inference_results_Q,  # Actor-specific queue
                    shared_param_prompt_Q,
                    shared_evaluation_inference_results_Q,
                ),
            ])
            LOGGER.info("======== ✅ Shared engines set on both actors ========")

        if rubric_judge_engines is not None:
            ray_get_with_progress(
                [engine.ready.remote() for engine in rubric_judge_engines],
                "Initializing rubric judge vLLM engines",
                timeout=1200  # 20 minutes timeout for engine initialization
            )
            LOGGER.info("======== ✅ rubric judge vLLM engines initialized for joint training ========")

        # Set cross-references between actors
        ray.get(policy_actor.set_rubric_actor.remote(rubric_actor))
        ray.get(rubric_actor.set_policy_actor.remote(policy_actor))
        
        # Create data provider Ray actor in main and pass to rubric actor
        # This must happen after set_policy_actor since data provider needs policy_actor
        question_key = getattr(args.grpo_args, "question_key", "question") if args.grpo_args else "question"
        LOGGER.info(f"Creating data provider Ray actor with method={args.rejected_answer_method}")

        # In two-model mode, get the rubric model's GenerateTextActor so the data
        # provider can generate rubrics from the evolving rubric model directly.
        # We must NOT pass rubric_actor itself, because it is a single-threaded Ray
        # actor whose main thread is blocked in the training loop — calling
        # rubric_actor.create_rubric.remote() from the data provider would deadlock.
        if args.single_model_mode:
            rubric_gen_text_actor = shared_policy_generate_text_actor
        else:
            rubric_gen_text_actor = ray.get(rubric_actor.get_generate_text_actor.remote())
            LOGGER.info(
                "[two-model mode] Using rubric model's GenerateTextActor for data provider rubric generation"
            )

        data_provider = create_data_provider(
            policy_actor=policy_actor,
            rejected_answer_method=args.rejected_answer_method,
            args=args,
            question_key=question_key,
            rubric_actor=rubric_actor,
            inference_model_engines_obj=inference_model_engines_obj,
            rubric_judge_tokenizer=rubric_judge_tokenizer,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            policy_generate_text_actor=rubric_gen_text_actor,
            multi_policy_frozen_engines=multi_policy_frozen_engines,
            multi_judge_engines=multi_judge_engines,
            rubric_prompt_key=args.grpo_args.rubric_prompt_key,
        )
        ray.get(rubric_actor.set_data_provider.remote(data_provider))
        
        # Parse judge size curriculum if specified
        curriculum_state = parse_judge_curriculum(
            args.judge_size_curriculum,
            args.judge_curriculum_schedule,
            args.cycles,
        )
        
        # Create curriculum swap callback if curriculum is active
        curriculum_swap_callback = None
        current_rubric_judge_engines_obj = rubric_judge_engines_obj
        
        if curriculum_state is not None:
            # Store parameters needed for engine creation in a closure
            def make_swap_callback():
                # Capture these parameters in closure
                grpo_args = args.grpo_args
                tokenizer_config_closure = args.tokenizer_config
                num_engines = args.grpo_args.rubric_judge_num_engines
                tp_size = args.grpo_args.rubric_judge_tensor_parallel_size
                gpu_util = args.grpo_args.rubric_judge_gpu_memory_utilization
                max_len = args.grpo_args.rubric_judge_max_model_len
                
                # Track current engines for cleanup
                engines_holder = {"current": current_rubric_judge_engines_obj}
                
                def swap_callback(new_model_name: str) -> AuxiliaryModelEngines:
                    nonlocal rubric_judge_actor_manager
                    new_engines = swap_rubric_judge_model(
                        current_engines=engines_holder["current"],
                        new_model_name=new_model_name,
                        grpo_args=grpo_args,
                        tokenizer_config=tokenizer_config_closure,
                        num_engines=num_engines,
                        tensor_parallel_size=tp_size,
                        gpu_memory_utilization=gpu_util,
                        max_model_len=max_len,
                    )
                    engines_holder["current"] = new_engines
                    return new_engines
                
                return swap_callback
            
            curriculum_swap_callback = make_swap_callback()
            LOGGER.info("Judge size curriculum enabled: %s", curriculum_state.models)
        
        result = run_alternating_training(
            policy_actor=policy_actor,
            rubric_actor=rubric_actor,
            train_dataset=train_dataset,
            cycles=args.cycles,
            steps_per_phase=args.steps_per_phase,
            use_both_models=args.use_both_models,
            rubric_judge_actor_manager=rubric_judge_actor_manager,
            curriculum_state=curriculum_state,
            curriculum_swap_callback=curriculum_swap_callback,
            skip_rubric_training=skip_rubric_training,
            skip_policy_training=args.freeze_policy_model,
            extra_policy_actors=extra_policy_actors if extra_policy_actors else None,
            extra_policy_names=extra_policy_names if extra_policy_names else None,
            data_provider=data_provider,
            resume_from_step=args.grpo_args.resume_from_step,
            force_single_step_alternation=args.rubric_reward_mode == "rlcer_evolving",
            enqueue_rlcer_evolving_cached_results=args.rubric_reward_mode == "rlcer_evolving",
            rlcer_expected_cached_rollouts=(
                max(int(args.grpo_args.num_unique_prompts_rollout), 1)
                * max(int(args.grpo_args.num_samples_per_prompt_rollout), 1)
            ),
        )
        serialised = _serialise_result(result)
        _maybe_save_output(Path(args.output), serialised)
    finally:
        # Cleanup multi-judge engines if created
        if "multi_judge_engines" in locals() and multi_judge_engines is not None:
            LOGGER.info("Shutting down multi-judge engines...")
            shutdown_multi_judge_engines(multi_judge_engines)

        # Cleanup multi-policy frozen engines if created
        if "multi_policy_frozen_engines" in locals() and multi_policy_frozen_engines is not None:
            LOGGER.info("Shutting down multi-policy frozen engines...")
            shutdown_multi_policy_frozen_engines(multi_policy_frozen_engines)

        LOGGER.info("Shutting down Ray runtime.")
        ray.shutdown()


if __name__ == "__main__":
    main()
