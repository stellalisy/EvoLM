"""Data provider for creating accepted/rejected answer pairs for rubric training.

This module provides a unified interface for generating accepted and rejected
answer pairs using different methods (replay_buffer, inferred_question, rubric).
"""

from __future__ import annotations

import asyncio
import collections
import logging
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import ray
import vllm
from open_instruct.ground_truth_utils import remove_thinking_section
from open_instruct.grpo_fast import create_generation_configs
from open_instruct.search_rewards.utils.rubric_chat_templates import format_messages

LOGGER = logging.getLogger(__name__)


def build_rubric_sampling_params(args: Any) -> vllm.SamplingParams:
    """Build sampling params for rubric generation, derived from the training generation config."""
    generation_configs = create_generation_configs(args)
    params = generation_configs["train"].clone()
    params.n = 1
    return params


def build_judge_sampling_params(args: Any) -> vllm.SamplingParams:
    """Build sampling params for judge scoring, using rubric_judge_* config fields."""
    generation_configs = create_generation_configs(args)
    params = generation_configs["train"].clone()
    params.n = 1
    params.temperature = getattr(args, "rubric_judge_temperature", 0.6)
    params.max_tokens = getattr(args, "rubric_judge_max_tokens", 16384)
    return params


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer data provider."""
    size: int = 2048
    """Maximum size of the replay buffer for storing past policy rollouts."""
    min_age: int = 0
    """Minimum step age for sampling from replay buffer (0 = include most recent)."""
    max_age: int | None = None
    """Maximum step age for sampling from replay buffer (None = no limit)."""
    
    @classmethod
    def from_args(cls, args) -> "ReplayBufferConfig":
        """
        Create a ReplayBufferConfig from an Args object by automatically matching field names.
        Maps args fields (replay_buffer_size, replay_buffer_min_age, replay_buffer_max_age)
        to config fields (size, min_age, max_age).
        """
        # Map args field names to config field names
        field_mapping = {
            "replay_buffer_size": "size",
            "replay_buffer_min_age": "min_age",
            "replay_buffer_max_age": "max_age",
        }
        
        matching_kwargs = {}
        for args_field, config_field in field_mapping.items():
            if hasattr(args, args_field):
                matching_kwargs[config_field] = getattr(args, args_field)
        
        return cls(**matching_kwargs)


@dataclass
class InferenceEngineConfig:
    """Configuration for inference engine used in question inference."""
    model_for_question_inference: str | None = None
    """Model to use for question inference when rejected_answer_method='inferred_question'. 
    Options:
    - 'inference_engine': Use dedicated inference model engines (requires INFERENCE_MODEL and INFERENCE_NUM_ENGINES)
    - 'rubric_judge': Use rubric judge model
    - 'policy': Use policy model (requires single_model_mode=True)
    Must be explicitly set. Raises ValueError if not configured or if required resources are unavailable."""
    
    @classmethod
    def from_args(cls, args) -> "InferenceEngineConfig":
        """
        Create an InferenceEngineConfig from an Args object by automatically matching field names.
        Maps args field (inference_model_for_question_inference) to config field (model_for_question_inference).
        """
        # Map args field name to config field name
        field_mapping = {
            "inference_model_for_question_inference": "model_for_question_inference",
        }
        
        matching_kwargs = {}
        for args_field, config_field in field_mapping.items():
            if hasattr(args, args_field):
                matching_kwargs[config_field] = getattr(args, args_field)
        
        return cls(**matching_kwargs)


def _create_inference_components(
    args: Any,
    inference_model_engines_obj: Any | None = None,
    rubric_judge_tokenizer: Any | None = None,
    inference_model_for_question_inference: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    policy_generate_text_actor: Any | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Create inference components for question inference based on available resources.
    
    This function localizes the creation of inference components that were previously
    created in main() and passed through multiple layers. It determines which model
    to use for question inference based on configuration and available resources.
    
    Priority order:
    1. If inference_model_for_question_inference='inference_engine': Use dedicated inference model engines
       (requires INFERENCE_MODEL and INFERENCE_NUM_ENGINES to be configured)
    2. If inference_model_for_question_inference='rubric_judge': Use rubric judge model
    3. If inference_model_for_question_inference='policy': Use policy model (requires single_model_mode=True)
    
    Raises ValueError if inference_model_for_question_inference is not set or if required resources are unavailable.
    
    Args:
        args: ScriptArgs object containing grpo_args and other configuration
        inference_model_engines_obj: Optional AuxiliaryModelEngines object for dedicated inference engines
        rubric_judge_tokenizer: Optional tokenizer for rubric judge model (kept for backward compatibility but not used)
        inference_model_for_question_inference: Which model to use for question inference
        rubric_judge_generate_text_actor: GenerateTextActor for rubric judge model
        policy_generate_text_actor: GenerateTextActor for policy model (for single_model_mode)
        
    Returns:
        Tuple of (inference_generate_text_actor, inference_generation_kwargs)
        Note: Tokenizer is no longer returned as generate_text_from_messages handles tokenization internally
    """
    inference_generate_text_actor = None
    inference_generation_kwargs = {}

    # Get grpo_args from args for generation config creation
    grpo_args = getattr(args, "grpo_args", None)

    # Create generation_configs from grpo_args
    # This is done here because args is passed directly instead of trying to access Ray actor attributes
    if grpo_args is not None:
        generation_configs = create_generation_configs(grpo_args)
        # Clone the train config and set n=1 for generate_text_from_messages
        # (which only supports n=1 since it returns a single string)
        inference_generation_kwargs = generation_configs["train"].clone()
        inference_generation_kwargs.n = 1
        LOGGER.info(
            "[_create_inference_components] Created generation_configs from grpo_args (set n=1 for inference)",
        )
    else:
        raise RuntimeError(
            "No grpo_args found in args. Cannot create generation_configs. "
            f"args type={type(args)}, hasattr(args, 'grpo_args')={hasattr(args, 'grpo_args')}"
        )
    
    # Log diagnostic information
    LOGGER.info(
        "[_create_inference_components] Diagnostic info: "
        "inference_model_for_question_inference=%s, "
        "inference_model_engines_obj=%s, rubric_judge_generate_text_actor=%s, policy_generate_text_actor=%s",
        inference_model_for_question_inference,
        inference_model_engines_obj is not None,
        rubric_judge_generate_text_actor is not None,
        policy_generate_text_actor is not None,
    )
    
    # inference_model_for_question_inference is guaranteed to be set and valid by validate_args()
    inference_model = inference_model_for_question_inference
    
    # If inference_model is None, we can't proceed - this should have been caught by validation
    if inference_model is None:
        LOGGER.error(
            "[_create_inference_components] inference_model_for_question_inference is None! "
            "This should have been caught by validate_args()."
        )
        raise RuntimeError(
            "inference_model_for_question_inference is None. "
            "This must be set when rejected_answer_method='inferred_question'. "
            "It should be passed to _create_inference_components."
        )
    
    # Priority 1: Use dedicated inference model engines if explicitly requested
    if inference_model == "inference_engine":
        # Dedicated engines are required when inference_model='inference_engine'
        # Validation ensures they're configured, but we still need to check availability
        if inference_model_engines_obj is None:
            raise RuntimeError(
                "inference_model_for_question_inference='inference_engine' but inference_model_engines_obj is None. "
                "This may indicate that engine creation failed or engines were not properly initialized."
            )
        inference_generate_text_actor = inference_model_engines_obj.generate_text_actor
        if inference_generate_text_actor is None:
            raise RuntimeError(
                "inference_model_for_question_inference='inference_engine' but inference_model_engines_obj.generate_text_actor is None. "
                "This may indicate that the GenerateTextActor was not properly created during engine initialization."
            )
        LOGGER.info(
            "Using dedicated inference model engines for question inference (model=%s, engines=%d)",
            grpo_args.inference_model if grpo_args else "unknown",
            grpo_args.inference_num_engines if grpo_args else 0,
        )
    # Priority 2: Use rubric judge model if requested
    elif inference_model == "rubric_judge":
        if rubric_judge_generate_text_actor is None:
            raise RuntimeError(
                "inference_model_for_question_inference='rubric_judge' but rubric judge model is not available. "
                "This may indicate that rubric_judge_num_engines was not set or engines failed to initialize."
            )
        LOGGER.info("Using rubric judge model for question inference")
        inference_generate_text_actor = rubric_judge_generate_text_actor
    # Priority 3: Use policy model if requested
    elif inference_model == "policy":
        # single_model_mode is guaranteed to be True by validate_args()
        if policy_generate_text_actor is None:
            raise RuntimeError(
                "inference_model_for_question_inference='policy' but policy_generate_text_actor is not available. "
                "This may indicate that engines failed to initialize."
            )
        LOGGER.info("Using policy model (shared) for question inference")
        inference_generate_text_actor = policy_generate_text_actor
    
    return inference_generate_text_actor, inference_generation_kwargs


@dataclass
class AnswerPair:
    """Container for accepted and rejected answer pair."""
    question: str
    accepted_answer: str
    rejected_answer: str
    
    def get_extra_fields(self) -> dict[str, Any]:
        """Get provider-specific extra fields for logging.
        
        Returns:
            Empty dict for base AnswerPair
        """
        return {}


@dataclass
class ReplayBufferAnswerPair(AnswerPair):
    """Extended pair for replay buffer method that includes step gap information."""
    rejected_step: int = 0  # The policy step when the rejected sample was generated
    current_step: int = 0   # The current policy step
    step_gap: int = 0       # Difference: current_step - rejected_step
    
    def get_extra_fields(self) -> dict[str, Any]:
        """Get provider-specific extra fields for logging.
        
        Returns:
            Dict with step gap information
        """
        return {
            "rejected_step": self.rejected_step,
            "current_step": self.current_step,
            "step_gap": self.step_gap,
        }


class BaseDataProvider(ABC):
    """Base class for data providers that create accepted/rejected answer pairs."""
    
    def __init__(
        self,
        policy_actor: Any,
        question_key: str = "question",
    ):
        """Initialize the base data provider.
        
        Args:
            policy_actor: Ray actor handle for policy model (must have _generate_policy_answer
                         and _generate_rejected_with_inferred_question methods)
            question_key: Key to extract question from example dicts
        """
        self.policy_actor = policy_actor
        self.question_key = question_key
    
    def set_active_policy_actor(self, policy_actor: Any) -> None:
        """Switch which policy actor is used for generating answers.
        
        Used in multi-policy co-evolution to rotate between different policy models
        during rubric training, so the rubric sees responses from all policies.
        
        Args:
            policy_actor: Ray actor handle for the policy model to use
        """
        self.policy_actor = policy_actor
    
    def add_experience(self, question: str, answer: str, step: int = 0) -> None:
        """Add a policy rollout (optional, may be no-op for some providers).
        
        Args:
            question: The question that was answered
            answer: The policy's answer
            step: The policy training step when this experience was generated
        """
        pass  # Default implementation does nothing
    
    @abstractmethod
    def get_pair_context(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Get context/inputs needed for creating the answer pair synchronously.
        
        This method returns immediately with the data needed to create the answer pair.
        Call this first to get context, then call create_answer_pair separately.
        
        For providers that override the question (e.g., replay buffer), this returns
        the overridden question. For other providers, this may return None.
        
        Args:
            example: Example dict containing question and other data
            
        Returns:
            Dict with context data (e.g., {"question": buffer_question, "rejected_answer": ...}),
            or None if no context/overrides needed.
        """
        pass
    
    @abstractmethod
    async def create_answer_pair(self, example: dict[str, Any], pair_context: dict[str, Any] | None = None) -> AnswerPair:
        """Create an AnswerPair asynchronously.
        
        This method is async and creates the answer pair by awaiting internal async operations.
        When called on a Ray actor with .remote(), it returns an ObjectRef that can be
        awaited later.
        
        Args:
            example: Example dict containing question and other data
            pair_context: Optional dict from get_pair_context (some providers require this)
            
        Returns:
            The created AnswerPair
        """
        pass
    
    
    def format_log_message(
        self,
        index: int,
        total: int,
        question: str,
        rubric_length: int,
        accepted_answer_length: int,
        rejected_answer_length: int,
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        """Format a log message for reward task creation.
        
        Args:
            index: Task index (1-based)
            total: Total number of tasks
            question: Original question
            rubric_length: Length of rubric text
            accepted_answer_length: Length of accepted answer
            rejected_answer_length: Length of rejected answer
            extra_fields: Provider-specific fields (e.g., inferred_question)
            
        Returns:
            Formatted log message string
        """
        return (
            f"[RubricTrainerActor] Creating reward task {index}/{total}: "
            f"question={question[:50]}..., rubric_length={rubric_length}, "
            f"accepted_answer_length={accepted_answer_length}, rejected_answer_length={rejected_answer_length}"
        )


@ray.remote
class ReplayBufferDataProvider(BaseDataProvider):
    """Data provider that uses past policy rollouts from a replay buffer.
    
    IMPORTANT: This provider returns question/answer pairs from the replay buffer.
    The question in the returned AnswerPair is from the BUFFER, not from the input example.
    Calling code MUST use answer_pair.question for rubric generation to avoid mismatch.
    """
    
    def __init__(
        self,
        policy_actor: Any,
        replay_buffer_maxlen: int = 2048,
        replay_buffer_min_age: int = 0,
        replay_buffer_max_age: int | None = None,
        question_key: str = "question",
    ):
        """Initialize the replay buffer data provider.
        
        Args:
            policy_actor: Ray actor handle for policy model
            replay_buffer_maxlen: Maximum size of replay buffer
            replay_buffer_min_age: Minimum step age for sampling (0 = include most recent)
            replay_buffer_max_age: Maximum step age for sampling (None = no limit)
            question_key: Key to extract question from example dicts
        """
        # Replay buffer stores (step, question, answer) tuples
        self.replay_buffer = collections.deque(maxlen=replay_buffer_maxlen)
        self.buffer_lock = threading.Lock()
        self._replay_buffer_min_age = replay_buffer_min_age
        self._replay_buffer_max_age = replay_buffer_max_age
        # Track the latest policy step for accurate age calculation during sampling
        self._last_policy_step = 0
        self.policy_actor = policy_actor
        self.question_key = question_key
    
    def add_experience(self, question: str, answer: str, step: int = 0) -> None:
        """Add a policy rollout to the replay buffer with step number for age-based sampling.
        
        Args:
            question: The question that was answered
            answer: The policy's answer
            step: The policy training step when this experience was generated
        """
        with self.buffer_lock:
            self.replay_buffer.append((step, question, answer))
            # Track the latest policy step for accurate age calculation during rubric sampling
            self._last_policy_step = max(self._last_policy_step, step)
    
    def get_pair_context(self, example: dict[str, Any]) -> dict[str, Any]:
        """Sample from replay buffer and return context with the sampled question and rejected answer.
        
        Args:
            example: Example dict (not used for replay buffer, we sample from buffer instead)
            
        Returns:
            Dict with "question", "rejected_answer", "rejected_step", "current_step", and "step_gap"
            from the replay buffer
        """
        # Get question and rejected_answer from replay buffer with age-based filtering
        with self.buffer_lock:
            if not self.replay_buffer:
                raise ValueError(
                    f"Replay buffer is empty. Cannot create paired future without buffer entries."
                )
            
            # Filter buffer by step age
            # Use _last_policy_step (not training_step) for accurate age calculation
            # This ensures age is measured in policy training steps, not global steps
            current_step = self._last_policy_step
            min_age = self._replay_buffer_min_age
            max_age = self._replay_buffer_max_age
            
            # Filter experiences by age (step, question, answer)
            valid_experiences = [
                (s, q, a) for s, q, a in self.replay_buffer
                if (current_step - s) >= min_age and 
                   (max_age is None or (current_step - s) <= max_age)
            ]
            
            # Fallback mechanism if no experiences in the target range
            fallback_used = None
            if not valid_experiences:
                # Compute actual age range in buffer
                buffer_steps = [s for s, _, _ in self.replay_buffer]
                oldest_step = min(buffer_steps)
                newest_step = max(buffer_steps)
                oldest_age = current_step - oldest_step
                newest_age = current_step - newest_step
                
                # Fallback 1: Try min_age only (no max_age constraint)
                # This handles the case where all experiences are "too old"
                if newest_age >= min_age:
                    valid_experiences = [
                        (s, q, a) for s, q, a in self.replay_buffer
                        if (current_step - s) >= min_age
                    ]
                    if valid_experiences:
                        fallback_used = f"relaxed_max_age (was {max_age}, now unlimited)"
                
                # Fallback 2: Use any experience that's not too young (min_age/2)
                if not valid_experiences:
                    relaxed_min = max(1, min_age // 2)
                    valid_experiences = [
                        (s, q, a) for s, q, a in self.replay_buffer
                        if (current_step - s) >= relaxed_min
                    ]
                    if valid_experiences:
                        fallback_used = f"relaxed_min_age (was {min_age}, now {relaxed_min})"
                
                # Fallback 3: Use the oldest experiences available (at least some gap)
                if not valid_experiences and oldest_age > 0:
                    # Sample from the oldest half of the buffer to maximize gap
                    sorted_experiences = sorted(self.replay_buffer, key=lambda x: x[0])
                    half_idx = len(sorted_experiences) // 2
                    valid_experiences = sorted_experiences[:half_idx] if half_idx > 0 else sorted_experiences[:1]
                    fallback_used = f"oldest_half (buffer age range: {newest_age}-{oldest_age})"
                
                # Fallback 4: Use anything (last resort)
                if not valid_experiences:
                    valid_experiences = list(self.replay_buffer)
                    fallback_used = f"any_experience (buffer age range: {newest_age}-{oldest_age})"
                
                if fallback_used:
                    LOGGER.warning(
                        f"[ReplayBuffer] No experiences in target range [min_age={min_age}, max_age={max_age}] "
                        f"at step {current_step}. Buffer has {len(self.replay_buffer)} experiences "
                        f"(age range: {newest_age}-{oldest_age}). Using fallback: {fallback_used}"
                    )
            
            rejected_step, question, rejected_answer = random.choice(valid_experiences)
            step_gap = current_step - rejected_step
        
        return {
            "question": question,
            "rejected_answer": rejected_answer,
            "rejected_step": rejected_step,
            "current_step": current_step,
            "step_gap": step_gap,
        }
    
    async def create_answer_pair(self, example: dict[str, Any], pair_context: dict[str, Any] | None = None) -> ReplayBufferAnswerPair:
        """Create a ReplayBufferAnswerPair using the question/rejected_answer from pair_context.
        
        Args:
            example: Example dict (not used, pair_context contains the data)
            pair_context: Dict with "question", "rejected_answer", "rejected_step", 
                         "current_step", and "step_gap" from get_pair_context
            
        Returns:
            The created ReplayBufferAnswerPair with step gap information
        """
        if not pair_context:
            raise ValueError("ReplayBufferDataProvider.create_answer_pair requires pair_context from get_pair_context")
        
        question = pair_context["question"]
        rejected_answer = pair_context["rejected_answer"]
        rejected_step = pair_context["rejected_step"]
        current_step = pair_context["current_step"]
        step_gap = pair_context["step_gap"]
        
        # Generate accepted answer (policy rollout on question)
        accepted_answer = await self.policy_actor._generate_policy_answer.remote(question, "")
        
        return ReplayBufferAnswerPair(
            question=question,
            accepted_answer=accepted_answer,
            rejected_answer=rejected_answer,
            rejected_step=rejected_step,
            current_step=current_step,
            step_gap=step_gap,
        )


@dataclass
class InferredAnswerPair(AnswerPair):
    """Extended pair for inferred question method that includes inferred question."""
    inferred_question: str = ""
    
    def get_extra_fields(self) -> dict[str, Any]:
        """Get provider-specific extra fields for logging.
        
        Returns:
            Dict with inferred_question field
        """
        return {"inferred_question": self.inferred_question}


@ray.remote
class InferredQuestionDataProvider(BaseDataProvider):
    """Data provider that generates rejected answers by inferring questions from accepted answers."""
    
    def __init__(
        self,
        policy_actor: Any,
        question_key: str = "question",
        inference_generate_text_actor: Any | None = None,
        inference_generation_kwargs: dict[str, Any] | None = None,
        frozen_policy_engines: dict[str, tuple[Any, Any]] | None = None,
    ):
        """Initialize the inferred question data provider.
        
        Args:
            policy_actor: Ray actor handle for policy model (used when no frozen engines)
            question_key: Key to extract question from example dicts
            inference_generate_text_actor: Optional Ray actor handle for generating text during inference.
                                         If not provided, uses policy_actor for inference.
                                         The actor must have generate_text_from_messages method.
            inference_generation_kwargs: Optional generation kwargs for inference (e.g. temperature).
            frozen_policy_engines: Optional dict mapping model names to (generate_text_actor, sampling_params)
                                  tuples. When provided, response generation uses frozen engines instead of
                                  policy_actor, cycling through models. Both responses in each pair come from
                                  the same frozen model.
        """
        self.policy_actor = policy_actor
        self.question_key = question_key
        self.inference_generate_text_actor = inference_generate_text_actor
        self.inference_generation_kwargs = inference_generation_kwargs or {}
        self.frozen_policy_engines = frozen_policy_engines
        self._frozen_model_names = list(frozen_policy_engines.keys()) if frozen_policy_engines else []
        if self._frozen_model_names:
            LOGGER.info(
                "[InferredQuestionDataProvider] Using %d frozen policy engines for response generation: %s",
                len(self._frozen_model_names), self._frozen_model_names,
            )
    
    async def _generate_rejected_with_inferred_question(self, original_question: str, accepted_answer: str) -> tuple[str, str]:
        """Generate a rejected answer by first inferring the question, then generating an answer to it.
        
        This creates a rejected sample by:
        1. Asking the model what question the accepted_answer was trying to answer
        2. Generating a new policy answer to that inferred question
        
        The intuition is that if the rubric is bad, the policy might be answering
        a different question than intended, and we want to capture that as a "rejected" sample.
        
        IMPORTANT: We strip thinking tokens (<think>...</think>) from the accepted_answer
        before question inference. The thinking process is internal reasoning and should
        not be shown to the question inference model - only the final answer matters.
        
        Args:
            original_question: The original question (for reference/logging)
            accepted_answer: The current policy's answer to the original question (may contain thinking)
            
        Returns:
            Tuple of (inferred_question, rejected_answer)
        """
        # Step 1: Infer what question the model thinks it was answering
        inferred_question = ""
        
        # CRITICAL: Strip thinking tokens before question inference!
        # The inference model should only see the final answer, not the thinking process.
        answer_for_inference = remove_thinking_section(accepted_answer)
        
        if self.inference_generate_text_actor:
            # Use specific inference actor (e.g. rubric judge or other model)
            # Format messages without tokenization - the actor will handle tokenization internally
            messages = format_messages(
                "question_inference",
                {"answer": answer_for_inference},  # Use cleaned answer
                tokenize=False,
            )
            
            # Build sampling params - handle both SamplingParams object and dict
            if isinstance(self.inference_generation_kwargs, vllm.SamplingParams):
                # Clone the SamplingParams object and override temperature
                sampling_params = self.inference_generation_kwargs.clone()
                sampling_params.temperature = 0.3  # Lower temperature for more deterministic question inference
            else:
                # Build from dict kwargs
                kwargs = dict(self.inference_generation_kwargs)
                kwargs["temperature"] = 0.3  # Lower temperature for more deterministic question inference
                
                # Filter kwargs to only include valid vllm.SamplingParams arguments
                valid_params = {
                    "n", "best_of", "presence_penalty", "frequency_penalty", "repetition_penalty",
                    "temperature", "top_p", "top_k", "min_p", "seed", "use_beam_search", "length_penalty",
                    "early_stopping", "stop", "stop_token_ids", "include_stop_str_in_output", "ignore_eos",
                    "max_tokens", "logprobs", "prompt_logprobs", "skip_special_tokens", "spaces_between_special_tokens",
                    "logits_processors"
                }
                sampling_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                
                sampling_params = vllm.SamplingParams(**sampling_kwargs)
            
            # Use generate_text_from_messages which handles tokenization internally
            result = await self.inference_generate_text_actor.generate_text_from_messages.remote(messages, sampling_params)
            
            # Strip thinking tokens from the inferred question using the utility function
            inferred_question = remove_thinking_section(result)
        
        # Log the inferred question (or that it's empty)
        if inferred_question:
            LOGGER.info(
                "[InferredQuestionDataProvider] Inferred question: original='%s...', inferred='%s...'",
                original_question[:100] if len(original_question) > 100 else original_question,
                inferred_question[:100] if len(inferred_question) > 100 else inferred_question,
            )
        else:
            LOGGER.warning(
                "[InferredQuestionDataProvider] Inferred question is EMPTY. "
                "original_question='%s...', accepted_answer_length=%d, "
                "inference_generate_text_actor=%s",
                original_question[:100] if len(original_question) > 100 else original_question,
                len(accepted_answer),
                self.inference_generate_text_actor is not None,
            )
        
        # Step 2: Generate a policy answer to the inferred question
        rejected_answer = await self.policy_actor._generate_policy_answer.remote(inferred_question, "")
        
        return inferred_question, rejected_answer

    async def _generate_rejected_with_inferred_question_frozen(
        self, original_question: str, accepted_answer: str,
        engine_actor: Any, sampling_params: Any,
    ) -> tuple[str, str]:
        """Like _generate_rejected_with_inferred_question but uses a frozen engine for the rejected answer.
        
        Question inference still uses self.inference_generate_text_actor (the training model),
        but the rejected answer is generated by the same frozen engine that produced the accepted answer.
        """
        inferred_question = ""
        answer_for_inference = remove_thinking_section(accepted_answer)
        
        if self.inference_generate_text_actor:
            messages = format_messages(
                "question_inference",
                {"answer": answer_for_inference},
                tokenize=False,
            )
            if isinstance(self.inference_generation_kwargs, vllm.SamplingParams):
                inf_params = self.inference_generation_kwargs.clone()
                inf_params.temperature = 0.3
            else:
                kwargs = dict(self.inference_generation_kwargs)
                kwargs["temperature"] = 0.3
                valid_params = {
                    "n", "best_of", "presence_penalty", "frequency_penalty", "repetition_penalty",
                    "temperature", "top_p", "top_k", "min_p", "seed", "use_beam_search", "length_penalty",
                    "early_stopping", "stop", "stop_token_ids", "include_stop_str_in_output", "ignore_eos",
                    "max_tokens", "logprobs", "prompt_logprobs", "skip_special_tokens", "spaces_between_special_tokens",
                    "logits_processors"
                }
                inf_params = vllm.SamplingParams(**{k: v for k, v in kwargs.items() if k in valid_params})
            result = await self.inference_generate_text_actor.generate_text_from_messages.remote(messages, inf_params)
            inferred_question = remove_thinking_section(result)
        
        if inferred_question:
            LOGGER.info(
                "[InferredQuestionDataProvider/frozen] Inferred question: original='%s...', inferred='%s...'",
                original_question[:100] if len(original_question) > 100 else original_question,
                inferred_question[:100] if len(inferred_question) > 100 else inferred_question,
            )
        else:
            LOGGER.warning(
                "[InferredQuestionDataProvider/frozen] Inferred question is EMPTY for original='%s...'",
                original_question[:100] if len(original_question) > 100 else original_question,
            )
        
        # Generate rejected answer from the SAME frozen engine
        rejected_answer = await self._generate_from_frozen_engine(
            "policy", {"question": inferred_question}, engine_actor, sampling_params,
        )
        
        return inferred_question, rejected_answer

    def get_pair_context(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Sample a frozen policy model for this pair (if frozen engines available).
        
        Returns:
            Dict with sampled frozen_model_name, or None if no frozen engines.
        """
        if self._frozen_model_names:
            model_name = random.choice(self._frozen_model_names)
            return {"frozen_model_name": model_name}
        return None
    
    async def _generate_from_frozen_engine(
        self, message_type: str, content: dict[str, Any], engine_actor: Any, sampling_params: Any,
    ) -> str:
        """Generate text using a frozen policy engine.
        
        Uses format_messages with tokenize=False and generate_text_from_messages,
        which handles tokenization internally with the engine's own tokenizer.
        """
        messages = format_messages(message_type, content, tokenize=False)
        return await engine_actor.generate_text_from_messages.remote(messages, sampling_params)

    async def create_answer_pair(self, example: dict[str, Any], pair_context: dict[str, Any] | None = None) -> InferredAnswerPair:
        """Create an InferredAnswerPair asynchronously.
        
        When frozen_policy_engines are available, both responses come from the same
        frozen model (selected in get_pair_context). Otherwise falls back to policy_actor.
        """
        question = example.get(self.question_key, "")
        
        if pair_context and "frozen_model_name" in pair_context:
            model_name = pair_context["frozen_model_name"]
            engine_actor, sampling_params = self.frozen_policy_engines[model_name]
            
            accepted_answer = await self._generate_from_frozen_engine(
                "policy", {"question": question}, engine_actor, sampling_params,
            )
            
            inferred_question, rejected_answer = await self._generate_rejected_with_inferred_question_frozen(
                question, accepted_answer, engine_actor, sampling_params,
            )
        else:
            accepted_answer = await self.policy_actor._generate_policy_answer.remote(question, "")
            inferred_question, rejected_answer = await self._generate_rejected_with_inferred_question(
                question, accepted_answer
            )
        
        return InferredAnswerPair(
            question=question,
            accepted_answer=accepted_answer,
            rejected_answer=rejected_answer,
            inferred_question=inferred_question,
        )
    
    def format_log_message(
        self,
        index: int,
        total: int,
        question: str,
        rubric_length: int,
        accepted_answer_length: int,
        rejected_answer_length: int,
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        """Format a log message for reward task creation with inferred question support.
        
        Args:
            index: Task index (1-based)
            total: Total number of tasks
            question: Original question
            rubric_length: Length of rubric text
            accepted_answer_length: Length of accepted answer
            rejected_answer_length: Length of rejected answer
            extra_fields: Provider-specific fields containing inferred_question
            
        Returns:
            Formatted log message string
        """
        if extra_fields and "inferred_question" in extra_fields:
            inferred_question = extra_fields["inferred_question"]
            return (
                f"[RubricTrainerActor] Creating reward task {index}/{total}: "
                f"original_question={question[:50]}..., inferred_question={inferred_question[:50]}..., "
                f"rubric_length={rubric_length}, "
                f"accepted_answer_length={accepted_answer_length}, rejected_answer_length={rejected_answer_length}"
            )
        else:
            return (
                f"[RubricTrainerActor] Creating reward task {index}/{total}: "
                f"question={question[:50]}..., rubric_length={rubric_length}, "
                f"accepted_answer_length={accepted_answer_length}, rejected_answer_length={rejected_answer_length}"
            )

@dataclass
class RubricAnswerPair(AnswerPair):
    """Extended pair for rubric method that includes the generated rubric."""
    generated_rubric: str = ""
    
    def get_extra_fields(self) -> dict[str, Any]:
        """Get provider-specific extra fields for logging.
        
        Returns:
            Empty dict - the generated rubric is already captured in the main rubric field
            of GenerationExample via decoded_responses, so no extra fields needed.
        """
        return {}


# Note: RubricAnswerPair inherits `question` field from AnswerPair


@ray.remote
class RubricDataProvider(BaseDataProvider):
    """Data provider that uses rubric in context to generate chosen answer and no rubric to generate rejected answer.
    
    This provider:
    1. First generates a rubric for the given question using the rubric model
    2. Generates chosen/accepted answer conditioned on both question and rubric
    3. Generates rejected answer conditioned only on question (no rubric)
    
    The intuition is that conditioning on a rubric should produce higher quality answers,
    so rubric-conditioned answers are "chosen" and non-rubric answers are "rejected".
    
    When frozen_policy_engines are provided, response generation uses frozen engines
    (cycling through models per pair), while rubric generation uses rubric_generate_text_actor
    (the training model).
    """
    
    def __init__(
        self,
        policy_actor: Any,
        question_key: str = "question",
        frozen_policy_engines: dict[str, tuple[Any, Any]] | None = None,
        rubric_generate_text_actor: Any | None = None,
        rubric_sampling_params: vllm.SamplingParams | None = None,
        rubric_prompt_key: str = "rubric_generation",
    ):
        """Initialize the rubric data provider.

        Args:
            policy_actor: Ray actor handle for policy model (used when no frozen engines)
            question_key: Key to extract question from example dicts
            frozen_policy_engines: Optional dict mapping model names to (generate_text_actor, sampling_params).
                                  When provided, response generation uses frozen engines.
            rubric_generate_text_actor: Optional GenerateTextActor for rubric generation.
                                       In two-model mode this is the rubric model's own
                                       GenerateTextActor (evolving), ensuring rubrics improve
                                       as the rubric model trains. When not provided, falls
                                       back to policy_actor._generate_rubric() (single-model).
                                       Required when frozen_policy_engines is set.
            rubric_sampling_params: Sampling params for rubric generation. Built from
                                   build_rubric_sampling_params() to stay consistent with training config.
            rubric_prompt_key: Registry key for rubric generation system prompt.
        """
        super().__init__(policy_actor, question_key)
        self.rubric_prompt_key = rubric_prompt_key
        self.frozen_policy_engines = frozen_policy_engines
        self._frozen_model_names = list(frozen_policy_engines.keys()) if frozen_policy_engines else []
        self.rubric_generate_text_actor = rubric_generate_text_actor
        self.rubric_sampling_params = rubric_sampling_params
        if self._frozen_model_names:
            if rubric_generate_text_actor is None:
                raise ValueError(
                    "rubric_generate_text_actor must be provided when frozen_policy_engines is set. "
                    "Rubric generation must use the training model, not the frozen policies."
                )
            LOGGER.info(
                "[RubricDataProvider] Using %d frozen policy engines for response generation: %s",
                len(self._frozen_model_names), self._frozen_model_names,
            )
        if rubric_generate_text_actor is not None:
            LOGGER.info(
                "[RubricDataProvider] Using rubric_generate_text_actor for rubric generation "
                "(evolving rubric model in two-model mode)",
            )

    def get_pair_context(self, example: dict[str, Any]) -> dict[str, Any] | None:
        """Sample a frozen policy model for this pair (if frozen engines available).
        
        Returns:
            Dict with sampled frozen_model_name, or None if no frozen engines.
        """
        if self._frozen_model_names:
            model_name = random.choice(self._frozen_model_names)
            return {"frozen_model_name": model_name}
        return None
    
    async def create_answer_pair(self, example: dict[str, Any], pair_context: dict[str, Any] | None = None) -> RubricAnswerPair:
        """Create a RubricAnswerPair asynchronously.
        
        When frozen_policy_engines are available, both responses come from the same frozen
        model (selected in get_pair_context). Rubric generation always uses the training model
        (rubric_generate_text_actor when set, otherwise policy_actor).
        
        Strips thinking tokens (<think>...</think>) from the generated rubric
        before using it to condition the policy answer.
        """
        question = example.get(self.question_key, "")
        
        if pair_context and "frozen_model_name" in pair_context:
            model_name = pair_context["frozen_model_name"]
            engine_actor, sampling_params = self.frozen_policy_engines[model_name]
            
            rubric_messages = format_messages(self.rubric_prompt_key, {"question": question}, tokenize=False)
            rubric_future = self.rubric_generate_text_actor.generate_text_from_messages.remote(
                rubric_messages, self.rubric_sampling_params,
            )
            
            # Rejected answer from frozen engine (no rubric conditioning) -- start in parallel
            rejected_messages = format_messages("policy", {"question": question}, tokenize=False)
            rejected_future = engine_actor.generate_text_from_messages.remote(rejected_messages, sampling_params)
            
            # Wait for rubric
            rubric_text_raw = await rubric_future
            rubric_text = remove_thinking_section(rubric_text_raw)
            
            # Accepted answer from the SAME frozen engine (with rubric)
            accepted_messages = format_messages(
                "policy_with_rubric", {"question": question, "rubric": rubric_text}, tokenize=False,
            )
            accepted_future = engine_actor.generate_text_from_messages.remote(accepted_messages, sampling_params)
            
            accepted_answer, rejected_answer = await asyncio.gather(accepted_future, rejected_future)
        else:
            if self.rubric_generate_text_actor is not None:
                rubric_messages = format_messages(self.rubric_prompt_key, {"question": question}, tokenize=False)
                rubric_future = self.rubric_generate_text_actor.generate_text_from_messages.remote(
                    rubric_messages, self.rubric_sampling_params,
                )
            else:
                rubric_future = self.policy_actor._generate_rubric.remote(question)
            
            rejected_future = self.policy_actor._generate_policy_answer.remote(question, "")
            
            rubric_text_raw = await rubric_future
            rubric_text = remove_thinking_section(rubric_text_raw)
            
            accepted_future = self.policy_actor._generate_policy_answer_with_rubric.remote(
                question, rubric_text,
            )
            
            accepted_answer, rejected_answer = await asyncio.gather(accepted_future, rejected_future)
        
        return RubricAnswerPair(
            question=question,
            accepted_answer=accepted_answer,
            rejected_answer=rejected_answer,
            generated_rubric=rubric_text,
        )
    
    def format_log_message(
        self,
        index: int,
        total: int,
        question: str,
        rubric_length: int,
        accepted_answer_length: int,
        rejected_answer_length: int,
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        """Format a log message for reward task creation with rubric support.
        
        Args:
            index: Task index (1-based)
            total: Total number of tasks
            question: Original question
            rubric_length: Length of rubric text
            accepted_answer_length: Length of accepted answer
            rejected_answer_length: Length of rejected answer
            extra_fields: Provider-specific fields containing rubric
            
        Returns:
            Formatted log message string
        """
        return (
            f"[RubricTrainerActor] Creating reward task {index}/{total}: "
            f"question={question[:50]}..., rubric_length={rubric_length}, "
            f"accepted_answer_length={accepted_answer_length}, "
            f"rejected_answer_length={rejected_answer_length}"
        )


@dataclass
class CombinedWeights:
    """Weights for combined data provider sampling."""
    replay_buffer: float = 0.0
    inferred_question: float = 0.0
    rubric: float = 0.0
    multi_policy_frozen: float = 0.0
    
    @classmethod
    def from_string(cls, weight_string: str) -> "CombinedWeights":
        """Parse a weight string like 'replay_buffer:1,inferred_question:1,rubric:1,multi_policy_frozen:1'.
        
        Weights don't need to sum to 1 - they are normalized automatically.
        
        Args:
            weight_string: Comma-separated key:value pairs
            
        Returns:
            CombinedWeights instance
        """
        weights = {"replay_buffer": 0.0, "inferred_question": 0.0, "rubric": 0.0, "multi_policy_frozen": 0.0}
        
        for pair in weight_string.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise ValueError(f"Invalid weight format '{pair}', expected 'method:weight'")
            method, weight = pair.split(":", 1)
            method = method.strip()
            weight = float(weight.strip())
            if method not in weights:
                raise ValueError(f"Unknown method '{method}', expected one of: {list(weights.keys())}")
            weights[method] = weight
        
        return cls(**weights)
    
    def normalize(self) -> "CombinedWeights":
        """Return a new CombinedWeights with weights normalized to sum to 1.0."""
        total = self.replay_buffer + self.inferred_question + self.rubric + self.multi_policy_frozen
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        return CombinedWeights(
            replay_buffer=self.replay_buffer / total,
            inferred_question=self.inferred_question / total,
            rubric=self.rubric / total,
            multi_policy_frozen=self.multi_policy_frozen / total,
        )
    
    def to_list(self) -> list[tuple[str, float]]:
        """Return list of (method, weight) tuples with non-zero weights."""
        result = []
        if self.replay_buffer > 0:
            result.append(("replay_buffer", self.replay_buffer))
        if self.inferred_question > 0:
            result.append(("inferred_question", self.inferred_question))
        if self.rubric > 0:
            result.append(("rubric", self.rubric))
        if self.multi_policy_frozen > 0:
            result.append(("multi_policy_frozen", self.multi_policy_frozen))
        return result


@ray.remote
class CombinedDataProvider(BaseDataProvider):
    """Data provider that combines multiple methods with configurable weights.
    
    Randomly samples from replay_buffer, inferred_question, and rubric methods
    based on provided weights. Useful for diverse preference pairs.
    """
    
    def __init__(
        self,
        policy_actor: Any,
        weights: CombinedWeights,
        question_key: str = "question",
        replay_buffer_provider: Any | None = None,
        inferred_question_provider: Any | None = None,
        rubric_provider: Any | None = None,
        multi_policy_frozen_provider: Any | None = None,
    ):
        """Initialize the combined data provider.
        
        Args:
            policy_actor: Ray actor handle for policy model
            weights: CombinedWeights specifying sampling probabilities
            question_key: Key to extract question from example dicts
            replay_buffer_provider: Ray actor handle for replay buffer provider (if weight > 0)
            inferred_question_provider: Ray actor handle for inferred question provider (if weight > 0)
            rubric_provider: Ray actor handle for rubric provider (if weight > 0)
            multi_policy_frozen_provider: Ray actor handle for multi-policy frozen provider (if weight > 0)
        """
        self.policy_actor = policy_actor
        self.question_key = question_key
        self.weights = weights.normalize()
        
        # Store providers
        self.replay_buffer_provider = replay_buffer_provider
        self.inferred_question_provider = inferred_question_provider
        self.rubric_provider = rubric_provider
        self.multi_policy_frozen_provider = multi_policy_frozen_provider
        
        # Build sampling list: (method_name, weight, provider)
        self._sampling_list: list[tuple[str, float, Any]] = []
        if self.weights.replay_buffer > 0 and replay_buffer_provider:
            self._sampling_list.append(("replay_buffer", self.weights.replay_buffer, replay_buffer_provider))
        if self.weights.inferred_question > 0 and inferred_question_provider:
            self._sampling_list.append(("inferred_question", self.weights.inferred_question, inferred_question_provider))
        if self.weights.rubric > 0 and rubric_provider:
            self._sampling_list.append(("rubric", self.weights.rubric, rubric_provider))
        if self.weights.multi_policy_frozen > 0 and multi_policy_frozen_provider:
            self._sampling_list.append(("multi_policy_frozen", self.weights.multi_policy_frozen, multi_policy_frozen_provider))
        
        if not self._sampling_list:
            raise ValueError("CombinedDataProvider requires at least one provider with positive weight")
        
        LOGGER.info(
            "[CombinedDataProvider] Initialized with %d active methods: %s",
            len(self._sampling_list),
            [(name, weight) for name, weight, _ in self._sampling_list],
        )
    
    def _sample_provider(self) -> tuple[str, Any]:
        """Sample a provider based on weights.
        
        Returns:
            Tuple of (method_name, provider_actor)
        """
        methods = [name for name, _, _ in self._sampling_list]
        weights = [w for _, w, _ in self._sampling_list]
        providers = [p for _, _, p in self._sampling_list]
        
        # Weighted random choice
        idx = random.choices(range(len(methods)), weights=weights, k=1)[0]
        return methods[idx], providers[idx]
    
    def set_active_policy_actor(self, policy_actor: Any) -> None:
        """Switch active policy actor on self and all sub-providers.
        
        Used in multi-policy co-evolution to rotate between policy models.
        Note: multi_policy_frozen_provider is NOT updated since it has its own
        independent frozen engines that don't depend on the active policy.
        """
        self.policy_actor = policy_actor
        # Propagate to sub-providers (skip multi_policy_frozen — it uses its own engines)
        futures = []
        if self.replay_buffer_provider is not None:
            futures.append(self.replay_buffer_provider.set_active_policy_actor.remote(policy_actor))
        if self.inferred_question_provider is not None:
            futures.append(self.inferred_question_provider.set_active_policy_actor.remote(policy_actor))
        if self.rubric_provider is not None:
            futures.append(self.rubric_provider.set_active_policy_actor.remote(policy_actor))
        if futures:
            ray.get(futures)
    
    def add_experience(self, question: str, answer: str, step: int = 0) -> None:
        """Add experience to the replay buffer provider (if it exists).
        
        Args:
            question: The question that was answered
            answer: The policy's answer
            step: The policy training step when this experience was generated
        """
        if self.replay_buffer_provider is not None:
            # Call add_experience on the replay buffer provider
            ray.get(self.replay_buffer_provider.add_experience.remote(question, answer, step))
    
    def get_pair_context(self, example: dict[str, Any]) -> dict[str, Any]:
        """Sample a method and get its pair context.
        
        Args:
            example: Example dict containing question and other data
            
        Returns:
            Dict with "selected_method" and provider-specific context
        """
        method_name, provider = self._sample_provider()
        
        # Get context from the selected provider
        provider_context = ray.get(provider.get_pair_context.remote(example))
        
        return {
            "selected_method": method_name,
            "selected_provider": provider,
            "provider_context": provider_context,
        }
    
    async def create_answer_pair(self, example: dict[str, Any], pair_context: dict[str, Any] | None = None) -> AnswerPair:
        """Create an answer pair using the selected provider.
        
        Args:
            example: Example dict containing question and other data
            pair_context: Dict from get_pair_context containing selected method/provider
            
        Returns:
            The created AnswerPair (type depends on selected method)
        """
        if not pair_context:
            raise ValueError("CombinedDataProvider.create_answer_pair requires pair_context from get_pair_context")
        
        provider = pair_context["selected_provider"]
        provider_context = pair_context.get("provider_context")
        
        # Delegate to the selected provider
        answer_pair = await provider.create_answer_pair.remote(example, provider_context)
        
        return answer_pair


@ray.remote
class MultiPolicyFrozenDataProvider(BaseDataProvider):
    """Data provider that uses multiple frozen policies to generate diverse responses.
    
    Samples two different frozen policy models for each question, generates responses
    from both, then uses rubric+judge to determine which is better (accepted vs rejected).
    This enables training the rubric generator on diverse policy outputs without training
    those policies.
    """
    
    def __init__(
        self,
        policy_engines: dict[str, tuple[Any, Any]],  # {model_name: (generate_text_actor, sampling_params)}
        rubric_generate_text_actor: Any,  # GenerateTextActor for rubric generation (required to avoid deadlock)
        judge_actor: Any | None = None,  # Optional single judge
        multi_judge_engines: Any | None = None,  # Optional multi-judge
        sampling_strategy: str = "uniform",
        question_key: str = "question",
        rubric_sampling_params: vllm.SamplingParams | None = None,
        judge_sampling_params: vllm.SamplingParams | None = None,
        rubric_prompt_key: str = "rubric_generation",
    ):
        """Initialize multi-policy frozen data provider.
        
        Args:
            policy_engines: Dict mapping model names to (generate_text_actor, sampling_params) tuples
            rubric_generate_text_actor: GenerateTextActor for rubric generation (avoids deadlock)
            judge_actor: Optional single judge actor (generate_text_actor)
            multi_judge_engines: Optional multi-judge engines object  
            sampling_strategy: How to sample policies ("uniform" or "round_robin")
            question_key: Key to extract question from example dicts
            rubric_sampling_params: Sampling params for rubric generation (from build_rubric_sampling_params)
            judge_sampling_params: Sampling params for judge scoring (from build_judge_sampling_params)
            rubric_prompt_key: Registry key for rubric generation system prompt.
        """
        if len(policy_engines) < 2:
            raise ValueError(f"MultiPolicyFrozenDataProvider requires at least 2 policy models, got {len(policy_engines)}")
        
        if judge_actor is None and multi_judge_engines is None:
            raise ValueError("MultiPolicyFrozenDataProvider requires either judge_actor or multi_judge_engines")
        
        if rubric_generate_text_actor is None:
            raise ValueError(
                "MultiPolicyFrozenDataProvider requires rubric_generate_text_actor. "
                "This avoids deadlock by generating rubrics directly instead of calling back to rubric_actor."
            )
        
        self.policy_engines = policy_engines
        self.policy_models = list(policy_engines.keys())
        self.rubric_generate_text_actor = rubric_generate_text_actor
        self.judge_actor = judge_actor
        self.multi_judge_engines = multi_judge_engines
        self.sampling_strategy = sampling_strategy
        self.question_key = question_key
        self.rubric_prompt_key = rubric_prompt_key
        self.rubric_sampling_params = rubric_sampling_params
        self.judge_sampling_params = judge_sampling_params
        self._round_robin_index = 0
        
        LOGGER.info(
            "[MultiPolicyFrozenDataProvider] Initialized with %d policies: %s (strategy=%s)",
            len(self.policy_models),
            self.policy_models,
            sampling_strategy,
        )
    
    def _sample_two_policies(self) -> tuple[str, str]:
        """Sample two different policies based on sampling strategy."""
        if self.sampling_strategy == "uniform":
            return tuple(random.sample(self.policy_models, 2))
        elif self.sampling_strategy == "round_robin":
            policy_a = self.policy_models[self._round_robin_index % len(self.policy_models)]
            policy_b = self.policy_models[(self._round_robin_index + 1) % len(self.policy_models)]
            self._round_robin_index += 1
            return policy_a, policy_b
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")
    
    def get_pair_context(self, example: dict[str, Any]) -> dict[str, Any]:
        """Sample two policies for this example."""
        question = example[self.question_key]
        policy_a_name, policy_b_name = self._sample_two_policies()
        
        return {
            "question": question,
            "policy_a_name": policy_a_name,
            "policy_b_name": policy_b_name,
        }
    
    async def create_answer_pair(
        self, example: dict[str, Any], pair_context: dict[str, Any] | None = None
    ) -> AnswerPair:
        """Create answer pair by generating from two frozen policies and judging."""
        if not pair_context:
            raise ValueError("MultiPolicyFrozenDataProvider.create_answer_pair requires pair_context")
        
        question = pair_context["question"]
        policy_a_name = pair_context["policy_a_name"]
        policy_b_name = pair_context["policy_b_name"]
        
        # Get engine actors and sampling params for both policies
        policy_a_actor, policy_a_params = self.policy_engines[policy_a_name]
        policy_b_actor, policy_b_params = self.policy_engines[policy_b_name]
        
        # Format question as messages
        from open_instruct.search_rewards.utils.rubric_chat_templates import format_messages
        
        messages = format_messages("policy", {"question": question}, tokenize=False)
        
        # Generate responses from both policies in parallel
        response_a_task = policy_a_actor.generate_text_from_messages.remote(messages, policy_a_params)
        response_b_task = policy_b_actor.generate_text_from_messages.remote(messages, policy_b_params)
        
        response_a, response_b = await asyncio.gather(response_a_task, response_b_task)
        
        rubric_messages = format_messages(self.rubric_prompt_key, {"question": question}, tokenize=False)
        
        rubric_text = await self.rubric_generate_text_actor.generate_text_from_messages.remote(
            rubric_messages, self.rubric_sampling_params
        )
        
        # Judge both responses with the rubric
        if self.multi_judge_engines is not None:
            # Multi-judge mode
            from open_instruct.search_rewards.rubric_judge_rewards import compute_rubric_judge_reward_multi_judge_async
            
            judge_actors = self.multi_judge_engines.get_judge_actors_with_sampling_params(self.judge_sampling_params)
            aggregation_mode = self.multi_judge_engines.aggregation_mode
            tie_breaker = self.multi_judge_engines.tie_breaker
            alpha = self.multi_judge_engines.alpha
            beta = self.multi_judge_engines.beta
            
            # Judge both responses together (function evaluates accepted vs rejected)
            result = await compute_rubric_judge_reward_multi_judge_async(
                question=question,
                accepted_answer=response_a,
                rejected_answer=response_b,
                generated_rubric=rubric_text,
                judge_actors=judge_actors,
                aggregation_mode=aggregation_mode,
                alpha=alpha,
                beta=beta,
                tie_breaker=tie_breaker,
            )
            winner = result.get("winner", "accepted")
        else:
            # Single judge mode
            from open_instruct.search_rewards.rubric_judge_rewards import compute_rubric_judge_reward_async
            
            # Judge both responses together (function evaluates accepted vs rejected)
            result = await compute_rubric_judge_reward_async(
                question=question,
                accepted_answer=response_a,
                rejected_answer=response_b,
                generated_rubric=rubric_text,
                rubric_judge_generate_text_actor=self.judge_actor,
                sampling_params=self.judge_sampling_params,
            )
            winner = "accepted" if result["accepted_score"] > result["rejected_score"] else "rejected"
        
        # Determine which is better
        if winner == "accepted":
            return AnswerPair(accepted_answer=response_a, rejected_answer=response_b, question=question)
        else:
            return AnswerPair(accepted_answer=response_b, rejected_answer=response_a, question=question)
    
    def format_log_message(
        self,
        index: int,
        total: int,
        question: str,
        rubric_length: int,
        accepted_answer_length: int,
        rejected_answer_length: int,
        extra_fields: dict[str, Any] | None = None,
    ) -> str:
        """Format a log message for this answer pair."""
        return (
            f"[MultiPolicyFrozen] Creating reward task {index}/{total}: "
            f"question={question[:50]}..., rubric_length={rubric_length}, "
            f"accepted_answer_length={accepted_answer_length}, rejected_answer_length={rejected_answer_length}"
        )
    
    def add_experience(self, question: str, answer: str, step: int) -> None:
        """Multi-policy frozen is stateless, no experience tracking needed."""
        pass
def create_data_provider(
    policy_actor: Any,
    rejected_answer_method: str,
    args: Any,
    question_key: str = "question",
    rubric_actor: Any | None = None,  # Kept for backward compatibility; not used by data providers
    inference_model_engines_obj: Any | None = None,
    rubric_judge_tokenizer: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    policy_generate_text_actor: Any | None = None,
    multi_policy_frozen_engines: Any | None = None,
    multi_judge_engines: Any | None = None,
    rubric_prompt_key: str = "rubric_generation",
) -> ray.actor.ActorHandle:
    """Factory function to create the appropriate data provider as a Ray actor.

    Args:
        policy_actor: Ray actor handle for policy model
        rejected_answer_method: Method for generating rejected answers.
                               Options: "replay_buffer", "inferred_question", "rubric", "multi_policy_frozen", "combined"
        args: Args object containing configuration (replay_buffer_size, replay_buffer_min_age,
              replay_buffer_max_age, inference_model_for_question_inference, grpo_args, etc.)
        question_key: Key to extract question from example dicts
        rubric_actor: Kept for backward compatibility; not used by data providers
        inference_model_engines_obj: Optional AuxiliaryModelEngines object for dedicated inference engines
        rubric_judge_tokenizer: Optional tokenizer for rubric judge model (kept for backward compatibility)
        rubric_judge_generate_text_actor: GenerateTextActor for rubric judge model
        policy_generate_text_actor: GenerateTextActor for rubric generation. In two-model mode,
                                   this is the rubric model's own GenerateTextActor (evolving).
                                   In single-model mode, this is the shared GenerateTextActor.
        multi_policy_frozen_engines: Optional MultiPolicyFrozenEngines object for multi-policy training
        multi_judge_engines: Optional MultiJudgeEngines object for multi-judge evaluation

    Returns:
        Ray actor handle to the appropriate data provider instance
    """
    grpo_args = getattr(args, "grpo_args", args)
    rubric_sp = build_rubric_sampling_params(grpo_args)
    judge_sp = build_judge_sampling_params(grpo_args)
    if rejected_answer_method == "replay_buffer":
        # Create config from args
        config = ReplayBufferConfig.from_args(args)
        
        return ReplayBufferDataProvider.remote(
            policy_actor=policy_actor,
            replay_buffer_maxlen=config.size,
            replay_buffer_min_age=config.min_age,
            replay_buffer_max_age=config.max_age,
            question_key=question_key,
        )
    elif rejected_answer_method == "inferred_question":
        # Create config from args
        inference_config = InferenceEngineConfig.from_args(args)
        
        # Get inference_model_for_question_inference from config (already set from command line args)
        inference_model_for_question_inference = inference_config.model_for_question_inference
        
        if inference_model_for_question_inference is None:
            raise ValueError(
                "inference_model_for_question_inference must be set when rejected_answer_method='inferred_question'. "
                "Check that --inference-model-for-question-inference is set in command line args."
            )
        
        LOGGER.info(
            "[create_data_provider] Using inference_model_for_question_inference='%s'",
            inference_model_for_question_inference,
        )
        
        # Create inference components using args directly (not via Ray actor)
        # This avoids issues with getattr on Ray actor handles
        inference_generate_text_actor, inference_generation_kwargs = _create_inference_components(
            args=args,
            inference_model_engines_obj=inference_model_engines_obj,
            rubric_judge_tokenizer=rubric_judge_tokenizer,
            inference_model_for_question_inference=inference_model_for_question_inference,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            policy_generate_text_actor=policy_generate_text_actor,
        )
        
        LOGGER.info(
            "[create_data_provider] _create_inference_components returned: "
            "inference_generate_text_actor=%s",
            inference_generate_text_actor is not None,
        )
        
        # Pass frozen engines when available (standalone inferred_question mode)
        frozen_engines_dict = None
        if multi_policy_frozen_engines is not None:
            frozen_engines_dict = multi_policy_frozen_engines.get_policy_engines_dict()
        
        return InferredQuestionDataProvider.remote(
            policy_actor=policy_actor,
            question_key=question_key,
            inference_generate_text_actor=inference_generate_text_actor,
            inference_generation_kwargs=inference_generation_kwargs,
            frozen_policy_engines=frozen_engines_dict,
        )
    elif rejected_answer_method == "rubric":
        # Pass frozen engines and rubric generation actor when available
        frozen_engines_dict = None
        rubric_gen_actor_for_provider = None
        if multi_policy_frozen_engines is not None:
            frozen_engines_dict = multi_policy_frozen_engines.get_policy_engines_dict()
        # Use policy_generate_text_actor for rubric generation when available.
        # In two-model mode this is the rubric model's GenerateTextActor (evolving);
        # in single-model mode it's the shared GenerateTextActor.
        if policy_generate_text_actor is not None:
            rubric_gen_actor_for_provider = policy_generate_text_actor

        return RubricDataProvider.remote(
            policy_actor=policy_actor,
            question_key=question_key,
            frozen_policy_engines=frozen_engines_dict,
            rubric_generate_text_actor=rubric_gen_actor_for_provider,
            rubric_sampling_params=rubric_sp,
            rubric_prompt_key=rubric_prompt_key,
        )
    elif rejected_answer_method == "combined":
        # Parse combined weights from args
        combined_weights_str = getattr(args, "combined_data_provider_weights", "")
        if not combined_weights_str:
            raise ValueError(
                "combined_data_provider_weights must be set when rejected_answer_method='combined'. "
                "Example: --combined-data-provider-weights 'replay_buffer:0.5,inferred_question:0.25,rubric:0.25'"
            )
        
        weights = CombinedWeights.from_string(combined_weights_str)
        LOGGER.info("[create_data_provider] Combined weights: %s", weights)
        
        # Create sub-providers for each method with non-zero weight
        replay_buffer_provider = None
        inferred_question_provider = None
        rubric_provider = None
        
        if weights.replay_buffer > 0:
            config = ReplayBufferConfig.from_args(args)
            replay_buffer_provider = ReplayBufferDataProvider.remote(
                policy_actor=policy_actor,
                replay_buffer_maxlen=config.size,
                replay_buffer_min_age=config.min_age,
                replay_buffer_max_age=config.max_age,
                question_key=question_key,
            )
            LOGGER.info("[create_data_provider] Created replay_buffer provider (weight=%.2f)", weights.replay_buffer)
        
        if weights.inferred_question > 0:
            inference_config = InferenceEngineConfig.from_args(args)
            inference_model_for_question_inference = inference_config.model_for_question_inference
            
            if inference_model_for_question_inference is None:
                raise ValueError(
                    "inference_model_for_question_inference must be set when using inferred_question in combined mode. "
                    "Check that --inference-model-for-question-inference is set in command line args."
                )
            
            inference_generate_text_actor, inference_generation_kwargs = _create_inference_components(
                args=args,
                inference_model_engines_obj=inference_model_engines_obj,
                rubric_judge_tokenizer=rubric_judge_tokenizer,
                inference_model_for_question_inference=inference_model_for_question_inference,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                policy_generate_text_actor=policy_generate_text_actor,
            )
            
            # Pass frozen engines to IQ provider when available
            frozen_engines_dict = None
            if multi_policy_frozen_engines is not None:
                frozen_engines_dict = multi_policy_frozen_engines.get_policy_engines_dict()
                LOGGER.info(
                    "[create_data_provider] Passing %d frozen policy engines to inferred_question provider",
                    len(frozen_engines_dict),
                )
            
            inferred_question_provider = InferredQuestionDataProvider.remote(
                policy_actor=policy_actor,
                question_key=question_key,
                inference_generate_text_actor=inference_generate_text_actor,
                inference_generation_kwargs=inference_generation_kwargs,
                frozen_policy_engines=frozen_engines_dict,
            )
            LOGGER.info("[create_data_provider] Created inferred_question provider (weight=%.2f)", weights.inferred_question)
        
        if weights.rubric > 0:
            # Pass frozen engines and rubric generation actor when available
            frozen_engines_dict = None
            rubric_gen_actor_for_provider = None
            if multi_policy_frozen_engines is not None:
                frozen_engines_dict = multi_policy_frozen_engines.get_policy_engines_dict()
                LOGGER.info(
                    "[create_data_provider] Passing %d frozen policy engines to rubric provider",
                    len(frozen_engines_dict),
                )
            # Use policy_generate_text_actor for rubric generation when available.
            if policy_generate_text_actor is not None:
                rubric_gen_actor_for_provider = policy_generate_text_actor
            elif multi_policy_frozen_engines is not None:
                raise ValueError(
                    "policy_generate_text_actor must be provided for rubric provider when "
                    "multi_policy_frozen_engines is set. This actor generates rubrics from "
                    "the training model while frozen engines handle response generation."
                )

            rubric_provider = RubricDataProvider.remote(
                policy_actor=policy_actor,
                question_key=question_key,
                frozen_policy_engines=frozen_engines_dict,
                rubric_generate_text_actor=rubric_gen_actor_for_provider,
                rubric_sampling_params=rubric_sp,
                rubric_prompt_key=rubric_prompt_key,
            )
            LOGGER.info("[create_data_provider] Created rubric provider (weight=%.2f)", weights.rubric)

        # Create multi_policy_frozen sub-provider if weight > 0 and engines are available
        multi_policy_frozen_provider = None
        if weights.multi_policy_frozen > 0:
            if multi_policy_frozen_engines is None:
                raise ValueError(
                    "multi_policy_frozen_engines must be provided when using multi_policy_frozen in combined mode. "
                    "Set --multi-policy-models and ensure create_multi_policy_frozen_engines() is called in main()."
                )
            policy_engines_dict = multi_policy_frozen_engines.get_policy_engines_dict()
            judge_actor = rubric_judge_generate_text_actor if multi_judge_engines is None else None
            rubric_gen_actor = policy_generate_text_actor
            if rubric_gen_actor is None:
                raise ValueError(
                    "policy_generate_text_actor must be provided for multi_policy_frozen in combined mode. "
                    "This is required to generate rubrics without deadlocking the rubric_actor."
                )
            multi_policy_frozen_provider = MultiPolicyFrozenDataProvider.remote(
                policy_engines=policy_engines_dict,
                rubric_generate_text_actor=rubric_gen_actor,
                judge_actor=judge_actor,
                multi_judge_engines=multi_judge_engines,
                sampling_strategy=multi_policy_frozen_engines.sampling_strategy,
                question_key=question_key,
                rubric_sampling_params=rubric_sp,
                judge_sampling_params=judge_sp,
                rubric_prompt_key=rubric_prompt_key,
            )
            LOGGER.info(
                "[create_data_provider] Created multi_policy_frozen provider (weight=%.2f) with %d policies",
                weights.multi_policy_frozen,
                len(multi_policy_frozen_engines.policy_models),
            )

        return CombinedDataProvider.remote(
            policy_actor=policy_actor,
            weights=weights,
            question_key=question_key,
            replay_buffer_provider=replay_buffer_provider,
            inferred_question_provider=inferred_question_provider,
            rubric_provider=rubric_provider,
            multi_policy_frozen_provider=multi_policy_frozen_provider,
        )
    elif rejected_answer_method == "multi_policy_frozen":
        # Multi-policy frozen mode - requires multi_policy_frozen_engines
        if multi_policy_frozen_engines is None:
            raise ValueError(
                "multi_policy_frozen_engines must be provided when rejected_answer_method='multi_policy_frozen'. "
                "This should be created by create_multi_policy_frozen_engines() in main()."
            )

        # Get policy engines dict for data provider
        policy_engines_dict = multi_policy_frozen_engines.get_policy_engines_dict()

        # Determine judge setup (either single judge or multi-judge)
        judge_actor = rubric_judge_generate_text_actor if multi_judge_engines is None else None

        # Require rubric_generate_text_actor for rubric generation (avoids deadlock)
        rubric_gen_actor = policy_generate_text_actor
        if rubric_gen_actor is None:
            raise ValueError(
                "policy_generate_text_actor must be provided for multi_policy_frozen mode. "
                "This is required to generate rubrics without deadlocking the rubric_actor."
            )

        LOGGER.info(
            "[create_data_provider] Creating multi_policy_frozen provider with %d policies (strategy=%s)",
            len(multi_policy_frozen_engines.policy_models),
            multi_policy_frozen_engines.sampling_strategy,
        )

        if multi_judge_engines:
            LOGGER.info(
                "[create_data_provider] Using multi-judge mode with %d judges",
                len(multi_judge_engines.judge_models),
            )

        return MultiPolicyFrozenDataProvider.remote(
            policy_engines=policy_engines_dict,
            rubric_generate_text_actor=rubric_gen_actor,
            judge_actor=judge_actor,
            multi_judge_engines=multi_judge_engines,
            sampling_strategy=multi_policy_frozen_engines.sampling_strategy,
            question_key=question_key,
            rubric_sampling_params=rubric_sp,
            judge_sampling_params=judge_sp,
            rubric_prompt_key=rubric_prompt_key,
        )
    else:
        raise ValueError(
            f"Invalid rejected_answer_method '{rejected_answer_method}'. "
            f"Must be one of: {{'replay_buffer', 'inferred_question', 'rubric', 'combined', 'multi_policy_frozen'}}"
        )
