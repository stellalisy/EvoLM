from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from open_instruct.queue_types import GenerationResult
from open_instruct.search_rewards.rubric_judge_rewards import (
    _remove_redacted_reasoning,
    parse_rlcer_rubric_items_with_scores,
)
from open_instruct.search_rewards.utils.rubric_chat_templates import format_messages
from open_instruct.search_rewards.utils.run_utils import run_litellm_async


@dataclass
class RLCERRubricSpec:
    """Container for an RL-CER rubric plus parsed per-item scores."""

    question: str
    rubric_text: str
    rubric_items: list[str]
    rubric_scores: list[float]
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: list[int] | None = None
    prompt_text: str | None = None
    generation_result: GenerationResult | None = None


async def generate_rlcer_rubric_spec(
    question: str,
    reference_response: str,
    *,
    api_rubric_generator: str | None,
    rubric_model: str,
    tokenizer: Any,
    generation_kwargs: dict[str, Any],
    build_sampling_params: Callable[..., Any],
    policy_generate_text_actor: Any | None,
    metadata: dict[str, Any] | None = None,
) -> RLCERRubricSpec:
    """Generate an RL-CER-format rubric and parse its scored items."""
    response = _remove_redacted_reasoning(reference_response)
    response = response.strip() if response else ""
    prompt_token_ids: list[int] | None = None
    prompt_text: str | None = None
    generation_result: GenerationResult | None = None

    if api_rubric_generator:
        messages = format_messages(
            "rlcer_rubric_generation",
            {"question": question, "response": response},
            tokenize=False,
        )
        prompt_text = "\n\n".join(message["content"] for message in messages)
        rubric_text = await run_litellm_async(
            model_name=api_rubric_generator,
            messages=messages,
        )
        model_name = api_rubric_generator
    else:
        kwargs = dict(generation_kwargs)
        kwargs.setdefault("n", 1)
        sampling_params = build_sampling_params(config_key="train", **kwargs)
        prompt_token_ids, messages = format_messages(
            "rlcer_rubric_generation",
            {"question": question, "response": response},
            tokenizer=tokenizer,
            add_generation_prompt=True,
            return_messages=True,
        )
        prompt_text = "\n\n".join(message["content"] for message in messages)
        if not policy_generate_text_actor:
            raise RuntimeError("policy_generate_text_actor is not initialized")
        if hasattr(policy_generate_text_actor, "generate_text_result_from_token_ids"):
            result_payload = await policy_generate_text_actor.generate_text_result_from_token_ids.remote(
                prompt_token_ids, sampling_params
            )
            rubric_text = str(result_payload.get("text", ""))
            generation_result = result_payload.get("result")
        else:
            rubric_text = await policy_generate_text_actor.generate_text_from_token_ids.remote(
                prompt_token_ids, sampling_params
            )
        model_name = rubric_model

    rubric_items, rubric_scores = parse_rlcer_rubric_items_with_scores(rubric_text)
    return RLCERRubricSpec(
        question=question,
        rubric_text=rubric_text,
        rubric_items=rubric_items,
        rubric_scores=rubric_scores,
        model_name=model_name,
        metadata={
            "reference_response_present": bool(response),
            **(metadata or {}),
        },
        prompt_token_ids=prompt_token_ids,
        prompt_text=prompt_text,
        generation_result=generation_result,
    )


async def precompute_rlcer_evolving_rollout_rubrics(
    *,
    questions: list[str],
    answers: list[str],
    rubric_actor: Any,
) -> list[dict[str, Any]]:
    """Propose one RL-CER rubric per rollout answer."""
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    rubric_futures = [
        rubric_actor.create_rlcer_rubric.remote(question, answer)
        for question, answer in zip(questions, answers)
    ]

    rubric_specs = await asyncio.gather(*rubric_futures)
    return [
        {
            "question": rubric_spec.question,
            "rubric_text": rubric_spec.rubric_text,
            "rubric_items": list(rubric_spec.rubric_items),
            "rubric_scores": [float(score) for score in rubric_spec.rubric_scores],
            "model_name": rubric_spec.model_name,
            "metadata": dict(rubric_spec.metadata),
            "prompt_token_ids": list(rubric_spec.prompt_token_ids or []),
            "prompt_text": rubric_spec.prompt_text,
            "generation_result": rubric_spec.generation_result,
        }
        for rubric_spec in rubric_specs
    ]
