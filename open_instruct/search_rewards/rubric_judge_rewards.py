"""
Reward function for rubric generation and LLM-as-a-judge evaluation.

This module implements a reward system where:
1. A model generates a rubric from a question
2. An LLM-as-a-judge evaluates both accepted and rejected answers
3. Reward is produced if the accepted answer is correct

Multi-judge support:
- Multiple judges can evaluate the same rubric+answer pair
- Rewards can be aggregated as an average vote, a hard majority vote, an
  average-minus-variance penalty based on Fleiss's kappa, or the legacy
  agreement bonus over pairwise accuracy
"""

import asyncio
import json as _json
import logging
import math
import os
import re
import time
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from open_instruct.search_rewards.utils.rubric_chat_templates import format_messages
from open_instruct.search_rewards.utils.run_utils import extract_json_from_response, run_litellm_async

LOGGER = logging.getLogger(__name__)


def is_valid_rubric_format(rubric_text: str) -> float:
    """Check if rubric text is valid JSON with criteria, weights, and scoring levels.

    Returns 1.0 for a well-formed rubric, 0.0 otherwise.  The check is applied
    to the text *after* stripping ``<think>...</think>`` reasoning blocks so
    that Qwen3 thinking-mode outputs are handled correctly.
    """
    cleaned = _remove_redacted_reasoning(rubric_text or "").strip()
    if not cleaned:
        return 0.0

    parsed = extract_json_from_response(cleaned)
    if not parsed or not isinstance(parsed, dict):
        return 0.0

    criteria = parsed.get("criteria")
    if not isinstance(criteria, list) or len(criteria) < 2:
        return 0.0

    total_weight = 0.0
    for entry in criteria:
        if not isinstance(entry, dict):
            return 0.0
        if "criterion" not in entry or "weight" not in entry:
            return 0.0
        try:
            w = float(entry["weight"])
        except (TypeError, ValueError):
            return 0.0
        if w <= 0:
            return 0.0
        total_weight += w

    if abs(total_weight - 1.0) > 0.15:
        return 0.0

    return 1.0

# LOGGER.setLevel(logging.DEBUG)

# Global semaphore limiting concurrent vLLM actor requests to prevent
# task queue buildup (>10K pending tasks) and OOM in heavy scoring modes
# (rar_implicit, rlcer, rrd). API calls are not limited.
_VLLM_ACTOR_SEMAPHORE: asyncio.Semaphore | None = None
VALID_MULTI_JUDGE_AGGREGATIONS = frozenset(
    {"average_vote", "majority_vote", "average_minus_variance", "agreement_bonus", "margin_kappa_format"}
)
VALID_MULTI_JUDGE_TIE_BREAKERS = frozenset({"mean_score", "first_judge"})


def _get_vllm_actor_semaphore() -> asyncio.Semaphore:
    """Lazy-init a per-event-loop semaphore for vLLM actor calls."""
    global _VLLM_ACTOR_SEMAPHORE
    if _VLLM_ACTOR_SEMAPHORE is None:
        limit = int(os.environ.get("MAX_CONCURRENT_JUDGE_REQUESTS", "128"))
        _VLLM_ACTOR_SEMAPHORE = asyncio.Semaphore(limit)
    return _VLLM_ACTOR_SEMAPHORE


def _extract_json_rubric(text: str) -> str | None:
    """Extract a JSON rubric object from text that may contain a reasoning prefix.

    Models like OLMo-3-7B-Think embed chain-of-thought reasoning directly in
    their output (without <think> tags) before the actual JSON rubric.  This
    helper finds the outermost ``{"criteria": ...}`` JSON structure and returns
    it, discarding any surrounding reasoning text.

    Returns the extracted JSON string, or None if no valid JSON rubric is found.
    """
    import json as _json

    match = re.search(r'\{\s*"criteria"\s*:', text)
    if match:
        start = match.start()
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : j + 1]
                    try:
                        _json.loads(candidate)
                        return candidate
                    except (ValueError, _json.JSONDecodeError):
                        break
    return None


def _remove_redacted_reasoning(text: str) -> str:
    """
    Remove <think> spans and reasoning prefixes from text.

    Handles two patterns:
    1. Explicit ``<think>...</think>`` blocks (Qwen-style thinking models).
    2. Implicit reasoning prefixes followed by a JSON rubric (OLMo-style
       thinking models that don't use <think> tags).

    Args:
        text: The text to clean

    Returns:
        Text with reasoning removed
    """
    if not text:
        return text
    # Remove closed <think>...</think> spans first.
    cleaned = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # If an unmatched <think> remains, the model never finished reasoning.
    # Only salvage content if structured tags (RUBRIC/EVALUATION) appear after <think>;
    # otherwise the entire tail is unstructured thinking — discard it so that
    # downstream rubric parsing sees empty text and r_format = 0.
    open_match = re.search(r"<\s*think\s*>", cleaned, flags=re.IGNORECASE)
    if open_match:
        before_think = cleaned[: open_match.start()].strip()
        tail = cleaned[open_match.end() :]
        structured_positions: list[int] = []
        for pattern in [r"<\s*RUBRIC\s*>", r"<\s*/\s*RUBRIC\s*>", r"<\s*EVALUATION\s*>", r"<\s*/\s*EVALUATION\s*>"]:
            match = re.search(pattern, tail, flags=re.IGNORECASE)
            if match:
                structured_positions.append(match.start())

        if structured_positions:
            cleaned = tail[min(structured_positions) :]
        else:
            # No structured output after unclosed <think> — the model only produced
            # reasoning, not rubric content.  Keep any text that appeared *before*
            # the <think> tag (rare but possible), discard everything inside.
            cleaned = before_think

    # Final cleanup in case orphan tags remain.
    cleaned = re.sub(r"<\s*/?\s*think\s*>", "", cleaned, flags=re.IGNORECASE)

    # If the cleaned text contains a JSON rubric object preceded by a
    # reasoning prefix (common with models that don't use <think> tags,
    # like OLMo-3-7B-Think), extract just the JSON part.  Only strip when
    # the JSON object starts well into the text (reasoning prefix > 200 chars).
    criteria_match = re.search(r'\{\s*"criteria"\s*:', cleaned)
    if criteria_match and criteria_match.start() > 200:
        json_rubric = _extract_json_rubric(cleaned)
        if json_rubric is not None:
            return json_rubric

    # Fallback for thinking models without <think> tags (e.g. OLMo-3-7B-Think):
    # if the text is very long and no structured content was extracted, the model
    # likely produced a long reasoning prefix followed by the actual content.
    # Take the last 8000 characters to capture the answer/rubric while discarding
    # the bulk of the thinking trace.
    _THINKING_FALLBACK_CHARS = 8000
    if len(cleaned) > _THINKING_FALLBACK_CHARS * 2:
        cleaned = cleaned[-_THINKING_FALLBACK_CHARS:]

    return cleaned


# Throttling for verbose logging - log one complete computation every 3 seconds
_last_verbose_log_time = 0.0
_VERBOSE_LOG_INTERVAL = 3.0
_logging_enabled = False


def verbose_debug(message: str) -> None:
    """
    Throttled debug logging - only logs if currently in a logging window.

    Args:
        message: The debug message to log
    """
    global _logging_enabled
    if _logging_enabled:
        LOGGER.debug(message)


def _should_enable_logging() -> bool:
    """
    Check if logging should be enabled for the next computation.
    Enables logging only if logging is not already enabled for another computation
    AND 3 seconds have passed since last log.

    Returns:
        True if logging should be enabled, False otherwise
    """
    global _last_verbose_log_time, _logging_enabled
    # If logging is already enabled for another computation, don't enable it for this one
    if _logging_enabled:
        return False
    current_time = time.time()
    if current_time - _last_verbose_log_time >= _VERBOSE_LOG_INTERVAL:
        _logging_enabled = True
        _last_verbose_log_time = current_time
        return True
    return False


def _disable_logging() -> None:
    """Disable logging after a computation completes."""
    global _logging_enabled
    _logging_enabled = False


_RUBRIC_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|(?:\(?\d+[.)]))\s+(.*)")
_YES_NO_EVALUATION_PATTERN = re.compile(
    r"<\s*EVALUATION\s*>\s*(YES|NO)\s*<\s*/\s*EVALUATION\s*>", re.IGNORECASE | re.DOTALL
)
_PLACEHOLDER_RUBRIC_PATTERN = re.compile(r"^(?:new\s+)?rubric\s*\d*$", re.IGNORECASE)


def parse_rubric_items(rubric_text: str) -> list[str]:
    """Parse rubric text into atomic rubric items."""
    cleaned = _remove_redacted_reasoning(rubric_text or "").strip()
    if not cleaned:
        return []

    tagged_items = re.findall(r"<\s*RUBRIC\s*>\s*(.*?)\s*<\s*/\s*RUBRIC\s*>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if tagged_items:
        return [item.strip() for item in tagged_items if item and item.strip()]

    items: list[str] = []
    current_item: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = _RUBRIC_ITEM_PATTERN.match(line)
        if matched:
            if current_item:
                item = " ".join(current_item).strip()
                if item:
                    items.append(item)
            current_item = [matched.group(1).strip()]
        elif current_item:
            current_item.append(line)
        else:
            current_item = [line]

    if current_item:
        item = " ".join(current_item).strip()
        if item:
            items.append(item)

    if not items:
        return [cleaned]
    return items


def _is_placeholder_rubric_item(item: str) -> bool:
    cleaned = " ".join((item or "").strip().split())
    if not cleaned:
        return True
    if _PLACEHOLDER_RUBRIC_PATTERN.fullmatch(cleaned):
        return True
    lowered = cleaned.lower()
    # Guard against models emitting schema placeholders instead of real criteria.
    if lowered in {"rubric", "new rubric", "criterion", "new criterion"}:
        return True
    return False


def _filter_valid_rubric_items(items: list[str]) -> list[str]:
    return [item for item in _dedupe_rubric_items(items) if not _is_placeholder_rubric_item(item)]


def parse_rlcer_rubric_items_with_scores(rubric_text: str) -> tuple[list[str], list[float]]:
    """Parse RLCER rubric JSON into rubric items and per-item scores.

    Returns an empty result when the text is not valid RLCER JSON.
    """
    cleaned = _remove_redacted_reasoning(rubric_text or "").strip()
    if not cleaned:
        return [], []

    parsed = extract_json_from_response(cleaned)
    if parsed and "rubrics" in parsed and isinstance(parsed["rubrics"], list):
        items: list[str] = []
        scores: list[float] = []
        for entry in parsed["rubrics"]:
            if not isinstance(entry, dict):
                continue
            criterion = entry.get("criterion", "")
            if not criterion or not str(criterion).strip():
                continue
            points = entry.get("points", 1)
            try:
                points = float(points)
            except (TypeError, ValueError):
                points = 1.0
            items.append(str(criterion).strip())
            scores.append(points)

        filtered_items = _filter_valid_rubric_items(items)
        if filtered_items:
            filtered_set = set(filtered_items)
            kept_indices: list[int] = []
            seen_cleaned: set[str] = set()
            for i, item in enumerate(items):
                cleaned_item = " ".join(item.strip().split())
                if cleaned_item in filtered_set and cleaned_item not in seen_cleaned:
                    kept_indices.append(i)
                    seen_cleaned.add(cleaned_item)
            filtered_scores = [scores[i] for i in kept_indices]
            return filtered_items, filtered_scores

    return [], []


def parse_rlcer_rubric_entries(rubric_text: str) -> list[dict[str, Any]]:
    """Parse RL-CER rubric JSON into structured rubric entries."""
    cleaned = _remove_redacted_reasoning(rubric_text or "").strip()
    if not cleaned:
        return []

    parsed = extract_json_from_response(cleaned)
    if parsed and "rubrics" in parsed and isinstance(parsed["rubrics"], list):
        entries: list[dict[str, Any]] = []
        seen_cleaned: set[str] = set()
        for entry in parsed["rubrics"]:
            if not isinstance(entry, dict):
                continue
            criterion = str(entry.get("criterion", "")).strip()
            if not criterion or _is_placeholder_rubric_item(criterion):
                continue
            normalized = " ".join(criterion.split())
            if normalized in seen_cleaned:
                continue
            seen_cleaned.add(normalized)
            points = entry.get("points", 1)
            try:
                points = float(points)
            except (TypeError, ValueError):
                points = 1.0
            category = str(entry.get("category", "Other Question-Specific Aspects")).strip()
            if not category:
                category = "Other Question-Specific Aspects"
            entries.append(
                {
                    "category": category,
                    "criterion": normalized,
                    "points": points,
                }
            )
        if entries:
            return entries

    return []


def _extract_binary_judgment(response: str) -> float:
    """Extract YES/NO output from binary judge response and map to 1.0/0.0."""
    if not response:
        return 0.0

    response = _remove_redacted_reasoning(response)
    if not response:
        return 0.0

    match = _YES_NO_EVALUATION_PATTERN.search(response)
    if match:
        return 1.0 if match.group(1).upper() == "YES" else 0.0

    fallback = re.search(r"\b(YES|NO)\b", response, flags=re.IGNORECASE)
    if fallback:
        return 1.0 if fallback.group(1).upper() == "YES" else 0.0
    return 0.0


def _coerce_rlcer_judgement(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) != 0.0 else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return 1.0
        if lowered in {"false", "no", "0"}:
            return 0.0
    return None


def _estimate_covariance(score_matrix: np.ndarray, shrinkage: float = 0.1, ridge: float = 1e-4) -> np.ndarray:
    """Estimate a stable covariance matrix from rubric binary score vectors.

    Uses uncentered second moment (1/N) X^T X as defined in RRD Lemma 2,
    NOT mean-centered covariance.
    """
    num_samples, num_rubrics = score_matrix.shape
    if num_rubrics == 0:
        return np.zeros((0, 0), dtype=float)

    if num_samples <= 1:
        cov = np.eye(num_rubrics, dtype=float)
    else:
        # RRD Lemma 2: Sigma_hat = (1/N) sum Xi Xi^T (no mean subtraction)
        cov = (score_matrix.T @ score_matrix) / num_samples
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if cov.shape != (num_rubrics, num_rubrics):
            cov = np.eye(num_rubrics, dtype=float)

    cov = 0.5 * (cov + cov.T)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.eye(num_rubrics, dtype=float)
    cov += ridge * np.eye(num_rubrics, dtype=float)
    return cov


def _compute_wu_weights(covariance: np.ndarray) -> np.ndarray:
    """Compute whitened-uniform weights: w_wu is proportional to Sigma^{-1/2} 1.

    Per RRD Eq. 1, weights are constrained to w_k >= 0.
    """
    num_rubrics = covariance.shape[0]
    if num_rubrics == 0:
        return np.array([], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    raw_weights = inv_sqrt @ np.ones(num_rubrics, dtype=float)

    if not np.all(np.isfinite(raw_weights)) or np.allclose(raw_weights, 0.0):
        raw_weights = np.ones(num_rubrics, dtype=float)

    # Paper requires w_k >= 0 (Eq. 1)
    raw_weights = np.maximum(raw_weights, 0.0)

    l1_norm = float(np.sum(raw_weights))
    if l1_norm <= 1e-12:
        return np.ones(num_rubrics, dtype=float) / float(num_rubrics)
    return raw_weights / l1_norm


def _aggregate_binary_scores_with_wu_weights(binary_scores: list[float], wu_weights: np.ndarray) -> float:
    """Aggregate per-rubric binary scores into [0, 1] using WU weights.

    With non-negative weights (per paper Eq. 1), this is simply a weighted average.
    """
    if len(binary_scores) == 0:
        return 0.0

    scores = np.asarray(binary_scores, dtype=float)
    raw = float(np.dot(wu_weights, scores))

    w_sum = float(wu_weights.sum())
    if w_sum <= 1e-12:
        return 0.0

    return float(np.clip(raw / w_sum, 0.0, 1.0))


async def judge_answer_binary_with_rubric_item(
    question: str,
    rubric_item: str,
    answer: str,
    model_name: str | None = None,
    answer_type: str = "answer",
    prompt_template: str = "judge_binary",
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Judge if answer satisfies a single rubric item using strict YES/NO output."""
    del question  # Prompt currently only needs rubric + answer.

    if model_name is None:
        model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")

    answer = _remove_redacted_reasoning(answer)
    rubric_item = _remove_redacted_reasoning(rubric_item)

    try:
        messages = format_messages(
            prompt_template, {"question": "", "rubric": rubric_item, "answer": answer}, tokenize=False
        )

        if rubric_judge_generate_text_actor:
            assert sampling_params is not None, "Sampling params are required for rubric judge actor"
            async with _get_vllm_actor_semaphore():
                response = await rubric_judge_generate_text_actor.generate_text_from_messages.remote(
                    messages, sampling_params
                )
        else:
            response = await run_litellm_async(model_name=model_name, messages=messages)

        score = _extract_binary_judgment(response)
        return {"score": score, "reasoning": f"{answer_type} satisfies rubric={bool(score)}", "raw_response": response}
    except Exception as e:
        LOGGER.error(f"Error in binary rubric judgment: {e}")
        return {"score": 0.0, "reasoning": f"Error: {str(e)}", "raw_response": ""}


async def _score_answer_against_rubric_items(
    question: str,
    rubric_items: list[str],
    answer: str,
    model_name: str | None = None,
    prompt_template: str = "judge_binary",
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
    answer_type: str = "answer",
) -> dict[str, Any]:
    """Score one answer against all rubric items with binary judgments."""
    if not rubric_items:
        return {"binary_scores": [], "reasoning": "No rubric items"}

    tasks = [
        judge_answer_binary_with_rubric_item(
            question=question,
            rubric_item=item,
            answer=answer,
            model_name=model_name,
            answer_type=answer_type,
            prompt_template=prompt_template,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )
        for item in rubric_items
    ]
    results = await asyncio.gather(*tasks)
    binary_scores = [float(r.get("score", 0.0)) for r in results]
    satisfied = int(sum(1 for s in binary_scores if s >= 0.5))
    reasoning = f"Satisfied {satisfied}/{len(rubric_items)} rubric items"
    return {"binary_scores": binary_scores, "reasoning": reasoning}


async def _score_answer_against_rubric_items_likert(
    question: str,
    rubric_items: list[str],
    answer: str,
    model_name: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
    answer_type: str = "answer",
) -> dict[str, Any]:
    """Score one answer against all rubric items with Likert 1-10 ratings normalized to [0,1].

    Per-item Likert conformity scoring as described in the Query-Specific Rubrics
    paper (Eq. 2). Each rubric item gets a 1-10 rating linearly mapped to [0, 1].
    """
    if not rubric_items:
        return {"likert_scores": [], "reasoning": "No rubric items"}

    if model_name is None:
        model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")

    answer_cleaned = _remove_redacted_reasoning(answer)

    async def _score_single_item(item: str) -> float:
        try:
            messages = format_messages(
                "judge_likert_per_item",
                {"answer": answer_cleaned, "rubric": _remove_redacted_reasoning(item)},
                tokenize=False,
            )
            if rubric_judge_generate_text_actor:
                assert sampling_params is not None
                response = await rubric_judge_generate_text_actor.generate_text_from_messages.remote(
                    messages, sampling_params
                )
            else:
                response = await run_litellm_async(model_name=model_name, messages=messages)
            return _extract_likert_rating(response)
        except Exception as e:
            LOGGER.error("Error in per-item Likert scoring: %s", e)
            return 0.0

    likert_scores = await asyncio.gather(*[_score_single_item(item) for item in rubric_items])
    likert_scores = [float(s) for s in likert_scores]
    mean_score = sum(likert_scores) / len(likert_scores) if likert_scores else 0.0
    reasoning = f"{answer_type}: mean Likert={mean_score:.3f} across {len(rubric_items)} items"
    return {"likert_scores": likert_scores, "reasoning": reasoning}


async def judge_answer_with_rubric(
    question: str,
    rubric: str,
    answer: str,
    model_name: str | None = None,
    answer_type: str = "answer",
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
    scoring_mode: str = "rubric_judge",
    concurrency_semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    """
    Use LLM-as-a-judge to evaluate an answer based on a rubric.

    The ``scoring_mode`` selects how the rubric + answer are scored:
      - ``"rubric_judge"`` (default): JSON-based scoring with the ``judge`` template.
      - ``"rar_implicit"``: Holistic 1-10 Likert rating via the ``rar_implicit`` template.
      - ``"rar_explicit"``: Binary judging per rubric item + weighted sum (Eq. 1 of RaR paper).

    Args:
        question: The original question
        rubric: The rubric to use for evaluation
        answer: The answer to evaluate
        model_name: The model to use for judging (defaults to RUBRIC_JUDGE_MODEL env var)
        answer_type: Type of answer being judged (e.g., "accepted" or "rejected") for logging
        rubric_judge_generate_text_actor: GenerateTextActor for load-balanced generation
        sampling_params: vLLM sampling parameters
        scoring_mode: Scoring strategy — ``"rubric_judge"``, ``"rar_implicit"``, or ``"rar_explicit"``.

    Returns:
        Dictionary containing score and reasoning
    """
    # ── RaR-Implicit: delegate to Likert scorer ──────────────────────────
    if scoring_mode == "rar_implicit":
        return await _score_single_answer_likert(
            question=question,
            answer=answer,
            template_name="rar_implicit",
            rubric_text=rubric,
            model_name=model_name,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )

    # ── RaR-Explicit: binary per-item + weighted sum ─────────────────────
    if scoring_mode == "rar_explicit":
        return await _judge_answer_rar_explicit(
            question=question,
            rubric_text=rubric,
            answer=answer,
            model_name=model_name,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )

    # ── Default: rubric_judge JSON scoring ───────────────────────────────
    if model_name is None:
        model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")

    # Remove thinking trace from answer
    rubric = _remove_redacted_reasoning(rubric)
    answer = _remove_redacted_reasoning(answer)

    verbose_debug(f"[Judge Evaluation] Evaluating {answer_type} answer...")
    verbose_debug(f"[Judge Evaluation] Question: {question}")
    verbose_debug(f"[Judge Evaluation] Answer: {answer}")
    verbose_debug(f"[Judge Evaluation] Using model: {model_name}")

    try:
        verbose_debug(f"[Judge Evaluation] Calling LLM judge for {answer_type} answer...")
        messages = format_messages("judge", {"question": question, "rubric": rubric, "answer": answer}, tokenize=False)

        # Use GenerateTextActor if available, otherwise fall back to API
        if rubric_judge_generate_text_actor:
            assert sampling_params is not None, "Sampling params are required for rubric judge actor"
            verbose_debug("[Judge Evaluation] Using GenerateTextActor for load-balanced generation")
            sem = concurrency_semaphore if concurrency_semaphore is not None else _get_vllm_actor_semaphore()
            async with sem:
                response = await rubric_judge_generate_text_actor.generate_text_from_messages.remote(
                    messages, sampling_params
                )
        else:
            verbose_debug(f"[Judge Evaluation] Using API call to model {model_name}")
            response = await run_litellm_async(model_name=model_name, messages=messages)

        verbose_debug(f"[Judge Evaluation] Received response, extracting score for {answer_type} answer...")
        verbose_debug(f"""[Judge Evaluation] call: 
================== messages ==================
{messages}
================== response ==================
{response}
================== """)
        result = extract_json_from_response(response)
        if result and "score" in result:
            score = float(result["score"])
            reasoning = result.get("reasoning", "")
            verbose_debug(f"[Judge Evaluation] {answer_type.capitalize()} answer score: {score}")
            verbose_debug(f"[Judge Evaluation] {answer_type.capitalize()} answer reasoning: {reasoning}")
            return {"score": score, "reasoning": reasoning}
        else:
            LOGGER.debug(f"Failed to extract score from judge response: {response}")
            return {"score": float("nan"), "reasoning": "Failed to extract score", "parse_error": True}
    except Exception as e:
        LOGGER.error(f"Error judging answer: {e}")
        return {"score": float("nan"), "reasoning": f"Error: {str(e)}", "parse_error": True}


async def _judge_answer_rar_explicit(
    question: str,
    rubric_text: str,
    answer: str,
    model_name: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Score a single answer using RaR-Explicit: binary judging per rubric item + weighted sum."""
    rubric_items = _parse_rar_rubric_items_from_data(rubric_text)
    if not rubric_items:
        parsed = parse_rubric_items(rubric_text)
        rubric_items = [{"description": item, "weight": 1} for item in parsed]
    if not rubric_items:
        return {"score": 0.0, "reasoning": "No rubric items", "rubric_items": [], "weights": [], "binary_scores": []}

    descriptions = [item.get("description", str(item)) for item in rubric_items]
    weights: list[float] = []
    for item in rubric_items:
        w = item.get("weight", 1)
        if isinstance(w, str):
            weights.append(RAR_EXPLICIT_CATEGORY_WEIGHTS.get(w.lower(), 0.7))
        elif isinstance(w, (int, float)) and w < 0:
            weights.append(RAR_EXPLICIT_CATEGORY_WEIGHTS["pitfall"])
        else:
            category = _get_category_from_description(item.get("description", ""))
            weights.append(RAR_EXPLICIT_CATEGORY_WEIGHTS.get(category, 0.7))

    eval_result = await _score_answer_against_rubric_items(
        question=question,
        rubric_items=descriptions,
        answer=answer,
        model_name=model_name,
        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
        sampling_params=sampling_params,
        answer_type="rar_explicit",
    )
    binary_scores = eval_result["binary_scores"]
    numerator = sum(w * s for w, s in zip(weights, binary_scores))
    denominator = sum(abs(w) for w in weights)
    score = max(0.0, min(1.0, numerator / denominator)) if denominator > 1e-12 else 0.0
    satisfied = int(sum(1 for s in binary_scores if s >= 0.5))
    return {
        "score": score,
        "reasoning": f"RaR-Explicit: {satisfied}/{len(descriptions)} satisfied, weighted={score:.3f}",
        "rubric_items": descriptions,
        "weights": weights,
        "binary_scores": binary_scores,
    }


async def compute_rubric_judge_reward_async(
    question: str,
    accepted_answer: str,
    rejected_answer: str,
    generated_rubric: str | None = None,
    judge_model: str | None = None,
    rubric_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
    score_separation: bool = False,
    score_separation_weight: float = 0.3,
    margin_reward: bool = False,
) -> dict[str, Any]:
    """
    Compute reward using rubric generation and LLM-as-a-judge.

    This function:
    1. Generates a rubric from the question (if not provided)
    2. Uses LLM-as-a-judge to evaluate both accepted and rejected answers
    3. Produces reward if accepted answer is correct (score > rejected score)

    When score_separation=True, the reward blends pairwise accuracy with a score
    margin bonus to encourage rubrics that produce well-separated scores:
        reward = (1 - w) * pairwise_accuracy + w * clamp(accepted - rejected, 0, 1)

    When margin_reward=True, the reward is the raw margin (accepted - rejected),
    which can be negative. This directly incentivizes rubrics that produce
    well-separated scores. Takes precedence over score_separation.

    Args:
        question: The question being answered
        accepted_answer: The accepted/correct answer
        rejected_answer: The rejected/incorrect answer
        generated_rubric: Optional pre-generated rubric (if None, will be generated)
        judge_model: Model to use for judging (defaults to RUBRIC_JUDGE_MODEL env var)
        rubric_model: Model to use for rubric generation (defaults to RUBRIC_JUDGE_MODEL env var)
        rubric_judge_generate_text_actor: GenerateTextActor for load-balanced generation
        sampling_params: vLLM sampling parameters
        score_separation: Whether to add score separation bonus
        score_separation_weight: Weight of score separation bonus (0-1)

    Returns:
        Dictionary containing:
        - reward: The computed reward
        - accepted_score: Score for accepted answer
        - rejected_score: Score for rejected answer
        - score_margin: accepted_score - rejected_score (for logging)
        - rubric: The rubric used for evaluation
        - accepted_reasoning: Reasoning for accepted answer evaluation
        - rejected_reasoning: Reasoning for rejected answer evaluation
        - error: Error message if any step failed
    """
    # Enable logging for this computation if 3 seconds have passed
    _should_enable_logging()
    try:
        verbose_debug("[Rubric Judge Reward] Starting reward computation")
        verbose_debug(f"[Rubric Judge Reward] Question: {question}")
        verbose_debug(f"[Rubric Judge Reward] Accepted answer: {accepted_answer}")
        verbose_debug(f"[Rubric Judge Reward] Rejected answer: {rejected_answer}")

        result = {
            "reward": 0.0,
            "accepted_score": 0.0,
            "rejected_score": 0.0,
            "rubric": None,
            "accepted_reasoning": "",
            "rejected_reasoning": "",
            "error": None,
        }

        assert generated_rubric is not None, "Generated rubric is required"
        result["rubric"] = _remove_redacted_reasoning(generated_rubric)
        verbose_debug(f"[Rubric Judge Reward] Using provided rubric: {result['rubric']}")

        # Evaluate both answers in parallel
        verbose_debug("[Rubric Judge Reward] Evaluating accepted and rejected answers in parallel...")
        accepted_task = judge_answer_with_rubric(
            question,
            result["rubric"],
            accepted_answer,
            judge_model,
            answer_type="accepted",
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )
        rejected_task = judge_answer_with_rubric(
            question,
            result["rubric"],
            rejected_answer,
            judge_model,
            answer_type="rejected",
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )

        accepted_eval, rejected_eval = await asyncio.gather(accepted_task, rejected_task)

        result["accepted_score"] = accepted_eval["score"]
        result["rejected_score"] = rejected_eval["score"]
        result["accepted_reasoning"] = accepted_eval.get("reasoning", "")
        result["rejected_reasoning"] = rejected_eval.get("reasoning", "")

        # If either judge call had a parse failure, the pairwise comparison
        # is meaningless — propagate NaN so GRPO can neutralize this rollout.
        if math.isnan(result["accepted_score"]) or math.isnan(result["rejected_score"]):
            result["reward"] = float("nan")
            result["score_margin"] = float("nan")
            result["parse_error"] = True
            LOGGER.debug(
                "[Rubric Judge Reward] Parse failure on accepted or rejected judge call; "
                "setting reward=NaN to exclude from training."
            )
            return result

        margin = result["accepted_score"] - result["rejected_score"]
        result["score_margin"] = margin
        pairwise_correct = 1.0 if margin > 0 else 0.0

        if margin_reward:
            result["reward"] = margin
            verbose_debug(
                f"[Rubric Judge Reward] margin_reward: accepted={result['accepted_score']:.3f}, "
                f"rejected={result['rejected_score']:.3f}, reward=margin={margin:.3f}"
            )
        elif score_separation:
            clamped_margin = max(0.0, min(1.0, margin))
            w = score_separation_weight
            result["reward"] = (1.0 - w) * pairwise_correct + w * clamped_margin
            verbose_debug(
                f"[Rubric Judge Reward] score_separation: pairwise={pairwise_correct}, margin={margin:.3f}, "
                f"clamped={clamped_margin:.3f}, w={w}, reward={result['reward']:.3f}"
            )
        else:
            result["reward"] = pairwise_correct
            verbose_debug(
                f"[Rubric Judge Reward] Accepted score ({result['accepted_score']}) "
                f"{'>' if pairwise_correct else '<='} Rejected score ({result['rejected_score']}), reward = {pairwise_correct}"
            )

        verbose_debug(f"[Rubric Judge Reward] Final reward: {result['reward']}")
        verbose_debug(
            f"[Rubric Judge Reward] Accepted score: {result['accepted_score']}, Rejected score: {result['rejected_score']}, Margin: {margin:.3f}"
        )

        return result
    finally:
        # Disable logging after this computation completes
        _disable_logging()


def _dedupe_rubric_items(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = " ".join(item.strip().split())
        if cleaned and cleaned not in seen:
            deduped.append(cleaned)
            seen.add(cleaned)
    return deduped


def _resolve_actor(explicit_actor: Any | None, model_name: str | None, fallback_actor: Any | None) -> Any | None:
    """Pick the generation actor for a proposer/judge role.

    Priority: explicit actor > None (signals API via model_name) > fallback.
    """
    if explicit_actor is not None:
        return explicit_actor
    if model_name:
        return None
    return fallback_actor


async def _run_generation_from_messages(
    messages: list[dict[str, str]],
    model_name: str | None,
    rubric_judge_generate_text_actor: Any | None,
    sampling_params: Any | None,
) -> str:
    if model_name is None:
        model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")
    if rubric_judge_generate_text_actor:
        assert sampling_params is not None, "Sampling params are required for rubric judge actor"
        async with _get_vllm_actor_semaphore():
            return await rubric_judge_generate_text_actor.generate_text_from_messages.remote(messages, sampling_params)
    return await run_litellm_async(model_name=model_name, messages=messages)


def _build_rlcer_rubric_entries_from_items(
    rubric_items: list[str],
    rubric_scores: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            "category": "Other Question-Specific Aspects",
            "criterion": str(item),
            "points": float(score),
        }
        for item, score in zip(rubric_items, rubric_scores)
    ]


def _extract_rlcer_verifier_result(
    response: str,
    rubric_entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    parsed = extract_json_from_response(_remove_redacted_reasoning(response or ""))
    if not isinstance(parsed, dict):
        return None

    raw_judgements = parsed.get("judgement")
    if not isinstance(raw_judgements, list) or len(raw_judgements) != len(rubric_entries):
        return None

    binary_scores: list[float] = []
    for raw in raw_judgements:
        coerced = _coerce_rlcer_judgement(raw)
        if coerced is None:
            return None
        binary_scores.append(coerced)

    computed_final_score = float(
        sum(
            float(entry.get("points", 0.0))
            for entry, binary_score in zip(rubric_entries, binary_scores)
            if binary_score >= 0.5
        )
    )
    try:
        final_score = float(parsed.get("final_score", computed_final_score))
    except (TypeError, ValueError):
        final_score = computed_final_score

    return {
        "binary_scores": binary_scores,
        "final_score": final_score,
    }


async def _score_answer_with_rlcer_verifier(
    question: str,
    rubric_entries: list[dict[str, Any]],
    answer: str,
    model_name: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Score one answer against the full RL-CER rubric list with the paper-style verifier."""
    if not rubric_entries:
        return {"binary_scores": [], "final_score": 0.0, "reasoning": "No rubric entries", "raw_response": ""}

    rubric_payload = []
    for entry in rubric_entries:
        rubric_payload.append(
            {
                "category": str(entry.get("category", "Other Question-Specific Aspects")),
                "criterion": str(entry.get("criterion", "")),
                "points": float(entry.get("points", 0.0)),
            }
        )

    messages = format_messages(
        "rlcer_verifier",
        {
            "question": question,
            "answer": _remove_redacted_reasoning(answer),
            "rubrics": _json.dumps(rubric_payload, ensure_ascii=False),
        },
        tokenize=False,
    )
    response = await _run_generation_from_messages(
        messages=messages,
        model_name=model_name,
        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
        sampling_params=sampling_params,
    )
    parsed = _extract_rlcer_verifier_result(response, rubric_payload)
    if parsed is None:
        return {
            "binary_scores": [0.0] * len(rubric_payload),
            "final_score": 0.0,
            "reasoning": "Failed to parse RL-CER verifier response",
            "raw_response": response,
        }
    parsed["reasoning"] = (
        f"Satisfied {int(sum(1 for score in parsed['binary_scores'] if score >= 0.5))}/{len(rubric_payload)} rubric items"
    )
    parsed["raw_response"] = response
    return parsed


async def _score_all_answers_with_rlcer_verifier(
    question: str,
    rubric_entries: list[dict[str, Any]],
    answers: list[str],
    model_name: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> np.ndarray:
    if not rubric_entries or not answers:
        return np.zeros((len(answers), len(rubric_entries)), dtype=float)

    results = await asyncio.gather(
        *[
            _score_answer_with_rlcer_verifier(
                question=question,
                rubric_entries=rubric_entries,
                answer=answer,
                model_name=model_name,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
            for answer in answers
        ]
    )
    return np.asarray([result["binary_scores"] for result in results], dtype=float)


async def _rrd_generate_initial_rubrics(
    question: str,
    sample_responses: list[str],
    proposer_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[str]:
    messages = format_messages(
        "rrd_initial_rubric_generation", {"question": question, "responses": sample_responses}, tokenize=False
    )
    for _ in range(3):
        response = await _run_generation_from_messages(
            messages=messages,
            model_name=proposer_model,
            rubric_judge_generate_text_actor=proposer_generate_text_actor,
            sampling_params=sampling_params,
        )
        parsed = _filter_valid_rubric_items(parse_rubric_items(response))
        if parsed:
            return parsed
    return []


async def _rlcer_generate_rubrics_with_scores(
    question: str,
    reference_response: str,
    proposer_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> tuple[list[str], list[float]]:
    """Generate RLCER-style rubrics with importance scores (points).

    Returns structured rubric items with per-item importance scores,
    following the paper's format: positive points for merits, negative for flaws.
    Returns an empty rubric when generation repeatedly fails to produce valid
    RL-CER JSON.
    """
    cleaned_response = _remove_redacted_reasoning(reference_response or "").strip()
    messages = format_messages(
        "rlcer_rubric_generation", {"question": question, "response": cleaned_response}, tokenize=False
    )
    for _ in range(3):
        response = await _run_generation_from_messages(
            messages=messages,
            model_name=proposer_model,
            rubric_judge_generate_text_actor=proposer_generate_text_actor,
            sampling_params=sampling_params,
        )
        filtered_items, filtered_scores = parse_rlcer_rubric_items_with_scores(response)
        if filtered_items:
            return filtered_items, filtered_scores

    return [], []


async def _rrd_decompose_rubric(
    question: str,
    sample_responses: list[str],
    current_rubric: str,
    other_rubrics: list[str],
    proposer_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[str]:
    messages = format_messages(
        "rrd_rubric_decomposition",
        {
            "question": question,
            "responses": sample_responses,
            "current_rubric": current_rubric,
            "other_rubrics": other_rubrics,
        },
        tokenize=False,
    )
    for _ in range(3):
        response = await _run_generation_from_messages(
            messages=messages,
            model_name=proposer_model,
            rubric_judge_generate_text_actor=proposer_generate_text_actor,
            sampling_params=sampling_params,
        )
        parsed = _filter_valid_rubric_items(parse_rubric_items(response))
        if parsed:
            return parsed
    return []


async def _rrd_run_binary_filter(
    message_type: str,
    *,
    existing_rubrics: list[str],
    new_rubric: str,
    judge_model: str | None = None,
    judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> bool:
    messages = format_messages(
        message_type, {"existing_rubrics": existing_rubrics, "new_rubric": new_rubric}, tokenize=False
    )
    response = await _run_generation_from_messages(
        messages=messages,
        model_name=judge_model,
        rubric_judge_generate_text_actor=judge_generate_text_actor,
        sampling_params=sampling_params,
    )
    return bool(_extract_binary_judgment(response))


async def _rrd_is_misaligned_rubric(
    question: str,
    rubric_item: str,
    strong_reference_answer: str | None,
    weak_reference_answer: str | None,
    judge_model: str | None = None,
    judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> bool:
    if not strong_reference_answer or not weak_reference_answer:
        return False
    strong_eval, weak_eval = await asyncio.gather(
        judge_answer_binary_with_rubric_item(
            question=question,
            rubric_item=rubric_item,
            answer=strong_reference_answer,
            model_name=judge_model,
            answer_type="strong",
            rubric_judge_generate_text_actor=judge_generate_text_actor,
            sampling_params=sampling_params,
        ),
        judge_answer_binary_with_rubric_item(
            question=question,
            rubric_item=rubric_item,
            answer=weak_reference_answer,
            model_name=judge_model,
            answer_type="weak",
            rubric_judge_generate_text_actor=judge_generate_text_actor,
            sampling_params=sampling_params,
        ),
    )
    return float(weak_eval.get("score", 0.0)) > float(strong_eval.get("score", 0.0))


async def _rrd_count_satisfied_responses(
    question: str,
    rubric_item: str,
    responses: list[str],
    judge_model: str | None = None,
    judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> tuple[int, list[str]]:
    """Return (count, list_of_satisfied_responses) for the given rubric item."""
    if not responses:
        return 0, []
    tasks = [
        judge_answer_binary_with_rubric_item(
            question=question,
            rubric_item=rubric_item,
            answer=response,
            model_name=judge_model,
            answer_type="rrd_sample",
            rubric_judge_generate_text_actor=judge_generate_text_actor,
            sampling_params=sampling_params,
        )
        for response in responses
    ]
    evals = await asyncio.gather(*tasks)
    satisfied = [r for r, e in zip(responses, evals) if float(e.get("score", 0.0)) >= 0.5]
    return len(satisfied), satisfied


async def _rrd_prune_redundant_rubrics(
    rubric_items: list[str],
    *,
    apply_overlap_filter: bool,
    apply_conflict_filter: bool,
    judge_model: str | None = None,
    judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[str]:
    """
    Prune semantically redundant rubrics using overlap/conflict filters.
    Keeps earliest rubric when conflicts/overlap are detected.
    """
    if not rubric_items:
        return []

    kept: list[str] = []
    for rubric in rubric_items:
        if not kept:
            kept.append(rubric)
            continue

        overlap_task = (
            _rrd_run_binary_filter(
                "rrd_overlap_check",
                existing_rubrics=kept,
                new_rubric=rubric,
                judge_model=judge_model,
                judge_generate_text_actor=judge_generate_text_actor,
                sampling_params=sampling_params,
            )
            if apply_overlap_filter
            else asyncio.sleep(0, result=False)
        )
        conflict_task = (
            _rrd_run_binary_filter(
                "rrd_conflict_check",
                existing_rubrics=kept,
                new_rubric=rubric,
                judge_model=judge_model,
                judge_generate_text_actor=judge_generate_text_actor,
                sampling_params=sampling_params,
            )
            if apply_conflict_filter
            else asyncio.sleep(0, result=False)
        )
        is_overlap, is_conflict = await asyncio.gather(overlap_task, conflict_task)
        if is_overlap or is_conflict:
            continue
        kept.append(rubric)
    return kept


async def build_rrd_rubric_async(
    question: str,
    sample_responses: list[str],
    *,
    initial_rubric_text: str | None = None,
    proposer_model: str | None = None,
    judge_model: str | None = None,
    decomposition_trigger: int = 2,
    termination_threshold: int = 15,
    max_rounds: int = 6,
    apply_overlap_filter: bool = True,
    apply_conflict_filter: bool = True,
    apply_misalignment_filter: bool = True,
    strong_reference_answer: str | None = None,
    weak_reference_answer: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    judge_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """
    Full RRD procedure from Appendix prompts:
    initial proposal -> recursive decomposition -> filtering -> rejection-threshold stop.
    """
    responses = [_remove_redacted_reasoning(x) for x in sample_responses if x and x.strip()]
    if not responses:
        return {"rubric_items": [], "rejected_count": 0, "iterations": 0}

    proposer_actor = _resolve_actor(proposer_generate_text_actor, proposer_model, rubric_judge_generate_text_actor)
    judge_actor = _resolve_actor(judge_generate_text_actor, judge_model, rubric_judge_generate_text_actor)

    if initial_rubric_text and initial_rubric_text.strip():
        rubric_items = _filter_valid_rubric_items(parse_rubric_items(initial_rubric_text))
    else:
        rubric_items = await _rrd_generate_initial_rubrics(
            question=question,
            sample_responses=responses,
            proposer_model=proposer_model,
            proposer_generate_text_actor=proposer_actor,
            sampling_params=sampling_params,
        )

    trace: dict[str, Any] = {
        "decomposition_trigger": int(decomposition_trigger),
        "termination_threshold": int(termination_threshold),
        "max_rounds": int(max_rounds),
        "initial_rubric_items": list(rubric_items),
        "initial_pruned_rubric_items": [],
        "rounds": [],
    }

    rubric_items = await _rrd_prune_redundant_rubrics(
        rubric_items,
        apply_overlap_filter=apply_overlap_filter,
        apply_conflict_filter=apply_conflict_filter,
        judge_model=judge_model,
        judge_generate_text_actor=judge_actor,
        sampling_params=sampling_params,
    )
    trace["initial_pruned_rubric_items"] = list(rubric_items)

    if not rubric_items:
        trace["final_rubric_items"] = []
        return {"rubric_items": [], "rejected_count": 0, "iterations": 0, "trace": trace}

    rejected_count = 0
    iterations = 0
    while rejected_count < termination_threshold and iterations < max_rounds:
        iterations += 1
        snapshot = list(rubric_items)
        new_items: list[str] = []
        round_trace: dict[str, Any] = {
            "iteration": iterations,
            "input_rubric_items": list(snapshot),
            "accepted_new_items": [],
            "decomposition_events": [],
            "rejected_by_reason": {
                "empty_or_placeholder": 0,
                "no_candidates": 0,
                "overlap": 0,
                "conflict": 0,
                "misaligned": 0,
                "duplicate": 0,
            },
        }

        for rubric in snapshot:
            satisfied_count, satisfied_responses = await _rrd_count_satisfied_responses(
                question=question,
                rubric_item=rubric,
                responses=responses,
                judge_model=judge_model,
                judge_generate_text_actor=judge_actor,
                sampling_params=sampling_params,
            )
            event: dict[str, Any] = {
                "source_rubric": rubric,
                "satisfied_count": int(satisfied_count),
                "triggered_decomposition": False,
                "num_candidates": 0,
            }
            # Paper behavior: decompose when satisfied_count is at least trigger.
            if satisfied_count < decomposition_trigger:
                round_trace["decomposition_events"].append(event)
                continue
            event["triggered_decomposition"] = True

            # Per RRD Algorithm 1: pass only the satisfied subset Rm to decomposition
            candidates = await _rrd_decompose_rubric(
                question=question,
                sample_responses=satisfied_responses,
                current_rubric=rubric,
                other_rubrics=[x for x in rubric_items if x != rubric],
                proposer_model=proposer_model,
                proposer_generate_text_actor=proposer_actor,
                sampling_params=sampling_params,
            )
            candidates = _filter_valid_rubric_items(candidates)
            event["num_candidates"] = len(candidates)
            round_trace["decomposition_events"].append(event)

            if not candidates:
                rejected_count += 1
                round_trace["rejected_by_reason"]["no_candidates"] += 1
                continue

            for candidate in candidates:
                if not candidate.strip() or _is_placeholder_rubric_item(candidate):
                    rejected_count += 1
                    round_trace["rejected_by_reason"]["empty_or_placeholder"] += 1
                    continue

                existing = rubric_items + new_items
                overlap_task = (
                    _rrd_run_binary_filter(
                        "rrd_overlap_check",
                        existing_rubrics=existing,
                        new_rubric=candidate,
                        judge_model=judge_model,
                        judge_generate_text_actor=judge_actor,
                        sampling_params=sampling_params,
                    )
                    if apply_overlap_filter and existing
                    else asyncio.sleep(0, result=False)
                )
                conflict_task = (
                    _rrd_run_binary_filter(
                        "rrd_conflict_check",
                        existing_rubrics=existing,
                        new_rubric=candidate,
                        judge_model=judge_model,
                        judge_generate_text_actor=judge_actor,
                        sampling_params=sampling_params,
                    )
                    if apply_conflict_filter and existing
                    else asyncio.sleep(0, result=False)
                )
                misaligned_task = (
                    _rrd_is_misaligned_rubric(
                        question=question,
                        rubric_item=candidate,
                        strong_reference_answer=strong_reference_answer,
                        weak_reference_answer=weak_reference_answer,
                        judge_model=judge_model,
                        judge_generate_text_actor=judge_actor,
                        sampling_params=sampling_params,
                    )
                    if apply_misalignment_filter
                    else asyncio.sleep(0, result=False)
                )
                is_overlap, is_conflict, is_misaligned = await asyncio.gather(
                    overlap_task, conflict_task, misaligned_task
                )

                if is_overlap or is_conflict or is_misaligned or candidate in existing:
                    rejected_count += 1
                    if is_overlap:
                        round_trace["rejected_by_reason"]["overlap"] += 1
                    elif is_conflict:
                        round_trace["rejected_by_reason"]["conflict"] += 1
                    elif is_misaligned:
                        round_trace["rejected_by_reason"]["misaligned"] += 1
                    else:
                        round_trace["rejected_by_reason"]["duplicate"] += 1
                    continue
                new_items.append(candidate)
                round_trace["accepted_new_items"].append(candidate)

        round_trace["num_new_items"] = len(new_items)
        if not new_items:
            trace["rounds"].append(round_trace)
            break

        rubric_items = await _rrd_prune_redundant_rubrics(
            _filter_valid_rubric_items(rubric_items + new_items),
            apply_overlap_filter=apply_overlap_filter,
            apply_conflict_filter=apply_conflict_filter,
            judge_model=judge_model,
            judge_generate_text_actor=judge_actor,
            sampling_params=sampling_params,
        )
        round_trace["output_rubric_items"] = list(rubric_items)
        trace["rounds"].append(round_trace)

    trace["final_rubric_items"] = list(rubric_items)
    return {"rubric_items": rubric_items, "rejected_count": rejected_count, "iterations": iterations, "trace": trace}


def _compute_uniform_weights(num_rubrics: int) -> np.ndarray:
    if num_rubrics <= 0:
        return np.array([], dtype=float)
    return np.ones(num_rubrics, dtype=float) / float(num_rubrics)


async def _compute_llm_weights(
    question: str,
    rubric_items: list[str],
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> np.ndarray:
    if not rubric_items:
        return np.array([], dtype=float)

    messages = format_messages(
        "rrd_weight_assignment", {"question": question, "rubrics": rubric_items}, tokenize=False
    )
    response = await _run_generation_from_messages(
        messages=messages,
        model_name=judge_model,
        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
        sampling_params=sampling_params,
    )
    parsed = extract_json_from_response(response)
    if not parsed or "weights" not in parsed or not isinstance(parsed["weights"], list):
        return _compute_uniform_weights(len(rubric_items))

    raw = np.asarray([max(0.0, float(x)) for x in parsed["weights"]], dtype=float)
    if raw.shape[0] != len(rubric_items):
        return _compute_uniform_weights(len(rubric_items))
    total = float(raw.sum())
    if total <= 1e-12:
        return _compute_uniform_weights(len(rubric_items))
    return raw / total


async def judge_answer_rrd(
    question: str,
    rubric: str,
    answer: str,
    *,
    weighting_method: str = "wu",
    rubric_items: list[str] | None = None,
    answer_item_scores: list[float] | np.ndarray | None = None,
    covariance_answers: list[str] | None = None,
    covariance_item_scores: list[list[float]] | np.ndarray | None = None,
    precomputed_weights: list[float] | np.ndarray | None = None,
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """
    Score a single answer with RRD rubric items and one of three methods:
    `uniform`, `llm`, `wu`.
    """
    parsed_items = rubric_items if rubric_items is not None else parse_rubric_items(rubric)
    parsed_items = _filter_valid_rubric_items(parsed_items)
    if not parsed_items:
        return {"score": 0.0, "reasoning": "No rubric items"}

    method = weighting_method.lower().replace("rrd_", "")
    if method == "wu":
        if answer_item_scores is None:
            raise ValueError("RRD-WU requires precomputed answer_item_scores")
        if precomputed_weights is None:
            raise ValueError("RRD-WU requires precomputed_weights")
    elif method == "llm" and precomputed_weights is None:
        raise ValueError("RRD-LLM requires precomputed_weights")

    if answer_item_scores is None:
        answer_eval = await _score_answer_against_rubric_items(
            question=question,
            rubric_items=parsed_items,
            answer=answer,
            model_name=judge_model,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
            answer_type="policy",
        )
        answer_scores = answer_eval["binary_scores"]
        answer_reasoning = answer_eval["reasoning"]
    else:
        answer_scores = np.asarray(answer_item_scores, dtype=float).tolist()
        if len(answer_scores) != len(parsed_items):
            raise ValueError("Provided answer_item_scores length does not match number of rubric items")
        satisfied = int(sum(1 for score in answer_scores if score >= 0.5))
        answer_reasoning = f"Satisfied {satisfied}/{len(parsed_items)} rubric items"

    if precomputed_weights is not None:
        weights = np.asarray(precomputed_weights, dtype=float)
        if weights.shape[0] != len(parsed_items):
            raise ValueError("Provided precomputed_weights length does not match number of rubric items")
    elif method == "uniform":
        weights = _compute_uniform_weights(len(parsed_items))
    else:
        raise ValueError(f"RRD-{method.upper()} requires precomputed_weights")

    score = _aggregate_binary_scores_with_wu_weights(answer_scores, weights)
    return {
        "score": score,
        "reasoning": answer_reasoning,
        "rubric_items": parsed_items,
        "weights": weights.tolist(),
        "binary_scores": answer_scores,
        "weighting_method": method,
    }


async def _score_all_answers_against_rubric_items(
    question: str,
    rubric_items: list[str],
    answers: list[str],
    model_name: str | None = None,
    prompt_template: str = "judge_binary",
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> np.ndarray:
    """Score all answers against all rubric items, returning a (num_answers, num_rubrics) binary matrix."""
    if not rubric_items or not answers:
        return np.zeros((len(answers), len(rubric_items)), dtype=float)

    tasks = [
        _score_answer_against_rubric_items(
            question=question,
            rubric_items=rubric_items,
            answer=answer,
            model_name=model_name,
            prompt_template=prompt_template,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
            answer_type="rlcer_rollout",
        )
        for answer in answers
    ]
    results = await asyncio.gather(*tasks)
    return np.asarray([r["binary_scores"] for r in results], dtype=float)


def _rlcer_filter_valid_rubrics(
    score_matrix: np.ndarray, correctness_vector: np.ndarray, alpha: float = 0.2
) -> list[int]:
    """Filter rubrics by correlation with answer correctness (RLCER validity).

    A rubric is valid if:
      (i)  corr(v_k, z) > alpha  (positively correlated with correctness)
      (ii) std(v_k) > 0          (discriminative across rollouts)

    Args:
        score_matrix: (num_answers, num_rubrics) binary satisfaction matrix.
        correctness_vector: (num_answers,) binary correctness vector.
        alpha: Correlation threshold (default 0.2, per RLCER paper).

    Returns:
        List of valid rubric indices.
    """
    num_answers, num_rubrics = score_matrix.shape
    if num_answers <= 1 or num_rubrics == 0:
        return list(range(num_rubrics))

    z = np.asarray(correctness_vector, dtype=float)
    # If all answers are correct or all wrong, correlation is undefined.
    # In this case, keep all rubrics that are discriminative.
    if np.std(z) < 1e-8:
        valid = []
        for k in range(num_rubrics):
            if np.std(score_matrix[:, k]) > 1e-8:
                valid.append(k)
        return valid if valid else list(range(num_rubrics))

    valid = []
    for k in range(num_rubrics):
        v_k = score_matrix[:, k]
        if np.std(v_k) < 1e-8:
            continue
        # Pearson correlation between rubric satisfaction and answer correctness
        corr = float(np.corrcoef(v_k, z)[0, 1])
        if np.isfinite(corr) and corr > alpha:
            valid.append(k)

    return valid


def _rlcer_compute_cot_reward(
    binary_scores: list[float], valid_indices: list[int], rubric_scores: list[float] | None = None
) -> float:
    """Compute RLCER CoT reward: norm(sum of satisfied valid rubric scores).

    Following Eq. 6 in the paper:
      r_cot = norm(sum_{valid rubrics} pi_phi(c_k, C) * s_k)
    where norm is min-max normalization to [0, 1].

    Args:
        binary_scores: Per-rubric binary satisfaction for this answer.
        valid_indices: Indices of valid rubrics.
        rubric_scores: Per-rubric importance scores (s_k). If None, uniform (1.0 each).

    Returns:
        CoT reward in [0, 1].
    """
    if not valid_indices:
        return 0.0

    scores = binary_scores
    if rubric_scores is None:
        rubric_scores = [1.0] * len(scores)

    # Compute raw sum for this answer
    raw = sum(scores[k] * rubric_scores[k] for k in valid_indices)

    # Min-max normalization: min is when no rubric is satisfied, max is when all are
    valid_rubric_scores = [rubric_scores[k] for k in valid_indices]
    min_val = sum(s for s in valid_rubric_scores if s < 0)
    max_val = sum(s for s in valid_rubric_scores if s > 0)

    denom = max_val - min_val
    if denom <= 1e-12:
        return 0.0

    normalized = (raw - min_val) / denom
    return float(np.clip(normalized, 0.0, 1.0))


def _rlcer_check_answer_correctness(policy_answer: str, ground_truth: str, verifier_type: str = "math") -> float:
    """Check if policy_answer is correct using a math verifier.

    Uses the appropriate verifier (GSM8K or MATH) for exact answer checking,
    consistent with RLVR (Reinforcement Learning with Verifiable Rewards).

    Args:
        policy_answer: The model-generated answer text.
        ground_truth: The ground truth answer string.
        verifier_type: Verifier type ("gsm8k", "math", or "strict_math").

    Returns:
        1.0 if the answer is correct, 0.0 otherwise.
    """
    from open_instruct.ground_truth_utils import GSM8KVerifier, MathVerifier, StrictMathVerifier

    if verifier_type == "gsm8k":
        verifier = GSM8KVerifier()
    elif verifier_type == "strict_math":
        verifier = StrictMathVerifier()
    else:
        verifier = MathVerifier()

    result = verifier([], policy_answer, ground_truth)
    return float(result.score)


async def score_policy_rollouts_with_rlcer(
    *,
    questions: list[str],
    answers: list[str],
    ground_truths: list[str],
    verifier_types: list[str],
    correlation_threshold: float = 0.2,
    outcome_reward_weight: float = 1.0,
    cot_reward_weight: float = 1.0,
    proposer_model: str | None = None,
    judge_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    judge_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
    precomputed_rollout_rubrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Score policy rollouts using the RLCER method (Sheng et al., 2026).

    For each question group:
      1. Generate one rubric per rollout from the question and that rollout.
      2. Score ALL rollouts against each rollout's rubric items.
      3. Check answer correctness for each rollout using math verifiers.
      4. Filter each rollout's rubric items by correlation with correctness (corr > alpha, std > 0).
      5. Compute CoT reward for each rollout from its valid rubric satisfaction.
      6. Total reward = outcome_reward + cot_reward.

    Args:
        questions: List of questions (repeated for each rollout).
        answers: List of policy-generated answers.
        ground_truths: List of ground-truth answers (e.g., "72" for GSM8K).
        verifier_types: List of verifier types (e.g., "gsm8k", "math").
        correlation_threshold: Alpha threshold for rubric validity (default 0.2).
        outcome_reward_weight: Weight for outcome reward component.
        cot_reward_weight: Weight for CoT reward component.
        proposer_model: Model for rubric generation.
        judge_model: Model for the full-rubric RL-CER verifier.
        proposer_generate_text_actor: Actor for rubric generation.
        judge_generate_text_actor: Actor for judging.
        rubric_judge_generate_text_actor: Fallback actor.
        sampling_params: vLLM sampling parameters.

    Returns:
        List of result dicts with 'score', 'reasoning', 'rubric_items',
        'valid_rubric_indices', 'binary_scores', 'correctness', etc.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if len(questions) != len(ground_truths):
        raise ValueError("questions and ground_truths must have the same length")
    if not questions:
        return []
    if precomputed_rollout_rubrics is not None and len(precomputed_rollout_rubrics) != len(questions):
        raise ValueError("precomputed_rollout_rubrics must align 1:1 with questions and answers")

    grouped_indices: dict[str, list[int]] = {}
    for idx, question in enumerate(questions):
        grouped_indices.setdefault(question, []).append(idx)

    results: list[dict[str, Any] | None] = [None] * len(questions)

    proposer_actor = _resolve_actor(proposer_generate_text_actor, proposer_model, rubric_judge_generate_text_actor)

    async def _score_question_group(question: str, indices: list[int]) -> None:
        group_answers = [answers[idx] for idx in indices]

        gt_answer = ground_truths[indices[0]]
        vtype = verifier_types[indices[0]] if indices[0] < len(verifier_types) else "math"
        # Handle list-typed ground truths from RLVR datasets
        if isinstance(gt_answer, list):
            gt_answer = gt_answer[0] if gt_answer else ""
        if isinstance(vtype, list):
            vtype = vtype[0] if vtype else "math"

        correctness_vector = np.asarray(
            [_rlcer_check_answer_correctness(answers[idx], str(gt_answer), str(vtype)) for idx in indices], dtype=float
        )

        if precomputed_rollout_rubrics is not None:
            rollout_rubrics = []
            for global_idx in indices:
                rubric_data = precomputed_rollout_rubrics[global_idx]
                if rubric_data is None:
                    raise ValueError(f"Missing precomputed RL-CER rubric for rollout index {global_idx}")
                rollout_rubrics.append(rubric_data)
        else:
            generated_rubrics = await asyncio.gather(
                *[
                    _rlcer_generate_rubrics_with_scores(
                        question=question,
                        reference_response=answers[idx],
                        proposer_model=proposer_model,
                        proposer_generate_text_actor=proposer_actor,
                        sampling_params=sampling_params,
                    )
                    for idx in indices
                ]
            )
            rollout_rubrics = [
                {
                    "rubric_items": items,
                    "rubric_scores": scores,
                    "rubric_entries": _build_rlcer_rubric_entries_from_items(items, scores),
                }
                for items, scores in generated_rubrics
            ]

        async def _score_single_rollout(local_i: int, global_idx: int, rubric_data: dict[str, Any]) -> None:
            rubric_items = list(rubric_data.get("rubric_items", []))
            rubric_scores = [float(score) for score in rubric_data.get("rubric_scores", [])]
            rubric_entries = list(rubric_data.get("rubric_entries") or [])
            if not rubric_entries and rubric_data.get("rubric_text"):
                rubric_entries = parse_rlcer_rubric_entries(str(rubric_data.get("rubric_text", "")))
            if not rubric_entries:
                rubric_entries = _build_rlcer_rubric_entries_from_items(rubric_items, rubric_scores)
            if len(rubric_scores) < len(rubric_items):
                rubric_scores.extend([1.0] * (len(rubric_items) - len(rubric_scores)))
            elif len(rubric_scores) > len(rubric_items):
                rubric_scores = rubric_scores[:len(rubric_items)]

            if not rubric_items:
                correctness = float(correctness_vector[local_i])
                outcome_reward = 1.0 if correctness >= 0.5 else -1.0
                total_reward = outcome_reward_weight * outcome_reward
                results[global_idx] = {
                    "score": total_reward,
                    "reasoning": "No rubric items generated",
                    "rubric_items": [],
                    "valid_rubric_indices": [],
                    "binary_scores": [],
                    "correctness": correctness,
                    "outcome_reward": outcome_reward,
                    "cot_reward": 0.0,
                    "num_valid_rubrics": 0,
                    "num_total_rubrics": 0,
                    "rlcer_correlation_threshold": correlation_threshold,
                    "rubric_scores": [],
                }
                return

            score_matrix = await _score_all_answers_with_rlcer_verifier(
                question=question,
                rubric_entries=rubric_entries,
                answers=group_answers,
                model_name=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
            valid_indices = _rlcer_filter_valid_rubrics(
                score_matrix=score_matrix, correctness_vector=correctness_vector, alpha=correlation_threshold
            )

            binary_scores = score_matrix[local_i].tolist()
            correctness = float(correctness_vector[local_i])
            outcome_reward = 1.0 if correctness >= 0.5 else -1.0
            cot_reward = _rlcer_compute_cot_reward(
                binary_scores=binary_scores, valid_indices=valid_indices, rubric_scores=rubric_scores
            )
            total_reward = outcome_reward_weight * outcome_reward + cot_reward_weight * cot_reward

            results[global_idx] = {
                "score": total_reward,
                "reasoning": (
                    f"RLCER: outcome={outcome_reward:.1f}, cot={cot_reward:.3f}, "
                    f"valid_rubrics={len(valid_indices)}/{len(rubric_items)}"
                ),
                "rubric_items": rubric_items,
                "valid_rubric_indices": valid_indices,
                "binary_scores": binary_scores,
                "correctness": correctness,
                "outcome_reward": outcome_reward,
                "cot_reward": cot_reward,
                "num_valid_rubrics": len(valid_indices),
                "num_total_rubrics": len(rubric_items),
                "rlcer_correlation_threshold": correlation_threshold,
                "rubric_scores": rubric_scores,
            }

        await asyncio.gather(
            *[
                _score_single_rollout(local_i, global_idx, rollout_rubrics[local_i])
                for local_i, global_idx in enumerate(indices)
            ]
        )

    max_concurrent_groups = int(os.environ.get("MAX_CONCURRENT_QUESTION_GROUPS", "16"))
    group_semaphore = asyncio.Semaphore(max_concurrent_groups)

    async def _bounded_score(q, idxs):
        async with group_semaphore:
            return await _score_question_group(q, idxs)

    await asyncio.gather(*[_bounded_score(q, idxs) for q, idxs in grouped_indices.items()])
    return [r if r is not None else {"score": 0.0, "reasoning": "Missing result"} for r in results]


async def compute_rlcer_rubricator_reward(
    *,
    question: str,
    rubric_text: str,
    rollout_answers: list[str],
    correctness_vector: list[float],
    correlation_threshold: float = 0.2,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Compute RLCER rubricator reward: K_valid/K + r_format.

    This implements the "with evolving" rubricator reward from Eq. 10 of the
    RLCER paper (Sheng et al., 2026).  The rubricator is trained to produce
    rubrics whose items *correlate* with answer correctness:

        r_evolving^Rub  =  K_valid / K  +  r_format

    where K_valid is the number of rubric items passing the correlation filter
    (Pearson > alpha, std > 0) across the supplied policy rollouts, K is the
    total number of parseable rubric items, and r_format is 1.0 when the rubric
    contains at least one parseable item.

    Args:
        question: The question the rubric was generated for.
        rubric_text: Full rubric text generated by the rubricator.
        rollout_answers: N policy rollout answers for this question.
        correctness_vector: Binary correctness labels (len = N).
        correlation_threshold: Alpha for rubric validity (default 0.2).
        rubric_judge_generate_text_actor: Actor for the full-rubric RL-CER verifier.
        sampling_params: vLLM sampling parameters.

    Returns:
        Dict with 'reward', 'k_valid', 'k_total', 'validity_fraction',
        'r_format', 'rubric_items', 'valid_indices'.
    """
    rubric_items, _ = parse_rlcer_rubric_items_with_scores(rubric_text)
    k_total = len(rubric_items)

    # Format reward: rubric must contain at least one valid item
    r_format = 1.0 if k_total > 0 else 0.0

    if k_total == 0 or len(rollout_answers) <= 1:
        return {
            "reward": r_format,
            "k_valid": 0,
            "k_total": k_total,
            "validity_fraction": 0.0,
            "r_format": r_format,
            "rubric_items": rubric_items,
            "valid_indices": [],
        }

    # Score all rollout answers against all rubric items → (N, K) binary matrix
    rubric_entries = parse_rlcer_rubric_entries(rubric_text)
    score_matrix = await _score_all_answers_with_rlcer_verifier(
        question=question,
        rubric_entries=rubric_entries,
        answers=rollout_answers,
        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
        sampling_params=sampling_params,
    )

    # Filter valid rubrics by correlation with answer correctness
    correctness_np = np.asarray(correctness_vector, dtype=float)
    valid_indices = _rlcer_filter_valid_rubrics(
        score_matrix=score_matrix, correctness_vector=correctness_np, alpha=correlation_threshold
    )

    k_valid = len(valid_indices)
    validity_fraction = k_valid / k_total

    # RLCER rubricator reward: K_valid/K + r_format  (Eq. 10)
    reward = validity_fraction + r_format

    return {
        "reward": reward,
        "k_valid": k_valid,
        "k_total": k_total,
        "validity_fraction": validity_fraction,
        "r_format": r_format,
        "rubric_items": rubric_items,
        "valid_indices": valid_indices,
    }


async def score_policy_rollouts_with_rrd_samples(
    *,
    questions: list[str],
    answers: list[str],
    weighting_method: str = "wu",
    proposer_model: str | None = None,
    judge_model: str | None = None,
    decomposition_trigger: int = 2,
    termination_threshold: int = 15,
    max_rounds: int = 6,
    strong_reference_answers: list[str] | None = None,
    weak_reference_answers: list[str] | None = None,
    proposer_generate_text_actor: Any | None = None,
    judge_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Build RRD rubrics directly from rollout samples (grouped by question) and
    score each rollout response.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")

    if not questions:
        return []

    strong_reference_answers = strong_reference_answers or []
    weak_reference_answers = weak_reference_answers or []

    grouped_indices: dict[str, list[int]] = {}
    for idx, question in enumerate(questions):
        grouped_indices.setdefault(question, []).append(idx)

    results: list[dict[str, Any] | None] = [None] * len(questions)

    async def _score_question_group(question: str, indices: list[int]) -> None:
        sample_response_indices = [idx for idx in indices if answers[idx] and answers[idx].strip()]
        sample_responses = [_remove_redacted_reasoning(answers[idx]) for idx in sample_response_indices]
        if not sample_responses:
            sample_responses = [""]

        strong_ref = strong_reference_answers[indices[0]] if len(strong_reference_answers) > indices[0] else None
        weak_ref = weak_reference_answers[indices[0]] if len(weak_reference_answers) > indices[0] else None

        rrd_result = await build_rrd_rubric_async(
            question=question,
            sample_responses=sample_responses,
            proposer_model=proposer_model,
            judge_model=judge_model,
            decomposition_trigger=decomposition_trigger,
            termination_threshold=termination_threshold,
            max_rounds=max_rounds,
            strong_reference_answer=strong_ref,
            weak_reference_answer=weak_ref,
            proposer_generate_text_actor=proposer_generate_text_actor,
            judge_generate_text_actor=judge_generate_text_actor,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )
        parsed_items = _filter_valid_rubric_items(rrd_result.get("rubric_items", []))
        rrd_trace = rrd_result.get("trace", {})
        method = weighting_method.lower().replace("rrd_", "")
        if not parsed_items:
            fallback = {
                "score": 0.0,
                "reasoning": "No rubric items",
                "rubric_items": [],
                "weights": [],
                "binary_scores": [],
                "weighting_method": method,
                "rrd_iterations": int(rrd_result.get("iterations", 0)),
                "rrd_rejected_count": int(rrd_result.get("rejected_count", 0)),
                "rrd_trace": rrd_trace,
            }
            for idx in indices:
                results[idx] = dict(fallback)
            return

        group_weights: np.ndarray | None = None
        answer_item_scores_by_index: dict[int, list[float]] = {}
        if method == "uniform":
            group_weights = _compute_uniform_weights(len(parsed_items))
        elif method == "llm":
            group_weights = await _compute_llm_weights(
                question=question,
                rubric_items=parsed_items,
                judge_model=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
        else:
            covariance_matrix_scores = await _score_all_answers_against_rubric_items(
                question=question,
                rubric_items=parsed_items,
                answers=sample_responses,
                model_name=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
            group_weights = _compute_wu_weights(_estimate_covariance(covariance_matrix_scores))
            answer_item_scores_by_index = {
                idx: covariance_matrix_scores[row_idx].tolist()
                for row_idx, idx in enumerate(sample_response_indices)
            }
            missing_answer_scores = [idx for idx in indices if idx not in answer_item_scores_by_index]
            if missing_answer_scores:
                raise ValueError(
                    "RRD-WU expected precomputed answer_item_scores for every answer in the question group"
                )

        eval_tasks = [
            judge_answer_rrd(
                question=question,
                rubric="",
                answer=answers[idx],
                weighting_method=weighting_method,
                rubric_items=parsed_items,
                answer_item_scores=answer_item_scores_by_index[idx] if method == "wu" else answer_item_scores_by_index.get(idx),
                covariance_answers=sample_responses,
                precomputed_weights=group_weights,
                judge_model=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
            for idx in indices
        ]
        eval_results = await asyncio.gather(*eval_tasks)
        for idx, eval_result in zip(indices, eval_results):
            eval_result["rrd_iterations"] = int(rrd_result.get("iterations", 0))
            eval_result["rrd_rejected_count"] = int(rrd_result.get("rejected_count", 0))
            eval_result["rrd_trace"] = rrd_trace
            results[idx] = eval_result

    # Limit concurrent question groups to avoid overwhelming API + judge
    max_concurrent_groups = int(os.environ.get("MAX_CONCURRENT_QUESTION_GROUPS", "16"))
    group_semaphore = asyncio.Semaphore(max_concurrent_groups)

    async def _bounded_score(q, idxs):
        async with group_semaphore:
            return await _score_question_group(q, idxs)

    await asyncio.gather(*[_bounded_score(q, idxs) for q, idxs in grouped_indices.items()])
    return [r if r is not None else {"score": 0.0, "reasoning": "Missing result"} for r in results]


async def compute_rrd_reward_async(
    question: str,
    accepted_answer: str,
    rejected_answer: str,
    *,
    weighting_method: str = "wu",
    generated_rubric: str | None = None,
    rubric_items: list[str] | None = None,
    accepted_item_scores: list[float] | None = None,
    rejected_item_scores: list[float] | None = None,
    covariance_answers: list[str] | None = None,
    covariance_item_scores: list[list[float]] | None = None,
    sample_responses: list[str] | None = None,
    apply_full_rrd_procedure: bool = False,
    proposer_model: str | None = None,
    decomposition_trigger: int = 2,
    termination_threshold: int = 15,
    max_rounds: int = 6,
    strong_reference_answer: str | None = None,
    weak_reference_answer: str | None = None,
    judge_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    judge_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """
    Compute pairwise reward using one of RRD-{Uniform, LLM, WU}.
    """
    _should_enable_logging()
    try:
        method = weighting_method.lower().replace("rrd_", "")
        result = {
            "reward": 0.0,
            "accepted_score": 0.0,
            "rejected_score": 0.0,
            "rubric": generated_rubric,
            "rubric_items": [],
            "weights": [],
            "weighting_method": method,
            "accepted_binary_scores": [],
            "rejected_binary_scores": [],
            "accepted_reasoning": "",
            "rejected_reasoning": "",
            "rrd_iterations": 0,
            "rrd_rejected_count": 0,
            "rrd_trace": {},
            "error": None,
        }

        parsed_items = rubric_items if rubric_items is not None else parse_rubric_items(generated_rubric or "")
        parsed_items = _filter_valid_rubric_items(parsed_items)

        if apply_full_rrd_procedure:
            rrd_responses = sample_responses if sample_responses is not None else [accepted_answer, rejected_answer]
            rrd_result = await build_rrd_rubric_async(
                question=question,
                sample_responses=rrd_responses,
                initial_rubric_text=generated_rubric if parsed_items else None,
                proposer_model=proposer_model,
                judge_model=judge_model,
                decomposition_trigger=decomposition_trigger,
                termination_threshold=termination_threshold,
                max_rounds=max_rounds,
                strong_reference_answer=strong_reference_answer,
                weak_reference_answer=weak_reference_answer,
                proposer_generate_text_actor=proposer_generate_text_actor,
                judge_generate_text_actor=judge_generate_text_actor,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
            parsed_items = _dedupe_rubric_items(rrd_result.get("rubric_items", []))
            parsed_items = _filter_valid_rubric_items(parsed_items)
            result["rrd_iterations"] = int(rrd_result.get("iterations", 0))
            result["rrd_rejected_count"] = int(rrd_result.get("rejected_count", 0))
            result["rrd_trace"] = rrd_result.get("trace", {})

        if not parsed_items:
            result["error"] = "No rubric items found"
            return result
        result["rubric_items"] = parsed_items

        if accepted_item_scores is None:
            accepted_eval = await _score_answer_against_rubric_items(
                question=question,
                rubric_items=parsed_items,
                answer=accepted_answer,
                model_name=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
                answer_type="accepted",
            )
            accepted_item_scores = accepted_eval["binary_scores"]
            result["accepted_reasoning"] = accepted_eval["reasoning"]
        else:
            result["accepted_reasoning"] = "Using provided accepted_item_scores"

        if rejected_item_scores is None:
            rejected_eval = await _score_answer_against_rubric_items(
                question=question,
                rubric_items=parsed_items,
                answer=rejected_answer,
                model_name=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
                answer_type="rejected",
            )
            rejected_item_scores = rejected_eval["binary_scores"]
            result["rejected_reasoning"] = rejected_eval["reasoning"]
        else:
            result["rejected_reasoning"] = "Using provided rejected_item_scores"

        if len(accepted_item_scores) != len(parsed_items) or len(rejected_item_scores) != len(parsed_items):
            result["error"] = "Provided rubric-item score length does not match number of rubric items"
            return result

        if method == "uniform":
            weights = _compute_uniform_weights(len(parsed_items))
        elif method == "llm":
            weights = await _compute_llm_weights(
                question=question,
                rubric_items=parsed_items,
                judge_model=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
        else:
            if covariance_item_scores is not None:
                cov_matrix_scores = np.asarray(covariance_item_scores, dtype=float)
            else:
                cov_answers = covariance_answers if covariance_answers is not None else []
                cov_tasks = [
                    _score_answer_against_rubric_items(
                        question=question,
                        rubric_items=parsed_items,
                        answer=cov_answer,
                        model_name=judge_model,
                        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                        sampling_params=sampling_params,
                        answer_type="covariance",
                    )
                    for cov_answer in cov_answers
                ]
                cov_results = await asyncio.gather(*cov_tasks) if cov_tasks else []
                cov_rows = [
                    list(map(float, accepted_item_scores)),
                    list(map(float, rejected_item_scores)),
                    *[r["binary_scores"] for r in cov_results],
                ]
                cov_matrix_scores = np.asarray(cov_rows, dtype=float)
            weights = _compute_wu_weights(_estimate_covariance(cov_matrix_scores))

        accepted_score = _aggregate_binary_scores_with_wu_weights(list(map(float, accepted_item_scores)), weights)
        rejected_score = _aggregate_binary_scores_with_wu_weights(list(map(float, rejected_item_scores)), weights)

        result["accepted_binary_scores"] = list(map(float, accepted_item_scores))
        result["rejected_binary_scores"] = list(map(float, rejected_item_scores))
        result["weights"] = weights.tolist()
        result["accepted_score"] = accepted_score
        result["rejected_score"] = rejected_score
        result["reward"] = 1.0 if accepted_score > rejected_score else 0.0
        return result
    except Exception as e:
        LOGGER.error(f"Error in RRD reward computation: {e}")
        return {
            "reward": 0.0,
            "accepted_score": 0.0,
            "rejected_score": 0.0,
            "rubric": generated_rubric,
            "rubric_items": [],
            "weights": [],
            "weighting_method": weighting_method.lower().replace("rrd_", ""),
            "accepted_binary_scores": [],
            "rejected_binary_scores": [],
            "accepted_reasoning": "",
            "rejected_reasoning": "",
            "rrd_iterations": 0,
            "rrd_rejected_count": 0,
            "rrd_trace": {},
            "error": str(e),
        }
    finally:
        _disable_logging()


# =============================================================================
# RaR paper baselines and methods (Gunjal et al., 2025)
# https://arxiv.org/abs/2507.17746
# =============================================================================

# Predefined static rubrics from Appendix A.5 of the RaR paper.
RAR_PREDEFINED_RUBRICS: list[str] = [
    "The response contains correct information without factual errors, inaccuracies, or hallucinations that could mislead the user.",
    "The response fully answers all essential parts of the question and provides sufficient detail where needed.",
    "The response is concise and to the point, avoiding unnecessary verbosity or repetition.",
    "The response effectively meets the user's practical needs, provides actionable information, and is genuinely helpful for their situation.",
]

# Categorical weight mapping from Section 4.4 of the paper.
RAR_EXPLICIT_CATEGORY_WEIGHTS: dict[str, float] = {"essential": 1.0, "important": 0.7, "optional": 0.3, "pitfall": 0.9}


def _extract_likert_rating(response: str) -> float:
    """Extract a 1-10 Likert rating from an LLM judge response and normalize to [0, 1].

    Attempts JSON extraction first (```json {"rating": N} ```), then falls back
    to regex search for a bare integer.
    """
    if not response:
        return 0.0

    parsed = extract_json_from_response(response)
    if parsed and "rating" in parsed:
        try:
            raw = float(parsed["rating"])
            return max(0.0, min(1.0, (raw - 1.0) / 9.0))
        except (ValueError, TypeError):
            pass

    # Fallback: look for a standalone integer 1-10
    match = re.search(r"\b(10|[1-9])\b", response)
    if match:
        raw = float(match.group(1))
        return max(0.0, min(1.0, (raw - 1.0) / 9.0))

    return 0.0


def _parse_rar_rubric_items_from_data(rubric_data: Any) -> list[dict[str, Any]]:
    """Parse rubric items from dataset rubric data.

    Handles multiple formats:
    1. JSON string: '[{"title": ..., "description": ..., "weight": ...}, ...]'
    2. List of dicts: [{"title": ..., "description": ..., "weight": ...}, ...]
    3. Plain text rubric: parsed into items via parse_rubric_items()

    Returns list of dicts with at least 'description' and 'weight' keys.
    """
    import json

    if not rubric_data:
        return []

    # If it's already a list of dicts, use directly
    if isinstance(rubric_data, list):
        if rubric_data and isinstance(rubric_data[0], dict):
            return rubric_data
        # List of strings → treat each as a description with weight 1
        return [{"description": str(item), "weight": 1} for item in rubric_data if item]

    # If it's a string, try JSON parsing first
    if isinstance(rubric_data, str):
        rubric_data = rubric_data.strip()
        try:
            parsed = json.loads(rubric_data)
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    return parsed
                return [{"description": str(item), "weight": 1} for item in parsed if item]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back to text parsing
        items = parse_rubric_items(rubric_data)
        return [{"description": item, "weight": 1} for item in items]

    return []


def _format_rubric_items_for_implicit(rubric_items: list[dict[str, Any]]) -> str:
    """Format rubric items into a numbered text list for the implicit judge prompt."""
    lines = []
    for i, item in enumerate(rubric_items, 1):
        desc = item.get("description", item.get("title", str(item)))
        weight = item.get("weight", 1)
        # Include category if present in description
        lines.append(f"{i}. {desc} (weight: {weight})")
    return "\n".join(lines)


def _get_category_from_description(description: str) -> str:
    """Extract category label from a rubric item description.

    The RaR paper prefixes descriptions with category labels like:
    'Essential Criteria: ...', 'Pitfall Criteria: Does not mention ...'
    """
    lower = description.lower().strip()
    if lower.startswith("essential"):
        return "essential"
    elif lower.startswith("important"):
        return "important"
    elif lower.startswith("optional"):
        return "optional"
    elif lower.startswith("pitfall"):
        return "pitfall"
    return "important"  # default


async def _rar_generate_rubrics(
    question: str,
    sample_responses: list[str],
    proposer_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Generate generic RaR-style rubric items for arbitrary prompts."""
    responses = [_remove_redacted_reasoning(x) for x in sample_responses if x and x.strip()]
    if not responses:
        return []

    messages = format_messages(
        "rar_rubric_generation",
        {"question": question, "responses": responses},
        tokenize=False,
    )
    for _ in range(3):
        try:
            response = await _run_generation_from_messages(
                messages=messages,
                model_name=proposer_model,
                rubric_judge_generate_text_actor=proposer_generate_text_actor,
                sampling_params=sampling_params,
            )
        except Exception:
            continue
        parsed_items = _parse_rar_rubric_items_from_data(response)
        if parsed_items:
            return parsed_items

    return []


async def _score_single_answer_likert(
    question: str,
    answer: str,
    template_name: str,
    *,
    reference_answer: str | None = None,
    rubric_text: str | None = None,
    model_name: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Score a single answer using a Likert-scale judge template.

    Supports direct_likert, reference_likert, and rar_implicit templates.
    """
    if model_name is None:
        model_name = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")

    answer = _remove_redacted_reasoning(answer)

    content: dict[str, Any] = {"question": question, "answer": answer}
    if reference_answer is not None:
        content["reference_answer"] = reference_answer
    if rubric_text is not None:
        content["rubric"] = rubric_text

    try:
        messages = format_messages(template_name, content, tokenize=False)
        if rubric_judge_generate_text_actor:
            assert sampling_params is not None
            async with _get_vllm_actor_semaphore():
                response = await rubric_judge_generate_text_actor.generate_text_from_messages.remote(
                    messages, sampling_params
                )
        else:
            response = await run_litellm_async(model_name=model_name, messages=messages)

        score = _extract_likert_rating(response)
        return {"score": score, "reasoning": f"Likert rating (normalized): {score:.3f}", "raw_response": response}
    except Exception as e:
        LOGGER.error(f"Error in Likert scoring ({template_name}): {e}")
        return {"score": 0.0, "reasoning": f"Error: {str(e)}", "raw_response": ""}


async def score_policy_rollouts_with_direct_likert(
    *,
    questions: list[str],
    answers: list[str],
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Score policy rollouts using Direct-Likert (RaR paper baseline).

    An LLM-as-judge provides a direct assessment for each response-prompt pair
    on a 1-10 Likert scale, normalized to [0,1]. No rubrics or references used.

    Section 4.3 of the RaR paper.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if not questions:
        return []

    tasks = [
        _score_single_answer_likert(
            question=question,
            answer=answer,
            template_name="direct_likert",
            model_name=judge_model,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )
        for question, answer in zip(questions, answers)
    ]
    return await asyncio.gather(*tasks)


async def score_policy_rollouts_with_reference_likert(
    *,
    questions: list[str],
    answers: list[str],
    reference_answers: list[str],
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Score policy rollouts using Reference-Likert (RaR paper baseline).

    An LLM-as-judge compares the generated response against a reference answer
    and assigns a 1-10 Likert score, normalized to [0,1].

    Section 4.3 of the RaR paper.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if len(questions) != len(reference_answers):
        raise ValueError("questions and reference_answers must have the same length")
    if not questions:
        return []

    tasks = [
        _score_single_answer_likert(
            question=question,
            answer=answer,
            template_name="reference_likert",
            reference_answer=ref,
            model_name=judge_model,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
        )
        for question, answer, ref in zip(questions, answers, reference_answers)
    ]
    return await asyncio.gather(*tasks)


async def score_policy_rollouts_with_rar_implicit(
    *,
    questions: list[str],
    answers: list[str],
    rubric_data_list: list[Any] | None = None,
    proposer_model: str | None = None,
    judge_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Score policy rollouts using RaR-Implicit (RaR paper main method).

    All rubric criteria along with categorical weights are passed to an
    LLM-as-judge, which produces a single holistic 1-10 Likert rating
    normalized to [0,1]. The judge handles aggregation implicitly.

    When rubric_data_list is None or contains empty entries, rubrics are
    generated on-the-fly from rollout responses (grouped by question)
    using the same proposer used by RRD modes.

    Section 2.2 (Implicit Aggregation) and Section 4.4 of the RaR paper.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if rubric_data_list is not None and len(questions) != len(rubric_data_list):
        raise ValueError("questions and rubric_data_list must have the same length")
    if not questions:
        return []

    # Pad rubric_data_list if not provided
    if rubric_data_list is None:
        rubric_data_list = [None] * len(questions)

    # Check if any rubrics need on-the-fly generation
    needs_generation = any(not _parse_rar_rubric_items_from_data(rd) for rd in rubric_data_list)

    # Group by question for on-the-fly rubric generation
    generated_rubrics: dict[str, list[dict[str, Any]]] = {}
    if needs_generation:
        grouped_indices: dict[str, list[int]] = {}
        for idx, question in enumerate(questions):
            grouped_indices.setdefault(question, []).append(idx)

        proposer_actor = _resolve_actor(proposer_generate_text_actor, proposer_model, rubric_judge_generate_text_actor)

        async def _generate_for_question(question: str, indices: list[int]) -> None:
            # Collect rollout responses for this question as generation context
            sample_responses = [
                _remove_redacted_reasoning(answers[idx]) for idx in indices if answers[idx] and answers[idx].strip()
            ]
            if not sample_responses:
                generated_rubrics[question] = []
                return
            rubric_items = await _rar_generate_rubrics(
                question=question,
                sample_responses=sample_responses,
                proposer_model=proposer_model,
                proposer_generate_text_actor=proposer_actor,
                sampling_params=sampling_params,
            )
            generated_rubrics[question] = rubric_items

        await asyncio.gather(
            *[
                _generate_for_question(q, idxs)
                for q, idxs in grouped_indices.items()
                # Only generate for questions that actually need rubrics
                if any(not _parse_rar_rubric_items_from_data(rubric_data_list[i]) for i in idxs)
            ]
        )

    # Score each answer
    tasks = []
    for question, answer, rubric_data in zip(questions, answers, rubric_data_list):
        rubric_items = _parse_rar_rubric_items_from_data(rubric_data)
        if rubric_items:
            # Use dataset-provided rubrics
            rubric_text = _format_rubric_items_for_implicit(rubric_items)
        elif question in generated_rubrics and generated_rubrics[question]:
            # Use on-the-fly generated rubrics
            rubric_text = _format_rubric_items_for_implicit(generated_rubrics[question])
        else:
            rubric_text = ""

        tasks.append(
            _score_single_answer_likert(
                question=question,
                answer=answer,
                template_name="rar_implicit",
                rubric_text=rubric_text,
                model_name=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
            )
        )
    return await asyncio.gather(*tasks)


async def score_policy_rollouts_with_rar_explicit(
    *,
    questions: list[str],
    answers: list[str],
    rubric_data_list: list[Any] | None = None,
    proposer_model: str | None = None,
    judge_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Score policy rollouts using RaR-Explicit (RaR paper method).

    Each criterion is independently evaluated using binary judging, and the
    final normalized reward is computed as a weighted sum:
        r(x, y_hat) = sum(w_j * c_j) / sum(w_j)

    Weights are assigned based on categorical labels from the rubric data:
    Essential=1.0, Important=0.7, Optional=0.3, Pitfall=0.9.

    When rubric_data_list is None or contains empty entries, rubrics are
    generated on-the-fly from rollout responses (grouped by question).
    On-the-fly items default to "important" weight (0.7) since they
    lack category prefixes.

    Section 2.2 (Explicit Aggregation) and Section 4.4 of the RaR paper.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if rubric_data_list is not None and len(questions) != len(rubric_data_list):
        raise ValueError("questions and rubric_data_list must have the same length")
    if not questions:
        return []

    # Pad rubric_data_list if not provided
    if rubric_data_list is None:
        rubric_data_list = [None] * len(questions)

    # Generate rubrics on-the-fly for entries that lack them
    needs_generation = any(not _parse_rar_rubric_items_from_data(rd) for rd in rubric_data_list)
    generated_rubrics: dict[str, list[dict[str, Any]]] = {}
    if needs_generation:
        grouped_indices: dict[str, list[int]] = {}
        for idx, question in enumerate(questions):
            grouped_indices.setdefault(question, []).append(idx)
        proposer_actor = _resolve_actor(proposer_generate_text_actor, proposer_model, rubric_judge_generate_text_actor)

        async def _generate_for_question(question: str, indices: list[int]) -> None:
            sample_responses = [
                _remove_redacted_reasoning(answers[idx]) for idx in indices if answers[idx] and answers[idx].strip()
            ]
            if not sample_responses:
                generated_rubrics[question] = []
                return
            generated_rubrics[question] = await _rar_generate_rubrics(
                question=question,
                sample_responses=sample_responses,
                proposer_model=proposer_model,
                proposer_generate_text_actor=proposer_actor,
                sampling_params=sampling_params,
            )

        await asyncio.gather(
            *[
                _generate_for_question(q, idxs)
                for q, idxs in grouped_indices.items()
                if any(not _parse_rar_rubric_items_from_data(rubric_data_list[i]) for i in idxs)
            ]
        )

    async def _score_single_explicit(question: str, answer: str, rubric_data: Any) -> dict[str, Any]:
        rubric_items = _parse_rar_rubric_items_from_data(rubric_data)
        # Fallback to on-the-fly generated rubrics
        if not rubric_items and question in generated_rubrics and generated_rubrics[question]:
            rubric_items = generated_rubrics[question]
        if not rubric_items:
            return {
                "score": 0.0,
                "reasoning": "No rubric items",
                "rubric_items": [],
                "weights": [],
                "binary_scores": [],
            }

        # Extract descriptions for binary judging
        descriptions = [item.get("description", str(item)) for item in rubric_items]

        # Compute weights from categorical labels or explicit numeric weights
        weights: list[float] = []
        for item in rubric_items:
            w = item.get("weight", 1)
            if isinstance(w, str):
                # Categorical label
                weights.append(RAR_EXPLICIT_CATEGORY_WEIGHTS.get(w.lower(), 0.7))
            elif isinstance(w, (int, float)):
                # Numeric weight; negative weights indicate pitfall items
                if w < 0:
                    weights.append(RAR_EXPLICIT_CATEGORY_WEIGHTS["pitfall"])
                else:
                    # Use category from description if weight is numeric (paper convention)
                    category = _get_category_from_description(item.get("description", ""))
                    weights.append(RAR_EXPLICIT_CATEGORY_WEIGHTS.get(category, 0.7))
            else:
                weights.append(0.7)

        # Binary judge each criterion
        eval_result = await _score_answer_against_rubric_items(
            question=question,
            rubric_items=descriptions,
            answer=answer,
            model_name=judge_model,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
            answer_type="rar_explicit",
        )
        binary_scores = eval_result["binary_scores"]

        # Weighted sum: r = sum(w_j * c_j) / sum(w_j)  (Eq. 1 in paper)
        numerator = sum(w * s for w, s in zip(weights, binary_scores))
        denominator = sum(abs(w) for w in weights)
        score = numerator / denominator if denominator > 1e-12 else 0.0
        score = max(0.0, min(1.0, score))

        satisfied = int(sum(1 for s in binary_scores if s >= 0.5))
        return {
            "score": score,
            "reasoning": f"RaR-Explicit: {satisfied}/{len(descriptions)} rubric items satisfied, weighted score={score:.3f}",
            "rubric_items": descriptions,
            "weights": weights,
            "binary_scores": binary_scores,
        }

    tasks = [_score_single_explicit(q, a, rd) for q, a, rd in zip(questions, answers, rubric_data_list)]
    return await asyncio.gather(*tasks)


async def score_policy_rollouts_with_rar_predefined(
    *,
    questions: list[str],
    answers: list[str],
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Score policy rollouts using RaR-Predefined (RaR paper method).

    Uses a fixed set of generic rubrics (Appendix A.5) for all prompts with
    Explicit Aggregation (Eq. 1) and uniform weights. Each criterion is
    independently binary-judged and scores are averaged.

    Section 4.4 of the RaR paper.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if not questions:
        return []

    async def _score_single_predefined(question: str, answer: str) -> dict[str, Any]:
        eval_result = await _score_answer_against_rubric_items(
            question=question,
            rubric_items=RAR_PREDEFINED_RUBRICS,
            answer=answer,
            model_name=judge_model,
            rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
            sampling_params=sampling_params,
            answer_type="rar_predefined",
        )
        binary_scores = eval_result["binary_scores"]

        # Uniform weights → simple average
        score = sum(binary_scores) / len(binary_scores) if binary_scores else 0.0
        satisfied = int(sum(1 for s in binary_scores if s >= 0.5))
        return {
            "score": score,
            "reasoning": f"RaR-Predefined: {satisfied}/{len(RAR_PREDEFINED_RUBRICS)} generic rubric items satisfied",
            "rubric_items": list(RAR_PREDEFINED_RUBRICS),
            "weights": [1.0 / len(RAR_PREDEFINED_RUBRICS)] * len(RAR_PREDEFINED_RUBRICS),
            "binary_scores": binary_scores,
        }

    tasks = [_score_single_predefined(q, a) for q, a in zip(questions, answers)]
    return await asyncio.gather(*tasks)


async def score_policy_rollouts_with_query_specific_pref(
    *,
    questions: list[str],
    answers: list[str],
    preferred_answers: list[str] | None = None,
    dispreferred_answers: list[str] | None = None,
    rubric_data_list: list[Any] | None = None,
    proposer_model: str | None = None,
    judge_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
    use_likert_scoring: bool = True,
) -> list[dict[str, Any]]:
    """Score policy rollouts with query-specific rubrics weighted by preference discrimination.

    Implements the scoring pipeline from the Query-Specific Rubrics paper (Eq. 2):
      S(r|y) = sum(w_k * v_k) / sum(|w_k|)

    For each query:
      1. Build query-specific rubric items (dataset-provided if available, else generated).
         When dataset items include a 'weight' field, it is used as the base importance prior.
      2. Score preferred/dispreferred reference answers per rubric item using Likert 1-10
         (normalized to [0,1]) for finer preference discrimination.
      3. Compute rubric weights from preference deltas, allowing negative weights for
         error-type criteria where the dispreferred answer scores higher.
      4. Score rollout answers with weighted Likert aggregation, normalized to [0,1].

    If preference pairs are missing or non-discriminative, weights fall back to
    dataset-provided weights (if available) or uniform.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if preferred_answers is not None and len(preferred_answers) != len(questions):
        raise ValueError("preferred_answers and questions must have the same length")
    if dispreferred_answers is not None and len(dispreferred_answers) != len(questions):
        raise ValueError("dispreferred_answers and questions must have the same length")
    if rubric_data_list is not None and len(rubric_data_list) != len(questions):
        raise ValueError("rubric_data_list and questions must have the same length")
    if not questions:
        return []

    if preferred_answers is None:
        preferred_answers = [""] * len(questions)
    if dispreferred_answers is None:
        dispreferred_answers = [""] * len(questions)
    if rubric_data_list is None:
        rubric_data_list = [None] * len(questions)

    grouped_indices: dict[str, list[int]] = {}
    for idx, question in enumerate(questions):
        grouped_indices.setdefault(question, []).append(idx)

    results: list[dict[str, Any] | None] = [None] * len(questions)

    proposer_actor = _resolve_actor(proposer_generate_text_actor, proposer_model, rubric_judge_generate_text_actor)

    # Choose scoring function based on mode
    _score_items = (
        _score_answer_against_rubric_items_likert if use_likert_scoring else _score_answer_against_rubric_items
    )
    _score_key = "likert_scores" if use_likert_scoring else "binary_scores"

    async def _score_group(question: str, indices: list[int]) -> None:
        rubric_items_dict = _parse_rar_rubric_items_from_data(rubric_data_list[indices[0]])
        dataset_weights: list[float] | None = None
        if rubric_items_dict:
            rubric_items = _filter_valid_rubric_items(
                [item.get("description", str(item)) for item in rubric_items_dict if item]
            )
            # Extract dataset-provided weights if present (paper's category-based importance)
            raw_weights = []
            for item in rubric_items_dict:
                if item and isinstance(item, dict) and "weight" in item:
                    try:
                        raw_weights.append(float(item["weight"]))
                    except (ValueError, TypeError):
                        raw_weights.append(1.0)
            if raw_weights and len(raw_weights) == len(rubric_items):
                dataset_weights = raw_weights
        else:
            sample_responses = [
                _remove_redacted_reasoning(answers[idx]) for idx in indices if answers[idx] and answers[idx].strip()
            ]
            pref_ref = preferred_answers[indices[0]].strip()
            dispref_ref = dispreferred_answers[indices[0]].strip()
            if pref_ref:
                sample_responses.append(_remove_redacted_reasoning(pref_ref))
            if dispref_ref:
                sample_responses.append(_remove_redacted_reasoning(dispref_ref))
            if not sample_responses:
                sample_responses = [""]
            rubric_items = await _rrd_generate_initial_rubrics(
                question=question,
                sample_responses=sample_responses,
                proposer_model=proposer_model,
                proposer_generate_text_actor=proposer_actor,
                sampling_params=sampling_params,
            )
            rubric_items = _filter_valid_rubric_items(rubric_items)

        if not rubric_items:
            fallback = {
                "score": 0.0,
                "reasoning": "No rubric items",
                "rubric_items": [],
                "weights": [],
                "item_scores": [],
                "weighting_method": "query_specific_pref",
                "trace": {},
            }
            for idx in indices:
                results[idx] = dict(fallback)
            return

        pref_ref = preferred_answers[indices[0]].strip()
        dispref_ref = dispreferred_answers[indices[0]].strip()
        use_preference_weights = bool(pref_ref and dispref_ref)
        pref_scores: list[float] = []
        dispref_scores: list[float] = []

        if use_preference_weights:
            pref_eval, dispref_eval = await asyncio.gather(
                _score_items(
                    question=question,
                    rubric_items=rubric_items,
                    answer=pref_ref,
                    model_name=judge_model,
                    rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                    answer_type="preferred_reference",
                ),
                _score_items(
                    question=question,
                    rubric_items=rubric_items,
                    answer=dispref_ref,
                    model_name=judge_model,
                    rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                    answer_type="dispreferred_reference",
                ),
            )
            pref_scores = [float(x) for x in pref_eval.get(_score_key, [])]
            dispref_scores = [float(x) for x in dispref_eval.get(_score_key, [])]
            if len(pref_scores) != len(rubric_items) or len(dispref_scores) != len(rubric_items):
                use_preference_weights = False

        if use_preference_weights:
            # Allow negative deltas — negative weight means criterion inversely
            # correlates with quality (e.g., error-type criteria the paper assigns
            # weight -1/-2).
            deltas = [p - d for p, d in zip(pref_scores, dispref_scores)]

            if dataset_weights is not None:
                # Hybrid: scale dataset importance by preference discrimination
                raw_weights = [dw * delta for dw, delta in zip(dataset_weights, deltas)]
            else:
                raw_weights = deltas

            abs_total = float(sum(abs(w) for w in raw_weights))
            if abs_total > 1e-12:
                weights = [w / abs_total for w in raw_weights]
                weighting_method = "preference_delta"
            else:
                # Non-discriminative — fall back to dataset weights or uniform
                if dataset_weights is not None:
                    dw_abs_total = float(sum(abs(w) for w in dataset_weights))
                    weights = (
                        [w / dw_abs_total for w in dataset_weights]
                        if dw_abs_total > 1e-12
                        else _compute_uniform_weights(len(rubric_items)).tolist()
                    )
                    weighting_method = "dataset_weights_fallback"
                else:
                    weights = _compute_uniform_weights(len(rubric_items)).tolist()
                    weighting_method = "uniform_fallback"
        elif dataset_weights is not None:
            # No preference pair but dataset weights available
            dw_abs_total = float(sum(abs(w) for w in dataset_weights))
            weights = (
                [w / dw_abs_total for w in dataset_weights]
                if dw_abs_total > 1e-12
                else _compute_uniform_weights(len(rubric_items)).tolist()
            )
            weighting_method = "dataset_weights"
        else:
            weights = _compute_uniform_weights(len(rubric_items)).tolist()
            weighting_method = "uniform_no_reference"

        eval_tasks = [
            _score_items(
                question=question,
                rubric_items=rubric_items,
                answer=answers[idx],
                model_name=judge_model,
                rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                sampling_params=sampling_params,
                answer_type="policy",
            )
            for idx in indices
        ]
        eval_results = await asyncio.gather(*eval_tasks)

        for idx, eval_result in zip(indices, eval_results):
            item_scores = [float(x) for x in (eval_result.get(_score_key, []) or [])]
            if len(item_scores) != len(rubric_items):
                score = 0.0
                item_scores = []
            else:
                raw_score = float(sum(w * s for w, s in zip(weights, item_scores)))
                # Min-max normalize: weights can be negative, so compute bounds
                w_pos = sum(w for w in weights if w > 0)
                w_neg = sum(w for w in weights if w < 0)
                denom = w_pos - w_neg
                if denom > 1e-12:
                    score = (raw_score - w_neg) / denom
                else:
                    score = raw_score
            score = max(0.0, min(1.0, score))
            satisfied = int(sum(1 for s in item_scores if s >= 0.5))
            results[idx] = {
                "score": score,
                "reasoning": (
                    f"QuerySpecific: {satisfied}/{len(rubric_items)} satisfied, "
                    f"weighting={weighting_method}, score={score:.3f}"
                ),
                "rubric_items": rubric_items,
                "weights": weights,
                "item_scores": item_scores,
                "weighting_method": weighting_method,
                "trace": {
                    "preferred_scores": pref_scores,
                    "dispreferred_scores": dispref_scores,
                    "dataset_weights": dataset_weights,
                    "has_preference_pair": bool(pref_ref and dispref_ref),
                    "scoring_mode": "likert" if use_likert_scoring else "binary",
                },
            }

    await asyncio.gather(*[_score_group(q, idxs) for q, idxs in grouped_indices.items()])
    return [r if r is not None else {"score": 0.0, "reasoning": "Missing result"} for r in results]


# ---------------------------------------------------------------------------
# Rubric-ARM (Xu et al., 2026) — Alternating RL for Rubric-Based Reward Modeling
# ---------------------------------------------------------------------------

_RUBRIC_ARM_WINNER_RE = re.compile(r"Winner:\s*Response\s*([AB])", re.IGNORECASE)


def _extract_rubric_arm_winner(response: str) -> str | None:
    """Extract the winner from a Rubric-ARM pairwise judge response."""
    match = _RUBRIC_ARM_WINNER_RE.search(response)
    return match.group(1).upper() if match else None


def _rubric_arm_format_reward(response: str) -> float:
    """R_fmt: 1.0 if judge output has all three required sections, 0.0 otherwise."""
    has_compliance = "--- Compliance Check ---" in response or "Compliance Check" in response
    has_analysis = "--- Analysis ---" in response or "**Response A:**" in response
    has_judgment = "--- Final Judgment ---" in response or "Winner:" in response
    return 1.0 if (has_compliance and has_analysis and has_judgment) else 0.0


async def _rubric_arm_generate_rubric(
    question: str,
    proposer_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[str]:
    """Generate Rubric-ARM style rubrics from prompt only (no sample responses needed)."""
    messages = format_messages("rubric_arm_rubric_generation", {"question": question}, tokenize=False)
    for _ in range(3):
        response = await _run_generation_from_messages(
            messages=messages,
            model_name=proposer_model,
            rubric_judge_generate_text_actor=proposer_generate_text_actor,
            sampling_params=sampling_params,
        )
        items = parse_rubric_items(response)
        items = _filter_valid_rubric_items(items)
        if items:
            return items
    return []


async def _rubric_arm_judge_pairwise(
    question: str,
    response_a: str,
    response_b: str,
    rubric_text: str,
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Run pairwise evaluation using the Rubric-ARM 3-phase judge template.

    Returns dict with 'winner' ('A' or 'B' or None), 'r_fmt', and 'raw_response'.
    """
    if judge_model is None:
        judge_model = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4o-mini")

    response_a_clean = _remove_redacted_reasoning(response_a)
    response_b_clean = _remove_redacted_reasoning(response_b)

    messages = format_messages(
        "rubric_arm_judge_pairwise",
        {
            "instruction": question,
            "rubric": rubric_text,
            "response_a": response_a_clean,
            "response_b": response_b_clean,
        },
        tokenize=False,
    )
    try:
        if rubric_judge_generate_text_actor:
            assert sampling_params is not None
            raw = await rubric_judge_generate_text_actor.generate_text_from_messages.remote(messages, sampling_params)
        else:
            raw = await run_litellm_async(model_name=judge_model, messages=messages)
        winner = _extract_rubric_arm_winner(raw)
        r_fmt = _rubric_arm_format_reward(raw)
        return {"winner": winner, "r_fmt": r_fmt, "raw_response": raw}
    except Exception as e:
        LOGGER.error("Error in Rubric-ARM pairwise judging: %s", e)
        return {"winner": None, "r_fmt": 0.0, "raw_response": ""}


async def score_policy_rollouts_with_rubric_arm(
    *,
    questions: list[str],
    answers: list[str],
    preferred_answers: list[str] | None = None,
    dispreferred_answers: list[str] | None = None,
    proposer_model: str | None = None,
    judge_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    """Score policy rollouts using Rubric-ARM debiased pairwise evaluation (Eq. 16).

    For each question group:
      1. Generate rubric from question only (prompt-conditioned, no responses).
      2. For each rollout, use the preferred answer as reference (a^(0)):
         - Forward order: judge(question, policy_answer, reference, rubric)
         - Reverse order: judge(question, reference, policy_answer, rubric)
      3. Debiased reward = 0.5 * (I[forward picks policy] + I[reverse picks policy])

    This implements the downstream policy reward from Section 4.3, Eq. 14-16.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    if not questions:
        return []

    if preferred_answers is None:
        preferred_answers = [""] * len(questions)
    if dispreferred_answers is None:
        dispreferred_answers = [""] * len(questions)

    grouped_indices: dict[str, list[int]] = {}
    for idx, question in enumerate(questions):
        grouped_indices.setdefault(question, []).append(idx)

    results: list[dict[str, Any] | None] = [None] * len(questions)

    proposer_actor = _resolve_actor(proposer_generate_text_actor, proposer_model, rubric_judge_generate_text_actor)

    async def _score_group(question: str, indices: list[int]) -> None:
        # Step 1: Generate rubric from question only
        rubric_items = await _rubric_arm_generate_rubric(
            question=question,
            proposer_model=proposer_model,
            proposer_generate_text_actor=proposer_actor,
            sampling_params=sampling_params,
        )

        if not rubric_items:
            fallback = {
                "score": 0.0,
                "reasoning": "No rubric items generated",
                "rubric_items": [],
                "winner_forward": None,
                "winner_reverse": None,
                "r_fmt_forward": 0.0,
                "r_fmt_reverse": 0.0,
                "weighting_method": "rubric_arm",
                "trace": {},
            }
            for idx in indices:
                results[idx] = dict(fallback)
            return

        rubric_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(rubric_items))
        reference = preferred_answers[indices[0]].strip()
        if not reference:
            reference = dispreferred_answers[indices[0]].strip()

        # Step 2: For each rollout, run debiased pairwise judging
        async def _score_single(idx: int) -> dict[str, Any]:
            policy_answer = answers[idx]
            if not reference:
                return {
                    "score": 0.0,
                    "reasoning": "No reference answer for pairwise comparison",
                    "rubric_items": rubric_items,
                    "winner_forward": None,
                    "winner_reverse": None,
                    "r_fmt_forward": 0.0,
                    "r_fmt_reverse": 0.0,
                    "weighting_method": "rubric_arm",
                    "trace": {"has_reference": False},
                }

            # Forward: policy = A, reference = B
            fwd_result, rev_result = await asyncio.gather(
                _rubric_arm_judge_pairwise(
                    question=question,
                    response_a=policy_answer,
                    response_b=reference,
                    rubric_text=rubric_text,
                    judge_model=judge_model,
                    rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                ),
                # Reverse: reference = A, policy = B
                _rubric_arm_judge_pairwise(
                    question=question,
                    response_a=reference,
                    response_b=policy_answer,
                    rubric_text=rubric_text,
                    judge_model=judge_model,
                    rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
                    sampling_params=sampling_params,
                ),
            )

            # Eq. 16: R = 0.5 * (I(forward picks A) + I(reverse picks B))
            fwd_wins = 1.0 if fwd_result["winner"] == "A" else 0.0
            rev_wins = 1.0 if rev_result["winner"] == "B" else 0.0
            score = 0.5 * (fwd_wins + rev_wins)

            return {
                "score": score,
                "reasoning": (
                    f"RubricARM: fwd={fwd_result['winner']}, rev={rev_result['winner']}, "
                    f"debiased={score:.2f}, r_fmt={fwd_result['r_fmt']:.0f}/{rev_result['r_fmt']:.0f}"
                ),
                "rubric_items": rubric_items,
                "winner_forward": fwd_result["winner"],
                "winner_reverse": rev_result["winner"],
                "r_fmt_forward": fwd_result["r_fmt"],
                "r_fmt_reverse": rev_result["r_fmt"],
                "weighting_method": "rubric_arm",
                "trace": {
                    "has_reference": True,
                    "forward_raw": fwd_result.get("raw_response", "")[:500],
                    "reverse_raw": rev_result.get("raw_response", "")[:500],
                },
            }

        scored = await asyncio.gather(*[_score_single(idx) for idx in indices])
        for idx, result in zip(indices, scored):
            results[idx] = result

    await asyncio.gather(*[_score_group(q, idxs) for q, idxs in grouped_indices.items()])
    return [r if r is not None else {"score": 0.0, "reasoning": "Missing result"} for r in results]


async def compute_rubric_arm_rubricator_reward(
    *,
    question: str,
    rubric_text: str,
    preferred_answer: str,
    dispreferred_answer: str,
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Compute Rubric-ARM rubricator reward: R_r = I[judge predicts correct preference].

    The rubric generator is rewarded when the (frozen) judge, using this rubric,
    correctly determines that the preferred answer is better (Eq. 8 in the paper).
    """
    rubric_items = parse_rubric_items(rubric_text)
    rubric_items = _filter_valid_rubric_items(rubric_items)

    while isinstance(preferred_answer, list):
        preferred_answer = preferred_answer[0] if preferred_answer else ""
    while isinstance(dispreferred_answer, list):
        dispreferred_answer = dispreferred_answer[0] if dispreferred_answer else ""
    preferred_answer = str(preferred_answer or "")
    dispreferred_answer = str(dispreferred_answer or "")

    r_format = 1.0 if rubric_items else 0.0
    if not rubric_items or not preferred_answer.strip() or not dispreferred_answer.strip():
        return {
            "score": r_format,
            "reasoning": "No rubric items or no preference pair",
            "rubric_items": rubric_items,
            "r_format": r_format,
            "judge_correct": False,
        }

    rubric_formatted = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(rubric_items))

    # Judge in forward order: preferred=A, dispreferred=B
    result = await _rubric_arm_judge_pairwise(
        question=question,
        response_a=preferred_answer,
        response_b=dispreferred_answer,
        rubric_text=rubric_formatted,
        judge_model=judge_model,
        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
        sampling_params=sampling_params,
    )

    # R_r = I[o = o*]: judge should pick A (preferred)
    judge_correct = result["winner"] == "A"
    r_acc = 1.0 if judge_correct else 0.0

    return {
        "score": r_acc + r_format,
        "reasoning": f"RubricARM rubricator: judge_correct={judge_correct}, r_format={r_format}",
        "rubric_items": rubric_items,
        "r_format": r_format,
        "r_acc": r_acc,
        "judge_correct": judge_correct,
        "judge_winner": result["winner"],
    }


async def judge_answer_rrd_wu(
    question: str,
    rubric: str,
    answer: str,
    rubric_items: list[str] | None = None,
    answer_item_scores: list[float] | None = None,
    covariance_answers: list[str] | None = None,
    covariance_item_scores: list[list[float]] | None = None,
    precomputed_weights: list[float] | None = None,
    judge_model: str | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for RRD-WU single-answer scoring."""
    return await judge_answer_rrd(
        question=question,
        rubric=rubric,
        answer=answer,
        weighting_method="wu",
        rubric_items=rubric_items,
        answer_item_scores=answer_item_scores,
        covariance_answers=covariance_answers,
        covariance_item_scores=covariance_item_scores,
        precomputed_weights=precomputed_weights,
        judge_model=judge_model,
        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
        sampling_params=sampling_params,
    )


async def compute_rrd_wu_reward_async(
    question: str,
    accepted_answer: str,
    rejected_answer: str,
    generated_rubric: str | None = None,
    rubric_items: list[str] | None = None,
    accepted_item_scores: list[float] | None = None,
    rejected_item_scores: list[float] | None = None,
    covariance_answers: list[str] | None = None,
    covariance_item_scores: list[list[float]] | None = None,
    judge_model: str | None = None,
    proposer_generate_text_actor: Any | None = None,
    judge_generate_text_actor: Any | None = None,
    rubric_judge_generate_text_actor: Any | None = None,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for pairwise RRD-WU reward."""
    return await compute_rrd_reward_async(
        question=question,
        accepted_answer=accepted_answer,
        rejected_answer=rejected_answer,
        weighting_method="wu",
        generated_rubric=generated_rubric,
        rubric_items=rubric_items,
        accepted_item_scores=accepted_item_scores,
        rejected_item_scores=rejected_item_scores,
        covariance_answers=covariance_answers,
        covariance_item_scores=covariance_item_scores,
        judge_model=judge_model,
        proposer_generate_text_actor=proposer_generate_text_actor,
        judge_generate_text_actor=judge_generate_text_actor,
        rubric_judge_generate_text_actor=rubric_judge_generate_text_actor,
        sampling_params=sampling_params,
    )


# ---------------------------------------------------------------------------
# Multi-Judge Support
# ---------------------------------------------------------------------------


_MULTI_JUDGE_TIMEOUT_S = float(os.environ.get("MULTI_JUDGE_TIMEOUT_S", "300"))

_PER_JUDGE_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _get_per_judge_semaphore(judge_idx: int, num_judges: int) -> asyncio.Semaphore:
    """Return a per-judge concurrency semaphore, created lazily.

    The global ``MAX_CONCURRENT_JUDGE_REQUESTS`` budget is split evenly across
    judges so that one slow judge cannot starve others of semaphore slots.
    """
    if judge_idx not in _PER_JUDGE_SEMAPHORES:
        total = int(os.environ.get("MAX_CONCURRENT_JUDGE_REQUESTS", "128"))
        per_judge = max(4, total // max(1, num_judges))
        _PER_JUDGE_SEMAPHORES[judge_idx] = asyncio.Semaphore(per_judge)
    return _PER_JUDGE_SEMAPHORES[judge_idx]


async def judge_answer_with_multiple_judges(
    question: str,
    rubric: str,
    answer: str,
    judge_actors: list[tuple[Any, Any]],  # List of (generate_text_actor, sampling_params) tuples
    answer_type: str = "answer",
    per_judge_timeout: float | None = None,
) -> list[dict[str, Any]]:
    """
    Use multiple LLM judges to evaluate an answer based on a rubric in parallel.

    Each judge is given ``per_judge_timeout`` seconds to complete.  If a judge
    exceeds this deadline its score is recorded as NaN so the aggregation layer
    can drop it without blocking the entire pipeline.

    Each judge gets its own concurrency semaphore (derived from
    ``MAX_CONCURRENT_JUDGE_REQUESTS / num_judges``) so that a slow judge cannot
    starve others of semaphore capacity.

    Args:
        question: The original question
        rubric: The rubric to use for evaluation
        answer: The answer to evaluate
        judge_actors: List of (generate_text_actor, sampling_params) tuples for each judge
        answer_type: Type of answer being judged (e.g., "accepted" or "rejected") for logging
        per_judge_timeout: Seconds before a single judge is considered timed-out.
            Defaults to ``MULTI_JUDGE_TIMEOUT_S`` env var (300 s).

    Returns:
        List of dictionaries, one per judge, each containing score and reasoning
    """
    if per_judge_timeout is None:
        per_judge_timeout = _MULTI_JUDGE_TIMEOUT_S

    num_judges = len(judge_actors)

    async def _judge_with_timeout(idx: int, generate_text_actor, sampling_params) -> dict[str, Any]:
        try:
            sem = _get_per_judge_semaphore(idx, num_judges)
            return await asyncio.wait_for(
                judge_answer_with_rubric(
                    question=question,
                    rubric=rubric,
                    answer=answer,
                    model_name=None,
                    answer_type=f"{answer_type}_judge{idx}",
                    rubric_judge_generate_text_actor=generate_text_actor,
                    sampling_params=sampling_params,
                    concurrency_semaphore=sem,
                ),
                timeout=per_judge_timeout,
            )
        except asyncio.TimeoutError:
            LOGGER.warning(
                "[Multi-Judge] Judge %d timed out after %.0fs for %s answer",
                idx, per_judge_timeout, answer_type,
            )
            return {"score": float("nan"), "reasoning": "timeout", "parse_error": True}

    tasks = [
        _judge_with_timeout(idx, actor, params)
        for idx, (actor, params) in enumerate(judge_actors)
    ]

    results = await asyncio.gather(*tasks)

    verbose_debug(f"[Multi-Judge] Completed {len(results)} judge evaluations for {answer_type} answer")
    return results


def normalize_multi_judge_aggregation_mode(aggregation_mode: str | None) -> str:
    """Normalize the requested multi-judge aggregation mode."""
    mode = (aggregation_mode or "majority_vote").strip().lower()
    aliases = {
        "avg": "average_vote",
        "average": "average_vote",
        "majority": "majority_vote",
        "majority_voting": "majority_vote",
        "avg_minus_var": "average_minus_variance",
        "average_minus_var": "average_minus_variance",
        "average_minus_kappa": "average_minus_variance",
        "agreement": "agreement_bonus",
        "alpha_beta": "agreement_bonus",
        "mkf": "margin_kappa_format",
        "margin_format_kappa": "margin_kappa_format",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_MULTI_JUDGE_AGGREGATIONS:
        raise ValueError(
            f"Unknown multi-judge aggregation_mode={aggregation_mode!r}. "
            f"Valid options: {sorted(VALID_MULTI_JUDGE_AGGREGATIONS)}"
        )
    return mode


def normalize_multi_judge_tie_breaker(tie_breaker: str | None) -> str:
    """Normalize the requested tie-breaker for multi-judge voting."""
    normalized = (tie_breaker or "mean_score").strip().lower()
    aliases = {
        "mean": "mean_score",
        "first": "first_judge",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_MULTI_JUDGE_TIE_BREAKERS:
        raise ValueError(
            f"Unknown multi-judge tie_breaker={tie_breaker!r}. "
            f"Valid options: {sorted(VALID_MULTI_JUDGE_TIE_BREAKERS)}"
        )
    return normalized


def compute_pairwise_accuracy(accepted_scores: list[float], rejected_scores: list[float]) -> float:
    """
    Compute pairwise accuracy: fraction of judges that ranked accepted > rejected.

    Args:
        accepted_scores: List of scores for accepted answer from each judge
        rejected_scores: List of scores for rejected answer from each judge

    Returns:
        Fraction of judges that correctly ranked (accepted > rejected), in [0.0, 1.0]
    """
    if len(accepted_scores) != len(rejected_scores):
        raise ValueError(
            f"Accepted and rejected scores must have same length. "
            f"Got {len(accepted_scores)} and {len(rejected_scores)}"
        )

    if len(accepted_scores) == 0:
        return 0.0

    correct_rankings = sum(compute_binary_judge_votes(accepted_scores, rejected_scores))
    return correct_rankings / len(accepted_scores)


def compute_binary_judge_votes(accepted_scores: list[float], rejected_scores: list[float]) -> list[int]:
    """
    Convert per-judge score pairs into binary comparison votes.

    Returns a length-num_judges list of 0/1 values where:
    - 1 means the judge ranked accepted > rejected
    - 0 means the judge did not rank accepted > rejected
    """
    if len(accepted_scores) != len(rejected_scores):
        raise ValueError(
            f"Accepted and rejected scores must have same length. "
            f"Got {len(accepted_scores)} and {len(rejected_scores)}"
        )

    return [1 if acc > rej else 0 for acc, rej in zip(accepted_scores, rejected_scores)]


def compute_fleiss_kappa(accepted_scores: list[float], rejected_scores: list[float]) -> float:
    """
    Compute Fleiss's kappa from binary per-judge comparison votes.

    Each judge contributes a binary label:
    - 1 if accepted_score > rejected_score
    - 0 otherwise (including exact ties)
    """
    if len(accepted_scores) != len(rejected_scores):
        raise ValueError(
            f"Accepted and rejected scores must have same length. "
            f"Got {len(accepted_scores)} and {len(rejected_scores)}"
        )

    num_judges = len(accepted_scores)
    if num_judges == 0:
        return 0.0
    if num_judges == 1:
        return 1.0

    binary_votes = compute_binary_judge_votes(accepted_scores, rejected_scores)
    positive_votes = sum(binary_votes)
    negative_votes = num_judges - positive_votes

    observed_agreement = (
        positive_votes * (positive_votes - 1) + negative_votes * (negative_votes - 1)
    ) / (num_judges * (num_judges - 1))
    positive_rate = positive_votes / num_judges
    negative_rate = negative_votes / num_judges
    expected_agreement = positive_rate**2 + negative_rate**2

    if np.isclose(1.0 - expected_agreement, 0.0):
        return 1.0 if np.isclose(observed_agreement, 1.0) else 0.0

    kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return float(np.clip(kappa, -1.0, 1.0))


def resolve_multi_judge_votes(
    accepted_scores: list[float],
    rejected_scores: list[float],
    *,
    tie_breaker: str = "mean_score",
) -> dict[str, Any]:
    """Resolve per-judge answer comparisons into a single winner."""
    if len(accepted_scores) != len(rejected_scores):
        raise ValueError(
            f"Accepted and rejected scores must have same length. "
            f"Got {len(accepted_scores)} and {len(rejected_scores)}"
        )

    tie_breaker = normalize_multi_judge_tie_breaker(tie_breaker)
    if not accepted_scores:
        return {
            "winner": None,
            "binary_votes": [],
            "per_judge_votes": [],
            "accepted_votes": 0,
            "rejected_votes": 0,
            "tied_judges": 0,
            "accepted_vote_share": 0.0,
            "rejected_vote_share": 0.0,
            "vote_margin": 0,
            "is_vote_tie": False,
            "tie_breaker": tie_breaker,
            "tie_breaker_used": None,
        }

    binary_votes = compute_binary_judge_votes(accepted_scores, rejected_scores)
    per_judge_votes = ["accepted" if vote == 1 else "rejected" for vote in binary_votes]
    accepted_votes = sum(binary_votes)
    rejected_votes = len(binary_votes) - accepted_votes
    tied_judges = sum(1 for accepted_score, rejected_score in zip(accepted_scores, rejected_scores) if accepted_score == rejected_score)

    winner: str | None = None
    tie_breaker_used: str | None = None
    if accepted_votes > rejected_votes:
        winner = "accepted"
    elif rejected_votes > accepted_votes:
        winner = "rejected"
    else:
        tie_breaker_used = tie_breaker
        if tie_breaker == "mean_score":
            mean_accepted = float(np.mean(accepted_scores)) if accepted_scores else 0.0
            mean_rejected = float(np.mean(rejected_scores)) if rejected_scores else 0.0
            if mean_accepted > mean_rejected:
                winner = "accepted"
            elif mean_rejected > mean_accepted:
                winner = "rejected"
            else:
                tie_breaker_used = "mean_score_then_first_judge"
                winner = next((vote for vote in per_judge_votes if vote != "tie"), None) or "accepted"
        elif tie_breaker == "first_judge":
            first_vote = per_judge_votes[0] if per_judge_votes else "accepted"
            if first_vote != "tie":
                winner = first_vote
            else:
                tie_breaker_used = "first_judge_then_mean_score"
                mean_accepted = float(np.mean(accepted_scores)) if accepted_scores else 0.0
                mean_rejected = float(np.mean(rejected_scores)) if rejected_scores else 0.0
                if mean_accepted > mean_rejected:
                    winner = "accepted"
                elif mean_rejected > mean_accepted:
                    winner = "rejected"
                else:
                    tie_breaker_used = "first_judge_then_mean_score_then_accepted"
                    winner = "accepted"

    num_judges = len(accepted_scores)
    return {
        "winner": winner,
        "binary_votes": binary_votes,
        "per_judge_votes": per_judge_votes,
        "accepted_votes": accepted_votes,
        "rejected_votes": rejected_votes,
        "tied_judges": tied_judges,
        "accepted_vote_share": (accepted_votes / num_judges) if num_judges else 0.0,
        "rejected_vote_share": (rejected_votes / num_judges) if num_judges else 0.0,
        "vote_margin": accepted_votes - rejected_votes,
        "is_vote_tie": accepted_votes == rejected_votes,
        "tie_breaker": tie_breaker,
        "tie_breaker_used": tie_breaker_used,
    }


def compute_judge_agreement(accepted_scores: list[float], rejected_scores: list[float]) -> float:
    """
    Compute judge agreement using Kendall's tau on score differences.

    Measures how much judges agree on the relative quality (score difference).
    Returns a value in [0.0, 1.0] where:
    - 1.0 = perfect agreement (all judges have same ranking and similar margins)
    - 0.0 = no agreement or anti-correlation

    Args:
        accepted_scores: List of scores for accepted answer from each judge
        rejected_scores: List of scores for rejected answer from each judge

    Returns:
        Agreement score in [0.0, 1.0]
    """
    if len(accepted_scores) != len(rejected_scores):
        raise ValueError(
            f"Accepted and rejected scores must have same length. "
            f"Got {len(accepted_scores)} and {len(rejected_scores)}"
        )

    if len(accepted_scores) < 2:
        # Need at least 2 judges to compute agreement
        return 1.0

    # Compute score differences (margin) for each judge
    score_diffs = [acc - rej for acc, rej in zip(accepted_scores, rejected_scores)]

    # Use Kendall's tau to measure agreement
    # We compare each judge's score diff with every other judge's score diff
    # Kendall's tau measures rank correlation, ranges from -1 to 1
    # We normalize to [0, 1] by (tau + 1) / 2
    try:
        # Create pairs for Kendall's tau
        # We need two orderable lists - use judge indices as one, score_diffs as ranking
        indices = list(range(len(score_diffs)))
        # Sort indices by score_diffs to get ranking
        sorted_indices = sorted(indices, key=lambda i: score_diffs[i])

        # Compute Kendall's tau between natural order and score-diff order
        # This measures how much the judges' rankings agree
        tau, p_value = kendalltau(indices, sorted_indices)

        # If all judges have identical score diffs, tau may be NaN
        if np.isnan(tau):
            # Check if all diffs are the same (perfect agreement)
            if len(set(score_diffs)) == 1:
                return 1.0
            else:
                return 0.0

        # Normalize tau from [-1, 1] to [0, 1]
        # tau = 1: perfect agreement
        # tau = 0: no correlation
        # tau = -1: perfect disagreement
        normalized_tau = (tau + 1.0) / 2.0

        return max(0.0, min(1.0, normalized_tau))

    except Exception as e:
        LOGGER.warning(f"Error computing Kendall's tau for judge agreement: {e}")
        return 0.0


def aggregate_multi_judge_reward(
    accepted_scores: list[float], rejected_scores: list[float], alpha: float = 0.7, beta: float = 0.3
) -> dict[str, Any]:
    """
    Aggregate rewards from multiple judges using pairwise accuracy and agreement.

    Reward = alpha * pairwise_accuracy + beta * agreement

    Args:
        accepted_scores: List of scores for accepted answer from each judge
        rejected_scores: List of scores for rejected answer from each judge
        alpha: Weight for pairwise accuracy (default: 0.7)
        beta: Weight for agreement (default: 0.3)

    Returns:
        Dictionary containing:
        - reward: Aggregated reward
        - pairwise_accuracy: Fraction of judges that ranked correctly
        - agreement: Judge agreement score (Kendall's tau normalized)
        - accepted_scores: List of accepted scores from each judge
        - rejected_scores: List of rejected scores from each judge
        - mean_accepted_score: Mean score for accepted answer
        - mean_rejected_score: Mean score for rejected answer
    """
    original_total = alpha + beta
    if original_total != 1.0:
        # Normalize weights to sum to 1
        if original_total > 0:
            alpha = alpha / original_total
            beta = beta / original_total
        else:
            alpha, beta = 0.5, 0.5
        LOGGER.warning(
            "Alpha and beta should sum to 1.0, got %.3f. Normalized to alpha=%.3f, beta=%.3f",
            original_total,
            alpha,
            beta,
        )

    pairwise_accuracy = compute_pairwise_accuracy(accepted_scores, rejected_scores)
    agreement = compute_judge_agreement(accepted_scores, rejected_scores)

    reward = alpha * pairwise_accuracy + beta * agreement

    return {
        "reward": reward,
        "pairwise_accuracy": pairwise_accuracy,
        "agreement": agreement,
        "accepted_scores": accepted_scores,
        "rejected_scores": rejected_scores,
        "mean_accepted_score": np.mean(accepted_scores) if accepted_scores else 0.0,
        "mean_rejected_score": np.mean(rejected_scores) if rejected_scores else 0.0,
        "num_judges": len(accepted_scores),
    }


def aggregate_multi_judge_comparison(
    accepted_scores: list[float],
    rejected_scores: list[float],
    *,
    aggregation_mode: str = "majority_vote",
    alpha: float = 0.7,
    beta: float = 0.3,
    tie_breaker: str = "mean_score",
    rubric_format_score: float = 0.0,
    margin_weight: float = 0.5,
    format_weight: float = 0.3,
    kappa_weight: float = 0.2,
) -> dict[str, Any]:
    """Summarize a multi-judge comparison and compute the configured reward."""
    aggregation_mode = normalize_multi_judge_aggregation_mode(aggregation_mode)
    tie_breaker = normalize_multi_judge_tie_breaker(tie_breaker)

    # Filter out judge pairs where either score is NaN (from timeouts / parse failures)
    num_original = len(accepted_scores)
    filtered = [
        (a, r) for a, r in zip(accepted_scores, rejected_scores)
        if not (math.isnan(a) or math.isnan(r))
    ]
    if filtered:
        accepted_scores, rejected_scores = [list(t) for t in zip(*filtered)]
    else:
        accepted_scores, rejected_scores = [], []
    num_dropped = num_original - len(accepted_scores)
    if num_dropped > 0:
        LOGGER.warning(
            f"[aggregate_multi_judge_comparison] Dropped {num_dropped}/{num_original} judge(s) "
            f"with NaN scores (timeouts/parse failures); {len(accepted_scores)} valid judge(s) remain"
        )

    if not accepted_scores:
        return {
            "reward": 0.0,
            "pairwise_accuracy": 0.0,
            "agreement": 0.0,
            "fleiss_kappa": 0.0,
            "clipped_fleiss_kappa": 0.0,
            "variance_penalty": 0.0,
            "winner": None,
            "binary_votes": [],
            "per_judge_votes": [],
            "accepted_votes": 0,
            "rejected_votes": 0,
            "tied_judges": 0,
            "accepted_vote_share": 0.0,
            "rejected_vote_share": 0.0,
            "vote_margin": 0,
            "is_vote_tie": False,
            "tie_breaker": tie_breaker,
            "tie_breaker_used": None,
            "accepted_scores": [],
            "rejected_scores": [],
            "mean_accepted_score": 0.0,
            "mean_rejected_score": 0.0,
            "num_judges": 0,
            "aggregation_mode": aggregation_mode,
            "agreement_bonus_reward": 0.0,
            "average_minus_variance_reward": 0.0,
            "margin_kappa_format_reward": 0.0,
            "avg_margin": 0.0,
            "rubric_format_score": rubric_format_score,
        }
    vote_summary = resolve_multi_judge_votes(
        accepted_scores,
        rejected_scores,
        tie_breaker=tie_breaker,
    )
    agreement_bonus = aggregate_multi_judge_reward(
        accepted_scores=accepted_scores,
        rejected_scores=rejected_scores,
        alpha=alpha,
        beta=beta,
    )
    binary_votes = vote_summary["binary_votes"]
    average_vote_reward = float(np.mean(binary_votes)) if binary_votes else 0.0
    fleiss_kappa = compute_fleiss_kappa(accepted_scores, rejected_scores)
    clipped_fleiss_kappa = float(np.clip(fleiss_kappa, 0.0, 1.0))
    variance_penalty = 1.0 - clipped_fleiss_kappa
    average_minus_variance_reward = average_vote_reward - variance_penalty

    # Avg margin: mean of per-judge (accepted - rejected) score differences
    margins = [a - r for a, r in zip(accepted_scores, rejected_scores)]
    avg_margin = float(np.mean(margins)) if margins else 0.0

    # margin_kappa_format: weighted combination of margin, format, and agreement
    margin_kappa_format_reward = (
        margin_weight * avg_margin
        + format_weight * rubric_format_score
        + kappa_weight * clipped_fleiss_kappa
    )

    if aggregation_mode == "majority_vote":
        reward = 1.0 if vote_summary["winner"] == "accepted" else 0.0
    elif aggregation_mode == "average_vote":
        reward = average_vote_reward
    elif aggregation_mode == "average_minus_variance":
        reward = average_minus_variance_reward
    elif aggregation_mode == "margin_kappa_format":
        reward = margin_kappa_format_reward
    else:
        reward = agreement_bonus["reward"]

    result = dict(agreement_bonus)
    result.update(vote_summary)
    result["reward"] = reward
    result["aggregation_mode"] = aggregation_mode
    result["agreement_bonus_reward"] = agreement_bonus["reward"]
    result["average_vote_reward"] = average_vote_reward
    result["fleiss_kappa"] = fleiss_kappa
    result["clipped_fleiss_kappa"] = clipped_fleiss_kappa
    result["variance_penalty"] = variance_penalty
    result["average_minus_variance_reward"] = average_minus_variance_reward
    result["margin_kappa_format_reward"] = margin_kappa_format_reward
    result["avg_margin"] = avg_margin
    result["rubric_format_score"] = rubric_format_score
    return result


async def compute_rubric_judge_reward_multi_judge_async(
    question: str,
    accepted_answer: str,
    rejected_answer: str,
    generated_rubric: str,
    judge_actors: list[tuple[Any, Any]],  # List of (generate_text_actor, sampling_params) tuples
    aggregation_mode: str = "majority_vote",
    alpha: float = 0.7,
    beta: float = 0.3,
    tie_breaker: str = "mean_score",
    margin_weight: float = 0.5,
    format_weight: float = 0.3,
    kappa_weight: float = 0.2,
) -> dict[str, Any]:
    """
    Compute reward using multiple judges and aggregate their evaluations.

    This function:
    1. Uses multiple LLM judges to evaluate both accepted and rejected answers
    2. Resolves per-judge votes for accepted vs rejected
    3. Computes auxiliary metrics (pairwise accuracy, agreement)
    4. Aggregates reward according to the configured mode

    Args:
        question: The question being answered
        accepted_answer: The accepted/correct answer
        rejected_answer: The rejected/incorrect answer
        generated_rubric: The rubric to use for evaluation
        judge_actors: List of (generate_text_actor, sampling_params) tuples for each judge
        aggregation_mode: One of {'average_vote', 'majority_vote', 'average_minus_variance',
            'agreement_bonus', 'margin_kappa_format'}
        alpha: Weight for pairwise accuracy in reward aggregation (default: 0.7)
        beta: Weight for agreement in the legacy agreement_bonus mode (default: 0.3)
        tie_breaker: Tie-breaker used when judges split evenly (default: 'mean_score')
        margin_weight: Weight for avg margin in margin_kappa_format mode (default: 0.5)
        format_weight: Weight for rubric format in margin_kappa_format mode (default: 0.3)
        kappa_weight: Weight for Fleiss's kappa in margin_kappa_format mode (default: 0.2)

    Returns:
        Dictionary containing:
        - reward: Aggregated reward from all judges
        - pairwise_accuracy: Fraction of judges that ranked accepted > rejected
        - agreement: Judge agreement score (Kendall's tau normalized to [0,1])
        - fleiss_kappa: Fleiss's kappa on binary judge votes
        - winner: Final resolved winner ('accepted' or 'rejected')
        - accepted_votes / rejected_votes: Per-answer vote counts across judges
        - accepted_scores: List of scores from each judge for accepted answer
        - rejected_scores: List of scores from each judge for rejected answer
        - mean_accepted_score: Mean score across judges for accepted answer
        - mean_rejected_score: Mean score across judges for rejected answer
        - num_judges: Number of judges used
        - rubric: The rubric used for evaluation
        - accepted_reasonings: List of reasoning from each judge for accepted answer
        - rejected_reasonings: List of reasoning from each judge for rejected answer
        - error: Error message if any step failed
    """
    # Enable logging for this computation if 3 seconds have passed
    _should_enable_logging()
    try:
        verbose_debug(f"[Multi-Judge Reward] Starting reward computation with {len(judge_actors)} judges")
        verbose_debug(f"[Multi-Judge Reward] Question: {question}")
        verbose_debug(f"[Multi-Judge Reward] Accepted answer: {accepted_answer}")
        verbose_debug(f"[Multi-Judge Reward] Rejected answer: {rejected_answer}")
        verbose_debug(
            f"[Multi-Judge Reward] aggregation_mode={aggregation_mode}, "
            f"tie_breaker={tie_breaker}, alpha={alpha}, beta={beta}"
        )

        result = {
            "reward": 0.0,
            "pairwise_accuracy": 0.0,
            "agreement": 0.0,
            "winner": None,
            "accepted_votes": 0,
            "rejected_votes": 0,
            "tied_judges": 0,
            "accepted_scores": [],
            "rejected_scores": [],
            "mean_accepted_score": 0.0,
            "mean_rejected_score": 0.0,
            "num_judges": len(judge_actors),
            "aggregation_mode": aggregation_mode,
            "rubric": None,
            "accepted_reasonings": [],
            "rejected_reasonings": [],
            "error": None,
        }

        # Clean rubric
        rubric = _remove_redacted_reasoning(generated_rubric)
        result["rubric"] = rubric
        verbose_debug(f"[Multi-Judge Reward] Using rubric: {rubric}")

        # Evaluate with all judges in parallel
        verbose_debug(f"[Multi-Judge Reward] Evaluating with {len(judge_actors)} judges in parallel...")

        accepted_task = judge_answer_with_multiple_judges(
            question=question, rubric=rubric, answer=accepted_answer, judge_actors=judge_actors, answer_type="accepted"
        )
        rejected_task = judge_answer_with_multiple_judges(
            question=question, rubric=rubric, answer=rejected_answer, judge_actors=judge_actors, answer_type="rejected"
        )

        accepted_evals, rejected_evals = await asyncio.gather(accepted_task, rejected_task)

        # Extract scores and reasonings
        accepted_scores = [eval_result["score"] for eval_result in accepted_evals]
        rejected_scores = [eval_result["score"] for eval_result in rejected_evals]
        accepted_reasonings = [eval_result.get("reasoning", "") for eval_result in accepted_evals]
        rejected_reasonings = [eval_result.get("reasoning", "") for eval_result in rejected_evals]

        result["accepted_scores"] = accepted_scores
        result["rejected_scores"] = rejected_scores
        result["accepted_reasonings"] = accepted_reasonings
        result["rejected_reasonings"] = rejected_reasonings

        verbose_debug(f"[Multi-Judge Reward] Accepted scores: {accepted_scores}")
        verbose_debug(f"[Multi-Judge Reward] Rejected scores: {rejected_scores}")

        # Compute rubric format score for margin_kappa_format mode
        rubric_format_score = is_valid_rubric_format(rubric) if aggregation_mode == "margin_kappa_format" else 0.0

        # Aggregate rewards
        aggregated = aggregate_multi_judge_comparison(
            accepted_scores=accepted_scores,
            rejected_scores=rejected_scores,
            aggregation_mode=aggregation_mode,
            alpha=alpha,
            beta=beta,
            tie_breaker=tie_breaker,
            rubric_format_score=rubric_format_score,
            margin_weight=margin_weight,
            format_weight=format_weight,
            kappa_weight=kappa_weight,
        )

        # Update result with aggregated metrics
        result.update(aggregated)

        # Add scalar accepted_score and rejected_score for compatibility with existing logging code
        # These are the mean scores across all judges
        result["accepted_score"] = result["mean_accepted_score"]
        result["rejected_score"] = result["mean_rejected_score"]

        verbose_debug(f"[Multi-Judge Reward] Pairwise accuracy: {result['pairwise_accuracy']:.4f}")
        verbose_debug(f"[Multi-Judge Reward] Agreement (Kendall's tau): {result['agreement']:.4f}")
        verbose_debug(
            f"[Multi-Judge Reward] Winner={result['winner']} "
            f"(accepted_votes={result['accepted_votes']}, rejected_votes={result['rejected_votes']})"
        )
        verbose_debug(f"[Multi-Judge Reward] Final aggregated reward: {result['reward']:.4f}")

        return result
    except Exception as e:
        LOGGER.error(f"Error in multi-judge reward computation: {e}", exc_info=True)
        return {
            "reward": 0.0,
            "pairwise_accuracy": 0.0,
            "agreement": 0.0,
            "winner": None,
            "accepted_votes": 0,
            "rejected_votes": 0,
            "tied_judges": 0,
            "accepted_scores": [],
            "rejected_scores": [],
            "mean_accepted_score": 0.0,
            "mean_rejected_score": 0.0,
            "num_judges": len(judge_actors) if judge_actors else 0,
            "aggregation_mode": aggregation_mode,
            "rubric": generated_rubric,
            "accepted_reasonings": [],
            "rejected_reasonings": [],
            "error": str(e),
        }
    finally:
        # Disable logging after this computation completes
        _disable_logging()
