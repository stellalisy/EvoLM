"""Quick validation of Rubric-ARM reward functions.

Usage:
  # With local vLLM (default — set VLLM_PORT or start vLLM on port 8000):
  uv run python scripts/validate_rubric_arm.py

  # With an API model:
  uv run python scripts/validate_rubric_arm.py --model gpt-4o-mini

  # Unit-tests only (no LLM calls):
  uv run python scripts/validate_rubric_arm.py --unit-only
"""
import argparse
import asyncio
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="LiteLLM model name (default: local vLLM)")
parser.add_argument("--vllm-port", type=int, default=8000)
parser.add_argument("--unit-only", action="store_true", help="Run unit tests only, no LLM calls")
_args = parser.parse_args()

if _args.model:
    os.environ["RUBRIC_JUDGE_MODEL"] = _args.model
else:
    vllm_base = f"http://localhost:{_args.vllm_port}/v1"
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_API_BASE"] = vllm_base
    os.environ["RUBRIC_JUDGE_MODEL"] = "openai/Qwen/Qwen3-1.7B"

from open_instruct.search_rewards.rubric_judge_rewards import (
    score_policy_rollouts_with_rubric_arm,
    compute_rubric_arm_rubricator_reward,
    _extract_rubric_arm_winner,
    _rubric_arm_format_reward,
    _rubric_arm_generate_rubric,
)


def test_winner_extraction():
    sample = (
        "--- Compliance Check ---\n"
        "Identified Gatekeeper Criterion: Criterion 1\n\n"
        "--- Analysis ---\n"
        "**Response A:**\n- Criterion 1: Justification: Good\n"
        "**Response B:**\n- Criterion 1: Justification: Bad\n\n"
        "--- Final Judgment ---\n"
        "Justification: A is better.\n"
        "Winner: Response A"
    )
    assert _extract_rubric_arm_winner(sample) == "A"
    assert _rubric_arm_format_reward(sample) == 1.0

    assert _extract_rubric_arm_winner("Winner: Response B") == "B"
    assert _extract_rubric_arm_winner("I think A is better") is None
    assert _rubric_arm_format_reward("I think A is better") == 0.0
    print("[PASS] Winner extraction and format reward")


async def test_rubric_generation():
    question = (
        "Write a two-paragraph essay about the benefits of renewable energy. "
        "Include the keywords 'sustainability' and 'innovation'. "
        "Use an enthusiastic tone."
    )
    items = await _rubric_arm_generate_rubric(question=question)
    print(f"[INFO] Generated {len(items)} rubric items for question")
    for i, item in enumerate(items):
        has_tag = "[Hard Rule]" in item or "[Principle]" in item
        print(f"  {i+1}. {item[:100]}... tag_present={has_tag}")
    assert len(items) > 0, "Should generate at least one rubric item"
    print("[PASS] Rubric generation")
    return items


async def test_pairwise_scoring():
    question = "Write a haiku about autumn."
    good_answer = "Crimson leaves descend\nWhispering through the cool breeze\nNature's final dance"
    bad_answer = "Autumn is nice. I like autumn. It is a season."

    results = await score_policy_rollouts_with_rubric_arm(
        questions=[question, question],
        answers=[good_answer, bad_answer],
        preferred_answers=[good_answer, good_answer],
        dispreferred_answers=[bad_answer, bad_answer],
    )
    for i, r in enumerate(results):
        print(
            f"  Answer {i+1}: score={r['score']:.2f}, "
            f"fwd={r.get('winner_forward')}, rev={r.get('winner_reverse')}, "
            f"rubric_items={len(r.get('rubric_items', []))}"
        )
    # The good answer (index 0) should have a higher score
    print(f"[INFO] Good answer score: {results[0]['score']:.2f}, Bad answer score: {results[1]['score']:.2f}")
    print("[PASS] Pairwise scoring")
    return results


async def test_rubricator_reward():
    question = "Explain photosynthesis in simple terms."
    preferred = (
        "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide "
        "into glucose and oxygen. Think of it like the plant's way of cooking its own food using "
        "sunlight as the stove! The chlorophyll in leaves captures light energy, which powers "
        "a chemical reaction that transforms simple ingredients into sugar that fuels the plant's growth."
    )
    dispreferred = "Plants eat sun. Sun good for plant."

    rubric_text = (
        "1. The response must explain the core mechanism of photosynthesis. [Hard Rule]\n"
        "2. The response must use simple, accessible language. [Hard Rule]\n"
        "3. The response should use analogies or comparisons to aid understanding. [Principle]\n"
        "4. The response should be scientifically accurate. [Principle]"
    )

    result = await compute_rubric_arm_rubricator_reward(
        question=question,
        rubric_text=rubric_text,
        preferred_answer=preferred,
        dispreferred_answer=dispreferred,
    )
    print(
        f"  Rubricator reward: score={result['score']:.2f}, "
        f"judge_correct={result['judge_correct']}, "
        f"r_format={result['r_format']:.1f}, r_acc={result['r_acc']:.1f}, "
        f"judge_winner={result['judge_winner']}"
    )
    print("[PASS] Rubricator reward")
    return result


async def main():
    print("=" * 60)
    print("Rubric-ARM Validation Tests")
    print("=" * 60)

    test_winner_extraction()

    if _args.unit_only:
        print("\n[SKIP] LLM-dependent tests (--unit-only)")
    else:
        await test_rubric_generation()
        await test_pairwise_scoring()
        await test_rubricator_reward()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
