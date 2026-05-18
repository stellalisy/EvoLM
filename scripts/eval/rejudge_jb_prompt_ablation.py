"""
Re-judge JudgeBench results with modified judge prompt templates.

Takes existing JB results (with pre-generated rubrics) and re-runs ONLY
the judge scoring with different prompt variants to isolate the effect of
the judge prompt template on accuracy.

Usage:
    python scripts/eval/rejudge_jb_prompt_ablation.py \
        --results_file <path_to_existing_jb_results.jsonl> \
        --judge_model Qwen/Qwen3-1.7B \
        --judge_port 8001 \
        --output_dir <output_dir> \
        --concurrency_limit 20
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reward-bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "JudgeBench"))
from rewardbench.generative_v2_rubric import strip_thinking_tokens
import utils.models as models


# ── Prompt variants to test ──────────────────────────────────────────────

PROMPT_VARIANTS = {
    "original": """Question: {question}

Rubric: {rubric}

Answer to evaluate: {response}

Evaluate the answer against the rubric. For each criterion, decide how well the answer satisfies it (0.0 = not at all, 1.0 = fully), then multiply by the criterion's weight. Sum the weighted scores to get the total (must be between 0.0 and 1.0).

Output ONLY valid JSON:
{{"reasoning": "<evaluate each criterion, give satisfaction * weight, then sum>", "score": <float 0.0-1.0>}}

Example 1 (rubric: Factual Accuracy 0.4, Completeness 0.35, Clarity 0.25):
{{"reasoning": "Factual Accuracy (weight 0.4): answer is fully correct, 1.0 * 0.4 = 0.4. Completeness (weight 0.35): covers main points but misses edge cases, 0.6 * 0.35 = 0.21. Clarity (weight 0.25): well organized and easy to follow, 1.0 * 0.25 = 0.25. Total = 0.86", "score": 0.86}}
Example 2 (rubric: Correctness of Solution 0.5, Use of Examples 0.3, Appropriate Detail 0.2):
{{"reasoning": "Correctness of Solution (weight 0.5): correct approach but has an arithmetic error in the final step, 0.8 * 0.5 = 0.4. Use of Examples (weight 0.3): no examples provided, 0.0 * 0.3 = 0.0. Appropriate Detail (weight 0.2): gives a brief answer without elaboration, 0.2 * 0.2 = 0.04. Total = 0.44", "score": 0.44}}

Your evaluation:""",

    "no_examples": """Question: {question}

Rubric: {rubric}

Answer to evaluate: {response}

Evaluate the answer against the rubric. For each criterion in the rubric, decide how well the answer satisfies it (0.0 = not at all, 1.0 = fully), then multiply by the criterion's weight. Sum the weighted scores to get the total (must be between 0.0 and 1.0).

You MUST evaluate using the exact criteria listed in the rubric above. Do not use any other criteria.

Output ONLY valid JSON:
{{"reasoning": "<for each criterion in the rubric: name the criterion, assess satisfaction 0.0-1.0, multiply by weight, then sum all>", "score": <float 0.0-1.0>}}

Your evaluation:""",

    "no_examples_strict": """Question: {question}

Rubric: {rubric}

Answer to evaluate: {response}

IMPORTANT: You must evaluate the answer using ONLY the criteria listed in the rubric above. Do NOT use generic criteria like "Factual Accuracy" or "Completeness" — use the specific criteria from the rubric.

For each criterion in the rubric:
1. State the criterion name exactly as written in the rubric
2. Assess how well the answer satisfies it (0.0 = not at all, 1.0 = fully)
3. Multiply the satisfaction score by the criterion's weight

Sum all weighted scores to get the total (must be between 0.0 and 1.0).

Output ONLY valid JSON:
{{"reasoning": "<for each rubric criterion: [criterion name] (weight X): satisfaction * weight = Y. Then sum all.>", "score": <float 0.0-1.0>}}

Your evaluation:""",
}

SYSTEM_PROMPT = "You are an expert evaluator judging answers based on a rubric."


def parse_score(text: str) -> float:
    if not text or not text.strip():
        return -1.0
    m = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text.strip())
    if m:
        return float(m.group(1))
    return -1.0


async def score_response(
    api, tokenizer, question: str, rubric: str, response_text: str,
    prompt_template: str, strip_fn,
) -> Tuple[float, str]:
    user_prompt = prompt_template.format(
        question=question, rubric=rubric, response=response_text,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        raw = await api.complete(
            prompt=prompt, temperature=0.6, top_p=0.95, max_tokens=16384,
        )
        cleaned = strip_fn(raw)
        score = parse_score(cleaned)
        return score, cleaned
    except Exception as e:
        print(f"Scoring failed: {e}")
        return -1.0, ""


async def rejudge_pair(
    api, tokenizer, pair: Dict[str, Any], prompt_template: str,
    strip_fn, semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        question = pair["question"]
        response_A = pair["response_A"]
        response_B = pair["response_B"]
        rubric = pair["judgments"][0]["judgment"]["rubric"]

        (score_A, raw_A), (score_B, raw_B) = await asyncio.gather(
            score_response(api, tokenizer, question, rubric, response_A, prompt_template, strip_fn),
            score_response(api, tokenizer, question, rubric, response_B, prompt_template, strip_fn),
        )

        if score_A < 0 and score_B < 0:
            decision = None
        elif score_A > score_B:
            decision = "A>B"
        elif score_B > score_A:
            decision = "B>A"
        else:
            decision = "A=B"

        return {
            "pair_id": pair["pair_id"],
            "source": pair["source"],
            "label": pair["label"],
            "decision": decision,
            "score_A": score_A,
            "score_B": score_B,
            "raw_judge_A": raw_A,
            "raw_judge_B": raw_B,
            "rubric": rubric,
        }


async def run_variant(
    api, tokenizer, pairs: List[Dict], variant_name: str,
    prompt_template: str, strip_fn, concurrency_limit: int,
    output_dir: str,
) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f"Running variant: {variant_name}")
    print(f"{'='*60}")

    semaphore = asyncio.Semaphore(concurrency_limit)
    output_file = os.path.join(output_dir, f"rejudge_{variant_name}.jsonl")

    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                d = json.loads(line)
                existing_ids.add(d["pair_id"])
        print(f"  Resuming: {len(existing_ids)} already done")

    remaining = [p for p in pairs if p["pair_id"] not in existing_ids]
    print(f"  Pairs to process: {len(remaining)}")

    tasks = [
        asyncio.create_task(
            rejudge_pair(api, tokenizer, pair, prompt_template, strip_fn, semaphore)
        )
        for pair in remaining
    ]

    file_lock = asyncio.Lock()
    results = []
    for future in tqdm_asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        async with file_lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Load all results (existing + new)
    all_results = []
    with open(output_file) as f:
        for line in f:
            all_results.append(json.loads(line))

    # Compute metrics
    cat_map = {
        "knowledge": ["mmlu-pro", "gpqa", "arc"],
        "reasoning": ["zebra-grid", "livebench-reasoning"],
        "math": ["math", "minerva", "livebench-math", "aime", "gsm8k"],
        "coding": ["livecodebench", "humaneval", "bigcode"],
    }

    def get_category(source):
        for cat, prefixes in cat_map.items():
            for p in prefixes:
                if p in source.lower():
                    return cat
        return "other"

    total = correct = none_count = 0
    by_cat = {}
    for r in all_results:
        total += 1
        cat = get_category(r["source"])
        by_cat.setdefault(cat, {"total": 0, "correct": 0, "none": 0})
        by_cat[cat]["total"] += 1
        if r["decision"] is None:
            none_count += 1
            by_cat[cat]["none"] += 1
        elif r["decision"] == r["label"]:
            correct += 1
            by_cat[cat]["correct"] += 1

    print(f"\n  Results for {variant_name}:")
    print(f"  Overall: {correct}/{total} = {correct/max(total,1)*100:.1f}% (None: {none_count})")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        acc = c["correct"] / max(c["total"], 1) * 100
        print(f"    {cat:15s}: {c['correct']}/{c['total']} = {acc:.1f}% (None: {c['none']})")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True,
                        help="Path to existing JB results .jsonl with rubrics")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--judge_port", type=int, default=8001)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--concurrency_limit", type=int, default=20)
    parser.add_argument("--variants", nargs="+",
                        default=["no_examples", "no_examples_strict"],
                        choices=list(PROMPT_VARIANTS.keys()))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = []
    with open(args.results_file) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs from {args.results_file}")

    api = models.get_chat_api_from_model(args.judge_model, port=args.judge_port)
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    strip_fn = strip_thinking_tokens

    for variant_name in args.variants:
        template = PROMPT_VARIANTS[variant_name]
        asyncio.run(run_variant(
            api, tokenizer, pairs, variant_name, template,
            strip_fn, args.concurrency_limit, args.output_dir,
        ))


if __name__ == "__main__":
    main()
