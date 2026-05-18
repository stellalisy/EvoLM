"""Extract rubrics from a rubric generator checkpoint for qualitative analysis.

Loads a checkpoint via vLLM, generates rubrics for a set of eval prompts,
and saves them as JSONL.

Usage:
    python scripts/eval/extract_rubrics.py \
        --model /path/to/checkpoint/step_100 \
        --prompts data/eval_prompts.jsonl \
        --output /path/to/output/rubrics_step_100.jsonl \
        --port 8000
"""

import argparse
import json
import sys
from pathlib import Path

import openai
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from open_instruct.search_rewards.utils.rubric_chat_templates import (
    RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT,
    get_rubric_system_prompt,
)


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def generate_rubrics(
    client: openai.OpenAI,
    tokenizer,
    model_name: str,
    prompts: list[dict],
    max_tokens: int = 16384,
    rubric_prompt_key: str = "rubric_generation",
) -> list[dict]:
    system_prompt = get_rubric_system_prompt(rubric_prompt_key)
    messages_batch = []
    for p in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p["question"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        messages_batch.append(text)

    results = []
    for i, text in enumerate(messages_batch):
        resp = client.completions.create(
            model=model_name,
            prompt=text,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
        )
        rubric = resp.choices[0].text.strip()
        results.append({
            "prompt_index": i,
            "question": prompts[i]["question"],
            "rubric": rubric,
        })
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} rubrics")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--prompts", required=True, help="Path to eval_prompts.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument(
        "--rubric_prompt", default="rubric_generation",
        choices=list(RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT.keys()),
        help="Registry key for rubric generation system prompt.",
    )
    args = parser.parse_args()

    client = openai.OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    results = generate_rubrics(client, tokenizer, args.model, prompts, args.max_tokens, args.rubric_prompt)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} rubrics to {args.output}")


if __name__ == "__main__":
    main()
