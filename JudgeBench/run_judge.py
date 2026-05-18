from typing import List, Dict, Any
import argparse
import asyncio
import json
import os
import random

from tqdm.asyncio import tqdm_asyncio

import utils.file_operations as file_operations
import utils.judges as judges
import utils.metrics as metrics


async def judge_pairs(pairs: List[Dict[str, Any]], judge, concurrency_limit: int = 1, reverse_order: int = False, output_file: str = None):
    semaphore = asyncio.Semaphore(concurrency_limit)
    file_lock = asyncio.Lock()
    is_rubric_judge = isinstance(judge, judges.RubricJudge)
    
    async def judge_pair(pair: Dict[str, Any]):
        async with semaphore:
            
            question = pair["question"]
            response_A = pair["response_A"]
            response_B = pair["response_B"]

            shared_rubric = None
            if is_rubric_judge and reverse_order:
                shared_rubric = await judge.generate_rubric(question)

            try:
                judgment_1 = await judge.get_judgment(question, response_A, response_B,
                                                      rubric=shared_rubric)
            except Exception as e:
                print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
                judgment_1 = None
            judgments = [judgment_1]
            
            if reverse_order:
                try:
                    judgment_2 = await judge.get_judgment(question, response_B, response_A,
                                                          rubric=shared_rubric)
                except Exception as e:
                    print(f"Failed to judge pair {pair['pair_id']} due to the following error: {e}.")
                    judgment_2 = None
                judgments.append(judgment_2)
            
            pair["judge_name"] = getattr(judge, 'judge_model_name', str(type(judge).__name__))
            pair["judgments"] = judgments
            return pair

    tasks = [asyncio.create_task(judge_pair(pair)) for pair in pairs]

    for future in tqdm_asyncio.as_completed(tasks):
        pair = await future
        if output_file is not None:
            async with file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    return pairs


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_health_summary(pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute scoring health metrics for JudgeBench runs.
    This focuses on rubric/no_rubric judges that emit score_A/score_B.
    """
    pair_total = len(pairs)
    pair_with_failed_score = 0
    pair_with_null_decision = 0

    game_total = 0
    game_with_failed_score = 0

    scoring_calls = 0
    failed_score_calls = 0
    empty_failed_score_calls = 0
    parse_failed_score_calls = 0

    decision_none = 0
    decision_tie = 0
    decision_non_tie = 0

    for pair in pairs:
        judgments = pair.get("judgments", [])
        pair_failed = False
        pair_has_null_decision = False

        for judgment in judgments:
            game_total += 1

            if judgment is None:
                decision_none += 1
                pair_failed = True
                pair_has_null_decision = True
                continue

            decision = judgment.get("decision")
            if decision is None:
                decision_none += 1
                pair_has_null_decision = True
            elif decision == "A=B":
                decision_tie += 1
            else:
                decision_non_tie += 1

            judgment_payload = judgment.get("judgment", {})
            game_failed = False

            for side in ("A", "B"):
                score_key = f"score_{side}"
                raw_key = f"raw_judge_{side}"
                if score_key not in judgment_payload:
                    continue

                score = judgment_payload.get(score_key)
                if not isinstance(score, (int, float)):
                    continue

                scoring_calls += 1
                if score < 0:
                    failed_score_calls += 1
                    game_failed = True
                    raw = judgment_payload.get(raw_key)
                    if isinstance(raw, str) and raw == "":
                        empty_failed_score_calls += 1
                    else:
                        # Includes parse failures where raw text exists but score couldn't be parsed.
                        parse_failed_score_calls += 1

            if game_failed:
                game_with_failed_score += 1
                pair_failed = True

        if pair_failed:
            pair_with_failed_score += 1
        if pair_has_null_decision:
            pair_with_null_decision += 1

    return {
        "pair_total": pair_total,
        "pair_with_failed_score": pair_with_failed_score,
        "pair_with_null_decision": pair_with_null_decision,
        "pair_failed_score_rate": _safe_rate(pair_with_failed_score, pair_total),
        "pair_null_decision_rate": _safe_rate(pair_with_null_decision, pair_total),
        "game_total": game_total,
        "game_with_failed_score": game_with_failed_score,
        "game_failed_score_rate": _safe_rate(game_with_failed_score, game_total),
        "decision_none": decision_none,
        "decision_tie": decision_tie,
        "decision_non_tie": decision_non_tie,
        "decision_none_rate": _safe_rate(decision_none, game_total),
        "scoring_calls": scoring_calls,
        "failed_score_calls": failed_score_calls,
        "failed_score_rate": _safe_rate(failed_score_calls, scoring_calls),
        "empty_failed_score_calls": empty_failed_score_calls,
        "empty_failed_score_rate": _safe_rate(empty_failed_score_calls, scoring_calls),
        "parse_failed_score_calls": parse_failed_score_calls,
        "parse_failed_score_rate": _safe_rate(parse_failed_score_calls, scoring_calls),
    }


def print_health_summary(summary: Dict[str, float]) -> None:
    print("Health check summary:")
    print(
        f"  pairs={summary['pair_total']} | "
        f"pairs_with_failed_score={summary['pair_with_failed_score']} "
        f"({100 * summary['pair_failed_score_rate']:.2f}%) | "
        f"pairs_with_null_decision={summary['pair_with_null_decision']} "
        f"({100 * summary['pair_null_decision_rate']:.2f}%)"
    )
    print(
        f"  games={summary['game_total']} | "
        f"games_with_failed_score={summary['game_with_failed_score']} "
        f"({100 * summary['game_failed_score_rate']:.2f}%) | "
        f"decision_none={summary['decision_none']} "
        f"({100 * summary['decision_none_rate']:.2f}%) | "
        f"decision_tie={summary['decision_tie']} | "
        f"decision_non_tie={summary['decision_non_tie']}"
    )
    print(
        f"  scoring_calls={summary['scoring_calls']} | "
        f"failed_score_calls={summary['failed_score_calls']} "
        f"({100 * summary['failed_score_rate']:.2f}%) | "
        f"empty_failed={summary['empty_failed_score_calls']} "
        f"({100 * summary['empty_failed_score_rate']:.2f}%) | "
        f"parse_failed={summary['parse_failed_score_calls']} "
        f"({100 * summary['parse_failed_score_rate']:.2f}%)"
    )


def main(args: argparse.Namespace) -> None:
    
    random.seed(args.seed)
    
    pairs = file_operations.read_jsonl(args.pairs)    

    dataset_name = os.path.basename(args.pairs).replace(".jsonl", "")

    if args.rubric_model:
        rubric_model_tag = os.path.basename(args.rubric_model.rstrip("/")).replace("/", "-")
    else:
        rubric_model_tag = "none"
    judge_model_tag = args.judge_model.replace("/", "-")
    file_name = (
        f"dataset=judgebench,"
        f"response_model=gpt-4o-2024-05-13,"
        f"judge_name={args.judge_name},"
        f"rubric_model={rubric_model_tag},"
        f"judge_model={judge_model_tag}.jsonl"
    )

    output_dir = args.output_dir or "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Resuming with retry of failed pairs...")
        original_num_pairs = len(pairs)
        existing_pairs = file_operations.read_jsonl(file_path)

        def _pair_succeeded(pair):
            """A pair succeeded if all judgments have valid (non-negative) scores."""
            for j in pair.get("judgments", []):
                if j is None:
                    return False
                payload = j.get("judgment", {})
                for side in ("A", "B"):
                    score = payload.get(f"score_{side}")
                    if not isinstance(score, (int, float)) or score < 0:
                        return False
            return True

        succeeded_ids = {p["pair_id"] for p in existing_pairs if _pair_succeeded(p)}
        failed_ids = {p["pair_id"] for p in existing_pairs} - succeeded_ids

        # Keep only the succeeded pairs in the output; failed ones will be retried.
        if failed_ids:
            kept_pairs = [p for p in existing_pairs if p["pair_id"] in succeeded_ids]
            with open(file_path, 'w') as f:
                for p in kept_pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')

        pairs = [p for p in pairs if p["pair_id"] not in succeeded_ids]
        n_retry = len([p for p in pairs if p["pair_id"] in failed_ids])
        n_new = len(pairs) - n_retry
        print(f"Skipped {len(succeeded_ids)} succeeded pairs, retrying {n_retry} failed pairs, {n_new} new pairs.")

    judge_kwargs = {}
    if args.judge_name == "rubric":
        judge_kwargs["rubric_model_name"] = args.rubric_model
        judge_kwargs["rubric_port"] = args.rubric_port
        judge_kwargs["judge_port"] = args.judge_port
        if args.rubric_prompt_key:
            judge_kwargs["rubric_prompt_key"] = args.rubric_prompt_key
    elif args.judge_name == "no_rubric":
        judge_kwargs["judge_port"] = args.judge_port

    judge = judges.get_judge_from_judge_name_and_model(
        args.judge_name, args.judge_model, **judge_kwargs
    )

    if pairs: 
        print("Judging pairs ...")
        pairs = asyncio.run(
            judge_pairs(
                pairs,
                judge,
                reverse_order=not args.single_game,
                concurrency_limit=args.concurrency_limit,
                output_file=file_path,
            )
        )

    print("Computing final metrics ...") 
    pairs = file_operations.read_jsonl(file_path)
    for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
        score = metrics.compute_final_metrics(pairs, not args.single_game, include_fn = lambda x: x["source"].startswith(source))
        print(f"{source if source else 'Overall'}: {score:.2f}%.")

    summary = compute_health_summary(pairs)
    print_health_summary(summary)

    violations = []
    if summary["failed_score_rate"] > args.health_max_failed_score_rate:
        violations.append(
            f"failed_score_rate={summary['failed_score_rate']:.4f} > "
            f"max={args.health_max_failed_score_rate:.4f}"
        )
    if summary["pair_failed_score_rate"] > args.health_max_pair_failure_rate:
        violations.append(
            f"pair_failed_score_rate={summary['pair_failed_score_rate']:.4f} > "
            f"max={args.health_max_pair_failure_rate:.4f}"
        )
    if summary["decision_none_rate"] > args.health_max_null_decision_rate:
        violations.append(
            f"null_decision_rate={summary['decision_none_rate']:.4f} > "
            f"max={args.health_max_null_decision_rate:.4f}"
        )

    if violations:
        print("Health check violations:")
        for violation in violations:
            print(f"  - {violation}")
        if args.health_enforce:
            print("Health check failed and enforcement is enabled. Exiting with non-zero status.")
            raise SystemExit(2)
        print("Health check failed, but enforcement is disabled.")
    else:
        print("Health check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_name', type=str, required=True)
    parser.add_argument('--judge_model', type=str, required=True)
    parser.add_argument('--rubric_model', type=str, default=None)
    parser.add_argument('--response_model', type=str, default=None)
    parser.add_argument('--rubric_port', type=int, default=8000)
    parser.add_argument('--judge_port', type=int, default=8001)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--single_game', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--concurrency_limit', type=int, default=1)
    parser.add_argument('--pairs', type=str, required=True)
    parser.add_argument('--rubric_prompt_key', type=str, default=None,
                        help='Rubric generation prompt key (e.g. "rubric_generation_v3"). '
                             'Must match the prompt used during training. '
                             'Defaults to None (uses DEFAULT_RUBRIC_GENERATION_SYSTEM_PROMPT).')
    parser.add_argument('--health_enforce', action="store_true",
                        help='Fail the run (exit non-zero) when health thresholds are exceeded.')
    parser.add_argument('--health_max_failed_score_rate', type=float, default=0.02,
                        help='Maximum allowed failed-score rate over score_A/score_B calls.')
    parser.add_argument('--health_max_pair_failure_rate', type=float, default=0.10,
                        help='Maximum allowed fraction of pairs with any failed score.')
    parser.add_argument('--health_max_null_decision_rate', type=float, default=0.05,
                        help='Maximum allowed fraction of game-level null decisions.')
    args = parser.parse_args()
    main(args)
