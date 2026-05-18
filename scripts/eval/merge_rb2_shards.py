#!/usr/bin/env python3
"""Merge sharded RB2 evaluation results.

After running eval_rewardbench2.sh with RB2_NUM_SHARDS>1, each shard saves
its own JSON under eval-set-scores/. This script merges them into a single
result and prints the per-subset accuracy.

Usage:
    python merge_rb2_shards.py <results_dir> <subset_slug> <num_shards> [judge_model_name]

Example:
    python merge_rb2_shards.py \
        /checkpoint/.../rubric/step_1000/rb2_results_v2_rubric_generation_v3 \
        factuality 20 "Qwen/Qwen3-32B"
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="RB2 results directory (REWARDBENCH_LOCAL_RESULTS_DIR)")
    parser.add_argument("subset_slug", help="Subset slug, e.g. factuality, focus, math, precise_if, safety")
    parser.add_argument("num_shards", type=int)
    parser.add_argument("--judge_model", default="Qwen/Qwen3-32B",
                        help="Judge model name as it appears in filenames (slashes become -)")
    args = parser.parse_args()

    scores_dir = os.path.join(args.results_dir, "eval-set-scores")
    judge_slug = args.judge_model.replace("/", "-")

    all_results = []
    missing = []
    for sid in range(args.num_shards):
        suffix = f"subset_{args.subset_slug}_shard{sid}of{args.num_shards}"
        # Filenames look like: <judge_slug>__<suffix>.json  (under judge org subdir)
        # e.g. eval-set-scores/Qwen/Qwen3-32B__subset_factuality_shard0of20.json
        fname = f"{judge_slug}__{suffix}.json"
        # Handle org/ prefix in judge name
        parts = args.judge_model.split("/")
        if len(parts) == 2:
            fpath = os.path.join(scores_dir, parts[0], fname)
        else:
            fpath = os.path.join(scores_dir, fname)
        if not os.path.exists(fpath):
            missing.append(sid)
            continue
        with open(fpath) as f:
            d = json.load(f)
        shard_results = d.get("results", [])
        n = len(shard_results)
        correct = sum(1 for r in shard_results if r)
        all_results.extend(shard_results)
        print(f"  shard {sid}: {correct}/{n} correct")

    if missing:
        print(f"\nWARNING: missing shards: {missing}")
        print("Partial results below.\n")

    total = len(all_results)
    correct = sum(1 for r in all_results if r)
    if total > 0:
        acc = correct / total
        print(f"\n{args.subset_slug}: {correct}/{total} ({acc*100:.1f}%)")
    else:
        print(f"\n{args.subset_slug}: no results found")

    # Also save merged result
    merged_path = os.path.join(
        scores_dir,
        *args.judge_model.split("/")[:-1],
        f"{judge_slug}__subset_{args.subset_slug}_merged.json"
    )
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w") as f:
        json.dump({"results": all_results, "subset": args.subset_slug,
                    "num_shards": args.num_shards, "missing_shards": missing}, f)
    print(f"Saved merged results to {merged_path}")


if __name__ == "__main__":
    main()
