#!/usr/bin/env python3
"""
Merge sharded evaluation metrics into one canonical metrics file.

Shard layout:
  <output_root>/eval_shards/<task_name>/shard_<i>/step_<step>/
      task-000-<task_name>-metrics.json

Merged output:
  <output_root>/step_<step>/task-000-<task_name>-metrics.json

Also supports copying the existing (pre-shard) checkpoint progress into shard
directories so that already-generated examples are not re-computed.

Usage:
  # Merge after all shards complete
  python merge_eval_shards.py --output-root /path/to/olmes_eval \
      --step 950 --task-name popqa --num-shards 16

  # Seed shard checkpoints from an existing partial run
  python merge_eval_shards.py --output-root /path/to/olmes_eval \
      --step 950 --task-name popqa --num-shards 16 --seed-checkpoints
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _weighted_metric(shards: list[dict[str, Any]], metric_name: str) -> float | None:
    total_weight = 0
    weighted_sum = 0.0
    for shard in shards:
        metrics = shard.get("metrics", {})
        if metric_name not in metrics:
            continue
        try:
            metric_value = float(metrics[metric_name])
        except (TypeError, ValueError):
            continue
        weight = int(shard.get("num_instances", 0))
        total_weight += weight
        weighted_sum += metric_value * weight
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def shard_dir(output_root: Path, task_name: str, shard_id: int, step: int) -> Path:
    return output_root / "eval_shards" / task_name / f"shard_{shard_id}" / f"step_{step}"


def gather_checkpoints(
    output_root: Path, step: int, task_name: str, old_num_shards: int
) -> int:
    """Merge checkpoint lines from all existing shards back into the central checkpoint.

    Tasks like MBPP+ produce multiple checkpoint lines per prompt hash (one per
    sample).  We deduplicate by exact line content so no samples are lost.

    Returns the number of unique checkpoint lines after merging.
    """
    step_dir = output_root / f"step_{step}"

    central_ckpt = None
    for f in step_dir.glob(f"*-{task_name}-generation-checkpoint.jsonl"):
        central_ckpt = f
        break

    unique_lines: set[str] = set()

    if central_ckpt is not None and central_ckpt.exists():
        with central_ckpt.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if stripped:
                    unique_lines.add(stripped)
        print(f"  Central checkpoint: {len(unique_lines)} lines")

    for sid in range(old_num_shards):
        sdir = shard_dir(output_root, task_name, sid, step)
        for ckpt_file in sdir.glob(f"*-generation-checkpoint.jsonl"):
            before = len(unique_lines)
            with ckpt_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.rstrip("\n")
                    if stripped:
                        unique_lines.add(stripped)
            added = len(unique_lines) - before
            if added > 0:
                print(f"  Shard {sid}: +{added} new lines")

    if central_ckpt is None:
        ckpt_name = f"task-000-{task_name}-generation-checkpoint.jsonl"
        central_ckpt = step_dir / ckpt_name

    central_ckpt.parent.mkdir(parents=True, exist_ok=True)
    with central_ckpt.open("w", encoding="utf-8") as fh:
        for line in unique_lines:
            fh.write(line + "\n")

    print(f"  Merged checkpoint: {len(unique_lines)} total lines -> {central_ckpt}")
    return len(unique_lines)


def seed_checkpoints(
    output_root: Path, step: int, task_name: str, num_shards: int
) -> None:
    """Copy the existing generation checkpoint to each shard directory.

    OLMES checkpoints are keyed by prompt hash (SHA-256 of the prompt text),
    NOT by doc_id. Each shard will build requests only for its subset of docs,
    then look up prompt hashes in the checkpoint — any matching hashes will be
    skipped. So we can safely copy the ENTIRE checkpoint to every shard: each
    shard will ignore hashes for prompts it doesn't own.
    """
    step_dir = output_root / f"step_{step}"

    ckpt_file = None
    for f in step_dir.glob(f"*-{task_name}-generation-checkpoint.jsonl"):
        ckpt_file = f
        break

    if ckpt_file is None or not ckpt_file.exists():
        print(f"  No existing checkpoint for {task_name} at step {step}, shards start fresh")
        return

    ckpt_size = ckpt_file.stat().st_size
    num_lines = sum(1 for _ in ckpt_file.open())
    print(f"  Existing checkpoint: {num_lines} lines ({ckpt_size / 1e6:.1f} MB)")

    import shutil
    for sid in range(num_shards):
        sdir = shard_dir(output_root, task_name, sid, step)
        sdir.mkdir(parents=True, exist_ok=True)

        dst = sdir / ckpt_file.name
        if dst.exists():
            print(f"  Shard {sid}: checkpoint already exists, skipping")
        else:
            shutil.copy2(str(ckpt_file), str(dst))
            print(f"  Shard {sid}: seeded with full checkpoint ({num_lines} lines)")


def merge_shards(
    output_root: Path, step: int, task_name: str, num_shards: int
) -> Path:
    """Merge per-shard metrics into a single canonical metrics file."""
    step_dir = output_root / f"step_{step}"
    shard_metrics: list[dict[str, Any]] = []
    shard_paths: list[Path] = []

    for sid in range(num_shards):
        sdir = shard_dir(output_root, task_name, sid, step)
        candidates = list(sdir.glob(f"*-{task_name}-metrics.json"))
        if not candidates:
            raise FileNotFoundError(
                f"Missing shard {sid} metrics in {sdir} "
                f"(looking for *-{task_name}-metrics.json)"
            )
        metrics_path = candidates[0]
        data = _load_json(metrics_path)
        shard_metrics.append(data)
        shard_paths.append(metrics_path)

    merged = copy.deepcopy(shard_metrics[0])
    merged["num_instances"] = sum(int(s.get("num_instances", 0)) for s in shard_metrics)
    merged["processing_time"] = sum(float(s.get("processing_time", 0.0)) for s in shard_metrics)
    merged["task_hash"] = f"{task_name}_sharded_merge_v1"

    task_cfg = merged.setdefault("task_config", {})
    task_cfg["eval_shard_id"] = None
    task_cfg["eval_num_shards"] = num_shards
    task_cfg.setdefault("metadata", {})
    task_cfg["metadata"]["merged_from_shards"] = num_shards

    metric_names: set[str] = set()
    for shard in shard_metrics:
        metric_names.update(shard.get("metrics", {}).keys())

    merged_metrics: dict[str, float] = {}
    for metric_name in sorted(metric_names):
        if metric_name == "primary_score":
            continue
        merged_value = _weighted_metric(shard_metrics, metric_name)
        if merged_value is not None:
            merged_metrics[metric_name] = merged_value

    primary_metric = task_cfg.get("primary_metric", "primary_score")
    if primary_metric in merged_metrics:
        merged_metrics["primary_score"] = merged_metrics[primary_metric]
    elif "primary_score" in shard_metrics[0].get("metrics", {}):
        primary_score = _weighted_metric(shard_metrics, "primary_score")
        if primary_score is not None:
            merged_metrics["primary_score"] = primary_score

    merged["metrics"] = merged_metrics
    merged["shard_summaries"] = [
        {
            "shard_id": i,
            "metrics_path": str(shard_paths[i]),
            "num_instances": int(shard_metrics[i].get("num_instances", 0)),
            "metrics": shard_metrics[i].get("metrics", {}),
        }
        for i in range(num_shards)
    ]

    out_path = step_dir / f"task-000-{task_name}-metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded eval metrics or seed shard checkpoints.")
    parser.add_argument("--output-root", required=True, help="Base OLMES output dir (contains step_N/)")
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--task-name", required=True, help="Task stem, e.g. popqa, mbppplus")
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument(
        "--seed-checkpoints", action="store_true",
        help="Instead of merging, seed shard dirs with existing checkpoint progress",
    )
    parser.add_argument(
        "--gather-checkpoints", action="store_true",
        help="Gather shard checkpoint progress back into the central checkpoint file",
    )
    parser.add_argument(
        "--old-num-shards", type=int, default=None,
        help="Previous number of shards (for --gather-checkpoints)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.gather_checkpoints:
        old_n = args.old_num_shards or args.num_shards
        gather_checkpoints(output_root, args.step, args.task_name, old_n)
    elif args.seed_checkpoints:
        seed_checkpoints(output_root, args.step, args.task_name, args.num_shards)
    else:
        merged_path = merge_shards(output_root, args.step, args.task_name, args.num_shards)
        print(f"Merged {args.num_shards} shards -> {merged_path}")


if __name__ == "__main__":
    main()
