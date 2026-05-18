#!/usr/bin/env python3
"""
Merge sharded LiveCodeBench metrics into one canonical metrics file.

Expected shard paths (supported layouts):
  A) <output_root>/step_<step>/livecodebench_shards/shard_<i>/
       task-000-livecodebench_codegeneration-metrics.json
  B) <output_root>/livecodebench_shards/shard_<i>/step_<step>/
       task-000-livecodebench_codegeneration-metrics.json

Merged output:
  <output_root>/step_<step>/task-000-livecodebench_codegeneration-metrics.json
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


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weighted_metric(shards: list[dict[str, Any]], metric_name: str) -> float | None:
    total_weight = 0
    weighted_sum = 0.0
    for shard in shards:
        metrics = shard.get("metrics", {})
        if metric_name not in metrics:
            continue
        metric_value = _coerce_float(metrics[metric_name])
        if metric_value is None:
            continue
        weight = int(shard.get("num_instances", 0))
        total_weight += weight
        weighted_sum += metric_value * weight
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def _validate_shard(
    shard_data: dict[str, Any], shard_id: int, num_shards: int, shard_path: Path
) -> None:
    task_cfg = shard_data.get("task_config", {})
    got_shard_id = int(task_cfg.get("livecodebench_shard_id", -1))
    got_num_shards = int(task_cfg.get("livecodebench_num_shards", -1))
    if got_shard_id != shard_id or got_num_shards != num_shards:
        raise ValueError(
            f"Shard config mismatch for {shard_path}: "
            f"expected shard_id={shard_id}, num_shards={num_shards}, "
            f"got shard_id={got_shard_id}, num_shards={got_num_shards}"
        )
    if int(shard_data.get("num_instances", 0)) <= 0:
        raise ValueError(f"Invalid/empty shard metrics for {shard_path}: num_instances <= 0")


def merge_livecodebench_shards(output_root: Path, step: int, num_shards: int) -> Path:
    step_dir = output_root / f"step_{step}"
    shard_root_layout_a = step_dir / "livecodebench_shards"
    shard_root_layout_b = output_root / "livecodebench_shards"

    shard_metrics: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    for shard_id in range(num_shards):
        candidates = [
            shard_root_layout_a
            / f"shard_{shard_id}"
            / "task-000-livecodebench_codegeneration-metrics.json",
            shard_root_layout_b
            / f"shard_{shard_id}"
            / f"step_{step}"
            / "task-000-livecodebench_codegeneration-metrics.json",
        ]
        metrics_path = next((p for p in candidates if p.exists()), None)
        if metrics_path is None:
            raise FileNotFoundError(
                "Missing shard metrics; looked at: "
                + " | ".join(str(p) for p in candidates)
            )
        data = _load_json(metrics_path)
        _validate_shard(data, shard_id, num_shards, metrics_path)
        shard_metrics.append(data)
        shard_paths.append(metrics_path)

    merged = copy.deepcopy(shard_metrics[0])
    merged["num_instances"] = int(sum(int(s.get("num_instances", 0)) for s in shard_metrics))
    merged["processing_time"] = float(
        sum(float(s.get("processing_time", 0.0)) for s in shard_metrics)
    )
    merged["task_hash"] = "livecodebench_sharded_merge_v1"

    task_cfg = merged.setdefault("task_config", {})
    task_cfg["livecodebench_shard_id"] = None
    task_cfg["livecodebench_num_shards"] = num_shards
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

    primary_metric = task_cfg.get("primary_metric", "pass_at_1")
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

    out_path = step_dir / "task-000-livecodebench_codegeneration-metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded LiveCodeBench metrics.")
    parser.add_argument("--output-root", required=True, help="Base OLMES output dir")
    parser.add_argument("--step", required=True, type=int, help="Checkpoint step number")
    parser.add_argument("--num-shards", required=True, type=int, help="Number of shards to merge")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    merged_path = merge_livecodebench_shards(output_root, args.step, args.num_shards)
    print(f"Merged LiveCodeBench shard metrics -> {merged_path}")


if __name__ == "__main__":
    main()
