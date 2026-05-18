#!/usr/bin/env python3
"""
Run OLMES evaluation on training checkpoints.

This script finds all checkpoints from a training run and evaluates them
using the OLMES evaluation framework.

Usage:
    # Evaluate all checkpoints in a training output directory
    python scripts/eval/run_olmes_on_checkpoints.py \
        --checkpoint-dir /path/to/output_checkpoints \
        --output-dir /path/to/eval_results \
        --task tulu_3_dev

    # Evaluate specific steps only
    python scripts/eval/run_olmes_on_checkpoints.py \
        --checkpoint-dir /path/to/output_checkpoints \
        --output-dir /path/to/eval_results \
        --task gsm8k::tulu minerva_math::tulu \
        --steps 25 50 100 200

    # Use vLLM for faster inference
    python scripts/eval/run_olmes_on_checkpoints.py \
        --checkpoint-dir /path/to/output_checkpoints \
        --output-dir /path/to/eval_results \
        --task tulu_3_dev \
        --model-type vllm
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_checkpoints(checkpoint_dir: str, steps: Optional[List[int]] = None) -> List[tuple]:
    """
    Find all checkpoint directories and their step numbers.
    
    Args:
        checkpoint_dir: Path to the checkpoints directory
        steps: Optional list of specific steps to evaluate
        
    Returns:
        List of (step_number, checkpoint_path) tuples, sorted by step
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    checkpoints = []
    
    # Look for directories named step_N
    for item in checkpoint_path.iterdir():
        if item.is_dir():
            match = re.match(r"step_(\d+)", item.name)
            if match:
                step = int(match.group(1))
                if steps is None or step in steps:
                    checkpoints.append((step, str(item)))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    
    return checkpoints


def run_olmes_eval(
    model_path: str,
    tasks: List[str],
    output_dir: str,
    model_type: str = "hf",
    model_args: Optional[dict] = None,
    task_args: Optional[dict] = None,
    limit: Optional[int] = None,
    gpus: int = 1,
    extra_args: Optional[List[str]] = None,
    reuse_alpaca_generations: bool = False,
) -> dict:
    """
    Run OLMES evaluation on a model checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        tasks: List of task names to evaluate
        output_dir: Directory to save results
        model_type: Model type (hf, vllm)
        model_args: Additional model arguments
        task_args: Task config overrides (e.g., generation_kwargs)
        limit: Limit number of instances per task (for debugging)
        extra_args: Extra arguments to pass to olmes
        
    Returns:
        Dictionary with evaluation results
    """
    cmd = [
        "python", "-m", "oe_eval.launch",
        "--model", model_path,
        "--model-type", model_type,
        "--output-dir", output_dir,
    ]
    
    cmd.append("--task")
    cmd.extend(tasks)
    
    if model_args:
        cmd.extend(["--model-args", json.dumps(model_args)])
    
    if task_args:
        cmd.extend(["--task-args", json.dumps(task_args)])
    
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if gpus is not None:
        cmd.extend(["--gpus", str(gpus)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command
    env = os.environ.copy()
    if reuse_alpaca_generations:
        env["OLMES_REUSE_ALPACA_GENERATIONS"] = "1"
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent.parent.parent / "olmes"),
        capture_output=False,
        env=env,
    )
    
    return {"returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser(description="Run OLMES evaluation on training checkpoints")
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the checkpoints directory (e.g., output_checkpoints)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=["tulu_3_dev"],
        help="Task(s) to evaluate (default: tulu_3_dev)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Specific steps to evaluate (default: all checkpoints)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vllm",
        choices=["hf", "vllm"],
        help="Model type for inference (default: vllm)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit instances per task (for debugging)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="Maximum model context length (default: 16384)",
    )
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=16384,
        help="Maximum generation tokens per request (default: 16384). "
             "Must be set high enough for thinking models that produce <think> blocks.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of evaluation workers (forwarded to oe_eval.launch)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to allocate to oe_eval.launch",
    )
    parser.add_argument(
        "--livecodebench-shard-id",
        type=int,
        default=None,
        help="Optional LiveCodeBench shard id (0-indexed).",
    )
    parser.add_argument(
        "--livecodebench-num-shards",
        type=int,
        default=None,
        help="Optional total shard count for LiveCodeBench.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Generic eval shard id (0-indexed). Works for any task.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Generic total shard count. Works for any task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--reuse-alpaca-generations",
        action="store_true",
        help="Reuse cached AlpacaEval generations if available (sets OLMES_REUSE_ALPACA_GENERATIONS=1)",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra arguments to pass to olmes",
    )
    
    args = parser.parse_args()
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args.steps)
    
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for step, path in checkpoints:
        print(f"  Step {step}: {path}")
    print()
    
    model_args = {"max_length": args.max_length}
    if args.trust_remote_code:
        model_args["trust_remote_code"] = True
    
    task_args = {
        "generation_kwargs": {
            "max_gen_toks": args.max_gen_toks,
            "truncate_context": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
        }
    }
    if args.livecodebench_shard_id is not None or args.livecodebench_num_shards is not None:
        if args.livecodebench_shard_id is None or args.livecodebench_num_shards is None:
            raise ValueError(
                "--livecodebench-shard-id and --livecodebench-num-shards must be set together"
            )
        if args.livecodebench_num_shards <= 0:
            raise ValueError("--livecodebench-num-shards must be > 0")
        if args.livecodebench_shard_id < 0 or args.livecodebench_shard_id >= args.livecodebench_num_shards:
            raise ValueError(
                f"Invalid shard config: shard_id={args.livecodebench_shard_id}, "
                f"num_shards={args.livecodebench_num_shards}"
            )
        task_args["livecodebench_shard_id"] = args.livecodebench_shard_id
        task_args["livecodebench_num_shards"] = args.livecodebench_num_shards

    if args.shard_id is not None or args.num_shards is not None:
        if args.shard_id is None or args.num_shards is None:
            raise ValueError("--shard-id and --num-shards must be set together")
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be > 0")
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError(
                f"Invalid shard config: shard_id={args.shard_id}, "
                f"num_shards={args.num_shards}"
            )
        task_args["eval_shard_id"] = args.shard_id
        task_args["eval_num_shards"] = args.num_shards
    
    forwarded_extra_args = list(args.extra_args)
    if args.num_workers is not None:
        forwarded_extra_args.extend(["--num-workers", str(args.num_workers)])

    results = {}
    for step, checkpoint_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating step {step}: {checkpoint_path}")
        print(f"{'='*60}\n")
        
        step_output_dir = os.path.join(args.output_dir, f"step_{step}")
        
        if args.dry_run:
            print(f"[DRY RUN] Would evaluate {checkpoint_path}")
            print(f"  Tasks: {args.task}")
            print(f"  Output: {step_output_dir}")
            continue
        
        result = run_olmes_eval(
            model_path=checkpoint_path,
            tasks=args.task,
            output_dir=step_output_dir,
            model_type=args.model_type,
            model_args=model_args,
            task_args=task_args,
            limit=args.limit,
            gpus=args.gpus,
            extra_args=forwarded_extra_args,
            reuse_alpaca_generations=args.reuse_alpaca_generations,
        )
        
        results[step] = result
        
        if result["returncode"] != 0:
            print(f"WARNING: Evaluation failed for step {step}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    failed_steps = []
    for step, result in results.items():
        status = "SUCCESS" if result["returncode"] == 0 else "FAILED"
        print(f"  Step {step}: {status}")
        if result["returncode"] != 0:
            failed_steps.append(step)

    if failed_steps:
        print(f"\nOne or more evaluations failed (steps: {failed_steps}). Exiting with non-zero status.")
        sys.exit(1)


if __name__ == "__main__":
    main()
