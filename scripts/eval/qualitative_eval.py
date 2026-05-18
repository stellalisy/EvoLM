#!/usr/bin/env python
"""Generate rubrics or policy responses from checkpoints on fixed eval prompts.

Iterates through an experiment's checkpoint directory, auto-detects checkpoint
types (rubric vs policy), and for each step launches a vLLM server, generates
outputs for all eval prompts, and saves JSONL files organized by type.

Supports parallel processing: with --num_workers N, runs N vLLM servers
simultaneously on different GPUs to process N checkpoints at once.

Usage:
    python scripts/eval/qualitative_eval.py \
        --checkpoint_dir /path/to/experiment_checkpoints \
        --prompts data/eval_prompts.jsonl \
        --step_interval 50

    # 8 parallel workers on 8 GPUs
    python scripts/eval/qualitative_eval.py \
        --checkpoint_dir /path/to/experiment_checkpoints \
        --steps 25,100,500,1000 \
        --rubric_prompt rubric_generation_v3 \
        --num_workers 8
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from open_instruct.search_rewards.utils.rubric_chat_templates import (
    DEFAULT_POLICY_SYSTEM_PROMPT,
    RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT,
    get_rubric_system_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("qualitative_eval")

CHECKPOINT_TYPE_CONFIGS: dict[str, dict[str, Any]] = {
    "rubric": {
        "temperature": 0.6,
        "max_tokens": 16384,
        "top_p": 0.95,
    },
    "policy": {
        "system_prompt": DEFAULT_POLICY_SYSTEM_PROMPT,
        "temperature": 0.6,
        "max_tokens": 16384,
        "top_p": 0.95,
    },
}

# Thread-safe tracking of all active vLLM server subprocesses.
_active_server_procs: dict[int, subprocess.Popen] = {}  # port -> proc
_server_lock = threading.Lock()


def _cleanup_servers() -> None:
    """Kill any lingering vLLM servers on script exit."""
    with _server_lock:
        for port, proc in list(_active_server_procs.items()):
            LOGGER.info("atexit: killing vLLM server on port %d (pid %d)", port, proc.pid)
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _active_server_procs.clear()


atexit.register(_cleanup_servers)


def _sigterm_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM (e.g. from SLURM) by cleaning up and exiting."""
    LOGGER.info("Received SIGTERM, cleaning up...")
    _cleanup_servers()
    sys.exit(143)  # 128 + 15 (SIGTERM)


signal.signal(signal.SIGTERM, _sigterm_handler)


@dataclass
class WorkItem:
    """A single checkpoint to evaluate."""
    step_num: int
    step_path: Path
    checkpoint_type: str
    output_file: Path
    system_prompt: str
    gen_config: dict[str, Any]


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def detect_checkpoint_types(checkpoint_dir: Path) -> dict[str, Path]:
    """Detect checkpoint subdirectory types.

    Returns a dict mapping logical type ("rubric" or "policy") to the
    directory containing step_N subdirectories.
    """
    types: dict[str, Path] = {}

    if (checkpoint_dir / "rubric").is_dir():
        types["rubric"] = checkpoint_dir / "rubric"
    if (checkpoint_dir / "policy").is_dir():
        types["policy"] = checkpoint_dir / "policy"

    if not types:
        if (checkpoint_dir / "policytrainernorubricmodelactor").is_dir():
            types["policy"] = checkpoint_dir / "policytrainernorubricmodelactor"

    if not types:
        flat_steps = [
            d for d in checkpoint_dir.iterdir()
            if d.is_dir() and re.match(r"step_\d+$", d.name)
        ]
        if flat_steps:
            types["policy"] = checkpoint_dir

    return types


def find_steps(
    subdir: Path,
    *,
    steps: set[int] | None = None,
    step_interval: int | None = None,
    max_step: int = 1000,
) -> list[tuple[int, Path]]:
    """Find and filter step_N directories under ``subdir``."""
    checkpoints: list[tuple[int, Path]] = []
    for item in subdir.iterdir():
        if not item.is_dir():
            continue
        match = re.match(r"step_(\d+)$", item.name)
        if not match:
            continue
        step_num = int(match.group(1))
        if step_num > max_step:
            continue
        if steps is not None and step_num not in steps:
            continue
        if step_interval is not None and step_num % step_interval != 0:
            continue
        checkpoints.append((step_num, item))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def start_vllm_server(
    model_path: str,
    port: int,
    num_gpus: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    log_file: Path | None = None,
    cuda_devices: str | None = None,
) -> tuple[subprocess.Popen, Any]:
    """Launch a vLLM OpenAI-compatible server as a subprocess.

    Returns (process, log_file_handle). Caller must close the file handle
    after stopping the server.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(num_gpus),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--max-model-len", str(max_model_len),
    ]
    LOGGER.info("Starting vLLM server: %s (CUDA_VISIBLE_DEVICES=%s)", " ".join(cmd), cuda_devices or "inherited")

    fh = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_file, "w")
        LOGGER.info("vLLM server logs -> %s", log_file)

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    proc = subprocess.Popen(
        cmd,
        stdout=fh if fh else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc, fh


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Poll the vLLM health endpoint until it responds or times out."""
    url = f"http://localhost:{port}/health"
    for i in range(1, timeout + 1):
        try:
            urllib.request.urlopen(url, timeout=2)
            LOGGER.info("vLLM server ready after %ds", i)
            return True
        except Exception:
            pass
        time.sleep(1)
        if i % 60 == 0:
            LOGGER.info("Still waiting for vLLM server... (%ds/%ds)", i, timeout)
    LOGGER.error("vLLM server did not start within %ds", timeout)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop a vLLM server subprocess."""
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        LOGGER.warning("vLLM server did not exit gracefully, killing")
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def load_prompts(path: str) -> list[dict]:
    """Load eval prompts from a JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def generate_outputs(
    client: openai.OpenAI,
    tokenizer: Any,
    model_name: str,
    prompts: list[dict],
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[dict]:
    """Generate outputs for all prompts via the vLLM completions API."""
    prompt_texts: list[str] = []
    for p in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p["question"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_texts.append(text)

    results: list[dict] = []
    for i, text in enumerate(prompt_texts):
        try:
            resp = client.completions.create(
                model=model_name,
                prompt=text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            generation = resp.choices[0].text.strip()
        except Exception as e:
            LOGGER.error("Generation failed for prompt %d: %s", i, e)
            generation = f"ERROR: {e}"

        results.append({
            "prompt_index": i,
            "question": prompts[i]["question"],
            "generation": generation,
        })

        if (i + 1) % 20 == 0:
            LOGGER.info("  Generated %d/%d", i + 1, len(prompts))

    return results


# ---------------------------------------------------------------------------
# Checkpoint processing
# ---------------------------------------------------------------------------


def process_checkpoint(
    step_num: int,
    step_path: Path,
    checkpoint_type: str,
    output_file: Path,
    prompts: list[dict],
    system_prompt: str,
    gen_config: dict[str, Any],
    port: int,
    num_gpus: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    server_timeout: int,
    cuda_devices: str | None = None,
) -> bool:
    """Process a single checkpoint: start server, generate, save, stop server.

    Returns True on success, False on failure.
    """
    model_path = str(step_path)
    LOGGER.info(
        "=== Processing %s/step_%d  (%s)  [GPUs: %s] ===",
        checkpoint_type, step_num, model_path, cuda_devices or "all",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not getattr(tokenizer, "chat_template", None):
            LOGGER.warning("Tokenizer from %s has no chat_template; falling back to Qwen/Qwen3-8B", model_path)
            _fb = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
            tokenizer.chat_template = _fb.chat_template
    except Exception as e:
        LOGGER.error("Failed to load tokenizer from %s: %s", model_path, e)
        return False

    server_log = output_file.with_suffix(".vllm.log")

    proc, log_fh = start_vllm_server(
        model_path, port, num_gpus, gpu_memory_utilization, max_model_len,
        log_file=server_log,
        cuda_devices=cuda_devices,
    )
    with _server_lock:
        _active_server_procs[port] = proc

    try:
        if not wait_for_server(port, timeout=server_timeout):
            LOGGER.error("Skipping %s/step_%d (server failed to start)", checkpoint_type, step_num)
            return False

        client = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )

        results = generate_outputs(
            client=client,
            tokenizer=tokenizer,
            model_name=model_path,
            prompts=prompts,
            system_prompt=system_prompt,
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"],
            top_p=gen_config["top_p"],
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        LOGGER.info("Saved %d outputs to %s", len(results), output_file)
        return True

    finally:
        stop_server(proc)
        with _server_lock:
            _active_server_procs.pop(port, None)
        if log_fh is not None:
            log_fh.close()
        time.sleep(5)


def write_manifest(
    output_dir: Path,
    checkpoint_dir: Path,
    prompts_path: str,
    rubric_prompt_key: str,
    types_processed: dict[str, list[int]],
    gen_configs: dict[str, dict[str, Any]],
) -> None:
    """Write a manifest.json summarizing the qualitative eval run."""
    manifest = {
        "checkpoint_dir": str(checkpoint_dir),
        "prompts_path": prompts_path,
        "rubric_prompt_key": rubric_prompt_key,
        "timestamp": datetime.now().isoformat(),
        "types_processed": {k: sorted(v) for k, v in types_processed.items()},
        "generation_configs": gen_configs,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    LOGGER.info("Manifest written to %s", manifest_path)


# ---------------------------------------------------------------------------
# Worker and main
# ---------------------------------------------------------------------------


def _run_worker(
    worker_id: int,
    work_items: list[WorkItem],
    prompts: list[dict],
    num_gpus: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    server_timeout: int,
    base_port: int,
    cuda_devices: str,
) -> tuple[int, int, dict[str, list[int]]]:
    """Process a list of checkpoints sequentially on assigned GPUs.

    Returns (processed_count, failed_count, {type: [steps]}).
    """
    port = base_port + worker_id
    processed = 0
    failed = 0
    steps_done: dict[str, list[int]] = {}

    LOGGER.info(
        "[Worker %d] Starting: %d items, GPUs=%s, port=%d",
        worker_id, len(work_items), cuda_devices, port,
    )

    for item in work_items:
        success = process_checkpoint(
            step_num=item.step_num,
            step_path=item.step_path,
            checkpoint_type=item.checkpoint_type,
            output_file=item.output_file,
            prompts=prompts,
            system_prompt=item.system_prompt,
            gen_config=item.gen_config,
            port=port,
            num_gpus=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            server_timeout=server_timeout,
            cuda_devices=cuda_devices,
        )
        if success:
            processed += 1
            steps_done.setdefault(item.checkpoint_type, []).append(item.step_num)
        else:
            failed += 1

    LOGGER.info("[Worker %d] Done: %d processed, %d failed", worker_id, processed, failed)
    return processed, failed, steps_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rubrics/responses from checkpoints on fixed eval prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir", required=True,
        help="Path to the _checkpoints/ directory.",
    )
    parser.add_argument(
        "--prompts", default="data/eval_prompts.jsonl",
        help="Path to eval prompts JSONL (default: data/eval_prompts.jsonl).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory. Default: {checkpoint_dir}/../qualitative_eval/",
    )
    parser.add_argument(
        "--steps", default=None,
        help="Comma-separated specific steps to evaluate, e.g. '25,100,500,1000'.",
    )
    parser.add_argument(
        "--step_interval", type=int, default=None,
        help="Evaluate every N steps (e.g. 50).",
    )
    parser.add_argument(
        "--max_step", type=int, default=1000,
        help="Maximum step number to evaluate (default: 1000).",
    )
    parser.add_argument(
        "--rubric_prompt", default="rubric_generation",
        choices=list(RUBRIC_PROMPT_KEY_TO_SYSTEM_PROMPT.keys()),
        help="Registry key for rubric generation system prompt (default: 'rubric_generation').",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs for vLLM tensor parallelism per worker (default: 1).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers, each processing checkpoints on its own GPU(s). "
             "Total GPUs used = num_gpus * num_workers. (default: 1).",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="vLLM GPU memory fraction (default: 0.9).",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=32768,
        help="vLLM maximum model length (default: 32768).",
    )
    parser.add_argument(
        "--server_timeout", type=int, default=600,
        help="Seconds to wait for vLLM server health check (default: 600).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate outputs even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    if not checkpoint_dir.is_dir():
        LOGGER.error("Checkpoint directory does not exist: %s", checkpoint_dir)
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = checkpoint_dir.parent / "qualitative_eval"

    steps_set: set[int] | None = None
    if args.steps:
        steps_set = {int(s.strip()) for s in args.steps.split(",")}

    prompts = load_prompts(args.prompts)
    LOGGER.info("Loaded %d eval prompts from %s", len(prompts), args.prompts)

    types = detect_checkpoint_types(checkpoint_dir)
    if not types:
        LOGGER.error(
            "No checkpoint subdirectories found in %s. "
            "Expected policy/, rubric/, policytrainernorubricmodelactor/, or step_N/ dirs.",
            checkpoint_dir,
        )
        sys.exit(1)

    LOGGER.info("Detected checkpoint types: %s", {k: str(v) for k, v in types.items()})

    rubric_system_prompt = get_rubric_system_prompt(args.rubric_prompt)
    LOGGER.info("Rubric prompt key: %s", args.rubric_prompt)

    gen_configs: dict[str, dict[str, Any]] = {}
    for type_name in types:
        base = dict(CHECKPOINT_TYPE_CONFIGS[type_name])
        if type_name == "rubric":
            base["system_prompt"] = rubric_system_prompt
        gen_configs[type_name] = base

    # Collect all work items, filtering already-completed checkpoints
    all_work_items: list[WorkItem] = []
    total_skipped = 0

    for type_name, type_dir in types.items():
        config = gen_configs[type_name]
        system_prompt = config["system_prompt"]

        found_steps = find_steps(
            type_dir,
            steps=steps_set,
            step_interval=args.step_interval,
            max_step=args.max_step,
        )

        if not found_steps:
            LOGGER.warning("No matching steps found for type '%s' in %s", type_name, type_dir)
            continue

        LOGGER.info(
            "Found %d checkpoints for type '%s': steps %s",
            len(found_steps), type_name, [s for s, _ in found_steps],
        )

        for step_num, step_path in found_steps:
            output_file = output_dir / type_name / f"step_{step_num}.jsonl"
            if output_file.exists() and not args.overwrite:
                LOGGER.info("Skipping %s/step_%d (output exists)", type_name, step_num)
                total_skipped += 1
                continue
            all_work_items.append(WorkItem(
                step_num=step_num,
                step_path=step_path,
                checkpoint_type=type_name,
                output_file=output_file,
                system_prompt=system_prompt,
                gen_config=config,
            ))

    if not all_work_items:
        LOGGER.info("Nothing to process (all outputs already exist or no matching steps).")
        return

    num_workers = min(args.num_workers, len(all_work_items))
    base_port = 10000 + (os.getpid() % 5000)

    LOGGER.info(
        "Processing %d checkpoints with %d worker(s), %d GPU(s)/worker, base port %d",
        len(all_work_items), num_workers, args.num_gpus, base_port,
    )

    # Distribute work items round-robin across workers
    worker_items: list[list[WorkItem]] = [[] for _ in range(num_workers)]
    for i, item in enumerate(all_work_items):
        worker_items[i % num_workers].append(item)

    types_processed: dict[str, list[int]] = {t: [] for t in types}
    total_processed = 0
    total_failed = 0

    if num_workers == 1:
        processed, failed, steps_done = _run_worker(
            worker_id=0,
            work_items=worker_items[0],
            prompts=prompts,
            num_gpus=args.num_gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            server_timeout=args.server_timeout,
            base_port=base_port,
            cuda_devices="0",
        )
        total_processed += processed
        total_failed += failed
        for t, steps_list in steps_done.items():
            types_processed[t].extend(steps_list)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {}
            for wid, items in enumerate(worker_items):
                if not items:
                    continue
                gpu_start = wid * args.num_gpus
                gpu_ids = list(range(gpu_start, gpu_start + args.num_gpus))
                cuda_devices = ",".join(str(g) for g in gpu_ids)
                futures[pool.submit(
                    _run_worker,
                    worker_id=wid,
                    work_items=items,
                    prompts=prompts,
                    num_gpus=args.num_gpus,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    server_timeout=args.server_timeout,
                    base_port=base_port,
                    cuda_devices=cuda_devices,
                )] = wid

            for future in as_completed(futures):
                wid = futures[future]
                try:
                    processed, failed, steps_done = future.result()
                    total_processed += processed
                    total_failed += failed
                    for t, steps_list in steps_done.items():
                        types_processed[t].extend(steps_list)
                except Exception:
                    LOGGER.exception("[Worker %d] crashed", wid)

    write_manifest(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        prompts_path=args.prompts,
        rubric_prompt_key=args.rubric_prompt,
        types_processed=types_processed,
        gen_configs={
            k: {kk: vv for kk, vv in v.items() if kk != "system_prompt"}
            for k, v in gen_configs.items()
        },
    )

    LOGGER.info(
        "Done. Processed: %d, Skipped: %d, Failed: %d. Output: %s",
        total_processed, total_skipped, total_failed, output_dir,
    )


if __name__ == "__main__":
    main()
