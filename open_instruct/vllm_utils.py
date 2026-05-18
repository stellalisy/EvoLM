# Taken and modified from https://github.com/huggingface/trl
# Copyright 2024 The AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file is copied from https://github.com/OpenRLHF/OpenRLHF"""

import asyncio
import contextlib
import os
import queue
import sys
import threading
import time
import types
import uuid
from collections import defaultdict
from collections.abc import Awaitable
from concurrent import futures
from datetime import timedelta
from typing import Any

import ray
import torch
import torch.distributed
import vllm
from ray.util import queue as ray_queue
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    ProcessGroup,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.v1.core import kv_cache_utils

from open_instruct import logger_utils
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.tool_utils.tools import MaxCallsExceededTool, Tool
from open_instruct.utils import ModelDims, ray_get_with_progress

logger = logger_utils.setup_logger(__name__)

NUM_PREFETCH_WORKERS = 3
NUM_TOOL_WORKERS = 20
DRAIN_ACTIVE_TASKS_SLEEP_S = 1
SHOULD_STOP_TIMEOUT_S = 0.1


def assert_threaded_actor(instance):
    """Assert that an instance's class is suitable for use in a threaded (non-async) Ray actor.

    This function performs two checks:
      1. The class must not define any `async def` methods
         (including async generators, staticmethods, or classmethods).
      2. There must not be a running asyncio event loop in the current thread.

    Args:
        instance: The instance whose class to inspect.

    Raises:
        AssertionError: If the class defines one or more async methods, or a running asyncio event loop is detected.
    """
    try:
        loop = asyncio.get_running_loop()
        raise AssertionError(
            f"{instance.__class__.__name__} must run in a threaded Ray actor (no running event loop). "
            f"Detected RUNNING loop={loop!r} on thread='{threading.current_thread().name}'. "
            f"Python={sys.version.split()[0]}."
        )
    except RuntimeError:
        return


def _truncate_tool_output_tokens(
    tool_output_token_ids: list[int],
    current_prompt_token_ids: list[int],
    accumulated_tokens: list[int],
    max_model_len: int,
    max_tokens: int,
    current_mask_len: int,
) -> tuple[list[int], int, list[int]]:
    prompt_and_tool_output = current_prompt_token_ids + accumulated_tokens + tool_output_token_ids
    excess = len(prompt_and_tool_output) - max_model_len
    if excess > 0:
        tool_output_token_ids = tool_output_token_ids[:-excess]

    remaining = max_tokens - current_mask_len
    if remaining <= 0:
        return [], excess, prompt_and_tool_output
    elif len(tool_output_token_ids) > remaining:
        return tool_output_token_ids[:remaining], excess, prompt_and_tool_output

    return tool_output_token_ids, excess, prompt_and_tool_output


async def process_request_async(
    actor: "LLMRayActor",
    sub_request_id: str,
    base_request_id: str,
    prompt: vllm.TokensPrompt,
    sampling_params: vllm.SamplingParams,
):
    """Process a single async request with tool support, awaiting tools inline."""
    logger.info(f"[process_request_async {actor._get_display_name()}] START processing {sub_request_id}")

    accumulated_tokens = []
    accumulated_logprobs = []
    masks = []
    num_calls = 0
    timeout = False
    tool_error = ""
    tool_output = ""
    tool_runtime = 0.0
    tool_called = False

    current_prompt = prompt
    current_prompt_token_ids = actor.request_metadata[base_request_id]["prompt_token_ids"]
    current_sampling_params = sampling_params.clone()
    final_prompt_token_ids = None
    iteration = 0
    request_output = None  # Will be set if generation succeeds

    while True:
        iteration_request_id = f"{sub_request_id}_iter{iteration}"
        actor.inflight_engine_request_ids.add(iteration_request_id)
        logger.info(
            f"[process_request_async {actor._get_display_name()}] Calling llm_engine.generate for {iteration_request_id}"
        )
        outputs = [
            o
            async for o in actor.llm_engine.generate(current_prompt, current_sampling_params, iteration_request_id)
            if o.finished
        ]
        actor.inflight_engine_request_ids.discard(iteration_request_id)
        logger.info(
            f"[process_request_async {actor._get_display_name()}] Got {len(outputs)} outputs for {iteration_request_id}"
        )

        if len(outputs) == 0:
            logger.warning(
                f"[process_request_async {actor._get_display_name()}] Request {iteration_request_id} "
                f"produced 0 outputs (likely aborted during phase transition)"
            )
            return
        assert len(outputs) == 1, f"Expected exactly 1 output, got {len(outputs)} for request {iteration_request_id}"
        request_output = outputs[0]
        iteration += 1
        output = request_output.outputs[0]

        if final_prompt_token_ids is None:
            final_prompt_token_ids = request_output.prompt_token_ids

        accumulated_tokens.extend(output.token_ids)
        accumulated_logprobs.extend(output.logprobs)
        masks.extend([1] * len(output.token_ids))

        if not actor.tools or not actor.max_tool_calls:
            break

        triggered_tool, stop_str = get_triggered_tool(
            output.text, actor.tools, actor.max_tool_calls, num_calls, sampling_params
        )
        if triggered_tool is None:
            break

        assert actor.executor is not None, f"executor is None for request {sub_request_id}"

        loop = asyncio.get_running_loop()
        tool_result = await loop.run_in_executor(actor.executor, triggered_tool, output.text)

        tool_called = True
        num_calls += 1
        timeout = timeout or tool_result.timeout
        tool_error += "" if tool_result.error is None else tool_result.error
        tool_output += tool_result.output
        tool_runtime += tool_result.runtime

        tool_output_token_ids = actor.llm_engine.tokenizer.encode(
            "<output>\n" + tool_result.output + "</output>\n", add_special_tokens=False
        )

        tool_output_token_ids, excess, prompt_and_tool_output = _truncate_tool_output_tokens(
            tool_output_token_ids,
            current_prompt_token_ids,
            accumulated_tokens,
            actor.llm_engine.model_config.max_model_len,
            sampling_params.max_tokens,
            len(masks),
        )

        accumulated_tokens.extend(tool_output_token_ids)
        accumulated_logprobs.extend(
            [{token_id: types.SimpleNamespace(logprob=0.0)} for token_id in tool_output_token_ids]
        )
        masks.extend([0] * len(tool_output_token_ids))

        new_sample_tokens = sampling_params.max_tokens - len(masks)
        if excess > 0 or new_sample_tokens <= 0:
            break

        current_prompt = vllm.TokensPrompt(prompt_token_ids=prompt_and_tool_output, cache_salt=base_request_id)
        current_prompt_token_ids = prompt_and_tool_output
        final_prompt_token_ids = prompt_and_tool_output
        current_sampling_params = sampling_params.clone()
        current_sampling_params.max_tokens = new_sample_tokens

    complete_output = vllm.CompletionOutput(
        index=split_request_id(sub_request_id)["request_index"],
        text="",
        token_ids=accumulated_tokens,
        cumulative_logprob=output.cumulative_logprob,
        logprobs=accumulated_logprobs,
        finish_reason=output.finish_reason,
        stop_reason=output.stop_reason,
    )

    if actor.tools:
        complete_output.mask = masks
        complete_output.num_calls = num_calls
        complete_output.timeout = timeout
        complete_output.tool_error = tool_error
        complete_output.tool_output = tool_output
        complete_output.tool_runtime = tool_runtime
        complete_output.tool_called = tool_called

    actor.active_tasks.pop(sub_request_id, None)

    meta = actor.request_metadata.get(base_request_id)
    if meta is None:
        logger.warning(
            f"[process_request_async {actor._get_display_name()}] {sub_request_id} "
            f"metadata gone (phase transition), dropping result"
        )
        return

    actor.completion_queue.put(
        {
            "base_request_id": base_request_id,
            "expected_n": meta["original_sampling_params"].n,
            "request_output": vllm.RequestOutput(
                request_id=sub_request_id,
                prompt=request_output.prompt,
                prompt_token_ids=final_prompt_token_ids,
                prompt_logprobs=request_output.prompt_logprobs,
                outputs=[complete_output],
                finished=True,
            ),
            "tools": actor.tools,
        }
    )
    logger.info(f"[process_request_async {actor._get_display_name()}] COMPLETED {sub_request_id}")


# Edited from: https://github.com/OpenRLHF/OpenRLHF/pull/971/files
# Turns out Ray doesnt necessarily place bundles together,
# so this function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
# avoids unnecessary communication for TP>1 with vllm.
def get_bundle_indices_list(placement_group: ray.util.placement_group) -> list[int]:
    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = defaultdict(list)
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles[node_id].append(bundle)

    flattened_bundle_indices = []
    for bundles in node_id_to_bundles.values():
        flattened_bundle_indices.extend(bundles)
    return flattened_bundle_indices


def make_request_id(request: PromptRequest) -> str:
    """Generate a unique tracking key for a request."""
    prefix = "eval" if request.is_eval else "train"
    return f"{prefix}_{request.epoch_number}_{request.training_step}_{request.dataset_index}"


def split_request_id(full_request_id: str) -> dict:
    """Split request ID into base ID and request index.

    >>> split_request_id("train_0_1_43039_0")
    {'base_id': 'train_0_1_43039', 'request_index': 0}
    >>> split_request_id("eval_0_5_12345_2")
    {'base_id': 'eval_0_5_12345', 'request_index': 2}
    """
    parts = full_request_id.split("_")
    return {"base_id": "_".join(parts[:-1]), "request_index": int(parts[-1])}


def get_triggered_tool(
    output_text: str,
    tools: dict[str, Tool],
    max_tool_calls: dict[str, int],
    num_calls: int,
    sampling_params: vllm.SamplingParams,
) -> tuple[Tool | None, str | None]:
    """Check if any tool was triggered and return the tool and stop_str if found.

    Args:
        output_text: The generated text to check for tool triggers
        tools: Dictionary mapping stop strings to Tool instances
        max_tool_calls: Dictionary mapping stop strings to their call limits
        num_calls: Current number of tool calls for this request
        sampling_params: Sampling parameters containing stop strings

    Returns:
        Tuple of (tool, stop_str) if a tool was triggered, (None, None) otherwise.
    """
    for stop_str in sampling_params.stop:
        if stop_str in tools and output_text.endswith(stop_str):
            if num_calls < max_tool_calls.get(stop_str, 0):
                return tools[stop_str], stop_str
            else:
                return MaxCallsExceededTool(start_str="<tool>", end_str="</tool>"), stop_str
    return None, None


def process_completed_request(request_id, outs, current_time, tools, request_metadata):
    """Process a completed request with all its samples and return the result.

    Args:
        request_id: The base request ID
        outs: List of vllm.RequestOutput objects for all sub-requests
        current_time: Current timestamp for performance metrics
        tools: Dictionary of available tools (may be None or empty)
        request_metadata: Dictionary containing metadata for all requests

    Returns:
        Tuple of (result, is_eval) where result is a GenerationResult and is_eval is a boolean
    """
    final_output = vllm.RequestOutput(
        request_id=request_id,
        prompt=outs[0].prompt,
        prompt_token_ids=outs[0].prompt_token_ids,
        prompt_logprobs=outs[0].prompt_logprobs,
        outputs=[completion for out in outs for completion in out.outputs],
        finished=outs[0].finished,
    )

    total_generation_tokens = sum(len(completion.token_ids) for out in outs for completion in out.outputs)
    metadata = request_metadata[request_id]  # Don't pop yet, _poll_tool_futures might need it

    # Process the vLLM RequestOutput into GenerationResult format
    response_ids = [list(out.token_ids) for out in final_output.outputs]
    finish_reasons = [out.finish_reason for out in final_output.outputs]
    use_tools = bool(tools)

    logprobs = []
    for idx, out in enumerate(final_output.outputs):
        assert len(out.token_ids) == len(out.logprobs), (
            f"vLLM CompletionOutput {idx}: token_ids length ({len(out.token_ids)}) "
            f"!= logprobs length ({len(out.logprobs)})"
        )
        logprobs.append(
            [logprob_dict[token_id].logprob for token_id, logprob_dict in zip(out.token_ids, out.logprobs)]
        )

    # Extract attributes based on whether tools are used
    if use_tools:
        # Extract tool-specific attributes from outputs
        masks = [getattr(out, "mask", [1] * len(out.token_ids)) for out in final_output.outputs]
        num_calls = [getattr(out, "num_calls", 0) for out in final_output.outputs]
        timeouts = [getattr(out, "timeout", False) for out in final_output.outputs]
        tool_errors = [getattr(out, "tool_error", "") for out in final_output.outputs]
        tool_outputs = [getattr(out, "tool_output", "") for out in final_output.outputs]
        tool_runtimes = [getattr(out, "tool_runtime", 0.0) for out in final_output.outputs]
        tool_calleds = [getattr(out, "tool_called", False) for out in final_output.outputs]
    else:
        # Use default values when tools are not used
        masks = [[1] * len(resp) for resp in response_ids]
        num_calls = [0] * len(response_ids)
        timeouts = [False] * len(response_ids)
        tool_errors = [""] * len(response_ids)
        tool_outputs = [""] * len(response_ids)
        tool_runtimes = [0.0] * len(response_ids)
        tool_calleds = [False] * len(response_ids)

    result = GenerationResult(
        responses=response_ids,
        finish_reasons=finish_reasons,
        masks=masks,
        request_info=RequestInfo(
            num_calls=num_calls,
            timeouts=timeouts,
            tool_errors=tool_errors,
            tool_outputs=tool_outputs,
            tool_runtimes=tool_runtimes,
            tool_calleds=tool_calleds,
        ),
        dataset_index=metadata["dataset_index"],
        epoch_number=metadata["epoch_number"],
        training_step=metadata.get("training_step"),  # Propagate training step for replay buffer age
        token_statistics=TokenStatistics(
            num_prompt_tokens=len(metadata["prompt_token_ids"]),
            num_response_tokens=total_generation_tokens,
            generation_time=current_time - metadata["start_time"],
        ),
        start_time=metadata["start_time"],
        logprobs=logprobs,
        actor_id=metadata.get("actor_id"),  # For routing in single_model_mode
    )
    return result, metadata["is_eval"], metadata["is_generate_text_request"]


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str | None = None,
    pg_options: Any | None = None,
) -> ProcessGroup:
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend) if backend else Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def _prefetch_worker(actor: "LLMRayActor") -> None:
    consecutive_errors = 0
    max_backoff_s = 30.0
    display_name = actor._get_display_name() if hasattr(actor, "_get_display_name") else "unknown"
    total_requests = 0
    last_request_time = time.perf_counter()
    while True:
        try:
            if actor._should_stop() or (
                actor.inference_batch_size is not None and len(actor.active_tasks) >= actor.inference_batch_size
            ):
                time.sleep(DRAIN_ACTIVE_TASKS_SLEEP_S)
                continue

            request = actor.prompt_queue.get()
            now = time.perf_counter()
            idle_seconds = now - last_request_time
            consecutive_errors = 0  # Reset on success
            total_requests += 1

            # Log first request and any request after a long idle (>60s)
            if total_requests == 1 or idle_seconds > 60:
                logger.info(
                    f"[_prefetch_worker {display_name}] Got request #{total_requests} "
                    f"after {idle_seconds:.1f}s idle, active_tasks={len(actor.active_tasks)}"
                )

            last_request_time = now
            add_request(actor, request)
        except Exception as e:
            consecutive_errors += 1
            backoff = min(2**consecutive_errors, max_backoff_s)
            logger.error(
                f"[_prefetch_worker {display_name}] Error getting from prompt_queue "
                f"(attempt {consecutive_errors}): {e}. Retrying in {backoff:.1f}s...",
                exc_info=(consecutive_errors <= 3),  # Full traceback for first 3 errors
            )
            time.sleep(backoff)


def add_request(actor: "LLMRayActor", request: PromptRequest) -> None:
    request_id = request.generate_text_request_id if request.is_generate_text_request else make_request_id(request)

    # Skip requests that were already aborted (timed-out on the coordinator side)
    if request.is_generate_text_request and request_id in actor._aborted_generate_text_ids:
        actor._aborted_generate_text_ids.discard(request_id)
        logger.info(f"[add_request {actor._get_display_name()}] Skipping aborted request {request_id}")
        return

    sampling_params = request.generation_config.clone()
    sampling_params.n = 1  # Use n=1 for tool processing

    actor.request_metadata[request_id] = {
        "is_eval": request.is_eval,
        "is_generate_text_request": request.is_generate_text_request,
        "dataset_index": request.dataset_index,
        "epoch_number": request.epoch_number,
        "training_step": request.training_step,
        "sampling_params": sampling_params,
        "original_sampling_params": request.generation_config,
        "prompt_token_ids": list(request.prompt),
        "start_time": time.perf_counter(),
        "actor_id": request.actor_id,  # For routing in single_model_mode
    }

    tokens_prompt = vllm.TokensPrompt(prompt_token_ids=request.prompt, cache_salt=request_id)

    for j in range(request.generation_config.n):
        sub_sampling_params = sampling_params.clone()
        if request.generation_config.seed is not None:
            sub_sampling_params.seed = request.generation_config.seed + j
        sub_request_id = f"{request_id}_{j}"
        logger.info(
            f"[add_request {actor._get_display_name()}] Adding request {sub_request_id}, active_tasks={len(actor.active_tasks)}"
        )
        try:
            future = asyncio.run_coroutine_threadsafe(
                process_request_async(actor, sub_request_id, request_id, tokens_prompt, sub_sampling_params),
                actor.loop,
            )
            actor.active_tasks[sub_request_id] = future
            logger.info(
                f"[add_request {actor._get_display_name()}] Request {sub_request_id} scheduled successfully, active_tasks={len(actor.active_tasks)}"
            )
        except Exception as e:
            logger.error(
                f"[add_request {actor._get_display_name()}] FAILED to schedule request {sub_request_id}: {e}",
                exc_info=True,
            )


@ray.remote
class LLMRayActor:
    """Ray actor for LLM generation with optional tool support."""

    def __init__(
        self,
        *args,
        tools: dict[str, Tool] | None = None,
        max_tool_calls: dict[str, int] | None = None,
        bundle_indices: list[int] | None = None,
        prompt_queue: ray_queue.Queue,
        results_queue: ray_queue.Queue,
        eval_results_queue: ray_queue.Queue,
        actor_manager: ray.actor.ActorHandle,
        inference_batch_size: int | None,
        inflight_updates: bool,
        is_rubric_judge_engine: bool = False,
        generate_text_results_queue=None,
        name: str | None = None,
        actor_results_queues: dict[str, ray_queue.Queue] | None = None,
        **kwargs,
    ):
        assert_threaded_actor(self)
        self.name = name
        self._init_config(tools, max_tool_calls, inference_batch_size, inflight_updates)
        self._init_queues(
            prompt_queue,
            results_queue,
            eval_results_queue,
            actor_manager,
            generate_text_results_queue,
            actor_results_queues,
        )
        self.is_rubric_judge_engine = is_rubric_judge_engine
        self._init_executor()

        noset_visible_devices = kwargs.pop("noset_visible_devices")
        distributed_executor_backend = kwargs.get("distributed_executor_backend")
        self._setup_gpu_visibility(noset_visible_devices, distributed_executor_backend)
        self._setup_and_start_async_engine(args, bundle_indices, kwargs)

        # Start active tasks reporting thread
        self._start_active_tasks_reporting_thread()

    def _init_config(
        self,
        tools: dict[str, Tool] | None,
        max_tool_calls: dict[str, int] | None,
        inference_batch_size: int | None,
        inflight_updates: bool,
    ) -> None:
        self.tools = tools or {}
        self.max_tool_calls = max_tool_calls or {}
        self.inference_batch_size = inference_batch_size
        self.inflight_updates = inflight_updates
        self.request_metadata = {}
        self.active_tasks = {}
        self.request_outputs = {}
        self.inflight_engine_request_ids: set[str] = set()
        self._aborted_generate_text_ids: set[str] = set()

    def _init_queues(
        self,
        prompt_queue,
        results_queue,
        eval_results_queue,
        actor_manager,
        generate_text_results_queue=None,
        actor_results_queues=None,
    ) -> None:
        self.completion_queue = queue.Queue()
        self.prompt_queue = prompt_queue
        self.results_queue = results_queue
        self.eval_results_queue = eval_results_queue
        self.actor_manager = actor_manager
        self.generate_text_results_queue = generate_text_results_queue
        # For single_model_mode: route results to actor-specific queues
        # Maps actor_id -> Queue (e.g., {"rubric": rubric_Q, "policy": policy_Q})
        self.actor_results_queues = actor_results_queues or {}

        # For caching should_stop status.
        self._last_should_stop_update = float("-inf")
        self._should_stop_value = False

    def _init_executor(self) -> None:
        max_workers = NUM_PREFETCH_WORKERS + (NUM_TOOL_WORKERS if self.tools else 0)
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._prefetch_future = self.executor.submit(_prefetch_worker, self)
        self._process_future = self.executor.submit(self.process_from_queue)

        # Initialize active tasks reporting thread state
        self._active_tasks_reporting_thread = None
        self._active_tasks_reporting_stop_event = threading.Event()

    def _setup_gpu_visibility(self, noset_visible_devices: bool, distributed_executor_backend: str) -> None:
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        if distributed_executor_backend == "ray":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    def _setup_and_start_async_engine(self, args, bundle_indices, kwargs) -> None:
        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            display_name = self.name if self.name is not None else "unnamed"
            logger.debug(f"engine {display_name}: creating LLM with bundle_indices={bundle_indices}")

        self._ensure_cuda_platform()

        engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        engine_args.disable_log_stats = True
        engine_args.disable_cascade_attn = True

        display_name = self.name if self.name is not None else "unnamed"
        max_retries = 3
        init_timeout = 300

        for attempt in range(max_retries):
            if attempt > 0:
                self._ensure_cuda_platform()

            init_complete = threading.Event()
            init_error: list[BaseException | None] = [None]
            self.loop = None
            self.llm_engine = None

            async def _init_engine():
                running_loop = asyncio.get_running_loop()
                assert running_loop == self.loop, f"Loop mismatch! running={running_loop}, actor.loop={self.loop}"
                return vllm.AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=False)

            def _run_loop():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                try:
                    self.llm_engine = self.loop.run_until_complete(_init_engine())
                except Exception as e:
                    init_error[0] = e
                finally:
                    init_complete.set()
                if self.llm_engine is not None:
                    self.loop.run_forever()

            self.loop_thread = threading.Thread(target=_run_loop, daemon=True)
            self.loop_thread.start()

            if not init_complete.wait(timeout=init_timeout):
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 5
                    logger.warning(
                        f"Engine {display_name} init timed out after {init_timeout}s "
                        f"(attempt {attempt + 1}/{max_retries}). Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Engine {display_name} init timed out after {max_retries} attempts"
                )

            if init_error[0] is not None:
                if attempt < max_retries - 1:
                    delay = (attempt + 1) * 5
                    logger.warning(
                        f"Engine {display_name} init failed "
                        f"(attempt {attempt + 1}/{max_retries}): {init_error[0]}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Engine {display_name} failed to initialize after "
                    f"{max_retries} attempts: {init_error[0]}"
                ) from init_error[0]

            break

    @staticmethod
    def _ensure_cuda_platform():
        """Force-set vLLM's platform to CUDA using PyTorch detection.

        vLLM's default platform detection relies on NVML (pynvml), which
        can fail on cluster nodes due to driver race conditions or missing
        libnvidia-ml. This bypasses NVML entirely by checking PyTorch CUDA
        and directly setting the platform.

        CRITICAL: Must be called BEFORE any vLLM code accesses
        `current_platform` (e.g. before vllm.AsyncEngineArgs()), because
        `from vllm.platforms import current_platform` in modules like
        arg_utils.py creates a frozen local binding at import time.
        On retries, already-imported module bindings are also patched.
        """
        try:
            import sys
            import vllm.platforms as _platforms

            if (_platforms._current_platform is not None
                    and _platforms._current_platform.device_type == "cuda"):
                return

            if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
                logger.warning(
                    "Cannot force CUDA platform: "
                    "torch.cuda.is_available()=%s, device_count=%s",
                    torch.cuda.is_available(),
                    torch.cuda.device_count() if torch.cuda.is_available() else "N/A",
                )
                return

            from vllm.platforms.cuda import CudaPlatform
            from vllm.platforms.interface import Platform
            cuda_platform = CudaPlatform()
            _platforms._current_platform = cuda_platform

            for mod in sys.modules.values():
                if mod is not None and getattr(mod, 'current_platform', None) is not None:
                    if isinstance(mod.current_platform, Platform):
                        mod.current_platform = cuda_platform

            logger.info(
                "Forced vLLM CUDA platform via PyTorch detection "
                "(bypassing NVML)"
            )
        except Exception as e:
            logger.warning("Failed to ensure CUDA platform: %s", e)

    def get_model_dims(self):
        """Get only the model dimensions without loading weights."""
        return ModelDims.from_vllm_config(self.llm_engine.vllm_config)

    def _get_actor_id(self) -> str:
        """Get or cache the actor ID."""
        if not hasattr(self, "_actor_id"):
            runtime_context = ray.get_runtime_context()
            self._actor_id = runtime_context.get_actor_id()
        return self._actor_id

    def _get_display_name(self) -> str:
        """Get display name for logging: use name if available, otherwise actor_id."""
        return self.name if self.name is not None else self._get_actor_id()

    def _report_active_tasks(self) -> None:
        """Report current active tasks count to ActorManager."""
        if self.actor_manager is None:
            return

        active_task_count = len(self.active_tasks)
        actor_id = self._get_actor_id()
        with contextlib.suppress(Exception):
            # Silently fail if reporting fails (actor_manager might be unavailable)
            self.actor_manager.report_active_tasks.remote(actor_id, active_task_count, self.is_rubric_judge_engine)

    def _active_tasks_reporting_worker(self) -> None:
        """Background thread worker that periodically reports active tasks to ActorManager."""
        # Initial delay to allow actor to fully initialize
        time.sleep(2)

        # Report every 0.5 seconds for more responsive updates
        report_interval = 2

        while not self._active_tasks_reporting_stop_event.is_set():
            self._report_active_tasks()
            # Wait for interval or until stop event is set
            self._active_tasks_reporting_stop_event.wait(timeout=report_interval)

    def _start_active_tasks_reporting_thread(self) -> None:
        """Start the background thread for reporting active tasks."""
        if self._active_tasks_reporting_thread is not None:
            return  # Already started

        self._active_tasks_reporting_stop_event.clear()
        self._active_tasks_reporting_thread = threading.Thread(
            target=self._active_tasks_reporting_worker, daemon=True, name=f"ActiveTasksReporter-{id(self)}"
        )
        self._active_tasks_reporting_thread.start()

    def _stop_active_tasks_reporting_thread(self) -> None:
        """Stop the background thread for reporting active tasks."""
        if self._active_tasks_reporting_thread is not None:
            self._active_tasks_reporting_stop_event.set()
            self._active_tasks_reporting_thread.join(timeout=1.0)
            self._active_tasks_reporting_thread = None

    def set_should_stop(self, value: bool) -> None:
        """Update the should_stop flag for this engine.

        Called remotely by actor_manager during weight sync to pause/resume inference.
        """
        logger.info(f"[{self._get_display_name()}] set_should_stop({value})")
        self._should_stop_value = value
        self._last_should_stop_update = time.perf_counter()

    def _should_stop(self) -> bool:
        # If actor_manager is None (e.g., for rubric judge engines), never stop
        if self.is_rubric_judge_engine:
            return False

        if (time.perf_counter() - self._last_should_stop_update) > SHOULD_STOP_TIMEOUT_S:
            should_stop_ref = self.actor_manager.should_stop.remote()
            ready_refs, _ = ray.wait([should_stop_ref], timeout=SHOULD_STOP_TIMEOUT_S)
            if ready_refs:
                self._should_stop_value = ray.get(ready_refs[0])
                self._last_should_stop_update = time.perf_counter()
            else:
                ray.cancel(should_stop_ref)
        return self._should_stop_value

    def _accumulate_sub_request(self, sub_request: dict) -> None:
        base_request_id = sub_request["base_request_id"]
        expected_n = sub_request["expected_n"]

        if base_request_id not in self.request_outputs:
            self.request_outputs[base_request_id] = {
                "outputs": [],
                "expected_n": expected_n,
                "tools": sub_request["tools"],
            }

        entry = self.request_outputs.get(base_request_id)
        if entry is None:
            # Race: clear_pending_requests removed the entry between the
            # check above and this access. Drop the stale sub-request.
            return

        entry["outputs"].append(sub_request["request_output"])

        is_complete = len(entry["outputs"]) == expected_n
        if is_complete:
            self._finalize_completed_request(base_request_id)

    def _finalize_completed_request(self, base_request_id: str) -> None:
        if base_request_id not in self.request_metadata:
            logger.warning(
                f"[{self._get_display_name()}] _finalize_completed_request: "
                f"metadata missing for {base_request_id} (stale request after phase transition), skipping"
            )
            self.request_outputs.pop(base_request_id, None)
            return

        # Pop atomically to avoid race with clear_pending_requests which may
        # call request_outputs.clear() from another thread between our read
        # and the pop.
        entry = self.request_outputs.pop(base_request_id, None)
        if entry is None:
            logger.warning(
                f"[{self._get_display_name()}] _finalize_completed_request: "
                f"request_outputs already cleared for {base_request_id} (race with phase transition), skipping"
            )
            return

        outputs = entry["outputs"]
        ordered_outs = sorted(outputs, key=lambda x: split_request_id(x.request_id)["request_index"])

        current_time = time.perf_counter()
        result, is_eval, is_generate_text_request = process_completed_request(
            base_request_id,
            ordered_outs,
            current_time,
            entry["tools"],
            self.request_metadata,
        )
        self.request_metadata.pop(base_request_id, None)

        if is_generate_text_request:
            if self.actor_manager and result.token_statistics:
                self.actor_manager.report_generate_text_usage.remote(result.token_statistics)
            text = (
                self.llm_engine.tokenizer.decode(result.responses[0], skip_special_tokens=True)
                if result.responses
                else ""
            )

            # Send result to shared queue for routing
            routed_result = {"request_id": base_request_id, "text": text, "result": result}
            logger.debug(
                f"Putting result in generate_text_results_queue for request_id={base_request_id}, text_len={len(text)}, queue_size={self.generate_text_results_queue.qsize() if hasattr(self.generate_text_results_queue, 'qsize') else 'N/A'}"
            )
            self.generate_text_results_queue.put(routed_result)
            logger.debug(f"Result put in queue for request_id={base_request_id}")
            return

        # Route to actor-specific queue if actor_id is set and we have actor queues
        actor_id = result.actor_id
        if actor_id and actor_id in self.actor_results_queues:
            self.actor_results_queues[actor_id].put(result)
        else:
            # Fallback to is_eval-based routing
            results_queue = self.eval_results_queue if is_eval else self.results_queue
            results_queue.put(result)

    def process_from_queue(self) -> None:
        while True:
            sub_request = self.completion_queue.get()
            self._accumulate_sub_request(sub_request)

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
        use_ray: bool = False,
        timeout_minutes: int = 120,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self.llm_engine.collective_rpc(
                "init_process_group",
                args=(
                    master_address,
                    master_port,
                    rank_offset,
                    world_size,
                    group_name,
                    backend,
                    use_ray,
                    timeout_minutes,
                ),
            ),
            self.loop,
        )
        return future.result(timeout=timeout_minutes * 60)

    def _run_async(self, coro: Awaitable[Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def _prepare_weight_update(self, name: str, dtype: str, drain_timeout_s: float = 300.0) -> None:
        if not self.inflight_updates and len(self.active_tasks) > 0:
            display_name = self._get_display_name()
            start = time.perf_counter()
            initial_count = len(self.active_tasks)
            logger.info(
                f"[_prepare_weight_update {display_name}] Draining {initial_count} active tasks "
                f"(timeout={drain_timeout_s}s)..."
            )
            while len(self.active_tasks) > 0:
                self.check_background_threads()
                elapsed = time.perf_counter() - start
                if elapsed > drain_timeout_s:
                    remaining = len(self.active_tasks)
                    logger.warning(
                        f"[_prepare_weight_update {display_name}] Drain timeout after {elapsed:.0f}s "
                        f"with {remaining}/{initial_count} tasks remaining. "
                        f"Aborting engine requests and proceeding with weight update."
                    )
                    # Abort active requests on the vLLM engine so its event loop
                    # can process the upcoming weight-update RPC.  Just cancelling
                    # the Python futures (below) doesn't stop the engine's async
                    # inference coroutines, which can block _run_async forever.
                    engine_request_ids = list(self.request_metadata.keys())
                    if engine_request_ids:
                        try:
                            abort_future = asyncio.run_coroutine_threadsafe(
                                self.llm_engine.abort(engine_request_ids), self.loop
                            )
                            abort_future.result(timeout=30)
                            logger.info(
                                f"[_prepare_weight_update {display_name}] Aborted "
                                f"{len(engine_request_ids)} engine request(s)"
                            )
                        except Exception as e:
                            logger.warning(
                                f"[_prepare_weight_update {display_name}] "
                                f"Failed to abort engine requests: {e}"
                            )
                    for task_id, future in list(self.active_tasks.items()):
                        if not future.done():
                            future.cancel()
                    self.active_tasks.clear()
                    self.request_metadata.clear()
                    break
                time.sleep(DRAIN_ACTIVE_TASKS_SLEEP_S)

        expected_dtype = str(self.llm_engine.model_config.dtype)
        assert dtype == expected_dtype, f"Mismatched dtype for {name}: received {dtype!r}, expected {expected_dtype!r}"

    def update_weight(self, name: str, dtype: str, shape: tuple[int, ...], empty_cache: bool = False) -> None:
        self._prepare_weight_update(name, dtype)
        return self._run_async(self.llm_engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache)))

    def update_weight_cuda_ipc(
        self, name: str, dtype: str, shape: tuple[int, ...], ipc_handles: list[Any], empty_cache: bool = False
    ) -> None:
        self._prepare_weight_update(name, dtype)
        return self._run_async(
            self.llm_engine.collective_rpc(
                "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
            )
        )

    def reset_prefix_cache(self) -> None:
        return self._run_async(self.llm_engine.reset_prefix_cache())

    def ready(self) -> bool:
        return True

    def warmup(self, timeout: float = 120.0) -> dict:
        """Send a short warmup inference to verify the engine can actually process CUDA operations.

        This is critical after long idle periods (e.g., when an extra policy actor waits
        while the main policy trains). CUDA contexts can become stale, and a warmup ensures
        the engine is functional before the training pipeline depends on it.

        Returns a dict with warmup status. Raises RuntimeError on failure.
        """
        import vllm as _vllm

        name = self._get_display_name() if hasattr(self, "_get_display_name") else "unknown"
        logger.info(f"[warmup {name}] Starting warmup inference...")
        warmup_params = _vllm.SamplingParams(temperature=0.0, max_tokens=1, n=1, logprobs=1)
        warmup_input = _vllm.TokensPrompt(prompt_token_ids=[1])
        request_id = f"warmup_{name}_{id(self)}"

        async def _warmup():
            result_gen = self.llm_engine.generate(warmup_input, warmup_params, request_id)
            async for _ in result_gen:
                pass
            return True

        try:
            future = asyncio.run_coroutine_threadsafe(_warmup(), self.loop)
            future.result(timeout=timeout)
            logger.info(f"[warmup {name}] Warmup succeeded")
            return {"success": True, "name": name}
        except TimeoutError:
            logger.error(f"[warmup {name}] Warmup timed out after {timeout}s - engine is likely hung")
            raise RuntimeError(
                f"vLLM engine '{name}' warmup timed out after {timeout}s. Engine is likely hung after long idle."
            )
        except Exception as e:
            logger.error(f"[warmup {name}] Warmup failed: {e}")
            raise RuntimeError(f"vLLM engine '{name}' warmup failed: {e}") from e

    def health_check(self) -> dict:
        """Comprehensive health check that verifies all internal threads and the engine are alive.

        Returns a dict with health status. Raises RuntimeError if any component is dead.
        """
        status = {
            "prefetch_alive": not self._prefetch_future.done(),
            "process_alive": not self._process_future.done(),
            "loop_alive": self.loop_thread.is_alive() if hasattr(self, "loop_thread") else False,
            "active_tasks": len(self.active_tasks),
            "name": self._get_display_name() if hasattr(self, "_get_display_name") else "unknown",
        }
        # Check for crashed threads and propagate exceptions
        if self._prefetch_future.done():
            try:
                self._prefetch_future.result()
            except Exception as e:
                status["prefetch_error"] = str(e)
                logger.error(f"[health_check {status['name']}] _prefetch_worker crashed: {e}")
                raise RuntimeError(f"_prefetch_worker thread is dead: {e}") from e
        if self._process_future.done():
            try:
                self._process_future.result()
            except Exception as e:
                status["process_error"] = str(e)
                logger.error(f"[health_check {status['name']}] process_from_queue crashed: {e}")
                raise RuntimeError(f"process_from_queue thread is dead: {e}") from e
        if not status["loop_alive"]:
            raise RuntimeError("vLLM engine loop thread has died")
        return status

    def clear_pending_requests(self) -> dict:
        """Clear all in-flight requests to reset engine state during phase transitions.

        Engines accumulate active tasks during training. At phase transitions
        these stale requests contend for GPU resources and starve the next
        actor's requests. Clearing them allows the engine to serve the incoming
        phase immediately.

        This method aborts requests in the underlying vLLM engine (not just the
        Python tracking dict) to prevent an ever-growing internal queue that
        causes progressive throughput collapse in multi-policy setups.
        """
        display_name = self._get_display_name()
        cleared_active = len(self.active_tasks)

        # Abort all pending requests in the vLLM engine's internal scheduler.
        # Without this, requests submitted via llm_engine.generate() keep
        # running even after active_tasks is cleared, creating an unbounded
        # backlog that starves fresh requests from the next phase.
        aborted = 0
        engine_ids = list(self.inflight_engine_request_ids)
        if self.llm_engine is not None and engine_ids:
            try:
                self._run_async(self.llm_engine.abort(engine_ids))
                aborted = len(engine_ids)
            except Exception as e:
                logger.warning(f"[{display_name}] Failed to abort engine requests: {e}")
        self.inflight_engine_request_ids.clear()

        # Cancel the asyncio futures so coroutines don't linger in the event loop
        for future in self.active_tasks.values():
            if not future.done():
                future.cancel()

        self.active_tasks.clear()

        cleared_outputs = len(self.request_outputs)
        self.request_outputs.clear()

        # NOTE: request_metadata is intentionally NOT cleared here.
        # Stale coroutines that survived cancellation (between two awaits) will
        # still access request_metadata when putting results into completion_queue.
        # Clearing it causes a KeyError crash in process_from_queue.  Entries are
        # tiny and get cleaned up naturally in _finalize_completed_request via pop().

        logger.info(
            f"[{display_name}] clear_pending_requests: "
            f"cleared {cleared_active} active tasks, {cleared_outputs} partial outputs, "
            f"aborted {aborted} engine requests"
        )
        return {"cleared_active": cleared_active, "cleared_outputs": cleared_outputs, "aborted": aborted}

    def abort_generate_text_request(self, request_id: str) -> dict:
        """Abort a specific generate_text request by its request_id.

        Called by GenerateTextActor when a timeout occurs.  This ensures that
        the timed-out request is cancelled in the vLLM engine itself (not just
        on the coordinator side), freeing GPU memory and compute so new
        requests aren't starved by orphaned in-flight work.
        """
        display_name = self._get_display_name()

        # Blacklist so _prefetch_worker / add_request skips it if still queued
        self._aborted_generate_text_ids.add(request_id)

        # Snapshot keys before iterating (dict may be modified by background threads)
        matching_sub_ids = [k for k in list(self.active_tasks.keys()) if k.startswith(request_id)]
        matching_engine_ids = [k for k in list(self.inflight_engine_request_ids) if k.startswith(request_id)]

        # Abort inside the vLLM engine scheduler
        if matching_engine_ids and self.llm_engine is not None:
            try:
                self._run_async(self.llm_engine.abort(matching_engine_ids))
            except Exception as e:
                logger.warning(f"[{display_name}] Failed to abort engine requests for {request_id}: {e}")

        for eid in matching_engine_ids:
            self.inflight_engine_request_ids.discard(eid)

        # Cancel the asyncio futures so coroutines don't linger
        for sub_id in matching_sub_ids:
            future = self.active_tasks.pop(sub_id, None)
            if future and not future.done():
                future.cancel()

        # Clean up metadata / partial outputs
        self.request_metadata.pop(request_id, None)
        self.request_outputs.pop(request_id, None)

        aborted_tasks = len(matching_sub_ids)
        aborted_engine = len(matching_engine_ids)
        if aborted_tasks or aborted_engine:
            logger.info(
                f"[{display_name}] abort_generate_text_request({request_id}): "
                f"cancelled {aborted_tasks} active task(s), "
                f"aborted {aborted_engine} engine request(s)"
            )
        return {"aborted_tasks": aborted_tasks, "aborted_engine_requests": aborted_engine}

    def drain_prompt_queue(self) -> dict:
        """Drain all pending requests from the shared prompt queue.

        Used during phase transitions to remove queued-but-unfetched requests
        left over from the previous actor's phase. Since all engines in a
        placement group share the same prompt_queue, calling this on one
        engine clears it for all.
        """
        drained = 0
        try:
            while not self.prompt_queue.empty():
                self.prompt_queue.get(block=False)
                drained += 1
        except Exception:
            pass
        if drained > 0:
            logger.info(f"[{self._get_display_name()}] drain_prompt_queue: drained {drained} pending request(s)")
        return {"drained": drained}

    def check_background_threads(self) -> None:
        if self._prefetch_future.done():
            self._prefetch_future.result()
        if self._process_future.done():
            self._process_future.result()
        for task in self.active_tasks.values():
            if task.done():
                task.result()
        if not self.loop_thread.is_alive():
            raise RuntimeError(
                "vLLM engine loop thread has died. Check logs for errors in EngineCore or async engine."
            )

    def get_kv_cache_info(self) -> int:
        """Get KV cache max concurrency from the vLLM engine."""
        kv_cache_specs = self._run_async(self.llm_engine.collective_rpc("get_kv_cache_spec"))

        vllm_config = self.llm_engine.vllm_config
        gpu_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = int(gpu_memory_utilization * total_gpu_memory)

        kv_cache_groups = kv_cache_utils.get_kv_cache_groups(vllm_config, kv_cache_specs[0])

        kv_cache_config = kv_cache_utils.get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, kv_cache_specs[0], available_memory
        )

        max_concurrency = kv_cache_utils.get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)

        return int(max_concurrency)


@ray.remote
class GenerateTextActor:
    """Async actor for generating text through vLLM engine queues.

    This actor provides a simple async interface for text generation,
    communicating with engines only through shared queues.
    """

    def __init__(
        self,
        prompt_queue: ray_queue.Queue,
        generate_text_results_queue: ray_queue.Queue,
        tokenizer: Any,
        name: str | None = None,
        generate_text_timeout: float = 600.0,
    ):
        """Initialize the GenerateTextActor.

        Args:
            prompt_queue: Queue to submit generation requests to engines
            generate_text_results_queue: Queue to receive results from engines
            tokenizer: Tokenizer for converting messages to token IDs
            name: Optional name for logging purposes
            generate_text_timeout: Timeout in seconds for generate_text requests (default: 600s)
        """
        self.prompt_queue = prompt_queue
        self.generate_text_results_queue = generate_text_results_queue
        self.tokenizer = tokenizer
        self.name = name or "GenerateTextActor"
        self.generate_text_timeout = generate_text_timeout

        self.generate_text_futures = {}  # request_id -> concurrent.futures.Future
        self.generate_text_results = {}  # request_id -> result string
        self._timed_out_ids: set[str] = set()  # request IDs that timed out (discard late results)
        self._engines: list = []  # LLMRayActor handles for abort-on-timeout

        # Start result routing worker
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        self._router_future = self.executor.submit(self._generate_text_results_worker)

        logger.info(f"{self.name} initialized with generate_text_timeout={generate_text_timeout}s")

    def set_engines(self, engines: list) -> None:
        """Register vLLM engine actor handles so timed-out requests can be aborted."""
        self._engines = list(engines)
        logger.info(f"{self.name} registered {len(self._engines)} engine(s) for abort-on-timeout")

    def _generate_text_results_worker(self) -> None:
        """Worker thread that routes results back to waiting futures."""
        while True:
            routed_result = self.generate_text_results_queue.get()

            request_id = routed_result.get("request_id")

            if request_id is not None:
                # Discard results for requests that already timed out
                if request_id in self._timed_out_ids:
                    self._timed_out_ids.discard(request_id)
                    continue

                # Store the result
                self.generate_text_results[request_id] = routed_result

                # Signal the concurrent.futures.Future (thread-safe)
                future = self.generate_text_futures.get(request_id)
                if future is not None and not future.done():
                    future.set_result(routed_result)

    async def generate_text(self, prompt: str, sampling_params: Any) -> str:
        """Generate text from a prompt (string). Async method."""
        logger.warning("""
=========================
generate_text is deprecated. Use generate_text_from_messages instead. Call this only for debugging purposes.
=========================
""")
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_text_from_messages(messages, sampling_params)

    async def generate_text_from_messages(self, messages: list[dict], sampling_params: Any) -> str:
        """Generate text using messages format. Async method."""
        prompt_token_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return await self.generate_text_from_token_ids(prompt_token_ids, sampling_params)

    async def generate_text_result_from_messages(self, messages: list[dict], sampling_params: Any) -> dict[str, Any]:
        """Generate text using messages format and return the full routed result."""
        prompt_token_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return await self.generate_text_result_from_token_ids(prompt_token_ids, sampling_params)

    async def generate_text_from_token_ids(self, prompt_token_ids: list[int], sampling_params: Any) -> str:
        """Generate text using fully async approach through shared queues."""
        result = await self.generate_text_result_from_token_ids(prompt_token_ids, sampling_params)
        return result.get("text", "")

    async def generate_text_result_from_token_ids(
        self, prompt_token_ids: list[int], sampling_params: Any
    ) -> dict[str, Any]:
        """Generate text using fully async approach through shared queues.

        This async method submits a request to the prompt queue and awaits the result
        from the results queue. Load balancing is handled by the engines themselves.

        NOTE: This method always uses n=1 since it only returns a single decoded
        string result plus the corresponding GenerationResult.
        For multiple samples per prompt, use the regular PromptRequest flow through param_prompt_Q.

        sampling_params can be a vLLM SamplingParams object or a plain dict.
        Dicts are preferred for Ray remote calls to avoid cross-process pydantic
        deserialization issues (the pickle payload for SamplingParams triggers
        pydantic module imports which can fail on nodes with stale NFS caches).
        """
        assert isinstance(prompt_token_ids, list) and all(isinstance(x, int) for x in prompt_token_ids), (
            "prompt_token_ids must be a list of integers"
        )
        logger.debug(f"{self.name} is generating text (async), time: {time.time()}")

        if isinstance(sampling_params, dict):
            from vllm import SamplingParams
            sampling_params = SamplingParams(**sampling_params)

        # Force n=1 since generate_text methods only return a single string (result.responses[0])
        # Multiple samples per prompt should use the regular PromptRequest flow, not generate_text methods
        sampling_params = sampling_params.clone()
        if sampling_params.n != 1:
            logger.warning(
                f"generate_text_from_token_ids: Forcing n=1 (was {sampling_params.n}). "
                "For multiple samples, use the regular PromptRequest flow through param_prompt_Q."
            )
            sampling_params.n = 1

        logger.debug(f"sampling_params: {sampling_params}")
        # Default logprobs to 1 if not set (required for generate_text)
        if sampling_params.logprobs is None:
            sampling_params.logprobs = 1

        from open_instruct.queue_types import PromptRequest

        # Use actor name as prefix to identify which vLLM (rubric vs policy) is handling the request
        request_id = f"{self.name}_{str(uuid.uuid4())}"

        # Create a concurrent.futures.Future (thread-safe, can be awaited from any loop)
        future = futures.Future()
        self.generate_text_futures[request_id] = future

        prompt_request = PromptRequest(
            prompt=prompt_token_ids,
            generation_config=sampling_params,
            epoch_number=None,
            training_step=None,
            dataset_index=None,
            is_eval=False,
            start_time=time.perf_counter(),
            is_generate_text_request=True,
            generate_text_request_id=request_id,
        )

        self.prompt_queue.put(prompt_request)
        logger.debug(f"{self.name} put prompt in queue, time: {time.time()}")

        try:
            # Use asyncio.wrap_future to await a concurrent.futures.Future
            # This allows efficient async waiting without polling
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=self.generate_text_timeout)
            logger.debug(f"{self.name} received result for {request_id}, time: {time.time()}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"generate_text request {request_id} timed out after {self.generate_text_timeout}s")
            self._timed_out_ids.add(request_id)
            # Abort the orphaned request in all engines (fire-and-forget)
            for engine in self._engines:
                try:
                    engine.abort_generate_text_request.remote(request_id)
                except Exception:
                    pass
            raise TimeoutError(f"generate_text request {request_id} timed out after {self.generate_text_timeout}s")
        finally:
            # Clean up
            self.generate_text_futures.pop(request_id, None)
            self.generate_text_results.pop(request_id, None)


def get_cuda_arch_list() -> str:
    """Get CUDA compute capabilities and format them for TORCH_CUDA_ARCH_LIST."""
    if not torch.cuda.is_available():
        return ""

    cuda_capabilities = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_capabilities.append(f"{major}.{minor}")

    # Remove duplicates and sort
    cuda_capabilities = sorted(set(cuda_capabilities))
    cuda_arch_list = ";".join(cuda_capabilities)
    logger.info(
        f"Detected CUDA compute capabilities: {cuda_capabilities}, setting TORCH_CUDA_ARCH_LIST={cuda_arch_list}"
    )
    return cuda_arch_list


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    tokenizer_name_or_path: str,
    pretrain: str,
    revision: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int | None,
    vllm_gpu_memory_utilization: float = 0.9,
    single_gpu_mode: bool = False,
    pg: PlacementGroup | None = None,
    tools: dict[str, Tool] | None = None,
    max_tool_calls: tuple[int, ...] = (5,),
    prompt_queue=None,
    results_queue=None,
    eval_results_queue=None,
    actor_manager=None,
    inference_batch_size: int | None = None,
    use_fp8_kv_cache=False,
    inflight_updates: bool = False,
    is_rubric_judge_engine: bool = False,
    generate_text_results_queue=None,
    name: str | list[str] | None = None,
    actor_results_queues: dict[str, ray_queue.Queue] | None = None,
    bundle_index_offset: int = 0,
) -> list[LLMRayActor]:
    # Convert max_tool_calls to a dict mapping tool end strings to their limits
    if tools:
        assert len(max_tool_calls) == 1 or len(max_tool_calls) == len(tools), (
            "max_tool_calls must have length 1 (applies to all tools) or same length as tools (per-tool limit)"
        )
        # tool key is the end_str
        if len(max_tool_calls) == 1:
            max_tool_calls_dict = {end_str: max_tool_calls[0] for end_str in tools}
        else:
            max_tool_calls_dict = {end_str: limit for end_str, limit in zip(tools.keys(), max_tool_calls)}
    else:
        max_tool_calls_dict = {}

    vllm_engines = []
    # Use Ray backend for tensor parallelism - it handles GPU allocation for workers
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = pg is not None
    # For TP=1: parent needs 1 GPU; for TP>1: parent coordinates, Ray workers get GPUs
    num_gpus = int(tensor_parallel_size == 1)

    logger.info(f"num_gpus: {num_gpus}")

    if not use_hybrid_engine:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        logger.info(f"Creating PACK placement group for {num_engines} vLLM engines ({len(bundles)} bundles)")
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
        logger.info("vLLM placement group ready")

    bundle_indices_list = get_bundle_indices_list(pg)

    # Process name parameter: convert to list of names, one per engine
    if name is None:
        actor_names = [None] * num_engines
    elif isinstance(name, str):
        actor_names = [f"{name}_{i}" for i in range(num_engines)]
    elif isinstance(name, list):
        assert len(name) == num_engines, f"name list length ({len(name)}) must match num_engines ({num_engines})"
        actor_names = name
    else:
        raise TypeError(f"name must be None, str, or list[str], got {type(name)}")

    for i in range(num_engines):
        bundle_indices = bundle_indices_list[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0],
        )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env=ray.runtime_env.RuntimeEnv(
                    env_vars={"TORCH_CUDA_ARCH_LIST": get_cuda_arch_list()}
                ),
            ).remote(
                model=pretrain,
                revision=revision,
                tokenizer=tokenizer_name_or_path,
                tokenizer_revision=revision,
                worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                dtype="bfloat16",
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                noset_visible_devices=ray_noset_visible_devices(),
                prompt_queue=prompt_queue,
                results_queue=results_queue,
                eval_results_queue=eval_results_queue,
                actor_manager=actor_manager,
                tools=tools,
                max_tool_calls=max_tool_calls_dict,
                inference_batch_size=inference_batch_size,
                inflight_updates=inflight_updates,
                kv_cache_dtype="auto" if not use_fp8_kv_cache else "fp8",
                calculate_kv_scales=use_fp8_kv_cache,
                is_rubric_judge_engine=is_rubric_judge_engine,
                generate_text_results_queue=generate_text_results_queue,
                name=actor_names[i],
                actor_results_queues=actor_results_queues,
            )
        )

    ray_get_with_progress(
        [engine.ready.remote() for engine in vllm_engines], "Initializing vLLM engines", timeout=1200
    )

    return vllm_engines
