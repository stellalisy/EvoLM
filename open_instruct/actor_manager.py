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

"""ActorManager for controlling evaluation and weight updates across all LLMRayActors."""

import collections
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from open_instruct import logger_utils


def find_free_port():
    """Find and return a free port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class ActorManager:
    """Centralized manager for controlling evaluation and weight updates across all LLMRayActors."""

    def __init__(self, queues: dict, args):
        self._should_stop = False
        self._last_updated = datetime.now()
        self._dashboard_port = None
        self._queues = queues or {}
        self._queue_sizes = {}
        self._queue_info = {}
        self._queue_size_history = {}  # queue_name -> deque of sizes
        self._sample_window = 100
        self._active_tasks_history = collections.deque(maxlen=self._sample_window)  # History of total active tasks
        self._policy_active_tasks_history = collections.deque(
            maxlen=self._sample_window
        )  # History of policy active tasks
        self._rubric_judge_active_tasks_history = collections.deque(
            maxlen=self._sample_window
        )  # History of rubric judge active tasks
        self._token_history = collections.deque(maxlen=self._sample_window)
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._training_step_history = collections.deque(maxlen=self._sample_window)
        self._generation_batch_history = collections.deque(maxlen=self._sample_window)
        self._generate_text_usage_history = collections.deque(maxlen=self._sample_window)
        self._total_generate_text_calls = 0
        self._kv_cache_max_concurrency = None
        self._args = args
        # Track active tasks per vLLM actor
        # actor_id -> {"active_tasks": int, "is_rubric_judge": bool, "last_updated": float}
        self._actor_active_tasks = {}
        self._actor_active_tasks_lock = threading.Lock()
        if self._args.enable_queue_dashboard:
            self._setup_queue_monitoring()
            self._start_dashboard()

    def _setup_queue_monitoring(self):
        """Setup queue monitoring and active tasks monitoring with background polling thread."""
        for queue_name, q in self._queues.items():
            self._queue_info[queue_name] = {"maxsize": q.maxsize if hasattr(q, "maxsize") else 0, "queue": q}
            self._queue_sizes[queue_name] = 0
            self._queue_size_history[queue_name] = collections.deque(maxlen=self._sample_window)

        self._polling_active = True
        self._poll_thread = threading.Thread(target=self._poll_queue_sizes_and_active_tasks, daemon=True)
        self._poll_thread.start()

    def _poll_queue_sizes_and_active_tasks(self):
        """Background thread to poll queue sizes and active tasks."""
        import ray

        logger = logger_utils.setup_logger(__name__)

        while self._polling_active:
            try:
                # Poll queue sizes
                for queue_name, info in self._queue_info.items():
                    try:
                        current_size = info["queue"].qsize()
                        self._queue_sizes[queue_name] = current_size
                        self._queue_size_history[queue_name].append(current_size)
                    except (ray.exceptions.RayActorError, ray.exceptions.ActorUnavailableError) as e:
                        # Queue actor is unavailable, skip this queue and continue
                        logger.warning(f"Queue '{queue_name}' actor unavailable, skipping: {e}")
                        # Keep the last known size
                        if queue_name in self._queue_sizes:
                            self._queue_size_history[queue_name].append(self._queue_sizes[queue_name])

                # Poll active tasks
                try:
                    actor_stats = self.get_vllm_actor_stats()
                    total_active_tasks = actor_stats.get("total_active_tasks", 0)
                    policy_active_tasks = actor_stats.get("policy_active_tasks", 0)
                    rubric_judge_active_tasks = actor_stats.get("rubric_judge_active_tasks", 0)

                    self._active_tasks_history.append(total_active_tasks)
                    self._policy_active_tasks_history.append(policy_active_tasks)
                    self._rubric_judge_active_tasks_history.append(rubric_judge_active_tasks)
                except Exception as e:
                    logger.warning(f"Failed to get vLLM actor stats: {e}")
                    # Append zeros to maintain history consistency
                    self._active_tasks_history.append(0)
                    self._policy_active_tasks_history.append(0)
                    self._rubric_judge_active_tasks_history.append(0)
            except Exception as e:
                # Catch any other unexpected errors to prevent thread crash
                logger.error(f"Unexpected error in queue polling thread: {e}", exc_info=True)

            time.sleep(0.5)

    def _start_dashboard(self):
        """Start the FastAPI dashboard server in a background thread."""
        if self._args.queue_dashboard_port is None:
            self._dashboard_port = find_free_port()
        else:
            self._dashboard_port = self._args.queue_dashboard_port
        app = FastAPI(title="ActorManager Dashboard")

        static_dir = Path(__file__).parent / "static"
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the HTML dashboard."""
            html_path = Path(__file__).parent / "static" / "dashboard.html"
            with open(html_path) as f:
                return f.read()

        @app.get("/api/status")
        async def api_status():
            """Return the current status as JSON."""
            queues_data = {
                queue_name: {"current": self._queue_sizes.get(queue_name, 0), "maxsize": info["maxsize"]}
                for queue_name, info in self._queue_info.items()
            }

            return {
                "should_stop": self._should_stop,
                "last_updated": self._last_updated.isoformat(),
                "queues": queues_data,
                "token_stats": self.get_token_stats(),
                "timing_stats": self.get_timing_stats(),
                "kv_cache_max_concurrency": self._kv_cache_max_concurrency,
                # This is less confusing to users.
                "inference_batch_size": self._args.inference_batch_size * self._args.num_samples_per_prompt_rollout,
                "vllm_actors": self.get_vllm_actor_stats(),
            }

        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=self._dashboard_port, log_level="error")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        hostname = socket.getfqdn()

        logger = logger_utils.setup_logger(__name__)
        logger.info(f"Dashboard server started at http://{hostname}:{self._dashboard_port}")

    def set_should_stop(self, should_stop: bool):
        """Set whether actors should stop processing."""
        self._should_stop = should_stop
        self._last_updated = datetime.now()

    def should_stop(self) -> bool:
        """Check if actors should stop processing."""
        return self._should_stop

    def report_token_stats(self, prompt_tokens: int, generation_tokens: int):
        """Report token statistics from main thread."""
        current_time = time.time()

        self._total_prefill_tokens += prompt_tokens
        self._total_decode_tokens += generation_tokens

        self._token_history.append(
            {"timestamp": current_time, "prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens}
        )

    def report_token_statistics(self, token_stats):
        """Report token statistics using TokenStatistics object."""
        current_time = time.time()

        self._total_prefill_tokens += token_stats.num_prompt_tokens
        self._total_decode_tokens += token_stats.num_response_tokens

        self._token_history.append(
            {
                "timestamp": current_time,
                "prompt_tokens": token_stats.num_prompt_tokens,
                "generation_tokens": token_stats.num_response_tokens,
            }
        )

        self._generation_batch_history.append(token_stats.generation_time)

    def report_generate_text_usage(self, token_stats):
        """Report generate_text API usage statistics."""
        self._total_generate_text_calls += 1

        self._generate_text_usage_history.append(
            {
                "prompt_tokens": token_stats.num_prompt_tokens,
                "generation_tokens": token_stats.num_response_tokens,
                "generation_time": token_stats.generation_time,
            }
        )

    def report_training_step_time(self, duration: float):
        """Report the time taken for a training step."""
        self._training_step_history.append(duration)

    def report_batch_generation_time(self, duration: float):
        """Report the time taken to generate a batch of data."""
        self._generation_batch_history.append(duration)

    def set_kv_cache_max_concurrency(self, max_concurrency: int):
        """Set the KV cache max concurrency value."""
        self._kv_cache_max_concurrency = max_concurrency

    def report_active_tasks(self, actor_id: str, active_tasks: int, is_rubric_judge: bool = False):
        """Report active tasks count for a vLLM actor.

        Args:
            actor_id: Unique identifier for the actor (e.g., Ray actor ID or custom identifier)
            active_tasks: Current number of active tasks in the actor
            is_rubric_judge: Whether this actor is a rubric judge engine
        """
        with self._actor_active_tasks_lock:
            self._actor_active_tasks[actor_id] = {
                "active_tasks": active_tasks,
                "is_rubric_judge": is_rubric_judge,
                "last_updated": time.time(),
            }

    def get_vllm_actor_stats(self):
        """Get statistics about all vLLM actors and their active tasks.

        Returns:
            Dictionary with actor statistics, including:
            - actors: List of actor info with active_tasks, is_rubric_judge, last_updated
            - total_actors: Total number of actors
            - total_active_tasks: Sum of all active tasks
            - rubric_judge_actors: Number of rubric judge actors
            - policy_actors: Number of policy (non-rubric) actors
            - rubric_judge_active_tasks: Sum of active tasks in rubric judge actors
            - policy_active_tasks: Sum of active tasks in policy actors
        """
        with self._actor_active_tasks_lock:
            actors_list = []
            total_active_tasks = 0
            rubric_judge_count = 0
            policy_count = 0
            rubric_judge_active_tasks = 0
            policy_active_tasks = 0

            current_time = time.time()
            # Remove stale entries (older than 60 seconds)
            stale_threshold = 60.0

            for actor_id, info in list(self._actor_active_tasks.items()):
                age = current_time - info["last_updated"]
                if age > stale_threshold:
                    # Remove stale entries
                    continue

                actors_list.append(
                    {
                        "actor_id": actor_id,
                        "active_tasks": info["active_tasks"],
                        "is_rubric_judge": info["is_rubric_judge"],
                        "last_updated": info["last_updated"],
                        "age_seconds": age,
                    }
                )

                total_active_tasks += info["active_tasks"]
                if info["is_rubric_judge"]:
                    rubric_judge_count += 1
                    rubric_judge_active_tasks += info["active_tasks"]
                else:
                    policy_count += 1
                    policy_active_tasks += info["active_tasks"]

            return {
                "actors": actors_list,
                "total_actors": len(actors_list),
                "total_active_tasks": total_active_tasks,
                "rubric_judge_actors": rubric_judge_count,
                "policy_actors": policy_count,
                "rubric_judge_active_tasks": rubric_judge_active_tasks,
                "policy_active_tasks": policy_active_tasks,
            }

    def get_token_stats(self):
        """Calculate and return current token statistics."""
        if not self._token_history:
            return {
                "total_prefill_tokens": self._total_prefill_tokens,
                "total_decode_tokens": self._total_decode_tokens,
                "prefill_tokens_per_sec": 0,
                "decode_tokens_per_sec": 0,
                "sample_count": 0,
            }

        current_time = time.time()

        window_prompt_tokens = 0
        window_generation_tokens = 0
        oldest_timestamp = self._token_history[0]["timestamp"]

        for entry in self._token_history:
            window_prompt_tokens += entry["prompt_tokens"]
            window_generation_tokens += entry["generation_tokens"]

        time_span = current_time - oldest_timestamp if len(self._token_history) > 1 else 1

        prompt_tokens_per_sec = window_prompt_tokens / time_span if time_span > 0 else 0
        generation_tokens_per_sec = window_generation_tokens / time_span if time_span > 0 else 0

        return {
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
            "prefill_tokens_per_sec": prompt_tokens_per_sec,
            "decode_tokens_per_sec": generation_tokens_per_sec,
            "sample_count": len(self._token_history),
        }

    def get_timing_stats(self):
        """Calculate and return current timing statistics."""
        avg_training_step_time = (
            sum(self._training_step_history) / len(self._training_step_history) if self._training_step_history else 0
        )

        avg_batch_generation_time = (
            sum(self._generation_batch_history) / len(self._generation_batch_history)
            if self._generation_batch_history
            else 0
        )

        return {
            "avg_training_step_time": avg_training_step_time,
            "avg_batch_generation_time": avg_batch_generation_time,
            "training_step_count": len(self._training_step_history),
            "batch_generation_count": len(self._generation_batch_history),
        }

    def get_queue_stats(self):
        """Calculate and return current queue statistics."""
        if not self._queue_info:
            return {}

        queue_stats = {}
        for queue_name, info in self._queue_info.items():
            current_size = self._queue_sizes.get(queue_name, 0)
            maxsize = info["maxsize"]
            size_history = self._queue_size_history.get(queue_name, collections.deque())

            # Average size over history window
            if size_history:
                avg_size = sum(size_history) / len(size_history)
                queue_stats[f"queue/{queue_name}/avg_size"] = avg_size
            else:
                queue_stats[f"queue/{queue_name}/avg_size"] = current_size

            # Utilization metrics
            if maxsize > 0:
                queue_stats[f"queue/{queue_name}/utilization"] = current_size / maxsize
                if size_history:
                    avg_utilization = (sum(size_history) / len(size_history)) / maxsize
                    queue_stats[f"queue/{queue_name}/avg_utilization"] = avg_utilization

        return queue_stats

    def get_active_tasks_stats(self):
        """Calculate and return current active tasks statistics with history.

        Returns:
            Dictionary with active tasks statistics including current values and averages.
        """
        actor_stats = self.get_vllm_actor_stats()
        total_active_tasks = actor_stats.get("total_active_tasks", 0)
        policy_active_tasks = actor_stats.get("policy_active_tasks", 0)
        rubric_judge_active_tasks = actor_stats.get("rubric_judge_active_tasks", 0)

        stats = {
            "active_tasks/total/current": total_active_tasks,
            "active_tasks/policy/current": policy_active_tasks,
            "active_tasks/rubric_judge/current": rubric_judge_active_tasks,
        }

        # Add average values from history
        if self._active_tasks_history:
            stats["active_tasks/total/avg"] = sum(self._active_tasks_history) / len(self._active_tasks_history)
        else:
            stats["active_tasks/total/avg"] = total_active_tasks

        if self._policy_active_tasks_history:
            stats["active_tasks/policy/avg"] = sum(self._policy_active_tasks_history) / len(
                self._policy_active_tasks_history
            )
        else:
            stats["active_tasks/policy/avg"] = policy_active_tasks

        if self._rubric_judge_active_tasks_history:
            stats["active_tasks/rubric_judge/avg"] = sum(self._rubric_judge_active_tasks_history) / len(
                self._rubric_judge_active_tasks_history
            )
        else:
            stats["active_tasks/rubric_judge/avg"] = rubric_judge_active_tasks

        return stats

    def _log_queue_usage_to_wandb(self, step: int | None = None):
        """Log queue usage statistics to wandb if initialized.

        Args:
            step: Optional step number to align with training steps. If None, logs without step.
        """
        try:
            import wandb

            # Check if wandb is initialized
            if wandb.run is None:
                return

            queue_metrics = self.get_queue_stats()
            if not queue_metrics:
                return

            # Add timestamp for periodic logging (when step is None)
            if step is None:
                queue_metrics["queue/timestamp"] = time.time()

            # Log to wandb with optional step alignment
            if step is not None:
                wandb.log(queue_metrics, step=step)
            else:
                wandb.log(queue_metrics)
        except ImportError:
            # wandb not installed, silently skip
            pass

    def get_dashboard_port(self):
        """Get the port number where the dashboard is running."""
        return self._dashboard_port

    def cleanup(self):
        """Clean up resources including stopping the polling thread."""
        logger = logger_utils.setup_logger(__name__)

        # Stop the polling thread if dashboard was enabled
        if self._args.enable_queue_dashboard:
            logger.info("Stopping queue polling thread...")
            self._polling_active = False
            # Wait for the thread to finish with a timeout
            self._poll_thread.join(timeout=2.0)
