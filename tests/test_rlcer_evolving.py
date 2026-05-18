import asyncio
from pathlib import Path

import numpy as np

from open_instruct.queue_types import GenerationResult, RequestInfo
from open_instruct.search_rewards import rubric_judge_rewards as rjr
from open_instruct.search_rewards.rlcer_rubric_utils import (
    RLCERRubricSpec,
    generate_rlcer_rubric_spec,
    precompute_rlcer_evolving_rollout_rubrics,
)


TRAINING_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "train_rubric_policy_joint.py"
)
RLCER_HELPER = (
    Path(__file__).resolve().parents[1] / "open_instruct" / "search_rewards" / "rlcer_rubric_utils.py"
)
TEMPLATE_FILE = (
    Path(__file__).resolve().parents[1] / "open_instruct" / "search_rewards" / "utils" / "rubric_chat_templates.py"
)
RUBRIC_JUDGE_FILE = (
    Path(__file__).resolve().parents[1] / "open_instruct" / "search_rewards" / "rubric_judge_rewards.py"
)
GRPO_FAST = (
    Path(__file__).resolve().parents[1] / "open_instruct" / "grpo_fast.py"
)


def test_parse_rlcer_rubric_items_with_scores_parses_json():
    rubric_text = """
    {
      "rubrics": [
        {"criterion": "Shows the key algebra step", "points": 4},
        {"criterion": "States the final answer clearly", "points": 2}
      ]
    }
    """

    items, scores = rjr.parse_rlcer_rubric_items_with_scores(rubric_text)

    assert items == [
        "Shows the key algebra step",
        "States the final answer clearly",
    ]
    assert scores == [4.0, 2.0]


def test_generate_rlcer_rubric_spec_local_actor_parses_json():
    class FakeTokenizer:
        pad_token_id = None
        chat_template = "unused"

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return [11, 22, 33]

    class FakeRemoteMethod:
        def __init__(self):
            self.calls = []

        async def remote(self, prompt_token_ids, sampling_params):
            self.calls.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "sampling_params": sampling_params,
                }
            )
            return {
                "text": """
                {
                  "rubrics": [
                    {"criterion": "Shows the key algebra step", "points": 4},
                    {"criterion": "States the final answer clearly", "points": 2}
                  ]
                }
                """,
                "result": GenerationResult(
                    responses=[[1, 2, 3]],
                    finish_reasons=["stop"],
                    masks=[[1, 1, 1]],
                    request_info=RequestInfo(
                        num_calls=[0],
                        timeouts=[0],
                        tool_errors=[""],
                        tool_outputs=[""],
                        tool_runtimes=[0.0],
                        tool_calleds=[False],
                    ),
                    logprobs=[[0.0, 0.0, 0.0]],
                ),
            }

    class FakeActor:
        def __init__(self):
            self.generate_text_result_from_token_ids = FakeRemoteMethod()

    tokenizer = FakeTokenizer()
    build_calls = []

    def fake_build_sampling_params(**kwargs):
        build_calls.append(kwargs)
        return {"sampling": kwargs}

    spec = asyncio.run(
        generate_rlcer_rubric_spec(
            "question",
            "<think>hidden</think>answer one",
            api_rubric_generator=None,
            rubric_model="Qwen/Qwen3-8B",
            tokenizer=tokenizer,
            generation_kwargs={"temperature": 0.7},
            build_sampling_params=fake_build_sampling_params,
            policy_generate_text_actor=FakeActor(),
            metadata={"created_at": "now"},
        )
    )

    prompt_text = "\n".join(message["content"] for message in tokenizer.calls[0]["messages"])
    assert "<think>" not in prompt_text
    assert "answer one" in prompt_text
    assert "answer two" not in prompt_text
    assert spec.question == "question"
    assert spec.rubric_items == [
        "Shows the key algebra step",
        "States the final answer clearly",
    ]
    assert spec.rubric_scores == [4.0, 2.0]
    assert spec.model_name == "Qwen/Qwen3-8B"
    assert spec.metadata["reference_response_present"] is True
    assert spec.prompt_token_ids == [11, 22, 33]
    assert "answer one" in (spec.prompt_text or "")
    assert spec.generation_result is not None
    assert build_calls == [{"config_key": "train", "temperature": 0.7, "n": 1}]


def test_precompute_rlcer_evolving_rollout_rubrics_keeps_one_rubric_per_answer():
    class FakeRemoteMethod:
        def __init__(self):
            self.calls = []

        async def remote(self, question, reference_response):
            self.calls.append((question, reference_response))
            return RLCERRubricSpec(
                question=question,
                rubric_text=f"rubric for {question}::{reference_response}",
                rubric_items=[f"criterion for {question}::{reference_response}"],
                rubric_scores=[1.0],
                model_name="stub-model",
                metadata={"reference_response": reference_response},
            )

    class FakeActor:
        def __init__(self):
            self.create_rlcer_rubric = FakeRemoteMethod()

    actor = FakeActor()
    result = asyncio.run(
        precompute_rlcer_evolving_rollout_rubrics(
            questions=["q1", "q1", "q2"],
            answers=["a1", "a2", "b1"],
            rubric_actor=actor,
        )
    )

    assert actor.create_rlcer_rubric.calls == [
        ("q1", "a1"),
        ("q1", "a2"),
        ("q2", "b1"),
    ]
    assert result[0]["rubric_items"] == ["criterion for q1::a1"]
    assert result[1]["metadata"]["reference_response"] == "a2"
    assert result[2]["rubric_scores"] == [1.0]


def test_score_policy_rollouts_with_rlcer_uses_precomputed_rollout_rubrics(monkeypatch):
    async def fail_generate(**kwargs):
        raise AssertionError("precomputed rlcer rubrics should bypass local generation")

    async def fake_score_all_answers_with_rlcer_verifier(**kwargs):
        criteria = [entry["criterion"] for entry in kwargs["rubric_entries"]]
        if criteria == ["positive rubric", "negative rubric"]:
            return np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=float)
        if criteria == ["unused rubric"]:
            return np.asarray([[0.0], [1.0]], dtype=float)
        raise AssertionError(f"unexpected rubric entries: {kwargs['rubric_entries']}")

    correctness = iter([1.0, 0.0])

    monkeypatch.setattr(rjr, "_rlcer_generate_rubrics_with_scores", fail_generate)
    monkeypatch.setattr(rjr, "_score_all_answers_with_rlcer_verifier", fake_score_all_answers_with_rlcer_verifier)
    monkeypatch.setattr(rjr, "_rlcer_check_answer_correctness", lambda *args, **kwargs: next(correctness))

    results = asyncio.run(
        rjr.score_policy_rollouts_with_rlcer(
            questions=["prompt", "prompt"],
            answers=["answer one", "answer two"],
            ground_truths=["gt", "gt"],
            verifier_types=["math", "math"],
            precomputed_rollout_rubrics=[
                {
                    "rubric_items": ["positive rubric", "negative rubric"],
                    "rubric_scores": [2.0, -1.0],
                },
                {
                    "rubric_items": ["unused rubric"],
                    "rubric_scores": [1.0],
                },
            ],
        )
    )

    assert [result["rubric_items"] for result in results] == [["positive rubric", "negative rubric"], ["unused rubric"]]
    assert [result["valid_rubric_indices"] for result in results] == [[0], []]
    assert [result["score"] for result in results] == [2.0, -1.0]


def test_rlcer_generate_rubrics_with_scores_returns_empty_without_valid_json(monkeypatch):
    responses = iter([
        "not json",
        '{"rubrics": [{"criterion": "", "points": 1}]}',
        '{"rubrics": "wrong-shape"}',
    ])

    async def fake_run_generation_from_messages(**kwargs):
        del kwargs
        return next(responses)

    monkeypatch.setattr(rjr, "_run_generation_from_messages", fake_run_generation_from_messages)

    items, scores = asyncio.run(
        rjr._rlcer_generate_rubrics_with_scores(
            question="q",
            reference_response="candidate",
        )
    )

    assert items == []
    assert scores == []


def test_score_policy_rollouts_with_rlcer_requires_aligned_precomputed_rollout_rubrics():
    try:
        asyncio.run(
            rjr.score_policy_rollouts_with_rlcer(
                questions=["prompt one", "prompt two"],
                answers=["answer one", "answer two"],
                ground_truths=["gt one", "gt two"],
                verifier_types=["math", "math"],
                precomputed_rollout_rubrics=[{"rubric_items": ["criterion"], "rubric_scores": [1.0]}],
            )
        )
    except ValueError as exc:
        assert "precomputed_rollout_rubrics must align" in str(exc)
    else:
        raise AssertionError("expected misaligned precomputed rollout rubrics to fail fast")


def test_training_script_routes_rlcer_evolving_through_rubric_actor():
    source = TRAINING_SCRIPT.read_text()
    helper_source = RLCER_HELPER.read_text()

    assert "async def create_rlcer_rubric(" in source
    assert "async def enqueue_rlcer_evolving_cached_generations(" in source
    assert "precompute_rlcer_evolving_rollout_rubrics" in source
    assert "precomputed_rollout_rubrics = await self._precompute_rlcer_evolving_rollout_rubrics(" in source
    assert "precomputed_rollout_rubrics=precomputed_rollout_rubrics" in source
    assert "proposer_generate_text_actor=None if (api_proposer or precomputed_rollout_rubrics is not None)" in source
    assert "create_rlcer_rubric.remote(question, answer)" in helper_source
    assert "generate_text_result_from_token_ids.remote" in helper_source
    assert "await self._build_rlcer_evolving_enriched_results(" in source
    assert "asyncio.run(" not in source[source.index("async def enqueue_rlcer_evolving_cached_generations("):source.index("def add_prompt_to_generator(")]
    assert "Preserved rlcer_evolving grouped rubric config after attaching shared engines " not in source
    assert "allow_world_padding=True" not in source


def test_training_script_reuses_cached_rlcer_generations_for_evolving():
    source = TRAINING_SCRIPT.read_text()
    grpo_source = GRPO_FAST.read_text()

    assert "self._rlcer_evolving_cached_generations_by_step" not in source
    assert "self._rlcer_cache_deque" in source
    assert "self._rlcer_cache_condition = threading.Condition()" in source
    assert "def _buffer_rlcer_evolving_cached_generations(" in source
    assert "self._rlcer_cache_deque.extend(dict(item) for item in cached_generations)" in source
    assert "self._rlcer_cache_condition.notify_all()" in source
    assert "def drain_rlcer_evolving_cached_generations(" in source
    assert "self._rlcer_cache_condition.wait_for(" in source
    assert "self._rlcer_cache_deque.popleft()" in source
    # Old step-keyed buffer code must be gone
    assert "pop_rlcer_evolving_cached_generations_for_step" not in source
    assert "wait_for_rlcer_evolving_cached_generations" not in source
    assert "_rlcer_evolving_last_flushed_step" not in source
    assert "_rlcer_evolving_cached_generation_step" not in source
    assert "def set_rlcer_evolving_cached_generations(self, cached_generations: list[dict[str, Any]]) -> None:" in source
    assert "self._rlcer_evolving_cached_generations = [dict(item) for item in cached_generations]" in source
    assert "Loaded %d cached RL-CER evolving generation(s) for the next rubric step" in source
    assert "unique_questions = {str(question) for question in questions}" in source
    assert "if len(unique_questions) != 1:" in source
    assert '"rlcer_evolving policy reward callback expects one prompt-group per call; "' in source
    assert "grouped: dict[str, list[int]] = {}" not in source
    assert "def _group_rlcer_evolving_cached_generations(" in source
    assert "prompt_group_index" in source
    assert "sample_index_within_prompt" in source
    assert 'if all(str(item.get("question", "")) for item in cached_generations):' in source
    assert 'question_grouped_items = _group_by_key(lambda item: str(item.get("question", "")))' in source
    assert "def _combine_rlcer_cached_group_result(" in source
    assert "def enqueue_rlcer_evolving_cached_generations(self, training_step: int) -> int:" in source
    assert "self._buffer_rlcer_evolving_cached_generations(" in source
    assert "policy_samples_per_prompt = max(" in source
    assert "for rollout_idx, cached_rubric in enumerate(precomputed_rollout_rubrics):" in source
    assert "eval_result = eval_results[rollout_idx]" in source
    assert '"question": questions[rollout_idx]' in source
    assert '"prompt_group_index": rollout_idx // policy_samples_per_prompt' in source
    assert '"rubricator_reward_detail": {' in source
    assert '"validity_fraction": validity_fraction' in source
    assert '"valid_indices": valid_indices' in source
    assert "grouped_cached_generations = self._group_rlcer_evolving_cached_generations(cached_generations)" in source
    assert 'cached_reward_detail = cached_item.get("rubricator_reward_detail")' in source
    assert "if isinstance(cached_reward_detail, dict):" in source
    assert "return dict(cached_reward_detail)" in source
    assert "scores=group_scores" in source
    assert 'Incomplete cached RL-CER evolving batch for rubric step %d: got %d/%d prompt groups; ' in source
    assert "selected_indices.append(" not in source
    assert "forcing num_samples_per_prompt_rollout=1" not in source
    assert "queued = ray.get(actor.enqueue_rlcer_evolving_cached_generations.remote(global_training_step + 1))" in source
    assert "actor.drain_rlcer_evolving_cached_generations.remote(" in source
    assert "rlcer_expected_cached_rollouts: int | None = None" in source
    assert "rlcer_expected_cached_rollouts=(" in source
    assert "rubric_actor.set_rlcer_evolving_cached_generations.remote(cached_rubric_generations)" in source
    assert "self.enriched_results_Q.put(enriched_result)" in source
    assert "EnrichedGenerationResult(" in source
    assert "force_single_step_alternation=args.rubric_reward_mode == \"rlcer_evolving\"" in source
    assert "enqueue_rlcer_evolving_cached_results=args.rubric_reward_mode == \"rlcer_evolving\"" in source
    assert 'if enqueue_rlcer_evolving_cached_results and not force_single_step_alternation:' in source
    assert '"rlcer_evolving cached-result handoff requires exactly one update per phase."' in source
    assert "effective_steps_per_phase = 1 if force_single_step_alternation else steps_per_phase" in source
    assert "keep_policy_running = (" in source
    assert "enqueue_rlcer_evolving_cached_results" in source
    assert "and active_actor is rubric_actor" in source
    assert "if keep_policy_running:" in source
    assert "continue" in source[source.index("for actor in all_policy_actors:"):source.index("# Always pause rubric actor when it's not the active actor.")]
    assert "def replenish_prompt_fn(" in source
    assert 'training only consumes cached policy-step' in source
    assert 'dummy_vllm_logprobs = torch.full_like(dummy_qr, float("nan"), dtype=torch.float)' in grpo_source
    assert "packed_sequences.vllm_logprobs.append(dummy_vllm_logprobs)" in grpo_source
    assert "set_rlcer_evolving_training_batch" not in source
    assert "prepare_rlcer_evolving_training_step" not in source
    assert "_enqueue_rlcer_evolving_prompt" not in source
    assert "_add_prompt_rlcer_evolving" not in source
    assert "set_training_active" not in source
    assert "lightweight_phase_switching" not in source
    assert "self.pending_queries_map.insert(" not in source[source.index("def enqueue_rlcer_evolving_cached_generations"):source.index("def add_prompt_to_generator")]
    assert 'or args.rubric_reward_mode == "rlcer"' not in source


def test_rlcer_evolving_rubric_packing_disables_zero_std_filtering():
    source = TRAINING_SCRIPT.read_text()

    assert "def _build_data_preparation_args(self) -> GrpoArgs:" in source
    assert 'actor_id == "rubric"' in source
    assert 'reward_mode = getattr(self, "_reward_mode", None) or getattr(self.args, "rubric_reward_mode", None)' in source
    assert 'if actor_id == "rubric" and reward_mode == "rlcer_evolving":' in source
    assert 'return replace(self.args, active_sampling=False, filter_zero_std_samples=False)' in source
    assert 'disabling active_sampling and filter_zero_std_samples for rlcer_evolving rubric packing' in source
    assert "packing_args = self._build_data_preparation_args()" in source
    assert source.count("self.args,") > 0
    assert "packing_args," in source[source.index("self.packing_future = self.executor.submit("):source.index("self.num_total_tokens = 0")]


def test_rlcer_evolving_cached_payload_stores_policy_time_rubricator_reward_detail():
    source = TRAINING_SCRIPT.read_text()

    assert '"rubricator_reward_detail": {' in source
    assert '"reward": validity_fraction + r_format' in source
    assert '"k_valid": k_valid' in source
    assert '"k_total": k_total' in source
    assert '"rubric_items": rubric_items' in source
    assert 'cached_reward_detail = cached_item.get("rubricator_reward_detail")' in source
    assert "compute_rlcer_rubricator_reward(" in source


def test_training_script_uses_strict_rlcer_prompt_and_parser_path():
    source = TRAINING_SCRIPT.read_text()
    helper_source = RLCER_HELPER.read_text()
    template_source = TEMPLATE_FILE.read_text()
    judge_source = RUBRIC_JUDGE_FILE.read_text()

    assert '"rlcer_rubric_generation"' in helper_source
    assert "DEFAULT_RLCER_VERIFIER_SYSTEM_PROMPT" in template_source
    assert '"rlcer_verifier": _rlcer_verifier_messages' in template_source
    assert '"judgement": [boolean, boolean, ...]' in template_source
    assert '"final_score": number' in template_source
    assert "Definition: The systematic breakdown of a complex problem" in template_source
    assert "Do not let the reference answer constrain your rubric system." in template_source
    assert "_score_all_answers_with_rlcer_verifier(" in source
    assert '{"question": question, "response": response}' in helper_source
    assert "parse_rlcer_rubric_items_with_scores(rubric_text)" in helper_source
    assert "reward = 1.0 if parsed_items else 0.0" in source
    assert "Falls back to the RRD-style generation" not in judge_source
    assert 'raise ValueError(' in source


def test_rlcer_verifier_parses_json_judgement_list():
    class FakeRemoteMethod:
        def __init__(self, response):
            self.response = response
            self.calls = []

        async def remote(self, messages, sampling_params):
            self.calls.append(
                {
                    "messages": messages,
                    "sampling_params": sampling_params,
                }
            )
            return self.response

    class FakeActor:
        def __init__(self, response):
            self.generate_text_from_messages = FakeRemoteMethod(response)

    actor = FakeActor(
        '<think>hidden reasoning</think>{"judgement": [true, false], "final_score": 4}'
    )
    result = asyncio.run(
        rjr._score_answer_with_rlcer_verifier(
            question="q",
            rubric_entries=[
                {"category": "A", "criterion": "criterion one", "points": 4},
                {"category": "B", "criterion": "criterion two", "points": -1},
            ],
            answer="candidate answer",
            rubric_judge_generate_text_actor=actor,
            sampling_params=object(),
        )
    )

    assert result["binary_scores"] == [1.0, 0.0]
    assert result["final_score"] == 4.0
    assert actor.generate_text_from_messages.calls
    content = actor.generate_text_from_messages.calls[0]["messages"][1]["content"]
    assert "Question:" in content
    assert "Response:" in content
    assert "Rubrics:" in content
    assert '"criterion": "criterion one"' in content
    assert '"final_score": number' in content


def test_rlcer_verifier_returns_zero_when_json_is_missing():
    class FakeRemoteMethod:
        async def remote(self, messages, sampling_params):
            del messages, sampling_params
            return "<think>only hidden reasoning</think>"

    class FakeActor:
        def __init__(self):
            self.generate_text_from_messages = FakeRemoteMethod()

    result = asyncio.run(
        rjr._score_answer_with_rlcer_verifier(
            question="q",
            rubric_entries=[
                {"category": "A", "criterion": "criterion one", "points": 4},
            ],
            answer="candidate answer",
            rubric_judge_generate_text_actor=FakeActor(),
            sampling_params=object(),
        )
    )

    assert result["binary_scores"] == [0.0]
    assert result["final_score"] == 0.0
