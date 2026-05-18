import asyncio

import numpy as np
import pytest

from open_instruct.search_rewards import rubric_judge_rewards as rjr


def test_score_policy_rollouts_with_rrd_samples_reuses_group_wu_scores(monkeypatch):
    async def fake_build_rrd_rubric_async(**kwargs):
        assert kwargs["sample_responses"] == ["answer one", "answer two"]
        return {
            "rubric_items": ["item 1", "item 2"],
            "iterations": 2,
            "rejected_count": 3,
            "trace": {"final_rubric_items": ["item 1", "item 2"]},
        }

    calls = {"score_all": 0, "score_single": 0}

    async def fake_score_all_answers_against_rubric_items(**kwargs):
        calls["score_all"] += 1
        assert kwargs["answers"] == ["answer one", "answer two"]
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    async def fail_score_answer_against_rubric_items(**kwargs):
        calls["score_single"] += 1
        raise AssertionError("per-answer scoring should reuse the precomputed WU group matrix")

    monkeypatch.setattr(rjr, "build_rrd_rubric_async", fake_build_rrd_rubric_async)
    monkeypatch.setattr(rjr, "_score_all_answers_against_rubric_items", fake_score_all_answers_against_rubric_items)
    monkeypatch.setattr(rjr, "_score_answer_against_rubric_items", fail_score_answer_against_rubric_items)
    monkeypatch.setattr(rjr, "_estimate_covariance", lambda scores: scores)
    monkeypatch.setattr(rjr, "_compute_wu_weights", lambda covariance: np.asarray([0.25, 0.75], dtype=float))

    results = asyncio.run(
        rjr.score_policy_rollouts_with_rrd_samples(
            questions=["prompt", "prompt"],
            answers=["answer one", "answer two"],
            weighting_method="wu",
        )
    )

    assert calls["score_all"] == 1
    assert calls["score_single"] == 0
    assert [result["binary_scores"] for result in results] == [[1.0, 0.0], [0.0, 1.0]]
    assert [result["weights"] for result in results] == [[0.25, 0.75], [0.25, 0.75]]
    assert [result["score"] for result in results] == [0.25, 0.75]
    assert [result["rrd_iterations"] for result in results] == [2, 2]
    assert [result["rrd_rejected_count"] for result in results] == [3, 3]


def test_judge_answer_rrd_wu_requires_precomputed_answer_scores():
    with pytest.raises(ValueError, match="precomputed answer_item_scores"):
        asyncio.run(
            rjr.judge_answer_rrd(
                question="prompt",
                rubric="",
                answer="answer",
                weighting_method="wu",
                rubric_items=["item 1"],
            )
        )


def test_judge_answer_rrd_wu_requires_precomputed_weights():
    with pytest.raises(ValueError, match="precomputed_weights"):
        asyncio.run(
            rjr.judge_answer_rrd(
                question="prompt",
                rubric="",
                answer="answer",
                weighting_method="wu",
                rubric_items=["item 1"],
                answer_item_scores=[1.0],
            )
        )


def test_judge_answer_rrd_llm_requires_precomputed_weights():
    with pytest.raises(ValueError, match="precomputed_weights"):
        asyncio.run(
            rjr.judge_answer_rrd(
                question="prompt",
                rubric="",
                answer="answer",
                weighting_method="llm",
                rubric_items=["item 1"],
            )
        )
