from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rar_scorers_use_rar_specific_generation_and_formatting():
    scorer_source = (REPO_ROOT / "open_instruct" / "search_rewards" / "rubric_judge_rewards.py").read_text()
    template_source = (
        REPO_ROOT / "open_instruct" / "search_rewards" / "utils" / "rubric_chat_templates.py"
    ).read_text()
    rar_helper_source = scorer_source.split("async def _rar_generate_rubrics(", 1)[1].split(
        "async def _score_single_answer_likert(", 1
    )[0]

    assert 'async def _rar_generate_rubrics(' in scorer_source
    assert 'messages = format_messages(\n        "rar_rubric_generation",' in rar_helper_source
    assert 'rubric_items = await _rar_generate_rubrics(' in scorer_source
    assert 'generated_rubrics[question] = await _rar_generate_rubrics(' in scorer_source
    assert 'rubric_text = _format_rubric_items_for_implicit(generated_rubrics[question])' in scorer_source
    assert 'if not responses:\n        return []' in rar_helper_source
    assert 'except Exception:\n            continue' in rar_helper_source
    assert 'return []' in rar_helper_source
    assert '_rrd_generate_initial_rubrics(' not in rar_helper_source
    assert '"rar_rubric_generation": lambda content: [' in template_source
