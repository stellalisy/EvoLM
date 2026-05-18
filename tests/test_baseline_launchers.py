from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "scripts" / "configs"


def test_multi_judge_qwen_llama_olmo_extends_three_judges():
    source = (CONFIGS_DIR / "multi_judge" / "qwen_llama_olmo.sh").read_text()

    assert 'source "$multi_judge_config_dir/three_judges.sh"' in source
    assert (
        'MULTI_JUDGE_MODELS="Qwen/Qwen3-1.7B,meta-llama/Llama-3.2-1B-Instruct,allenai/OLMo-2-0425-1B-Instruct,google/gemma-3-1b-it,Qwen/Qwen3-4B"'
        in source
    )
    assert 'configure_multi_judge_shared_rubric_settings' in source
    assert 'configure_multi_judge_request_concurrency' in source


def test_multi_judge_shared_settings_map_olmo():
    source = (CONFIGS_DIR / "multi_judge" / "shared_rubric_judge_settings.sh").read_text()
    olmo_config = (CONFIGS_DIR / "rubric_judge" / "olmo2_0425_1b_instruct.sh").read_text()
    llama_config = (CONFIGS_DIR / "rubric_judge" / "llama3_2_1b_instruct.sh").read_text()
    gemma_config = (CONFIGS_DIR / "rubric_judge" / "gemma3_1b_it.sh").read_text()
    qwen35_config = (CONFIGS_DIR / "rubric_judge" / "qwen3_5_2b.sh").read_text()

    assert '"meta-llama/Llama-3.2-1B-Instruct")' in source
    assert 'rubric_judge/llama3_2_1b_instruct.sh' in source
    assert 'RUBRIC_JUDGE_MODEL="meta-llama/Llama-3.2-1B-Instruct"' in llama_config
    assert '"allenai/OLMo-2-0425-1B-Instruct")' in source
    assert 'rubric_judge/olmo2_0425_1b_instruct.sh' in source
    assert 'RUBRIC_JUDGE_MODEL="allenai/OLMo-2-0425-1B-Instruct"' in olmo_config
    assert '"google/gemma-3-1b-it")' in source
    assert 'rubric_judge/gemma3_1b_it.sh' in source
    assert 'RUBRIC_JUDGE_MODEL="google/gemma-3-1b-it"' in gemma_config
    assert '"Qwen/Qwen3.5-2B")' in source
    assert 'rubric_judge/qwen3_5_2b.sh' in source
    assert 'RUBRIC_JUDGE_MODEL="Qwen/Qwen3.5-2B"' in qwen35_config
    assert 'configure_multi_judge_request_concurrency' in source
    assert 'MAX_CONCURRENT_MULTI_JUDGE_REQUESTS=$((judge_pool_gpus * 16))' in source
    assert 'MAX_CONCURRENT_JUDGE_REQUESTS=$((total_judge_gpus * 16))' in source


def test_multi_judge_aggregation_overlays_exist():
    average_vote = (CONFIGS_DIR / "multi_judge" / "aggregation" / "average_vote.sh").read_text()
    average_minus_variance = (
        CONFIGS_DIR / "multi_judge" / "aggregation" / "average_minus_variance.sh"
    ).read_text()

    assert 'MULTI_JUDGE_AGGREGATION="average_vote"' in average_vote
    assert 'MULTI_JUDGE_AGGREGATION="average_minus_variance"' in average_minus_variance
