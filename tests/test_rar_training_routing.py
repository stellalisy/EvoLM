from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_script_routes_rar_modes_through_standalone_policy_actor():
    source = (REPO_ROOT / "scripts" / "train_rubric_policy_joint.py").read_text()

    assert 'score_policy_rollouts_with_rar_implicit' in source
    assert 'score_policy_rollouts_with_rar_explicit' in source
    assert 'elif mode == "rar_implicit":' in source
    assert 'elif mode == "rar_explicit":' in source
    assert '"rar_implicit",' in source
    assert '"rar_explicit",' in source
    assert 'args.rubric_reward_mode not in ("rar_implicit", "rar_explicit")' not in source
