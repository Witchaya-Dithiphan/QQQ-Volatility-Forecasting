"""Expected execution coverage is configuration metadata, not model work."""
from src.modeling.expected_runs import expected_runs, delivery_complete
from src.modeling.configuration import load_config


def test_expected_runs_budgets_variants_roles_and_pca():
    config = load_config()
    runs = expected_runs(config)
    assert runs == expected_runs(config)
    assert len({row["run_key"] for row in runs}) == len(runs)
    final = [r for r in runs if r["role"] in ("finalized_scratch", "finalized_reference")]
    assert len(final) == 76 and all(r["required"] for r in final)
    assert {r["variant"] for r in final} == {"with_spike", "non_spike"}
    candidate = [r for r in runs if r["variant"] == "with_spike" and r["implementation"] == "scratch" and r["role"] == "candidate"]
    assert len([r for r in candidate if r["model"] == "polynomial"]) == 16
    assert len([r for r in candidate if r["model"] == "svm"]) == 12
    forest = [r for r in candidate if r["model"] == "random_forest"]
    assert len(forest) == 7
    assert len([r for r in forest if r["parameters"]["pca_components"] is not None]) == 3
    assert all(r["parameters"]["use_selected_no_pca_hyperparameters"] for r in forest if r["parameters"]["pca_components"] is not None)
    assert len([r for r in runs if r["role"] == "internal_fold"]) == 48
    assert len([r for r in runs if r["role"] == "pilot_only"]) == 4
    assert not any(r["role"] == "optional_stability" for r in runs)
    assert any(r["role"] == "optional_stability" for r in expected_runs(config, include_optional=True))


def test_delivery_completion_requires_fresh_completed_nonpilot():
    expected = [r for r in expected_runs(load_config()) if r["role"] in ("finalized_scratch", "finalized_reference")]
    actual = [{**r, "status": "completed"} for r in expected]
    assert delivery_complete(expected, actual)["complete"]
    for status in ["failed", "skipped", "incompatible"]:
        altered = [dict(r) for r in actual]
        altered[0]["status"] = status
        assert not delivery_complete(expected, altered)["complete"]
    actual[0]["pilot_only"] = True
    assert not delivery_complete(expected, actual)["complete"]
