"""The dependency lock must describe the imported, installed environment."""
from src.modeling.dependency_smoke import verify_dependencies


def test_dependency_gate_is_imported_and_lock_verified():
    result = verify_dependencies()
    assert result["python"] == "3.12.3"
    for package in ["scikit-learn", "xgboost", "joblib", "scipy", "cloudpickle", "narwhals", "threadpoolctl"]:
        assert package in result["imported_versions"]
    assert result["locked_distributions"] >= len(result["imported_versions"])
