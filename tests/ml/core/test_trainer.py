import inspect
import json

import numpy as np
import pytest

from src.ml.core.trainer import evaluate, expand_grid, train_model
from src.ml.regression.multiple_linear import MultipleLinear

REQUIRED_ARTIFACTS = (
    "config.json",
    "model.npz",
    "preprocessor.npz",
    "search_results.json",
    "validation_metrics.json",
    "validation_predictions.csv",
    "load_verification.json",
)


def test_classification_metrics_use_the_labels_written_to_predictions():
    class TiedProbability:
        def predict_proba(self, X):
            return [0.5, 0.8]

        def predict(self, X):
            return [0, 1]

    metrics, labels, scores = evaluate(TiedProbability(), [[0], [1]], [0, 0], [0, 1], "classification")
    assert labels.tolist() == [0, 1]
    assert scores.tolist() == [0.5, 0.8]
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["accuracy"]["value"] == 1.0


def test_expand_grid_produces_every_combination_in_a_stable_order():
    assert expand_grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_empty_grid_yields_one_default_candidate():
    assert expand_grid({}) == [{}]


def test_tuple_grid_lists_candidates_instead_of_multiplying_them():
    # Some models name whole candidates rather than ranges: five fields with a full
    # product would be dozens of fits, so the config picks the eight worth running.
    grid = {"tuples": [[50, 1, 0.1], [100, 2, 0.03]], "tuple_fields": ["n", "depth", "rate"]}
    assert expand_grid(grid) == [
        {"n": 50, "depth": 1, "rate": 0.1},
        {"n": 100, "depth": 2, "rate": 0.03},
    ]


def test_tuple_grid_keeps_the_listed_order():
    grid = {"tuples": [[3], [1], [2]], "tuple_fields": ["k"]}
    assert [candidate["k"] for candidate in expand_grid(grid)] == [3, 1, 2]


@pytest.mark.parametrize("grid", [
    {"tuples": [[1, 2]]},                                          # no field names
    {"tuples": [[1, 2]], "tuple_fields": ["a"]},                   # too many values
    {"tuples": [[1]], "tuple_fields": ["a", "b"]},                 # too few values
    {"tuples": [[1]], "tuple_fields": ["a"], "b": [1, 2]},         # mixed with a range
    {"tuple_fields": ["a"]},                                       # names without tuples
])
def test_malformed_tuple_grid_is_rejected(grid):
    with pytest.raises(ValueError):
        expand_grid(grid)


def test_a_range_grid_is_untouched_by_tuple_support():
    # A hyperparameter whose values happen to be lists must still multiply out.
    assert expand_grid({"hidden_layers": [[8], [16, 8]], "l2": [0.0]}) == [
        {"hidden_layers": [8], "l2": 0.0},
        {"hidden_layers": [16, 8], "l2": 0.0},
    ]


def test_train_model_writes_every_required_artifact(tmp_path):
    result = train_model(
        MultipleLinear, grid={"fit_intercept": [True, False]}, variant="with_spike", output_root=tmp_path
    )
    directory = tmp_path / "with_spike" / "regression" / "multiple_linear"
    for filename in REQUIRED_ARTIFACTS:
        assert (directory / filename).is_file(), filename
    assert result["load_verification"]["passed"] is True
    assert len(json.loads((directory / "search_results.json").read_text(encoding="utf-8"))) == 2


def test_artifacts_are_valid_json_without_stringified_arrays(tmp_path):
    """Guards the bug where json.dumps(default=str) wrote '[0.1 0.2 ...]' into metrics."""
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)
    directory = tmp_path / "with_spike" / "regression" / "multiple_linear"
    metrics = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
    assert isinstance(metrics["clipped_metrics"]["rmse"]["value"], float)
    assert "raw_prediction" not in metrics


def test_selected_candidate_is_the_best_on_validation(tmp_path):
    result = train_model(
        MultipleLinear, grid={"fit_intercept": [True, False]}, variant="with_spike", output_root=tmp_path
    )
    rmse = [c["metrics"]["clipped_metrics"]["rmse"]["value"] for c in result["search_results"]]
    assert result["validation_metrics"]["clipped_metrics"]["rmse"]["value"] == pytest.approx(min(rmse))


def test_both_variants_train_and_record_their_own_row_counts(tmp_path):
    for variant, rows in (("with_spike", 1733), ("non_spike", 1520)):
        train_model(MultipleLinear, variant=variant, output_root=tmp_path)
        config = json.loads(
            (tmp_path / variant / "regression" / "multiple_linear" / "config.json").read_text(encoding="utf-8")
        )
        assert config["train_rows"] == rows and config["variant"] == variant


def test_validation_predictions_cover_every_validation_row(tmp_path):
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)
    path = tmp_path / "with_spike" / "regression" / "multiple_linear" / "validation_predictions.csv"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 372  # header + 371 validation rows


def test_rerunning_the_same_model_overwrites_instead_of_crashing(tmp_path):
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path)


def test_preprocess_mode_comes_from_the_accepted_config():
    from src.ml.core.trainer import config_preprocess

    assert config_preprocess("simple_linear") == "none"
    assert config_preprocess("multiple_linear") == "standardize"
    assert config_preprocess("decision_tree") == "none"
    assert config_preprocess("not_a_model") == "standardize"


def test_model_declaring_no_preprocessing_is_not_standardized(tmp_path):
    """configs/modeling.json gives 7 models preprocess="none"; honour it."""
    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path, preprocess="none")
    directory = tmp_path / "with_spike" / "regression" / "multiple_linear"
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    _, metadata = __import__("src.ml.core.persist", fromlist=["load_npz"]).load_npz(
        directory / "preprocessor.npz"
    )
    assert config["preprocess"] == "none"
    assert metadata["standardized"] is False


def test_standardized_run_records_the_real_scaler(tmp_path):
    from src.ml.core.persist import load_npz

    train_model(MultipleLinear, variant="with_spike", output_root=tmp_path, preprocess="standardize")
    directory = tmp_path / "with_spike" / "regression" / "multiple_linear"
    arrays, metadata = load_npz(directory / "preprocessor.npz")
    assert metadata["standardized"] is True
    assert not (arrays["scale"] == 1.0).all()


def test_pca_forces_standardization_even_when_model_says_none(tmp_path):
    """PCA requires centered input, so it overrides preprocess="none"."""
    from src.ml.core.persist import load_npz

    train_model(
        MultipleLinear, variant="with_spike", output_root=tmp_path, preprocess="none", pca_components=4
    )
    directory = tmp_path / "with_spike" / "regression" / "multiple_linear"
    arrays, metadata = load_npz(directory / "preprocessor.npz")
    assert metadata["standardized"] is True
    assert arrays["components"].shape == (4, 8)


def test_trainer_source_never_references_the_test_split():
    """A monkeypatch cannot catch `from .data import load_test`, so check the source."""
    import src.ml.core.trainer as trainer_module

    source = inspect.getsource(trainer_module)
    assert "load_test" not in source
    assert "TEST_PATH" not in source


def test_random_forest_search_persists_the_pca_winner(tmp_path, monkeypatch):
    from src.ml.core.persist import load_npz
    import src.ml.core.trainer as trainer

    class TinyForest:
        name = "random_forest"
        task = "classification"

        def __init__(self, max_depth=3, n_estimators=2):
            self.max_depth, self.n_estimators = max_depth, n_estimators

        def fit(self, X, y):
            self.n_features_ = X.shape[1]
            return self

        def get_params(self):
            return {"max_depth": self.max_depth, "n_estimators": self.n_estimators}

        def predict(self, X):
            assert X.shape[1] == self.n_features_
            return X[:, 0].copy()

        def state_dict(self):
            return {"n_features": np.array([self.n_features_])}

        def load_state(self, arrays, metadata):
            self.n_features_ = int(arrays["n_features"][0])
            return self

    def score(candidate, X, *_):
        # Force PCA(4) above all raw and other PCA candidates.
        value = (0.9 if X.shape[1] == 4 else
                 0.7 if X.shape[1] == 8 and candidate.max_depth == 5 and candidate.n_estimators == 2
                 else 0.6 if X.shape[1] == 8 else 0.5)
        metrics = {key: {"value": value, "reason": None} for key in ("average_precision", "roc_auc")}
        return metrics, candidate.predict(X), np.full(len(X), value)

    monkeypatch.setattr(trainer, "evaluate", score)
    result = train_model(TinyForest, grid={"max_depth": [3, 5], "n_estimators": [2, 3]},
                         output_root=tmp_path)
    directory = result["directory"]
    rows = json.loads((directory / "search_results.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    arrays, metadata = load_npz(directory / "preprocessor.npz")

    assert len(rows) == 7 and len({row["id"] for row in rows}) == 7
    assert [row["pca_components"] for row in rows] == [None] * 4 + [2, 4, 6]
    assert all(row["params"] == rows[2]["params"] for row in rows[4:])
    assert config["selected_candidate"] == rows[5]["id"]
    assert config["params"] == rows[2]["params"]
    assert config["pca_components"] == metadata["pca_components"] == 4
    assert metadata["standardized"] is True
    assert arrays["components"].shape == (4, 8)
    assert result["load_verification"]["passed"] is True

    original_load = trainer.load_npz

    def altered_preprocessor(path):
        arrays, metadata = original_load(path)
        if path.name == "preprocessor.npz":
            arrays["components"] *= -1
        return arrays, metadata

    monkeypatch.setattr(trainer, "load_npz", altered_preprocessor)
    altered = train_model(TinyForest, grid={"max_depth": [3, 5], "n_estimators": [2, 3]},
                          output_root=tmp_path)
    assert altered["load_verification"]["passed"] is False


def test_random_forest_rank_failure_records_invalid_candidate(tmp_path, monkeypatch):
    from src.ml.classification.random_forest import RandomForest
    import src.ml.core.trainer as trainer

    original_fit = trainer.PCA.fit

    def fail_six(self, X):
        if self.n_components == 6:
            raise ValueError("PCA component count exceeds numerical rank")
        return original_fit(self, X)

    monkeypatch.setattr(trainer.PCA, "fit", fail_six)
    result = train_model(RandomForest, grid={"max_depth": [3, 5], "n_estimators": [2, 3]},
                         output_root=tmp_path)
    rows = json.loads((result["directory"] / "search_results.json").read_text(encoding="utf-8"))
    assert len(rows) == 7
    assert rows[-1]["pca_components"] == 6
    assert rows[-1]["status"] == "invalid"
    assert all(row["status"] == "valid" for row in rows[:-1])
    for metric in ("average_precision", "roc_auc"):
        assert rows[-1]["metrics"][metric] == {
            "value": None, "reason": "PCA component count exceeds numerical rank"
        }
    assert result["load_verification"]["passed"] is True

    def fail_unrelated(self, X):
        if self.n_components == 6:
            raise ValueError("unexpected covariance failure")
        return original_fit(self, X)

    monkeypatch.setattr(trainer.PCA, "fit", fail_unrelated)
    with pytest.raises(ValueError, match="unexpected covariance failure"):
        train_model(RandomForest, grid={"n_estimators": [2]}, output_root=tmp_path / "unexpected")


def test_reload_preprocessor_zeros_constant_feature(tmp_path, monkeypatch):
    from dataclasses import replace
    import src.ml.core.trainer as trainer

    original = trainer.load("with_spike")
    train_X = original.train.X.copy()
    validation_X = original.validation.X.copy()
    train_X[:, 0] = 0.0
    validation_X[:, 0] = 10.0
    monkeypatch.setattr(trainer, "load", lambda _: replace(
        original, train=replace(original.train, X=train_X),
        validation=replace(original.validation, X=validation_X)))

    class ConstantProbe:
        name = "constant_probe"
        task = "regression"

        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[:, 0]

        def get_params(self):
            return {}

        def state_dict(self):
            return {"fitted": np.array([1])}

        def load_state(self, arrays, metadata):
            return self

    result = train_model(ConstantProbe, output_root=tmp_path, preprocess="standardize")
    from src.ml.core.persist import load_npz

    arrays, _ = load_npz(result["directory"] / "preprocessor.npz")
    assert arrays["constant"][0]
    assert result["load_verification"]["passed"] is True


def test_svm_search_crosses_every_c_with_every_pca_size(tmp_path):
    """Unlike the forest, svm tries PCA with every C: 3 x (none, 2, 4, 6) = 12 in the config."""
    from src.ml.classification.svm import SVM

    original = SVM.__init__

    def quick(self, C=1.0, max_iter=300, fit_intercept=True, random_state=42):
        original(self, C=C, max_iter=max_iter, fit_intercept=fit_intercept, random_state=random_state)

    SVM.__init__ = quick
    try:
        result = train_model(SVM, grid={"C": [0.1, 1]}, variant="with_spike", output_root=tmp_path)
    finally:
        SVM.__init__ = original
    directory = tmp_path / "with_spike" / "classification" / "svm"
    rows = json.loads((directory / "search_results.json").read_text(encoding="utf-8"))
    assert [(row["params"]["C"], row["pca_components"]) for row in rows] == [
        (0.1, None), (1, None), (0.1, 2), (1, 2), (0.1, 4), (1, 4), (0.1, 6), (1, 6),
    ]
    assert all(row["status"] == "valid" for row in rows)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    winner = next(row for row in rows if row["id"] == result["selected"])
    assert config["pca_components"] == winner["pca_components"]
    assert config["preprocess"] == "standardize"
    assert result["load_verification"]["passed"]


def test_svm_reload_uses_the_pca_basis_of_the_winner(tmp_path, monkeypatch):
    from src.ml.classification.svm import SVM
    from src.ml.core import trainer
    from src.ml.core.persist import load_npz

    def pick_pca_four(candidates, task):
        return next(c for c in candidates if c["id"] == "svm-02")

    monkeypatch.setattr(trainer.metrics_module, "select_candidate", pick_pca_four)
    result = train_model(SVM, grid={"C": [1]}, variant="non_spike", output_root=tmp_path)
    directory = tmp_path / "non_spike" / "classification" / "svm"
    arrays, metadata = load_npz(directory / "preprocessor.npz")
    assert result["selected"] == "svm-02"
    assert metadata["pca_components"] == 4
    assert arrays["components"].shape == (4, 8)
    assert result["model"].coef_.shape == (4,)
    assert result["load_verification"]["passed"]


@pytest.mark.parametrize("variant,rows", [("with_spike", 1733), ("non_spike", 1520)])
def test_models_that_ask_for_positions_get_trading_day_indices(tmp_path, monkeypatch, variant, rows):
    """non_spike has rows removed, so a row index is not a trading-day index there."""
    from src.ml.classification.stacking import Stacking

    seen = {}
    original = Stacking.fit

    def capture(self, X, y=None, positions=None, n_positions=None):
        seen.update(positions=np.array(positions), n_positions=n_positions, X=np.array(X))
        return original(self, X, y, positions=positions, n_positions=n_positions)

    monkeypatch.setattr(Stacking, "fit", capture)
    result = train_model(Stacking, variant=variant, output_root=tmp_path)
    assert seen["n_positions"] == 1733
    assert len(seen["positions"]) == rows
    assert (np.diff(seen["positions"]) > 0).all()
    assert np.array_equal(seen["positions"], np.arange(1733)) == (variant == "with_spike")
    assert result["load_verification"]["passed"]


def test_base_specific_preprocessing_hands_the_model_raw_features(tmp_path, monkeypatch):
    from src.ml.classification.stacking import Stacking
    from src.ml.core.data import load
    from src.ml.core.persist import load_npz

    seen = {}
    original = Stacking.fit

    def capture(self, X, y=None, positions=None, n_positions=None):
        seen["X"] = np.array(X)
        return original(self, X, y, positions=positions, n_positions=n_positions)

    monkeypatch.setattr(Stacking, "fit", capture)
    train_model(Stacking, variant="with_spike", output_root=tmp_path)
    # Each base standardises for itself; a second scaler in front would be invisible to
    # the tree but would change what "raw" means in the saved artifact.
    assert np.array_equal(seen["X"], load("with_spike").train.X)
    _, metadata = load_npz(tmp_path / "with_spike" / "classification" / "stacking" / "preprocessor.npz")
    assert metadata["preprocess"] == "base_specific"
    assert metadata["standardized"] is False


def test_models_without_a_positions_argument_are_fitted_as_before(tmp_path, monkeypatch):
    seen = {}
    original = MultipleLinear.fit

    def capture(self, X, y=None, **extra):
        seen["extra"] = extra
        return original(self, X, y)

    monkeypatch.setattr(MultipleLinear, "fit", capture)
    train_model(MultipleLinear, variant="non_spike", output_root=tmp_path)
    assert seen["extra"] == {}
