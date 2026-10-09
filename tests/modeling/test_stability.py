import numpy as np
import pytest
from src.modeling.stability import SPECS, blocked_folds, _fit_predict


def test_blocked_folds_cover_validation_once_and_purge_neighbours():
    n, seen = 100, []
    for fit, test in blocked_folds(n, 5, purge=5):
        seen += list(test)
        assert not set(fit) & set(range(test[0] - 5, test[-1] + 6))
    assert sorted(seen) == list(range(n))


def test_unseeded_models_are_deterministic_and_seeded_ones_accept_seed():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 8)), rng.integers(0, 2, 120)
    dates = np.arange("2020-01-01", "2020-04-30", dtype="datetime64[D]")
    for name, spec in SPECS.items():
        if spec[0] == "classification":
            a, b = _fit_predict(spec, 42, X, y, X, dates=dates, original_dates=dates), _fit_predict(spec, 42, X, y, X, dates=dates, original_dates=dates)
            assert np.array_equal(a, b), name


def test_stacking_adapter_requires_actual_timeline():
    X = np.random.default_rng(0).normal(size=(120, 8))
    y = np.arange(120) % 2
    with pytest.raises(ValueError, match="dates"):
        _fit_predict(SPECS["Stacking"], 42, X, y, X)


def test_stacking_adapter_retains_original_day_purge(monkeypatch):
    from src.modeling import stability
    original = np.arange("2020-01-01", "2020-06-01", dtype="datetime64[D]")
    kept = np.delete(np.arange(len(original)), np.arange(30, 40))
    X = np.random.default_rng(0).normal(size=(len(kept), 8))
    y = np.arange(len(kept)) % 2
    model = stability.StackingClassifier(logistic_max_iter=15)
    spec = ("classification", None, "random_state", lambda seed: model)
    _fit_predict(spec, 42, X, y, X, dates=original[kept], original_dates=original)
    assert len(model.fold_records_) == 4
    assert model.oos_positions_[0] >= int(.4 * len(original))
    for fold in model.fold_records_:
        assert max(fold["train_positions"]) < min(fold["validation_positions"]) - 5


def test_stacking_stability_uses_expanding_cv_and_declares_shuffle_invalid(monkeypatch):
    from src.modeling import stability
    from types import SimpleNamespace
    from src.modeling.datasets import DatasetSplit
    original = np.arange("2020-01-01", "2020-06-01", dtype="datetime64[D]")
    X = np.random.default_rng(2).normal(size=(len(original),8))
    y = np.arange(len(X))%2
    train = DatasetSplit(original,X,np.ones(len(X)),y,"synthetic","synthetic")
    validation = SimpleNamespace(X=X[:30],dates=np.arange("2021-01-01","2021-01-31",dtype="datetime64[D]"))
    calls = []
    def observed(spec,seed,Xa,ya,Xb,**kwargs):
        assert "dates" in kwargs and "original_dates" in kwargs
        calls.append(kwargs)
        return np.arange(len(Xb))%2
    monkeypatch.setattr(stability,"_fit_predict",observed)
    result = stability.check_model("Stacking",train,validation,y,y[:30])
    assert len(calls)==len(stability.SEEDS)+5
    assert result["cv_protocol"]=="expanding_original_timeline_five_day_purge"
    assert result["shuffle_var_pct"] is None and result["shuffle_reason"]
    for call in calls[len(stability.SEEDS):]:
        assert call["original_dates"][-1]==call["dates"][-1]
