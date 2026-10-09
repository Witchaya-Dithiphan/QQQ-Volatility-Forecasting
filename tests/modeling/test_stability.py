import numpy as np
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
    for name, spec in SPECS.items():
        if spec[0] == "classification":
            a, b = _fit_predict(spec, 42, X, y, X), _fit_predict(spec, 42, X, y, X)
            assert np.array_equal(a, b), name
