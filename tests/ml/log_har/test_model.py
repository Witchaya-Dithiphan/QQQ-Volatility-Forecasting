"""Log-HAR features, the transfer-ridge closed form, sklearn parity and save/load."""
import numpy as np
import pytest

from src.ml.core.compare import ATOL, RTOL
from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.log_har.features import DAILY_FLOOR, back_transform, har_features
from src.ml.log_har.model import LogHAR, pooled_prior


def raw_features(n=400, seed=0):
    """Eight accepted-feature columns with plausible positive volatilities."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, len(FEATURE_COLUMNS)))
    col = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    X[:, col["return_1d"]] = rng.normal(0, 0.012, n)
    X[:, col["return_5d"]] = rng.normal(0, 0.025, n)
    X[:, col["historical_volatility_5d"]] = np.exp(rng.normal(np.log(0.18), 0.4, n))
    X[:, col["historical_volatility_20d"]] = np.exp(rng.normal(np.log(0.18), 0.3, n))
    X[:, col["intraday_range"]] = np.exp(rng.normal(np.log(0.012), 0.4, n))
    X[:, col["rsi_14"]] = rng.uniform(20, 80, n)
    return X


def target_for(X, seed=1):
    F = har_features(X, "X")
    rng = np.random.RandomState(seed)
    return np.exp(-0.3 + F @ np.array([0.05, 0.3, 0.45, 0.15, -2.0]) + rng.normal(0, 0.25, len(X)))


def test_features_follow_the_har_definition():
    X = raw_features(5)
    col = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    X[0, col["return_1d"]] = 0.0          # a flat day must not produce log(0)
    X[1, col["return_5d"]] = 0.03         # positive weekly return: no leverage term
    F = har_features(X, "X")
    assert F.shape == (5, 5) and np.isfinite(F).all()
    assert F[0, 0] == np.log(DAILY_FLOOR)
    assert np.isclose(F[2, 0], np.log(max(abs(X[2, col["return_1d"]]) * np.sqrt(252), DAILY_FLOOR)))
    assert np.allclose(F[:, 1], np.log(X[:, col["historical_volatility_5d"]]))
    assert np.allclose(F[:, 2], np.log(X[:, col["historical_volatility_20d"]]))
    assert np.allclose(F[:, 3], np.log(X[:, col["intraday_range"]]))
    assert F[1, 4] == 0.0 and np.all(F[:, 4] <= 0)
    assert har_features(X, "3").shape == (5, 3)


def test_back_transform_is_the_lognormal_mean():
    rng = np.random.RandomState(0)
    draws = np.exp(rng.normal(-1.5, 0.4, 2_000_000))
    assert abs(back_transform(np.array([-1.5]), 0.4 ** 2)[0] - draws.mean()) / draws.mean() < 1e-3


def test_ols_matches_lstsq_and_sklearn():
    X = raw_features()
    y = target_for(X)
    model = LogHAR(features="X", kappa=0.0).fit(X, y)
    F, z = har_features(X, "X"), np.log(y)
    solution, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(F)), F]), z, rcond=None)
    assert np.allclose(model.intercept_, solution[0]) and np.allclose(model.coef_, solution[1:])
    reference = model.reference().fit(X, y)
    assert np.allclose(model.predict(X[:50]), reference.predict(X[:50]), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kappa", [0.03, 1.0, 10.0, np.inf])
def test_transfer_ridge_matches_sklearn_ridge_on_the_shifted_target(kappa):
    X = raw_features()
    y = target_for(X)
    prior = np.array([0.1, 0.25, 0.5, 0.1, -1.0])
    model = LogHAR(features="X", kappa=kappa, prior=prior).fit(X, y)
    reference = model.reference().fit(X, y)
    held_out = raw_features(80, seed=7)
    assert np.allclose(model.predict(held_out), reference.predict(held_out), rtol=RTOL, atol=ATOL)
    assert np.allclose(model.coef_, reference.coef_, rtol=RTOL, atol=ATOL)


def test_kappa_endpoints_are_the_two_ablations():
    X = raw_features()
    y = target_for(X)
    prior = np.array([0.1, 0.25, 0.5, 0.1, -1.0])
    qqq_only = LogHAR(features="X", kappa=0.0).fit(X, y)
    at_zero = LogHAR(features="X", kappa=0.0, prior=prior).fit(X, y)
    assert np.allclose(at_zero.coef_, qqq_only.coef_)
    pooled = LogHAR(features="X", kappa=np.inf, prior=prior).fit(X, y)
    assert np.array_equal(pooled.coef_, prior)
    near = LogHAR(features="X", kappa=1e7, prior=prior).fit(X, y)
    assert np.allclose(near.coef_, prior, atol=1e-5)
    # Shrinkage moves monotonically from the QQQ estimate towards the prior.
    distances = [np.linalg.norm(LogHAR(features="X", kappa=k, prior=prior).fit(X, y).coef_ - prior)
                 for k in (0.0, 0.1, 1.0, 10.0)]
    assert all(a > b for a, b in zip(distances, distances[1:]))


def test_save_load_round_trip_predicts_identically():
    X = raw_features()
    y = target_for(X)
    model = LogHAR(features="X", kappa=0.3, prior=np.array([0.1, 0.25, 0.5, 0.1, -1.0])).fit(X, y)
    clone = LogHAR().load_state(model.state_dict(), {"params": model.get_params()})
    assert np.array_equal(model.predict(X), clone.predict(X))
    pooled = LogHAR(features="X", kappa=np.inf, prior=np.array([0.1, 0.25, 0.5, 0.1, -1.0])).fit(X, y)
    reloaded = LogHAR().load_state(pooled.state_dict(), {"params": pooled.get_params()})
    assert np.array_equal(pooled.predict(X), reloaded.predict(X))


def test_pooled_prior_is_the_within_estimator():
    from sklearn.linear_model import LinearRegression
    frames = []
    for seed, level in ((0, -0.2), (1, 0.4), (2, -0.6)):
        X = raw_features(300, seed=seed)
        frames.append((X, target_for(X, seed=seed + 10) * np.exp(level)))  # different volatility levels
    beta, per_asset = pooled_prior(frames, "X")
    stacked_F, stacked_z = [], []
    for X, y in frames:
        F, z = har_features(X, "X"), np.log(y)
        stacked_F.append(F - F.mean(axis=0))
        stacked_z.append(z - z.mean())
    reference = LinearRegression(fit_intercept=False).fit(np.vstack(stacked_F), np.concatenate(stacked_z))
    assert np.allclose(beta, reference.coef_, rtol=RTOL, atol=ATOL)
    assert len(per_asset) == 3 and all(b.shape == (5,) for b in per_asset)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        LogHAR(features="Y")
    with pytest.raises(ValueError):
        LogHAR(features="X", kappa=1.0).fit(raw_features(), target_for(raw_features()))  # kappa > 0 needs a prior
    with pytest.raises(ValueError):
        LogHAR(features="X").fit(raw_features(10), -np.ones(10))
    flat = raw_features(10)
    flat[0, FEATURE_COLUMNS.index("intraday_range")] = 0.0  # a halted day has no log range
    with pytest.raises(ValueError, match="positive"):
        har_features(flat, "X")
