"""Linear soft-margin SVM: optimality worked by hand, then objective parity with libsvm.

Hinge loss has kinks, so two correct solvers can stop at slightly different weights.
The comparison is therefore on the objective both are minimising, as ADR-002 in
docs/design/ARCHITECTURE.md allows for this one model.
"""

import numpy as np
import pytest

from src.ml.classification.svm import SVM
from src.ml.core.data import load
from src.ml.core.persist import load_npz, save_npz
from src.ml.core.preprocess import Standardizer
from src.ml.registry import build, model_names

# Two points per class on a line, mirrored around zero.
LINE_X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
LINE_Y = np.array([0, 0, 1, 1])


def objective(weights, intercept, X, y, C):
    signed = 2.0 * np.asarray(y) - 1.0
    return 0.5 * weights @ weights + C * np.maximum(0.0, 1.0 - signed * (X @ weights + intercept)).sum()


def test_wide_margin_puts_support_vectors_exactly_on_the_margin():
    # J(w) = w^2/2 + C * 2 * max(0, 1 - w) for w >= 1/2. With C = 1 the slope w - 2 is
    # negative until the hinge switches off at w = 1, so w* = 1 and b* = 0 by symmetry.
    model = SVM(C=1.0).fit(LINE_X, LINE_Y)
    assert np.allclose(model.coef_, [1.0], atol=1e-3)
    assert abs(model.intercept_) < 1e-3
    margin = (2 * LINE_Y - 1) * model.decision_function(LINE_X)
    assert np.allclose(margin, [2.0, 1.0, 1.0, 2.0], atol=2e-3)
    assert np.array_equal(model.predict(LINE_X), LINE_Y)


def test_small_c_accepts_margin_violations():
    # With C = 0.1 the slope on 1/2 < w < 1 is w - 0.2 > 0 and on w < 1/2 it is
    # w - 0.6 < 0, so the minimum sits on the kink at w = 1/2: the inner points
    # are deliberately left inside the margin.
    model = SVM(C=0.1).fit(LINE_X, LINE_Y)
    assert np.allclose(model.coef_, [0.5], atol=1e-3)
    margin = (2 * LINE_Y - 1) * model.decision_function(LINE_X)
    assert margin[1] < 1 and margin[2] < 1


def test_points_beyond_the_margin_do_not_move_the_solution():
    # KKT: a point with margin > 1 has zero multiplier, so pushing it further out or
    # deleting it must leave the optimum where it was.
    base = SVM(C=1.0).fit(LINE_X, LINE_Y)
    pushed = SVM(C=1.0).fit(np.array([[-50.0], [-1.0], [1.0], [50.0]]), LINE_Y)
    dropped = SVM(C=1.0).fit(LINE_X[1:3], LINE_Y[1:3])
    assert np.allclose(pushed.coef_, base.coef_, atol=2e-3)
    assert np.allclose(dropped.coef_, base.coef_, atol=2e-3)


def test_c_multiplies_the_hinge_term_not_the_norm():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int)
    loose, tight = SVM(C=0.01).fit(X, y), SVM(C=10.0).fit(X, y)
    # A larger C buys fewer violations at the price of a larger norm.
    assert np.linalg.norm(tight.coef_) > np.linalg.norm(loose.coef_)


def test_step_size_decays_as_one_over_t():
    assert [SVM.step_size(t) for t in (1, 2, 4, 100)] == [1.0, 0.5, 0.25, 0.01]


def test_returned_solution_is_the_best_iterate_and_history_records_the_path():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] - X[:, 1] + 0.7 * rng.normal(size=80) > 0).astype(int)
    model = SVM(C=1.0, max_iter=500).fit(X, y)
    final = objective(model.coef_, model.intercept_, X, y, 1.0)
    # Subgradient steps are not descent steps, so the raw path may go up; what is
    # returned is the lowest point visited.
    assert np.isclose(final, model.objective_, rtol=1e-12)
    assert final <= min(model.history_["loss"]) + 1e-9
    assert model.history_["iteration"][0] == 1
    assert len(model.history_["iteration"]) == len(model.history_["loss"])


def test_without_intercept_the_boundary_passes_through_the_origin():
    X = np.array([[1.0], [2.0], [3.0], [-1.0]])
    model = SVM(C=1.0, fit_intercept=False).fit(X, np.array([1, 1, 1, 0]))
    assert model.intercept_ == 0.0
    assert model.decision_function(np.zeros((1, 1)))[0] == 0.0


def test_readonly_input_is_preserved():
    X = np.arange(40.0).reshape(20, 2)
    X.setflags(write=False)
    SVM(max_iter=50).fit(X, np.array([0, 1] * 10))
    assert not X.flags.writeable


@pytest.mark.parametrize("kwargs", [{"C": 0.0}, {"C": -1.0}, {"max_iter": 0}, {"fit_intercept": "yes"}])
def test_invalid_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        SVM(**kwargs).fit(LINE_X, LINE_Y)


def test_invalid_training_data_fail():
    with pytest.raises(ValueError):
        SVM().fit([[0], [1]], [0, 2])
    with pytest.raises(ValueError):
        SVM().fit([[0], [np.nan]], [0, 1])


def test_has_no_probability_output():
    model = SVM(max_iter=50).fit(LINE_X, LINE_Y)
    with pytest.raises(NotImplementedError):
        model.predict_proba(LINE_X)


def assert_objective_parity(scratch, reference, X, y, held_out=None):
    """The svm criterion: same objective to 1e-3, never worse, labels agree on >= 99%."""
    ours = objective(scratch.coef_, scratch.intercept_, X, y, scratch.C)
    theirs = objective(reference.coef_[0], reference.intercept_[0], X, y, scratch.C)
    assert abs(ours - theirs) / theirs < 1e-3
    assert ours <= theirs * (1 + 1e-3)
    for rows in (X, held_out):
        if rows is not None:
            assert np.mean(scratch.predict(rows) == reference.predict(rows)) >= 0.99
    return ours, theirs


@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_synthetic_objective_parity(seed, C):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.8 * rng.normal(size=300) > 0).astype(int)
    scratch = SVM(C=C).fit(X[:200], y[:200])
    reference = scratch.reference().fit(X[:200], y[:200])
    assert_objective_parity(scratch, reference, X[:200], y[:200], held_out=X[200:])


def test_weights_match_reference_when_the_optimum_is_well_conditioned():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] + rng.normal(size=200) > 0).astype(int)
    scratch = SVM(C=0.1, max_iter=200000).fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert np.allclose(scratch.coef_, reference.coef_[0], atol=1e-4)
    assert abs(scratch.intercept_ - reference.intercept_[0]) < 1e-4


@pytest.mark.parametrize("variant", ["with_spike", "non_spike"])
@pytest.mark.parametrize("C", [0.1, 1, 10])
def test_real_train_and_validation_objective_parity(variant, C):
    data = load(variant)
    standardizer = Standardizer().fit(data.train.X)
    train_X = standardizer.transform(data.train.X)
    validation_X = standardizer.transform(data.validation.X)
    scratch = SVM(C=C).fit(train_X, data.train.y_classification)
    reference = scratch.reference().fit(train_X, data.train.y_classification)
    assert_objective_parity(scratch, reference, train_X, data.train.y_classification, held_out=validation_X)


def test_save_load_preserves_scores(tmp_path):
    rng = np.random.RandomState(2)
    X = rng.normal(size=(50, 3))
    y = (X[:, 0] > 0).astype(int)
    original = SVM(C=10.0, max_iter=300).fit(X, y)
    path = tmp_path / "svm.npz"
    save_npz(path, original.state_dict(), {"params": original.get_params()})
    arrays, metadata = load_npz(path)
    restored = SVM().load_state(arrays, metadata)
    assert restored.get_params() == original.get_params()
    assert np.array_equal(restored.decision_function(X), original.decision_function(X))
    assert np.array_equal(restored.predict(X), original.predict(X))


def test_registry_exposes_svm():
    assert "svm" in model_names()
    assert isinstance(build("svm"), SVM)


@pytest.mark.parametrize("shift", [5.0, 50.0])
@pytest.mark.parametrize("C", [1.0, 10.0])
def test_uncentred_features_reach_the_same_optimum(shift, C):
    """The intercept is free, so moving every feature by a constant must not change the fit.

    Without centring inside the solver this failed badly: at a shift of 5 the objective
    stopped 3% to 120% above the optimum and three million iterations did not close it.
    """
    rng = np.random.RandomState(0)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + rng.normal(size=120) > 0).astype(int)
    centred = SVM(C=C).fit(X, y)
    shifted = SVM(C=C).fit(X + shift, y)
    reference = shifted.reference().fit(X + shift, y)
    assert_objective_parity(shifted, reference, X + shift, y)
    # Same hyperplane, expressed in the shifted coordinates.
    assert np.allclose(shifted.coef_, centred.coef_, atol=1e-8)
    assert np.allclose(shifted.decision_function(X + shift), centred.decision_function(X), atol=1e-6)


def test_constant_column_does_not_stall_the_solver():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(150, 4))
    X[:, 2] = 3.0
    y = (X[:, 0] - X[:, 1] + rng.normal(size=150) > 0).astype(int)
    scratch = SVM(C=10.0).fit(X, y)
    assert_objective_parity(scratch, scratch.reference().fit(X, y), X, y)
    assert scratch.coef_[2] == 0.0


def test_without_intercept_features_are_not_centred():
    X = np.array([[1.0], [2.0], [3.0], [-1.0]])
    model = SVM(C=1.0, fit_intercept=False).fit(X, np.array([1, 1, 1, 0]))
    assert model.decision_function(np.zeros((1, 1)))[0] == 0.0
