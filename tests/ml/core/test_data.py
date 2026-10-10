import numpy as np
import pytest

from src.ml.core.contracts import FEATURE_COLUMNS
from src.ml.core.data import load, load_test


def test_with_spike_variant_matches_accepted_row_counts():
    data = load("with_spike")
    assert data.variant == "with_spike"
    assert data.train.X.shape == (1733, 8)
    assert data.validation.X.shape == (371, 8)


def test_non_spike_variant_has_the_accepted_reduced_train():
    data = load("non_spike")
    assert data.train.X.shape == (1520, 8)
    assert data.validation.X.shape == (371, 8)


def test_arrays_are_read_only_so_models_cannot_mutate_the_source():
    data = load("with_spike")
    assert data.train.X.flags.writeable is False
    with pytest.raises(ValueError):
        data.train.X[0, 0] = 1.0


def test_both_variants_share_the_same_validation_set():
    assert np.array_equal(load("with_spike").validation.X, load("non_spike").validation.X)


def test_non_spike_train_is_a_strict_subset_of_with_spike_train():
    """The accepted contract says rows were removed, nothing was recomputed."""
    full, reduced = load("with_spike").train, load("non_spike").train
    positions = np.searchsorted(full.dates, reduced.dates)
    assert np.array_equal(full.dates[positions], reduced.dates)
    assert np.array_equal(full.X[positions], reduced.X)
    assert np.array_equal(full.y_regression[positions], reduced.y_regression)


def test_train_ends_before_validation_begins():
    data = load("with_spike")
    assert data.train.dates[-1] < data.validation.dates[0]


def test_feature_order_is_the_accepted_order():
    assert load("with_spike").feature_names == list(FEATURE_COLUMNS)


def test_labels_agree_with_the_accepted_q75_threshold():
    data = load("with_spike")
    assert data.threshold == pytest.approx(0.2530580184684854)
    expected = (data.train.y_regression > data.threshold).astype(np.int64)
    assert np.array_equal(data.train.y_classification, expected)


def test_class_balance_matches_the_accepted_counts():
    y = load("with_spike").train.y_classification
    assert int((y == 0).sum()) == 1300 and int((y == 1).sum()) == 433


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset variant"):
        load("whatever")


def test_test_set_is_denied_by_default():
    with pytest.raises(PermissionError, match="finalize"):
        load_test()


def test_test_set_is_denied_for_truthy_non_true_values():
    with pytest.raises(PermissionError):
        load_test(allow_test=1)


def test_test_set_loads_only_with_explicit_opt_in():
    split = load_test(allow_test=True)
    assert split.X.shape == (373, 8)
