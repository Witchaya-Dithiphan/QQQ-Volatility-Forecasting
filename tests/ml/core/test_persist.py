import numpy as np
import pytest

from src.ml.core.persist import load_npz, save_npz, verify_reload


def test_save_load_round_trip_preserves_arrays_and_metadata(tmp_path):
    arrays = {"weights": np.array([1.5, -2.25, 0.0]), "intercept": np.array([0.75])}
    metadata = {"model": "multiple_linear", "seed": 42}
    path = tmp_path / "model.npz"

    save_npz(path, arrays, metadata)
    loaded_arrays, loaded_metadata = load_npz(path)

    assert loaded_metadata == metadata
    assert set(loaded_arrays) == set(arrays)
    for name, value in arrays.items():
        assert np.array_equal(loaded_arrays[name], value)


def test_save_npz_overwrites_so_reruns_do_not_crash(tmp_path):
    path = tmp_path / "model.npz"
    save_npz(path, {"a": np.array([1.0])}, {"run": 1})
    save_npz(path, {"a": np.array([2.0])}, {"run": 2})

    arrays, metadata = load_npz(path)
    assert arrays["a"] == np.array([2.0])
    assert metadata == {"run": 2}


def test_save_npz_rejects_nonfinite_arrays(tmp_path):
    with pytest.raises(ValueError, match="Nonfinite"):
        save_npz(tmp_path / "bad.npz", {"a": np.array([np.inf])}, {})


def test_save_npz_rejects_an_empty_model(tmp_path):
    with pytest.raises(ValueError, match="no arrays"):
        save_npz(tmp_path / "empty.npz", {}, {})


def test_save_npz_rejects_unsafe_array_names(tmp_path):
    with pytest.raises(ValueError, match="Unsafe array name"):
        save_npz(tmp_path / "bad.npz", {"../escape": np.array([1.0])}, {})


def test_verify_reload_passes_on_identical_predictions():
    before = np.array([0.1, 0.2, 0.3])
    assert verify_reload(before, before.copy())["passed"] is True


def test_verify_reload_fails_when_predictions_differ():
    result = verify_reload(np.array([0.1, 0.2]), np.array([0.1, 0.9]))
    assert result["passed"] is False and result["reason"]


def test_verify_reload_requires_exact_equality_for_labels():
    assert verify_reload(np.array([0, 1]), np.array([0, 1]), labels=True)["passed"] is True
    assert verify_reload(np.array([0, 1]), np.array([0, 0]), labels=True)["passed"] is False


def test_verify_reload_fails_on_shape_mismatch():
    assert verify_reload(np.array([1.0, 2.0]), np.array([1.0]))["passed"] is False
