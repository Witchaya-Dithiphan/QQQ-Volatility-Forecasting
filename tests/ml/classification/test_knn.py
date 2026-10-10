import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src.ml.core.compare import assert_parity
from src.ml.classification.knn import KNN


@pytest.fixture
def data():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(150, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


# --- unit test กลไกภายใน: ระยะทางที่คำนวณมือได้ ---


def test_picks_the_literal_nearest_neighbour_when_k_is_one():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    model = KNN(k=1).fit(X, np.array([0, 1]))
    assert model.predict(np.array([[0.1, 0.1]]))[0] == 0
    assert model.predict(np.array([[9.9, 9.9]]))[0] == 1


def test_uniform_weights_are_a_majority_vote():
    """Two zeros at distance 1, one one at distance 1 -> majority is 0."""
    X = np.array([[-1.0], [1.0], [2.0]])
    model = KNN(k=3, weights="uniform").fit(X, np.array([0, 0, 1]))
    assert model.predict(np.array([[0.0]]))[0] == 0


def test_distance_weights_let_a_closer_point_outvote_two_far_ones():
    """1/0.1 beats 1/5 + 1/6, so the near positive wins against two far negatives."""
    X = np.array([[0.1], [5.0], [6.0]])
    model = KNN(k=3, weights="distance").fit(X, np.array([1, 0, 0]))
    assert model.predict(np.array([[0.0]]))[0] == 1


def test_distance_weights_give_an_exact_match_full_weight():
    """A zero distance must not produce inf or nan."""
    X = np.array([[0.0], [1.0], [2.0]])
    model = KNN(k=3, weights="distance").fit(X, np.array([1, 0, 0]))
    probability = model.predict_proba(np.array([[0.0]]))[0]
    assert np.isfinite(probability) and probability == pytest.approx(1.0)


def test_probability_is_the_weighted_share_of_the_positive_class():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    model = KNN(k=4, weights="uniform").fit(X, np.array([1, 1, 0, 0]))
    assert model.predict_proba(np.array([[1.5]]))[0] == pytest.approx(0.5)


def test_k_larger_than_the_training_set_is_rejected():
    with pytest.raises(ValueError, match="k"):
        KNN(k=10).fit(np.zeros((3, 2)), np.array([0, 1, 0]))


# --- parity กับ reference ---


@pytest.mark.parametrize("k", [3, 5, 9, 15])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_matches_sklearn_across_the_whole_grid(data, k, weights):
    """The config grid is k x weights = 8 candidates, so all 8 must match.

    Queried on held-out points, which is what the trainer does — it fits on train
    and predicts on validation. See test_self_distance_* below for why predicting
    on the training set is a different and worse test.
    """
    X, y = data
    train_X, train_y, query_X = X[:100], y[:100], X[100:]
    scratch = KNN(k=k, weights=weights).fit(train_X, train_y)
    reference = scratch.reference().fit(train_X, train_y)
    assert_parity(scratch, reference, query_X, task="classification")


def test_self_distance_is_exactly_zero_unlike_sklearn():
    """sklearn's euclidean uses ||a||^2 + ||b||^2 - 2ab, which cancels badly.

    Querying a training point back, sklearn reports its distance to itself as
    ~4.2e-08 rather than 0, so with distance weights the point does not quite
    take all the weight. np.linalg.norm(x - x) is exactly 0, so the scratch
    model is the more accurate of the two here. Parity is therefore checked on
    held-out points, which is the only case the pipeline actually produces.
    """
    rng = np.random.RandomState(42)
    X = rng.normal(size=(150, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    scratch = KNN(k=3, weights="distance").fit(X, y)
    reference = scratch.reference().fit(X, y)

    distances, _ = reference.kneighbors(X[142:143], 3)
    assert distances[0][0] > 0, "sklearn no longer has the cancellation artefact"
    assert np.linalg.norm(X[142] - X[142]) == 0.0

    # Neighbour 142 is itself and carries label 0, so the exact answer is 0.0.
    assert scratch.predict_proba(X[142:143])[0] == 0.0
    assert reference.predict_proba(X[142:143])[0][1] > 0.0


def test_predictions_still_agree_with_sklearn_on_training_points(data):
    """The weighting artefact is tiny, so labels must still match everywhere."""
    X, y = data
    scratch = KNN(k=5, weights="distance").fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert np.array_equal(scratch.predict(X), reference.predict(X))


def test_tie_breaking_matches_sklearn_on_equidistant_points(data):
    """Points at identical distances are where implementations usually diverge."""
    X = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    query = np.array([[0.0, 0.0]])
    scratch = KNN(k=2, weights="uniform").fit(X, y)
    reference = scratch.reference().fit(X, y)
    assert scratch.predict(query)[0] == reference.predict(query)[0]


def test_reference_uses_brute_force_to_remove_tree_differences():
    reference = KNN().reference()
    assert isinstance(reference, KNeighborsClassifier)
    assert reference.get_params()["algorithm"] == "brute"
    assert reference.get_params()["metric"] == "euclidean"


# --- save / load ---


def test_state_round_trip_predicts_identically(data):
    X, y = data
    original = KNN(k=9, weights="distance").fit(X, y)
    restored = KNN().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(X), restored.predict(X))
    assert np.allclose(original.predict_proba(X), restored.predict_proba(X))


def test_state_round_trip_carries_the_training_set(data):
    """kNN has no learned weights — the training data is the model."""
    X, y = data
    original = KNN(k=5).fit(X, y)
    arrays = original.state_dict()
    assert arrays["X"].shape == X.shape and arrays["y"].shape == y.shape


def test_default_constructor_works_for_the_trainer(data):
    X, y = data
    assert np.isfinite(KNN().fit(X, y).predict(X)).all()
