import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering as SkAgglomerative

from src.ml.clustering.agglomerative import Agglomerative
from src.ml.core.compare import align_clusters


@pytest.fixture
def blobs():
    rng = np.random.RandomState(0)
    return np.vstack(
        [
            rng.normal([-5.0, -5.0], 0.4, size=(40, 2)),
            rng.normal([5.0, 5.0], 0.4, size=(40, 2)),
            rng.normal([-5.0, 5.0], 0.4, size=(40, 2)),
        ]
    )


# --- unit test กลไกภายใน ---


def test_merges_the_two_closest_points_first():
    """Four points on a line: 0 and 1 are closest, so they pair up first."""
    X = np.array([[0.0], [1.0], [10.0], [20.0]])
    model = Agglomerative(k=3, linkage="average").fit(X)
    labels = model.labels_
    assert labels[0] == labels[1]
    assert len({labels[0], labels[2], labels[3]}) == 3


def test_k_equal_to_n_leaves_every_point_alone():
    X = np.array([[0.0], [1.0], [2.0]])
    labels = Agglomerative(k=3).fit(X).labels_
    assert len(np.unique(labels)) == 3


def test_k_of_one_puts_everything_together(blobs):
    assert len(np.unique(Agglomerative(k=1).fit(blobs).labels_)) == 1


def test_finds_the_three_planted_blobs(blobs):
    labels = Agglomerative(k=3).fit(blobs).labels_
    for start in (0, 40, 80):
        assert len(np.unique(labels[start : start + 40])) == 1


def test_ward_merges_by_increase_in_within_cluster_variance():
    """Ward cost for two singletons is half their squared distance."""
    X = np.array([[0.0], [2.0]])
    model = Agglomerative(k=1, linkage="ward")
    cost = model._initial_cost(X)
    assert cost[0, 1] == pytest.approx(0.5 * 4.0)


def test_average_linkage_uses_the_mean_pairwise_distance():
    X = np.array([[0.0], [2.0]])
    model = Agglomerative(k=1, linkage="average")
    cost = model._initial_cost(X)
    assert cost[0, 1] == pytest.approx(2.0)


def test_unknown_linkage_is_rejected():
    with pytest.raises(ValueError, match="linkage"):
        Agglomerative(linkage="centroid").fit(np.zeros((4, 2)))


# --- out-of-sample extension ---


def test_predict_on_training_data_returns_the_fitted_labels(blobs):
    model = Agglomerative(k=3).fit(blobs)
    assert np.array_equal(model.predict(blobs), model.labels_)


def test_new_points_are_assigned_by_frozen_nearest_centroid(blobs):
    """Agglomerative has no native predict; the config fixes this extension."""
    model = Agglomerative(k=3).fit(blobs)
    near_first_blob = np.array([[-5.0, -5.0]])
    expected = int(np.argmin(np.linalg.norm(model.cluster_centers_ - near_first_blob, axis=1)))
    assert model.predict(near_first_blob)[0] == expected


def test_centroids_are_the_means_of_the_fitted_clusters(blobs):
    model = Agglomerative(k=3).fit(blobs)
    for index in range(3):
        assert np.allclose(model.cluster_centers_[index], blobs[model.labels_ == index].mean(axis=0))


# --- parity กับ reference ---


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("linkage", ["ward", "average"])
def test_partition_matches_sklearn(blobs, k, linkage):
    """Grid is k x linkage = 10 candidates; cluster ids are arbitrary."""
    scratch = Agglomerative(k=k, linkage=linkage).fit(blobs)
    reference = scratch.reference().fit(blobs)
    aligned = align_clusters(scratch.labels_, reference.labels_)
    assert np.array_equal(scratch.labels_, aligned)


@pytest.mark.parametrize("linkage", ["ward", "average"])
def test_partition_matches_sklearn_on_unseparated_data(linkage):
    """Clean blobs are easy; overlapping data exercises the tie-breaking."""
    rng = np.random.RandomState(5)
    X = rng.normal(size=(120, 4))
    scratch = Agglomerative(k=4, linkage=linkage).fit(X)
    reference = scratch.reference().fit(X)
    aligned = align_clusters(scratch.labels_, reference.labels_)
    assert np.array_equal(scratch.labels_, aligned)


def test_reference_mirrors_the_configured_metric():
    reference = Agglomerative(k=3, linkage="ward").reference()
    assert isinstance(reference, SkAgglomerative)
    assert reference.get_params()["metric"] == "euclidean"
    assert reference.get_params()["linkage"] == "ward"


# --- save / load ---


def test_state_round_trip_predicts_identically(blobs):
    original = Agglomerative(k=3).fit(blobs)
    restored = Agglomerative().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(blobs), restored.predict(blobs))


def test_default_constructor_works_for_the_trainer(blobs):
    assert np.isfinite(Agglomerative().fit(blobs).predict(blobs)).all()
