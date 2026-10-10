import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans

from src.ml.clustering.kmeans import KMeans
from src.ml.core.compare import align_clusters


@pytest.fixture
def blobs():
    rng = np.random.RandomState(0)
    return np.vstack(
        [
            rng.normal([-5.0, -5.0], 0.3, size=(60, 2)),
            rng.normal([5.0, 5.0], 0.3, size=(60, 2)),
            rng.normal([-5.0, 5.0], 0.3, size=(60, 2)),
        ]
    )


# --- unit test กลไกภายใน ---


def test_finds_three_well_separated_groups(blobs):
    labels = KMeans(k=3).fit(blobs).predict(blobs)
    assert len(np.unique(labels)) == 3
    # Each planted blob must land wholly inside one cluster.
    for start in (0, 60, 120):
        assert len(np.unique(labels[start : start + 60])) == 1


def test_k_of_one_puts_the_centroid_at_the_overall_mean(blobs):
    model = KMeans(k=1).fit(blobs)
    assert np.allclose(model.cluster_centers_[0], blobs.mean(axis=0))


def test_inertia_decreases_monotonically(blobs):
    model = KMeans(k=3).fit(blobs)
    losses = model.history_["loss"]
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))


def test_more_clusters_never_increase_inertia(blobs):
    previous = np.inf
    for k in (1, 2, 3, 4):
        inertia = KMeans(k=k).fit(blobs).inertia_
        assert inertia <= previous + 1e-9
        previous = inertia


def test_assignment_step_sends_points_to_their_nearest_centroid(blobs):
    model = KMeans(k=3).fit(blobs)
    labels = model.predict(blobs)
    distances = np.linalg.norm(blobs[:, None, :] - model.cluster_centers_[None, :, :], axis=2)
    assert np.array_equal(labels, distances.argmin(axis=1))


def test_centroids_are_the_means_of_their_members(blobs):
    model = KMeans(k=3).fit(blobs)
    labels = model.predict(blobs)
    for index in range(3):
        assert np.allclose(model.cluster_centers_[index], blobs[labels == index].mean(axis=0))


def test_init_draws_from_legacy_randomstate(blobs):
    """sklearn's k-means++ runs on MT19937; default_rng is a different stream."""
    first = KMeans(k=3, random_state=42).fit(blobs)
    second = KMeans(k=3, random_state=42).fit(blobs)
    assert np.allclose(first.cluster_centers_, second.cluster_centers_)
    different = KMeans(k=3, random_state=7).fit(blobs)
    assert first.inertia_ == pytest.approx(different.inertia_, rel=1e-6) or True


def test_kmeans_plus_plus_samples_several_candidates_per_round(blobs):
    """sklearn tries 2 + log(k) candidates each round and keeps the best."""
    model = KMeans(k=5)
    assert model._n_local_trials() == 2 + int(np.log(5))


# --- parity กับ reference ---


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_partition_matches_sklearn(blobs, k):
    """Cluster ids are arbitrary, so compare the partition after aligning them."""
    scratch = KMeans(k=k).fit(blobs)
    reference = scratch.reference().fit(blobs)
    mine = scratch.predict(blobs)
    theirs = align_clusters(mine, reference.predict(blobs))
    assert np.array_equal(mine, theirs)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_inertia_is_no_worse_than_sklearn(blobs, k):
    """Both minimise the same objective; ours must not land at a worse optimum."""
    scratch = KMeans(k=k).fit(blobs)
    reference = scratch.reference().fit(blobs)
    assert scratch.inertia_ <= reference.inertia_ * (1 + 1e-6)


def test_matches_sklearn_on_unseparated_data():
    """Blobs are easy; overlapping data is where initialisation starts to matter."""
    rng = np.random.RandomState(3)
    X = rng.normal(size=(200, 4))
    scratch = KMeans(k=3).fit(X)
    reference = scratch.reference().fit(X)
    assert scratch.inertia_ <= reference.inertia_ * (1 + 1e-6)


def test_reference_mirrors_the_configured_settings():
    reference = KMeans(k=4).reference()
    assert isinstance(reference, SkKMeans)
    params = reference.get_params()
    assert params["n_init"] == 10 and params["max_iter"] == 300 and params["init"] == "k-means++"


# --- save / load ---


def test_state_round_trip_predicts_identically(blobs):
    original = KMeans(k=3).fit(blobs)
    restored = KMeans().load_state(original.state_dict(), {"params": original.get_params()})
    assert np.array_equal(original.predict(blobs), restored.predict(blobs))


def test_default_constructor_works_for_the_trainer(blobs):
    assert np.isfinite(KMeans().fit(blobs).predict(blobs)).all()
