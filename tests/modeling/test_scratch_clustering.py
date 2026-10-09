import builtins
import numpy as np
import pytest
from src.modeling.clustering import KMeansClustering, AgglomerativeClustering


def points():
    return np.array([[0.,0.],[0.,.2],[.1,0.],[5.,5.],[5.,5.2],[5.1,5.]])


def test_kmeans_numpy_core_restarts_history_and_determinism(monkeypatch):
    original = builtins.__import__
    def guard(name,*a,**kw):
        assert not name.startswith('sklearn'), 'scratch imported sklearn'
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guard)
    a = KMeansClustering(2, n_init=10, random_state=7).fit(points())
    b = KMeansClustering(2, n_init=10, random_state=7).fit(points())
    assert len(a.restart_inertias_) == 10
    assert np.all(np.diff(a.inertia_history_) <= 1e-12)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_array_equal(a.predict(points()), a.labels_)
    assert a.inertia_ < .1


@pytest.mark.parametrize('linkage',['ward','average'])
def test_agglomerative_merges_match_reference_partitions(linkage):
    from sklearn.cluster import AgglomerativeClustering as Reference
    from sklearn.metrics import adjusted_rand_score
    X = np.random.default_rng(18).normal(size=(25,3))
    a = AgglomerativeClustering(3, linkage).fit(X)
    b = Reference(n_clusters=3,linkage=linkage).fit(X)
    assert adjusted_rand_score(a.labels_, b.labels_) == 1
    np.testing.assert_array_equal(a.children_, b.children_)
    assert a.children_.shape == (len(X)-1, 2)
    assert not hasattr(a,'predict')


def test_duplicates_empty_cluster_policy_and_invalid_inputs():
    X = np.array([[0.],[0.],[0.],[5.],[5.],[5.]])
    a = KMeansClustering(2).fit(X)
    assert len(np.unique(a.labels_)) == 2 and a.inertia_ == 0
    with pytest.raises(ValueError): KMeansClustering(3).fit(X)
    with pytest.raises(ValueError): AgglomerativeClustering(2,'single').fit(X)
    with pytest.raises(ValueError): KMeansClustering(2).fit(np.array([[np.nan],[1.]]))
