"""NumPy k-means++/Lloyd and Euclidean Ward/average-linkage scratch cores.

PCA is the existing covariance/eigh preprocessing mechanism, not a clustering algorithm.
"""
import numpy as np
from .preprocessing import _matrix, PCA


def _count(k,n):
    if type(k) is not int or not 1 <= k <= n: raise ValueError('Invalid cluster count')


def nearest_centroid(X,centers):
    X,centers = _matrix(X),_matrix(centers)
    squared = ((X[:,None,:]-centers[None,:,:])**2).sum(axis=2)
    labels = squared.argmin(axis=1)
    return labels,np.sqrt(squared[np.arange(len(X)),labels])


class KMeansClustering:
    def __init__(self,n_clusters=3,max_iter=300,*,n_init=10,tol=1e-6,random_state=42):
        self.n_clusters,self.max_iter,self.n_init,self.tol,self.random_state = n_clusters,max_iter,n_init,tol,random_state
        if type(max_iter) is not int or max_iter<1 or type(n_init) is not int or n_init<1 or not np.isfinite(tol) or tol<0:
            raise ValueError('Invalid KMeans optimization parameters')
        if type(random_state) is not int or not 0<=random_state<2**32: raise ValueError('Invalid seed')

    def _initialize(self,X,rng):
        centers = [X[rng.integers(len(X))].copy()]
        while len(centers)<self.n_clusters:
            squared = ((X[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=2).min(axis=1)
            if squared.sum()<=0: raise ValueError('Too few distinct points')
            centers.append(X[rng.choice(len(X),p=squared/squared.sum())].copy())
        return np.array(centers)

    def fit(self,X):
        X = _matrix(X)
        _count(self.n_clusters,len(X))
        if len(np.unique(X,axis=0))<self.n_clusters: raise ValueError('Too few distinct points')
        rng = np.random.default_rng(self.random_state)
        best = None
        self.restart_inertias_ = []
        for _ in range(self.n_init):
            centers = self._initialize(X,rng)
            history = []
            for iteration in range(1,self.max_iter+1):
                labels,distance = nearest_centroid(X,centers)
                history.append(float(distance@distance))
                updated = centers.copy()
                empty = []
                for k in range(self.n_clusters):
                    members = X[labels==k]
                    if len(members): updated[k] = members.mean(axis=0)
                    else: empty.append(k)
                # Deterministic farthest-point relocation; avoid donating singleton clusters.
                counts = np.bincount(labels,minlength=self.n_clusters)
                for k in empty:
                    eligible = counts[labels]>1
                    donor = int(np.argmax(np.where(eligible,distance,-1)))
                    old = labels[donor]
                    labels[donor] = k
                    counts[old]-=1
                    counts[k]+=1
                    updated[k] = X[donor]
                    updated[old] = X[labels==old].mean(axis=0)
                shift = np.linalg.norm(updated-centers)
                centers = updated
                if shift<=self.tol: break
            labels,distance = nearest_centroid(X,centers)
            inertia = float(distance@distance)
            history.append(inertia)
            if len(np.unique(labels)) != self.n_clusters: raise ValueError('Empty cluster after convergence')
            self.restart_inertias_.append(inertia)
            if best is None or inertia<best[0]: best = (inertia,centers.copy(),labels.copy(),history,iteration)
        self.inertia_,self.cluster_centers_,self.labels_,self.inertia_history_,self.n_iter_ = best
        self.n_features_ = X.shape[1]
        return self

    def predict(self,X):
        if not hasattr(self,'cluster_centers_'): raise ValueError('Model not fitted')
        return nearest_centroid(X,self.cluster_centers_)[0]

    def fit_predict(self,X): return self.fit(X).labels_

    def to_dict(self):
        return {'parameters':dict(n_clusters=self.n_clusters,max_iter=self.max_iter,n_init=self.n_init,tol=self.tol,random_state=self.random_state),
            'centers':self.cluster_centers_.tolist(),'labels':self.labels_.tolist(),'inertia':self.inertia_,
            'history':self.inertia_history_,'restarts':self.restart_inertias_,'n_iter':self.n_iter_}

    @classmethod
    def from_dict(cls,state):
        model = cls(**state['parameters'])
        model.cluster_centers_ = _matrix(state['centers'])
        model.labels_ = np.asarray(state['labels'],dtype=int)
        model.inertia_,model.inertia_history_,model.restart_inertias_,model.n_iter_ = state['inertia'],state['history'],state['restarts'],state['n_iter']
        model.n_features_ = model.cluster_centers_.shape[1]
        return model


class AgglomerativeClustering:
    """Full Euclidean linkage tree; intentionally no standard out-of-sample predict.

    Lance-Williams distances use O(n squared) memory. Per-row nearest distances
    are refreshed only when their neighbour is merged, avoiding repeated full scans.
    Equal distances choose the lowest pair of tree node IDs.
    """
    def __init__(self,n_clusters=3,linkage='ward'):
        if linkage not in ('ward','average'): raise ValueError('Unsupported linkage')
        self.n_clusters,self.linkage = n_clusters,linkage

    def fit(self,X):
        X = _matrix(X)
        n = len(X)
        _count(self.n_clusters,n)
        distance = np.sqrt(np.maximum(((X[:,None,:]-X[None,:,:])**2).sum(axis=2),0))
        np.fill_diagonal(distance,np.inf)
        active = np.ones(n,dtype=bool)
        sizes,ids = np.ones(n,dtype=int),np.arange(n)
        nearest = distance.argmin(axis=1)
        children,heights = [],[]
        members = {i:[i] for i in range(n)}
        partition = dict(members) if self.n_clusters==n else None
        for step in range(n-1):
            indices = np.flatnonzero(active)
            a = min(indices,key=lambda i:(distance[i,nearest[i]],min(ids[i],ids[nearest[i]]),max(ids[i],ids[nearest[i]])))
            b = nearest[a]
            pair = sorted([int(ids[a]),int(ids[b])])
            height = float(distance[a,b])
            children.append(pair)
            heights.append(height)
            rest = indices[(indices!=a)&(indices!=b)]
            na,nb = sizes[a],sizes[b]
            if self.linkage=='average': updated = (na*distance[a,rest]+nb*distance[b,rest])/(na+nb)
            else:
                nr = sizes[rest]
                updated = np.sqrt(np.maximum(((nr+na)*distance[a,rest]**2+(nr+nb)*distance[b,rest]**2-nr*height**2)/(nr+na+nb),0))
            distance[a,rest],distance[rest,a] = updated,updated
            distance[b,:],distance[:,b] = np.inf,np.inf
            distance[a,a] = np.inf
            active[b] = False
            sizes[a] = na+nb
            ids[a] = n+step
            members[a] = members[a]+members.pop(b)
            if len(members)==self.n_clusters: partition = {i:list(v) for i,v in members.items()}
            # Repair stale neighbours, compare all remaining rows with the merged cluster.
            for i in rest:
                if nearest[i] in (a,b):
                    candidates = np.flatnonzero(active)
                    nearest[i] = min(candidates,key=lambda j:(distance[i,j],ids[j]))
                elif distance[i,a]<distance[i,nearest[i]] or (distance[i,a]==distance[i,nearest[i]] and ids[a]<ids[nearest[i]]): nearest[i] = a
            if len(rest): nearest[a] = min(rest,key=lambda j:(distance[a,j],ids[j]))
        self.children_ = np.asarray(children,dtype=int).reshape(-1,2)
        self.distances_ = np.asarray(heights)
        self.labels_ = np.empty(n,dtype=int)
        for k,rows in enumerate(sorted(partition.values(),key=lambda r:min(r))): self.labels_[rows] = k
        self.cluster_centers_ = np.array([X[self.labels_==k].mean(axis=0) for k in range(self.n_clusters)])
        return self

    def fit_predict(self,X): return self.fit(X).labels_

    def to_dict(self):
        return {'parameters':dict(n_clusters=self.n_clusters,linkage=self.linkage),'centers':self.cluster_centers_.tolist(),
                'labels':self.labels_.tolist(),'children':self.children_.tolist(),'distances':self.distances_.tolist()}

    @classmethod
    def from_dict(cls,state):
        model = cls(**state['parameters'])
        model.cluster_centers_ = _matrix(state['centers'])
        model.labels_,model.children_,model.distances_ = np.array(state['labels'],dtype=int),np.array(state['children'],dtype=int),np.array(state['distances'],dtype=float)
        return model


class PCADecomposition:
    """Compatibility adapter: Train standardization followed by scratch covariance PCA."""
    def __init__(self,n_components=2):
        self.n_components = n_components

    def fit(self,X):
        from .preprocessing import Standardizer
        self.scaler_ = Standardizer().fit(X)
        self.pca_ = PCA(self.n_components).fit(self.scaler_.transform(X))
        return self

    def transform(self,X): return self.pca_.transform(self.scaler_.transform(X))
    def fit_transform(self,X): return self.fit(X).transform(X)
