import numpy as np


class KMeans:
    """Simple from-scratch K-Means clustering.

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iters : int
        Maximum iterations.
    tol : float
        Convergence tolerance on centroid movement (L2 norm).
    random_state : int | None
        Seed for deterministic centroid initialization and empty-cluster reinit.
    """

    def __init__(self, k: int = 20, max_iters: int = 100, tol: float = 1e-4, random_state: int | None = 42):
        self.k = int(k)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.centroids: np.ndarray | None = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples, n_features = X.shape
        if self.k <= 0:
            raise ValueError("k must be positive")
        if n_samples < self.k:
            raise ValueError("k cannot exceed number of samples")

        # 1) Initialize centroids randomly from data points
        random_indices = self._rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()

        clusters = np.zeros(n_samples, dtype=int)

        for _ in range(self.max_iters):
            # 2) Assign each point to closest centroid
            distances = self._compute_distances(X)
            clusters = np.argmin(distances, axis=1)

            # 3) Update centroids
            new_centroids = np.zeros((self.k, n_features), dtype=float)
            for cluster_idx in range(self.k):
                points_in_cluster = X[clusters == cluster_idx]
                if points_in_cluster.size:
                    new_centroids[cluster_idx] = points_in_cluster.mean(axis=0)
                else:
                    # empty cluster -> reinitialize randomly
                    new_centroids[cluster_idx] = X[self._rng.integers(0, n_samples)]

            # Check convergence
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        return clusters

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("fit must be called before predict")
        X = np.asarray(X)
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        return self.fit(X)

    def _compute_distances(self, X):
        # squared euclidean distance using (a-b)^2 = a^2 + b^2 - 2ab
        if self.centroids is None:
            raise ValueError("centroids not initialized")
        x_sq = np.sum(X * X, axis=1, keepdims=True)
        c_sq = np.sum(self.centroids * self.centroids, axis=1, keepdims=True).T
        return x_sq + c_sq - 2 * (X @ self.centroids.T)


def fit_cluster_label_mapping(cluster_ids, y_true):
    """Map each cluster id -> most frequent true label (majority vote)."""
    cluster_ids = np.asarray(cluster_ids)
    y_true = np.asarray(y_true)

    mapping = {}
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        if not np.any(mask):
            continue
        labels, counts = np.unique(y_true[mask], return_counts=True)
        mapping[int(cid)] = int(labels[np.argmax(counts)])
    return mapping


# Alias name expected by the notebook / assignment wording
fit_cluster_label_mapping = fit_cluster_label_mapping


def predict_labels_from_clusters(cluster_ids, mapping, unknown_label: int = -1):
    """Convert cluster ids to labels using mapping dict."""
    cluster_ids = np.asarray(cluster_ids)
    return np.array([mapping.get(int(c), unknown_label) for c in cluster_ids])


