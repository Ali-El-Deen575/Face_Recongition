import numpy as np


class GMM:
    def __init__(self, k: int = 40, max_iters: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6, random_state: int | None = 42):
        self.k = int(k)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.weights = None
        self.means = None
        self.covariances = None

    def fit(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        indices = self._rng.choice(n_samples, self.k, replace=False)
        self.means = X[indices].copy()

        self.covariances = np.array([np.eye(n_features) for _ in range(self.k)])
        self.weights = np.full(self.k, 1.0 / self.k)

        log_likelihood_old = -np.inf

        for i in range(self.max_iters):
            responsibilities = self._e_step(X)

            self._m_step(X, responsibilities)

            log_likelihood_new = self._compute_log_likelihood(X)
            if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

        return np.argmax(self._e_step(X), axis=1)

    def predict(self, X):
        X = np.asarray(X)
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def fit_predict(self, X):
        return self.fit(X)

    def _e_step(self, X):
        n_samples = X.shape[0]
        weighted_log_probs = np.zeros((n_samples, self.k))

        for k in range(self.k):
            weighted_log_probs[:, k] = np.log(self.weights[k] + 1e-10) + self._log_multivariate_normal_density(X, self.means[k], self.covariances[k])

        max_log_probs = np.max(weighted_log_probs, axis=1, keepdims=True)
        log_resp = weighted_log_probs - max_log_probs - np.log(np.sum(np.exp(weighted_log_probs - max_log_probs), axis=1, keepdims=True))
        return np.exp(log_resp)

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        nk = np.sum(responsibilities, axis=0)  # Total responsibility for each cluster

        self.weights = nk / n_samples
        self.means = (responsibilities.T @ X) / nk[:, np.newaxis]

        for k in range(self.k):
            diff = X - self.means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.covariances[k] = (weighted_diff.T @ diff) / nk[k]
            self.covariances[k] += np.eye(n_features) * self.reg_covar

    def _log_multivariate_normal_density(self, X, mean, cov):
        n_features = X.shape[1]
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov_reg = cov + np.eye(n_features) * 1e-4
            chol = np.linalg.cholesky(cov_reg)

        log_det_cov = 2.0 * np.sum(np.log(np.diag(chol)))

        diff = X - mean
        from scipy.linalg import solve_triangular
        sol = solve_triangular(chol, diff.T, lower=True).T
        mahalanobis = np.sum(np.square(sol), axis=1)

        return -0.5 * (n_features * np.log(2 * np.pi) + log_det_cov + mahalanobis)

    def _compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        weighted_log_probs = np.zeros((n_samples, self.k))
        for k in range(self.k):
            weighted_log_probs[:, k] = np.log(self.weights[k] + 1e-10) + self._log_multivariate_normal_density(X, self.means[k], self.covariances[k])

        max_log_probs = np.max(weighted_log_probs, axis=1, keepdims=True)
        log_likelihood = np.sum(max_log_probs + np.log(np.sum(np.exp(weighted_log_probs - max_log_probs), axis=1, keepdims=True)))
        return log_likelihood


def fit_cluster_label_mapping(cluster_ids, y_true):
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


def predict_labels_from_clusters(cluster_ids, mapping, unknown_label: int = -1):
    cluster_ids = np.asarray(cluster_ids)
    return np.array([mapping.get(int(c), unknown_label) for c in cluster_ids])
