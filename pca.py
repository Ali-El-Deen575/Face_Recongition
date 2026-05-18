import numpy as np

class PCA:
    def __init__(self):
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None  # shape: (n_components, n_features)

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # Covariance matrix and eigen decomposition
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx].T  # (n_features, n_features) → rows are components

        # Save eigenvalues for reuse
        np.save("eigenvalues.npy", self.eigenvalues)
        np.save("eigenvectors.npy", self.eigenvectors)

    def load(self):
        self.eigenvalues = np.load("eigenvalues.npy")
        self.eigenvectors = np.load("eigenvectors.npy")

    def n_components_for_variance(self, alpha):
        cumulative_variance = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
        return int(np.searchsorted(cumulative_variance, alpha) + 1)

    def project(self, X, alpha):
        k = self.n_components_for_variance(alpha)
        X_centered = X - self.mean
        return X_centered @ self.eigenvectors[:k].T  # (n_samples, k)
