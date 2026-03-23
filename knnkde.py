import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNKDE:
    def __init__(self, k_neighbors=15, metric='manhattan', sigma=1.0,shift = 1.0):
        """
        Shape-agnostic Density Estimator. 
        Works with [B, dim], [H, W, dim], or any [..., dim] shaped tensors.
        """
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.sigma = sigma
        self.shift = shift
        
        self.nn = None

    def _prepare_input(self, X : np.ndarray):
        """
        Flattens any input of shape [..., dim] to [B, dim].
        Returns the flattened array and the original shape for reconstruction.
        """
        if X.ndim==1: X=X[:,None]
        original_shape = X.shape[:-1]  # All dims except the last one (dim)
        feature_dim = X.shape[-1]
        X_flat = X.reshape(-1, feature_dim)
        return X_flat, original_shape

    def _compute_lse_density(self, dists, is_self_query=True):
        """
        Internal math for the LSE density.
        """
        # If querying self, skip the 0-distance column
        k_distances = dists[:, 1:] if is_self_query else dists
        
        if k_distances.shape[1] == 0:
            return np.zeros(dists.shape[0])

        # Aggregate distances using Soft-Max logic
        # log(sum(exp(shift + d/sigma))) - shift
        lse_dist = np.log(np.exp((self.shift + k_distances) / self.sigma).sum(axis=-1)) - self.shift/self.sigma
        
        # Return inverse as density
        return 1.0 / (lse_dist + 1e-8)

    def fit(self, X):
        """
        Fits on a cloud X and computes internal density.
        Accepts any shape [..., dim].
        """
        X_flat, original_shape = self._prepare_input(X)
        n_samples = X_flat.shape[0]

        # Handle edge cases for small clusters
        k_val = min(self.k_neighbors, n_samples - 1)
        if k_val < 1:
            return np.zeros(original_shape)

        self.nn = NearestNeighbors(n_neighbors=k_val + 1, metric=self.metric)
        self.nn.fit(X_flat)

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X_eval):
        """
        Computes density for eval points relative to the fitted cluster.
        Accepts any shape [..., dim], e.g., a grid [100, 100, 2].
        """
        if self.nn is None:
            raise ValueError("KNNKDE must be fitted before calling transform.")

        X_flat, original_shape = self._prepare_input(X_eval)
        
        k_query = min(self.k_neighbors, self.nn.n_samples_fit_)
        
        distances, _ = self.nn.kneighbors(X_flat, n_neighbors=k_query)
        density_flat = self._compute_lse_density(distances, is_self_query=False)
        
        return density_flat.reshape(original_shape)