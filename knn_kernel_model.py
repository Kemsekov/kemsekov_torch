import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import cg, LinearOperator, minres
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

def rbf_kernel(X, Y=None, sigma=1.0):
    """
    Dimension-agnostic RBF kernel.
    
    - Last axis: features
    - Second-to-last axis: samples
    - All preceding axes: batch dimensions
    
    Examples:
        (n,)              → (n, 1) features, treated as 1D samples
        (n, d)            → n samples, d features
        (b, n, d)         → batch of b, each with n samples, d features
        (b1, b2, n, d)    → nested batches, etc.
    """
    if Y is None:
        Y = X
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Handle 1D case by adding feature dimension
    if X.ndim == 1:
        X = X[..., np.newaxis]
    if Y.ndim == 1:
        Y = Y[..., np.newaxis]
    
    # Ensure feature dimension matches
    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(f"Feature dimensions must match: {X.shape[-1]} vs {Y.shape[-1]}")
    
    # Ensure batch dimensions match (all except last two)
    if X.shape[:-2] != Y.shape[:-2]:
        raise ValueError(f"Batch dimensions must match: {X.shape[:-2]} vs {Y.shape[:-2]}")
    
    # Compute squared norms along feature axis
    X_sq = np.sum(X**2, axis=-1, keepdims=True)      # [..., n, 1]
    Y_sq = np.sum(Y**2, axis=-1, keepdims=True)      # [..., m, 1]
    Y_sq = np.moveaxis(Y_sq, -2, -1)                 # [..., 1, m]
    
    # Compute pairwise squared distances
    dist_sq = X_sq + Y_sq - 2 * np.matmul(X, np.moveaxis(Y, -2, -1))  # [..., n, m]
    
    # Numerical stability
    dist_sq = np.maximum(dist_sq, 0)
    
    return np.exp(-dist_sq / (2 * sigma**2))


def get_sparse(partial_kernel: np.ndarray, index: np.ndarray) -> csr_matrix:
    """Convert KNN kernel values into CSR sparse matrix."""
    N, K = index.shape
    if partial_kernel.ndim == 3:
        if partial_kernel.shape[1] == 1:
            partial_kernel = partial_kernel[:, 0, :]
        else:
            raise ValueError(f"Expected shape (N,K) or (N,1,K); got {partial_kernel.shape}")
    if partial_kernel.shape != (N, K):
        raise ValueError(f"Shape mismatch: {partial_kernel.shape} vs expected ({N},{K})")
    
    row = np.repeat(np.arange(N), K)
    col = index.ravel()
    data = partial_kernel.ravel()
    return csr_matrix((data, (row, col)), shape=(N, N))


def solve_sparse_system(K: csr_matrix, Y: np.ndarray, lambda_reg: float = 1e-3,
                       tol: float = 1e-4, maxiter: int = 1000, 
                       use_minres: bool = False) -> np.ndarray:
    """
    Solve KX = Y using robust iterative solver.
    
    Args:
        K: Sparse symmetric matrix (may not be positive definite!)
        Y: Target matrix (N, D)
        lambda_reg: Regularization strength (MUST be >= 1e-3 for KNN kernels)
        use_minres: Use MINRES instead of CG (for indefinite matrices)
    """
    N, D = Y.shape
    
    # CRITICAL: Strong regularization for KNN kernels
    K_reg = K + lambda_reg * eye(N, format='csr')
    
    # Diagonal preconditioner (Jacobi) - NEVER fails, unlike ILU
    diag = K_reg.diagonal()
    if np.any(diag <= 0):
        raise ValueError(
            f"Matrix has non-positive diagonal entries after regularization. "
            f"Min diagonal: {diag.min():.2e}. Increase lambda_reg!"
        )
    M = LinearOperator(K_reg.shape, matvec=lambda x: x / diag)
    
    X = np.empty_like(Y)
    solver = minres if use_minres else cg
    
    for d in range(D):
        x_d, info = solver(K_reg, Y[:, d], M=M, rtol=tol, maxiter=maxiter)
        if info != 0:
            # Fallback to MINRES if CG fails (common for indefinite matrices)
            if not use_minres:
                x_d, info2 = minres(K_reg, Y[:, d], M=M, rtol=tol, maxiter=maxiter)
                if info2 == 0:
                    X[:, d] = x_d
                    continue
            raise RuntimeError(
                f"Solver failed for output dim {d} (info={info}). "
                f"Try increasing lambda_reg (current: {lambda_reg}) "
                f"or enabling use_minres=True."
            )
        X[:, d] = x_d
    return X


class FullKernelModel:
    """Dense RBF kernel regression with explicit kernel matrix."""
    
    def __init__(self, sigma=1.0, lambda_reg=1e-6):
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        self.X_train_ = None
        self.alpha_ = None
        self.kernel_ = None
    
    def fit(self, X, Y):
        """Fit kernel regression model: solve (K + λI)α = Y."""
        self.X_train_ = np.asarray(X)
        Y = np.asarray(Y)
        
        # Compute full dense kernel
        self.kernel_ = rbf_kernel(self.X_train_, sigma=self.sigma)
        
        # Regularize and solve (NEVER use np.linalg.inv)
        K_reg = self.kernel_ + self.lambda_reg * np.eye(len(X))
        self.alpha_ = np.linalg.solve(K_reg, Y)
        return self
    
    def predict(self, X_query=None):
        """Predict outputs for query points (or training points if None)."""
        X_query_shape = X_query.shape
        X_query=X_query.reshape(-1,X_query.shape[-1])
        if X_query is None:
            return self.kernel_ @ self.alpha_
        
        X_query = np.asarray(X_query)
        K_query = rbf_kernel(X_query, self.X_train_, sigma=self.sigma)
        pred = K_query @ self.alpha_
        return pred.reshape(*X_query_shape[:-1],-1)
    
    def score(self, X, Y):
        """Return R² score for predictions on X vs true Y."""
        return r2_score(Y, self.predict(X))

class KnnKernelModel:
    """Production-ready sparse kernel regression with automatic tuning."""
    
    def __init__(self,n_neighbors=32, kernel_sparsity = 0.005, metric='euclidean', lambda_reg='auto', 
                 symmetrize=True, solver_tol=1e-4, solver_maxiter=1000):
        """
        kernel_sparsity: desired minimum of non-neighbor elements in rbf kernel. This parameter helps regularize KNN kernel outputs to be smooth and make results independent of sigma
        """
        self.sigma = 1
        self.n_neighbors = n_neighbors
        self.lambda_reg = lambda_reg  # 'auto' enables adaptive regularization
        self.symmetrize = symmetrize
        self.solver_tol = solver_tol
        self.solver_maxiter = solver_maxiter
        
        self.X_train_ = None
        self.alpha_ = None
        self.sparse_kernel_ = None
        self.nn_ = None
        self._effective_lambda_ = None
        self.metric=metric
        self.rbf_kernel_expected_sparsity=kernel_sparsity
    
    def _auto_lambda(self, K_sparse):
        """Adaptive regularization based on kernel magnitude."""
        # Use 1% of mean diagonal as regularization strength
        diag_mean = np.mean(K_sparse.diagonal())
        return max(1e-4, 0.01 * diag_mean)
    
    def fit(self, X, Y):
        self.X_train_ = np.asarray(X)
        Y = np.asarray(Y)
        
        # Fit KNN index
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.nn_.fit(self.X_train_)
        nn_ind = self.nn_.kneighbors(self.X_train_, return_distance=False)
        
        mean_farthest_l2_distance_sq = ((X[nn_ind[:,0]]-X[nn_ind[:,-1]])**2).mean()
        self.sigma*=(-mean_farthest_l2_distance_sq/(2*np.log(self.rbf_kernel_expected_sparsity)))**0.5
        
        # Compute sparse kernel
        knn_kernel = rbf_kernel(
            self.X_train_[:, None, :], 
            self.X_train_[nn_ind], 
            sigma=self.sigma
        )
        self.sparse_kernel_ = get_sparse(knn_kernel, nn_ind)
        
        if self.symmetrize:
            self.sparse_kernel_ = self.sparse_kernel_.maximum(self.sparse_kernel_.T)
        
        # Adaptive regularization
        if self.lambda_reg == 'auto':
            self._effective_lambda_ = self._auto_lambda(self.sparse_kernel_)
        else:
            self._effective_lambda_ = self.lambda_reg
        
        # Solve with robust diagonal preconditioning
        self.alpha_ = solve_sparse_system(
            self.sparse_kernel_, 
            Y, 
            lambda_reg=self._effective_lambda_,
            tol=self.solver_tol,
            maxiter=self.solver_maxiter,
            use_minres=False  # CG usually sufficient with proper reg
        )
        return self
    
    def predict(self, X_query=None):
        X_query_shape = X_query.shape
        X_query=X_query.reshape(-1,X_query.shape[-1])
        if X_query is None:
            return self.sparse_kernel_.dot(self.alpha_)
        
        X_query = np.asarray(X_query)
        # n_queries = X_query.shape[0]
        # n_outputs = self.alpha_.shape[1]
        
        # Vectorized neighbor lookup + prediction
        nn_ind = self.nn_.kneighbors(X_query,return_distance=False)
        K_query = rbf_kernel(
            X_query[:, None, :], 
            self.X_train_[nn_ind], 
            sigma=self.sigma
        ).squeeze(axis=1)
        
        # Fully vectorized prediction (no Python loops!)
        pred = np.einsum('qk,qkd->qd', K_query, self.alpha_[nn_ind])
        return pred.reshape(*X_query_shape[:-1],-1)
    
    def score(self, X, Y):
        return r2_score(Y, self.predict(X))
    