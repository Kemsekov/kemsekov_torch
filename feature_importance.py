import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Optional, Tuple, Dict, Union
from sklearn.preprocessing import StandardScaler
from kemsekov_torch.metrics import r2_score
from kemsekov_torch.train import train_simple

@torch.no_grad()
def calculate_permutation_importance(model, compute_loss_and_metrics, X_test, y_test, 
                                      n_repeats=5, verbose=True, dtype=torch.float32, device='cpu'):
    """Calculate permutation importance for each feature."""
    def totensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.device != device or x.dtype != dtype:
            x = x.to(device=device, dtype=dtype)
        return x
    
    w = list(model.parameters())[0]
    orig_model_device = w.device
    orig_model_dtype = w.dtype
    
    model = model.eval().to(device, dtype=dtype)
    X_test = totensor(X_test)
    y_test = totensor(y_test)
    
    # Get baseline performance
    _, metrics = compute_loss_and_metrics(model, X_test, y_test)
    metric_name = list(metrics.keys())[0]
    baseline_score = metrics[metric_name]
    
    if verbose:
        print(f"Baseline {metric_name}: {baseline_score:.4f}")
    
    importances = []
    num_features = X_test.shape[1]
    
    for i in range(num_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.clone()
            perm = torch.randperm(X_test.shape[0])
            X_perm[:, i] = X_test[perm, i]
            _, p_metrics = compute_loss_and_metrics(model, X_perm, y_test)
            scores.append(p_metrics[metric_name])
        
        avg_score = sum(scores) / n_repeats
        importance = baseline_score - avg_score
        importances.append(importance)
        
        if verbose and (i + 1) % 40 == 0:
            print(f"Processed {i+1}/{num_features} features...")
    
    model.to(orig_model_device, dtype=orig_model_dtype)
    return torch.tensor(importances), metric_name


def compute_loss_and_metric(model, X, y):
    """Standard MSE loss + R² metric."""
    pred = model(X)
    loss = torch.nn.functional.mse_loss(pred, y)
    return loss, {'R2': r2_score(pred, y)}


class MLP(nn.Module):
    """Simple MLP model for regression."""
    def __init__(self, in_dim: int, out_dim: int, hid: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
        )
        self.fc = nn.Linear(hid, out_dim)
    
    def forward(self, x: torch.Tensor, return_emb: bool = False) -> torch.Tensor:
        emb = self.mlp(x)
        if return_emb:
            return self.fc(emb), emb
        return self.fc(emb)

class OptimalFeatureImportance:
    """
    Recursive Feature Elimination using Permutation Importance.
    
    Iteratively trains models, calculates feature importance, and drops 
    the least important features to find the optimal feature subset.
    """
    
    def __init__(
        self,
        hid: int = 128,
        epochs: int = 512,
        lr: float = 1e-3,
        batch_size: int = 32,
        drop_features_per_step: int = 10,
        max_steps: int = 15,
        n_model_init: int = 5,
        n_repeats: int = 5,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        seed: int = 42,
        rollback_K: int = 3,
    ):
        """
        Initialize the feature selection model.
        
        Args:
            hid: Hidden layer size for MLP
            epochs: Number of training epochs per model
            lr: Learning rate
            batch_size: Batch size for training
            test_size: Number of samples to hold out for testing
            drop_features_per_step: How many features to drop each iteration
            max_steps: Maximum number of elimination steps
            n_model_init: Number of random initializations to try per step
            n_repeats: Number of repeats for permutation importance
            device: Device to train on ('cuda' or 'cpu')
            dtype: Data type for training
            verbose: Whether to print progress
            seed: Random seed for reproducibility
            rollback_K: Rollback parameter for training
        """
        torch.manual_seed(seed)
        
        self.hid = hid
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.drop_features_per_step = drop_features_per_step
        self.max_steps = max_steps
        self.n_model_init = n_model_init
        self.n_repeats = n_repeats
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.seed = seed
        self.rollback_K = rollback_K
        
        # Fitted state
        self.is_fitted = False
        self.scaler = None
        self.best_features_id = None
        self.best_features_names = None
        self.best_features_r2 = -1e10
        self.best_fitted_model = None
        self.feature_importance_history = []
        self.all_columns = None
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.DataFrame, np.ndarray],
        X_test: pd.DataFrame,
        y_test: Union[pd.DataFrame, np.ndarray]
    ) -> 'OptimalFeatureImportance':
        """
        Fit the feature selection model.
        
        Args:
            X: Training input DataFrame of shape [samples, features]
            y: Training target DataFrame or ndarray
            X_test: Test input DataFrame (REQUIRED)
            y_test: Test target DataFrame or ndarray (REQUIRED)
            
        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame")
        
        # Store column names
        self.all_columns = list(X.columns)
        
        # Validate X_test has same columns as X
        missing_cols = set(self.all_columns) - set(X_test.columns)
        if missing_cols:
            raise ValueError(f"X_test missing columns: {missing_cols}")
        
        extra_cols = set(X_test.columns) - set(self.all_columns)
        if extra_cols:
            raise ValueError(f"X_test has extra columns not in X: {extra_cols}")
        def totensor(x):
            if isinstance(x,torch.Tensor): return x.float()
            return torch.tensor(x,dtype=torch.float32)
        # Convert training data to numpy and scale (fit scaler on TRAINING data only)
        X_np = X.to_numpy()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_np)
        X_full = totensor(X_scaled)
        y_full = totensor(y)
        
        # Align train shapes
        min_train_len = min(len(X_full), len(y_full))
        X_full = X_full[:min_train_len]
        y_full = y_full[:min_train_len]
        
        # Convert and scale test data using fitted scaler (NO refit!)
        X_test_np = X_test.to_numpy()
        X_test_scaled = self.scaler.transform(X_test_np)
  
        
        self.X_test_scaled = totensor(X_test_scaled)
        self.y_test_tensor = totensor(y_test)
        
        # Align test shapes
        min_test_len = min(len(self.X_test_scaled), len(self.y_test_tensor))
        self.X_test_scaled = self.X_test_scaled[:min_test_len]
        self.y_test_tensor = self.y_test_tensor[:min_test_len]
        
        # Store full scaled training data for potential recalculation
        self.X_full = X_full
        self.y_full = y_full
        
        # Initialize feature tracking
        features_id = list(range(X_full.shape[1]))
        
        # Reset best tracking
        self.best_features_id = None
        self.best_features_names = None
        self.best_features_r2 = -1e10
        self.best_fitted_model = None
        self.feature_importance_history = []
        
        # Recursive Feature Elimination Loop
        for step in range(self.max_steps):
            if len(features_id) <= self.drop_features_per_step:
                if self.verbose:
                    print(f"Step {step}: Stopping - too few features remaining ({len(features_id)})")
                break
            
            # Select current features for both train and test
            X_train_prep = X_full[:, features_id]
            X_test_prep = self.X_test_scaled[:, features_id]
            
            # Step 1: Find best model from multiple initializations
            best_model = None
            best_r2 = -1e10
            
            for i in range(self.n_model_init):
                mlp = MLP(X_train_prep.shape[-1], y_full.shape[-1], hid=self.hid)
                model, metrics = train_simple(
                    mlp,
                    compute_loss_and_metric,
                    X_train_prep,
                    y_full,
                    X_test_prep,
                    self.y_test_tensor,
                    rollback_K=self.rollback_K,
                    epochs=self.epochs,
                    lr=self.lr,
                    device=self.device,
                    dtype=self.dtype,
                    verbose=False
                )
                r2 = metrics['R2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
            
            # Step 2: Calculate feature importance on TEST set
            importances = calculate_permutation_importance(
                best_model,
                compute_loss_and_metric,
                X_test_prep,
                self.y_test_tensor,
                n_repeats=self.n_repeats,
                verbose=False,
                dtype=self.dtype,
                device=self.device
            )
            importance = importances[0]
            
            # Store importance history
            current_feature_names = [self.all_columns[idx] for idx in features_id]
            self.feature_importance_history.append({
                'step': step,
                'features': current_feature_names,
                'importance': importance.cpu().float().numpy(),
                'r2': best_r2
            })
            
            # Update best if this is the best so far (evaluated on TEST set)
            if best_r2 > self.best_features_r2:
                self.best_features_r2 = best_r2
                self.best_features_id = deepcopy(features_id)
                self.best_features_names = deepcopy(current_feature_names)
                self.best_fitted_model = deepcopy(best_model)
            
            if self.verbose:
                print("=" * 20)
                print(f"Step {step}: Features={len(features_id)}, Test R²={best_r2:.4f}")
                print(f"Current Features: {current_feature_names[:10]}{'...' if len(current_feature_names) > 10 else ''}")
            
            # Step 3: Drop least important features
            important_features_ind = (-importance).argsort()[:-self.drop_features_per_step]
            features_id = [features_id[i] for i in important_features_ind]
            
            if len(features_id) == 0:
                break
        
        self.is_fitted = True
        
        if self.verbose:
            print("\n" + "=" * 20)
            print(f"Feature Selection Complete!")
            print(f"Best Test R²: {self.best_features_r2:.4f}")
            print(f"Best Features ({len(self.best_features_id)}): {self.best_features_names}")
        
        return self
    import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Optional, Tuple, Dict, Union
from sklearn.preprocessing import StandardScaler
from kemsekov_torch.metrics import r2_score
from kemsekov_torch.train import train_simple

@torch.no_grad()
def calculate_permutation_importance(model, compute_loss_and_metrics, X_test, y_test, 
                                      n_repeats=5, verbose=True, dtype=torch.float32, device='cpu'):
    """Calculate permutation importance for each feature."""
    def totensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.device != device or x.dtype != dtype:
            x = x.to(device=device, dtype=dtype)
        return x
    
    w = list(model.parameters())[0]
    orig_model_device = w.device
    orig_model_dtype = w.dtype
    
    model = model.eval().to(device, dtype=dtype)
    X_test = totensor(X_test)
    y_test = totensor(y_test)
    
    # Get baseline performance
    _, metrics = compute_loss_and_metrics(model, X_test, y_test)
    metric_name = list(metrics.keys())[0]
    baseline_score = metrics[metric_name]
    
    if verbose:
        print(f"Baseline {metric_name}: {baseline_score:.4f}")
    
    importances = []
    num_features = X_test.shape[1]
    
    for i in range(num_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.clone()
            perm = torch.randperm(X_test.shape[0])
            X_perm[:, i] = X_test[perm, i]
            _, p_metrics = compute_loss_and_metrics(model, X_perm, y_test)
            scores.append(p_metrics[metric_name])
        
        avg_score = sum(scores) / n_repeats
        importance = baseline_score - avg_score
        importances.append(importance)
        
        if verbose and (i + 1) % 40 == 0:
            print(f"Processed {i+1}/{num_features} features...")
    
    model.to(orig_model_device, dtype=orig_model_dtype)
    return torch.tensor(importances), metric_name


def compute_loss_and_metric(model, X, y):
    """Standard MSE loss + R² metric."""
    pred = model(X)
    loss = torch.nn.functional.mse_loss(pred, y)
    return loss, {'R2': r2_score(pred, y)}


class MLP(nn.Module):
    """Simple MLP model for regression."""
    def __init__(self, in_dim: int, out_dim: int, hid: int = 256) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
        )
        self.fc = nn.Linear(hid, out_dim)
    
    def forward(self, x: torch.Tensor, return_emb: bool = False) -> torch.Tensor:
        emb = self.mlp(x)
        if return_emb:
            return self.fc(emb), emb
        return self.fc(emb)

class OptimalFeatureImportance:
    """
    Recursive Feature Elimination using Permutation Importance.
    
    Iteratively trains models, calculates feature importance, and drops 
    the least important features to find the optimal feature subset.
    """
    
    def __init__(
        self,
        hid: int = 128,
        epochs: int = 512,
        lr: float = 1e-3,
        batch_size: int = 32,
        drop_features_per_step: int = 10,
        max_steps: int = 15,
        n_model_init: int = 5,
        n_repeats: int = 5,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        seed: int = 42,
        rollback_K: int = 3,
    ):
        """
        Initialize the feature selection model.
        
        Args:
            hid: Hidden layer size for MLP
            epochs: Number of training epochs per model
            lr: Learning rate
            batch_size: Batch size for training
            test_size: Number of samples to hold out for testing
            drop_features_per_step: How many features to drop each iteration
            max_steps: Maximum number of elimination steps
            n_model_init: Number of random initializations to try per step
            n_repeats: Number of repeats for permutation importance
            device: Device to train on ('cuda' or 'cpu')
            dtype: Data type for training
            verbose: Whether to print progress
            seed: Random seed for reproducibility
            rollback_K: Rollback parameter for training
        """
        torch.manual_seed(seed)
        
        self.hid = hid
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.drop_features_per_step = drop_features_per_step
        self.max_steps = max_steps
        self.n_model_init = n_model_init
        self.n_repeats = n_repeats
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.seed = seed
        self.rollback_K = rollback_K
        
        # Fitted state
        self.is_fitted = False
        self.scaler = None
        self.best_features_id = None
        self.best_features_names = None
        self.best_features_r2 = -1e10
        self.best_fitted_model = None
        self.feature_importance_history = []
        self.all_columns = None
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.DataFrame, np.ndarray],
        X_test: pd.DataFrame,
        y_test: Union[pd.DataFrame, np.ndarray]
    ) -> 'OptimalFeatureImportance':
        """
        Fit the feature selection model.
        
        Args:
            X: Training input DataFrame of shape [samples, features]
            y: Training target DataFrame or ndarray
            X_test: Test input DataFrame (REQUIRED)
            y_test: Test target DataFrame or ndarray (REQUIRED)
            
        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame")
        
        # Store column names
        self.all_columns = list(X.columns)
        
        # Validate X_test has same columns as X
        missing_cols = set(self.all_columns) - set(X_test.columns)
        if missing_cols:
            raise ValueError(f"X_test missing columns: {missing_cols}")
        
        extra_cols = set(X_test.columns) - set(self.all_columns)
        if extra_cols:
            raise ValueError(f"X_test has extra columns not in X: {extra_cols}")
        def totensor(x):
            if isinstance(x,torch.Tensor): return x.float()
            return torch.tensor(x,dtype=torch.float32)
        # Convert training data to numpy and scale (fit scaler on TRAINING data only)
        X_np = X.to_numpy()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_np)
        X_full = totensor(X_scaled)
        y_full = totensor(y)
        
        # Align train shapes
        min_train_len = min(len(X_full), len(y_full))
        X_full = X_full[:min_train_len]
        y_full = y_full[:min_train_len]
        
        # Convert and scale test data using fitted scaler (NO refit!)
        X_test_np = X_test.to_numpy()
        X_test_scaled = self.scaler.transform(X_test_np)
  
        
        self.X_test_scaled = totensor(X_test_scaled)
        self.y_test_tensor = totensor(y_test)
        
        # Align test shapes
        min_test_len = min(len(self.X_test_scaled), len(self.y_test_tensor))
        self.X_test_scaled = self.X_test_scaled[:min_test_len]
        self.y_test_tensor = self.y_test_tensor[:min_test_len]
        
        # Store full scaled training data for potential recalculation
        self.X_full = X_full
        self.y_full = y_full
        
        # Initialize feature tracking
        features_id = list(range(X_full.shape[1]))
        
        # Reset best tracking
        self.best_features_id = None
        self.best_features_names = None
        self.best_features_r2 = -1e10
        self.best_fitted_model = None
        self.feature_importance_history = []
        # Recursive Feature Elimination Loop
        for step in range(self.max_steps):
            if len(features_id) <= self.drop_features_per_step:
                if self.verbose:
                    print(f"Step {step}: Stopping - too few features remaining ({len(features_id)})")
                break
            
            # Select current features for both train and test
            X_train_prep = X_full[:, features_id]
            X_test_prep = self.X_test_scaled[:, features_id]
            
            # Step 1: Find best model from multiple initializations
            best_model = None
            best_r2 = -1e10
            
            for i in range(self.n_model_init):
                mlp = MLP(X_train_prep.shape[-1], y_full.shape[-1], hid=self.hid)
                model, metrics = train_simple(
                    mlp,
                    compute_loss_and_metric,
                    X_train_prep,
                    y_full,
                    X_test_prep,
                    self.y_test_tensor,
                    rollback_K=self.rollback_K,
                    epochs=self.epochs,
                    lr=self.lr,
                    device=self.device,
                    dtype=self.dtype,
                    verbose=False
                )
                r2 = metrics['R2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
            
            
            # new we attach part of train dataset alongside test dataset
            # to make sure that permutation importance
            X_permutation = torch.concat([X_train_prep,X_test_prep],0)[-2*X_test_prep.shape[0]:]
            y_permutation = torch.concat([y_full,self.y_test_tensor],0)[-2*X_test_prep.shape[0]:]
            
            # Step 2: Calculate feature importance on TEST set
            importances = calculate_permutation_importance(
                best_model,
                compute_loss_and_metric,
                X_permutation,
                y_permutation,
                n_repeats=self.n_repeats,
                verbose=False,
                dtype=self.dtype,
                device=self.device
            )
            importance = importances[0]
            
            # Store importance history
            current_feature_names = [self.all_columns[idx] for idx in features_id]
            self.feature_importance_history.append({
                'step': step,
                'features': current_feature_names,
                'importance': importance.cpu().float().numpy(),
                'r2': best_r2
            })
            
            # Update best if this is the best so far (evaluated on TEST set)
            if best_r2 > self.best_features_r2:
                self.best_features_r2 = best_r2
                self.best_features_id = deepcopy(features_id)
                self.best_features_names = deepcopy(current_feature_names)
                self.best_fitted_model = deepcopy(best_model)
            
            if self.verbose:
                print("=" * 20)
                print(f"Step {step}: Features={len(features_id)}, Test R²={best_r2:.4f}")
                print(f"Current Features: {current_feature_names[:10]}{'...' if len(current_feature_names) > 10 else ''}")
            
            # Step 3: Drop least important features
            important_features_ind = (-importance).argsort()[:-self.drop_features_per_step]
            features_id = [features_id[i] for i in important_features_ind]
            
            if len(features_id) == 0:
                break
        
        self.is_fitted = True
        
        if self.verbose:
            print("\n" + "=" * 20)
            print(f"Feature Selection Complete!")
            print(f"Best Test R²: {self.best_features_r2:.4f}")
            print(f"Best Features ({len(self.best_features_id)}): {self.best_features_names}")
        
        return self
    
    def finetune(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.DataFrame, np.ndarray],
        X_test: pd.DataFrame,
        y_test: Union[pd.DataFrame, np.ndarray]
    ) -> 'OptimalFeatureImportance':
        """
        Finetune the best model on provided data using the selected features.
        
        Args:
            X: Training input DataFrame (must have same columns as fit)
            y: Training target DataFrame or ndarray
            X_test: Test input DataFrame (must have same columns as fit)
            y_test: Test target DataFrame or ndarray
            
        Returns:
            self
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before finetuning. Call fit() first.")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame")
        
        # Validate columns (like predict)
        missing_cols = set(self.best_features_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Ensure scaler compatibility (must match fit data structure)
        if set(X.columns) != set(self.all_columns):
            raise ValueError(f"X must have the same columns as the data used in fit().")
        
        # Prepare Data (like predict)
        # Scale
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.best_features_id]
        def totensor(x):
            if isinstance(x,torch.Tensor): return x.float()
            return torch.tensor(x,dtype=torch.float32)
        

        X_tensor = totensor(X_selected)
        y_tensor = totensor(y)
        
        # Align shapes
        min_train_len = min(len(X_tensor), len(y_tensor))
        X_tensor = X_tensor[:min_train_len]
        y_tensor = y_tensor[:min_train_len]
        
        # Test Data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = X_test_scaled[:, self.best_features_id]
        X_test_tensor = totensor(X_test_selected)
        
        y_test_tensor = totensor(y_test)
        
        min_test_len = min(len(X_test_tensor), len(y_test_tensor))
        X_test_tensor = X_test_tensor[:min_test_len]
        y_test_tensor = y_test_tensor[:min_test_len]
        
        # Train
        # Use the existing best model architecture/weights as starting point
        model = self.best_fitted_model
        
        trained_model, metrics = train_simple(
            model,
            compute_loss_and_metric,
            X_tensor,
            y_tensor,
            X_test_tensor,
            y_test_tensor,
            rollback_K=self.rollback_K,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            dtype=self.dtype,
            verbose=self.verbose
        )
        
        # Update State
        self.best_fitted_model = trained_model
        self.best_features_r2 = metrics['R2']
        
        if self.verbose:
            print(f"Finetuning Complete. New Test R²: {self.best_features_r2:.4f}")
            
        return self
    def predict(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> Dict:
        """
        Make predictions using the best feature subset.
        
        Args:
            X: Input DataFrame (must have same columns as fit)
            y: Optional target for evaluation
            
        Returns:
            Dictionary containing predictions, features used, and metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Validate columns
        missing_cols = set(self.best_features_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Select best features and scale (using fitted scaler)
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.best_features_id]
        X_tensor = torch.tensor(X_selected, dtype=torch.float32).to(self.device)
        
        # Ensure model parameters and input are on same device/dtype
        model_params = list(self.best_fitted_model.parameters())[0]
        X_tensor = X_tensor.to(device=model_params.device, dtype=model_params.dtype)
        
        # Make prediction
        self.best_fitted_model.eval()
        with torch.no_grad():
            predictions = self.best_fitted_model(X_tensor)
        
        result = {
            'predictions': predictions.cpu().numpy(),
            'features_used': self.best_features_names,
            'best_r2': self.best_features_r2
        }
        
        # Evaluate if y provided
        if y is not None:
            y_np = y.to_numpy() if isinstance(y, pd.DataFrame) else y
            y_tensor = torch.tensor(y_np, dtype=torch.float32).to(self.device)
            
            # Align shapes
            min_len = min(len(predictions), len(y_tensor))
            pred_trim = predictions[:min_len]
            y_trim = y_tensor[:min_len]
            
            r2 = float(r2_score(pred_trim.cpu().numpy(), y_trim.cpu().numpy()))
            result['r2'] = r2
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the final step.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        # Try to get importance from history for best features
        for hist in reversed(self.feature_importance_history):
            if set(hist['features']) == set(self.best_features_names):
                importance = hist['importance']
                return pd.DataFrame({
                    'feature': self.best_features_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
        
        # Fallback: recalculate on stored test data
        X_test_prep = self.X_test_scaled[:, self.best_features_id]
        
        importances = calculate_permutation_importance(
            self.best_fitted_model,
            compute_loss_and_metric,
            X_test_prep,
            self.y_test_tensor,
            n_repeats=self.n_repeats,
            verbose=False,
            dtype=self.dtype,
            device=self.device
        )
        importance = importances[0].cpu().numpy()
        
        return pd.DataFrame({
            'feature': self.best_features_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
