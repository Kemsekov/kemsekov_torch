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
        test_size: int = 128,
        drop_features_per_step: int = 10,
        max_steps: int = 15,
        n_model_init: int = 5,
        n_repeats: int = 5,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        seed: int = 42,
        rollback_K: int = 3,
        prediction_indices: Optional[List[int]] = None
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
            prediction_indices: Which columns to predict (None = all)
        """
        torch.manual_seed(seed)
        
        self.hid = hid
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.test_size = test_size
        self.drop_features_per_step = drop_features_per_step
        self.max_steps = max_steps
        self.n_model_init = n_model_init
        self.n_repeats = n_repeats
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.seed = seed
        self.rollback_K = rollback_K
        self.prediction_indices = prediction_indices
        
        # Fitted state
        self.is_fitted = False
        self.scaler = None
        self.best_features_id = None
        self.best_features_names = None
        self.best_features_r2 = -1e10
        self.best_fitted_model = None
        self.feature_importance_history = []
        self.all_columns = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame|np.ndarray] = None) -> 'OptimalFeatureImportance':
        """
        Fit the feature selection model.
        
        Args:
            X: Input DataFrame of shape [samples, features]
            y: Target DataFrame or ndarray (optional, if None uses X columns)
            
        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Store column names
        self.all_columns = list(X.columns)
        
        # Convert to numpy and scale
        X_np = X.to_numpy()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_np)
        X_full = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Handle target
        if y is not None:
            y_np = np.array(y)
            y_full = torch.tensor(y_np, dtype=torch.float32)
            if self.prediction_indices is None:
                self.prediction_indices = list(range(y_full.shape[1]))
        else:
            # Use X columns as target (auto-regressive)
            if self.prediction_indices is None:
                self.prediction_indices = list(range(X_full.shape[1]))
            y_full = X_full[1:, self.prediction_indices]
            X_full = X_full[:-1]
        
        # Align shapes if y was provided separately
        if y is not None and len(y_full) > len(X_full):
            y_full = y_full[:len(X_full)]
        elif y is not None and len(y_full) < len(X_full):
            X_full = X_full[:len(y_full)]
        
        # Store for later
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
            
            # Select current features
            X_prep = X_full[:, features_id]
            X_train = X_prep[:-self.test_size]
            y_train = y_full[:-self.test_size]
            X_test = X_prep[-self.test_size:]
            y_test = y_full[-self.test_size:]
            
            # Step 1: Find best model from multiple initializations
            best_model = None
            best_r2 = -1e10
            
            for i in range(self.n_model_init):
                mlp = MLP(X_train.shape[-1], y_train.shape[-1], hid=self.hid)
                model, metric = train_simple(
                    mlp,
                    compute_loss_and_metric,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    rollback_K=self.rollback_K,
                    epochs=self.epochs,
                    lr=self.lr,
                    device=self.device,
                    dtype=self.dtype,
                    verbose=False
                )
                r2 = metric['R2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
            
            # Step 2: Calculate feature importance
            importances = calculate_permutation_importance(
                best_model,
                compute_loss_and_metric,
                X_test,
                y_test,
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
            
            # Update best if this is the best so far
            if best_r2 > self.best_features_r2:
                self.best_features_r2 = best_r2
                self.best_features_id = deepcopy(features_id)
                self.best_features_names = deepcopy(current_feature_names)
                self.best_fitted_model = deepcopy(best_model)
            
            if self.verbose:
                print("=" * 20)
                print(f"Step {step}: Features={len(features_id)}, Best R²={best_r2:.4f}")
                print(f"Current Features: {current_feature_names[:10]}{'...' if len(current_feature_names) > 10 else ''}")
            
            # Step 3: Drop least important features
            # argsort ascending (worst first), keep top features
            important_features_ind = (-importance).argsort()[:-self.drop_features_per_step]
            features_id = [features_id[i] for i in important_features_ind]
            
            if len(features_id) == 0:
                break
        
        self.is_fitted = True
        
        if self.verbose:
            print("\n" + "=" * 20)
            print(f"Feature Selection Complete!")
            print(f"Best R²: {self.best_features_r2:.4f}")
            print(f"Best Features ({len(self.best_features_id)}): {self.best_features_names}")
        
        return self
    
    def predict(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> Dict:
        """
        Make predictions using the best feature subset.
        
        Args:
            X: Input DataFrame (must have same columns as fit)
            y: Optional target for evaluation
            
        Returns:
            Dictionary containing:
                - predictions: Model predictions
                - features_used: List of feature names used
                - r2: R² score if y provided
                - importance: Feature importance scores if available
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Validate columns
        missing_cols = set(self.best_features_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Select best features and scale
        X_selected = X[self.best_features_names]
        X_scaled = self.scaler.transform(X_selected.to_numpy())
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device, dtype=self.dtype)
        
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
            y_np = y.to_numpy()
            y_tensor = torch.tensor(y_np, dtype=torch.float32).to(self.device, dtype=self.dtype)
            
            # Align shapes
            if len(y_tensor) > len(X_tensor):
                y_tensor = y_tensor[:len(X_tensor)]
            elif len(y_tensor) < len(X_tensor):
                predictions = predictions[:len(y_tensor)]
            
            r2 = float(r2_score(predictions, y_tensor))
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
        
        # Get importance from last recorded step with best features
        for hist in reversed(self.feature_importance_history):
            if set(hist['features']) == set(self.best_features_names):
                importance = hist['importance']
                break
        else:
            # Fallback: recalculate
            X_prep = self.X_full[:, self.best_features_id]
            X_test = X_prep[-self.test_size:]
            y_test = self.y_full[-self.test_size:]
            
            importances = calculate_permutation_importance(
                self.best_fitted_model,
                compute_loss_and_metric,
                X_test,
                y_test,
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
    
    def get_selection_history(self) -> pd.DataFrame:
        """
        Get the history of feature selection across all steps.
        
        Returns:
            DataFrame with step, num_features, and best_r2
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        history = []
        for h in self.feature_importance_history:
            history.append({
                'step': h['step'],
                'num_features': len(h['features']),
                'r2': h['r2']
            })
        
        return pd.DataFrame(history)
    
    def get_best_features(self) -> List[str]:
        """Return the list of best feature names."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.best_features_names
    
    def save(self, path: str):
        """Save the model state."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving.")
        
        torch.save({
            'best_model_state': self.best_fitted_model.state_dict(),
            'best_features_id': self.best_features_id,
            'best_features_names': self.best_features_names,
            'best_features_r2': self.best_features_r2,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'config': {
                'hid': self.hid,
                'epochs': self.epochs,
                'lr': self.lr,
                'test_size': self.test_size,
                'drop_features_per_step': self.drop_features_per_step,
                'max_steps': self.max_steps,
                'n_model_init': self.n_model_init,
                'n_repeats': self.n_repeats,
                'device': self.device,
                'dtype': str(self.dtype),
                'all_columns': self.all_columns,
                'prediction_indices': self.prediction_indices
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'OptimalFeatureImportance':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device or 'cpu')
        config = checkpoint['config']
        
        # Reconstruct dtype
        config['dtype'] = torch.bfloat16 if config['dtype'] == 'torch.bfloat16' else torch.float32
        if device:
            config['device'] = device
        
        model = cls(**config)
        
        # Restore scaler
        model.scaler = StandardScaler()
        model.scaler.mean_ = checkpoint['scaler_mean']
        model.scaler.scale_ = checkpoint['scaler_scale']
        model.scaler.n_features_in_ = len(checkpoint['scaler_mean'])
        
        # Restore best model
        model.best_fitted_model = MLP(
            len(checkpoint['best_features_id']),
            len(config['prediction_indices']),
            hid=config['hid']
        )
        model.best_fitted_model.load_state_dict(checkpoint['best_model_state'])
        model.best_fitted_model.to(config['device'], dtype=config['dtype'])
        
        # Restore state
        model.best_features_id = checkpoint['best_features_id']
        model.best_features_names = checkpoint['best_features_names']
        model.best_features_r2 = checkpoint['best_features_r2']
        model.all_columns = config['all_columns']
        model.is_fitted = True
        
        return model