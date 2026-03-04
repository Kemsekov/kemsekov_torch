import numpy as np
import torch.nn as nn
import torch
import math
from kemsekov_torch.attention import SelfAttention
from kemsekov_torch.common_modules import Transpose, ChanLayerNorm
from copy import deepcopy
from typing import Optional, Tuple, List, Union
from kemsekov_torch.metrics import r2_score

class TimeSeriesModel(nn.Module):
    def __init__(self,in_dim,out_dim,hid,layers=2,dropout=0.1,heads=8,head_dim=64) -> None:
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.m = nn.Sequential(
            nn.Linear(in_dim,hid),
            Transpose(1,-1),
            # my self attention expects tensors [batch,dim,sequence]
            *[
                nn.Sequential(
                    nn.SiLU(),
                    ChanLayerNorm(hid),
                    SelfAttention(
                        hid,
                        is_causal=True,
                        heads=heads,
                        head_dim=head_dim,
                        add_absolute_pos=True,
                        add_rotary_embedding=True,
                        dimensions=1,
                        dropout=dropout,
                        prenorm=False
                    ),
                    ChanLayerNorm(hid),
                )
            for i in range(layers)  
            ],
            Transpose(1,-1)
        )
        self.fc = nn.Linear(hid,out_dim*2)
    def forward(self,x,return_emb=False):
        out = self.m(x)
        mu,logstd = self.fc(out).chunk(2,-1)
        if return_emb:
            return mu,logstd,out
        return mu,logstd
    
    def autoregressive(self,context,num_gen_steps,max_lagsuence_length):
        if self.in_dim!=self.out_dim:
            raise RuntimeError("Cannot do autoregression when in_dim!=out_dim")
        device = list(self.parameters())[0].device
        context = context.to(device)
        lists_mu = []
        lists_sigma = []

        with torch.no_grad():
            for i in range(num_gen_steps):
                # Forward pass: input [1, seq_len, dim] -> output [1, seq_len, dim*2]
                mu, logstd = self(context[:,-max_lagsuence_length:])
                
                # Extract prediction for the *next* step (the last output vector)
                next_mu = mu[:, -1:, :]      
                next_sigma = logstd[:, -1:, :] 
                
                lists_mu.append(next_mu.cpu())
                lists_sigma.append(next_sigma.exp().cpu())
                
                # Append predicted mean to context for next iteration (Greedy decoding)
                context = torch.cat([context, next_mu], dim=1)
        return lists_mu,lists_sigma
class TimeseriesRegression:
    """
    A probabilistic time series regression model with uncertainty estimation.
    
    Uses negative log-likelihood loss with Gaussian predictions (mu, log_std).
    Includes automatic best-model checkpointing based on test R².
    """
    
    def __init__(
        self,
        prediction_indices: List[int],
        hid: int = 64,
        layers: int = 1,
        dropout=0.0,
        lags: int = 100,
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 32,
        check_each: int = 30,
        sigma_threshold: float = 3.0,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 10,
    ):
        """
        Initialize the Time Series Regression model.
        
        Args:
            prediction_indices: Which columns to predict (None = all)
            hid: Hidden layer size
            layers: Number of hidden layers
            dropout: Dropout probability
            lags: Sequence length for input windows
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            future_steps: Steps to predict into the future
            train_size: Size of training set (None = use all except test_size)
            test_size: Size of test set (from end of data)
            check_each: Evaluate test R² every N epochs
            sigma_threshold: Filter predictions with high uncertainty
            device: Device to train on ('cuda' or 'cpu')
            dtype: Data type for training (torch.bfloat16, torch.float32, etc.)
            seed: Random seed for reproducibility
            model_class: Custom model class (default: TimeSeriesModel)
        """
        torch.manual_seed(seed)
        
        self.out_dim = len(prediction_indices)
        self.hid = hid
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.lags = lags
        self.future_steps = 1
        self.prediction_indices = prediction_indices
        self.check_each = check_each
        self.sigma_threshold = sigma_threshold
        self.device = device
        self.dtype = dtype
        # Training state
        self.model = None
        self.best_params = None
        self.best_test_r2 = -1e10
        self.training_history = []
        self.is_fitted = False
    
    def _normal_log_prob(self, x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Calculate Gaussian log probability."""
        sigma = log_sigma.exp()
        diff = (((x - mu) / sigma.clamp(1e-2)) ** 2)
        log_prob = -0.5 * (math.log(2 * math.pi) + 2 * log_sigma + diff)
        return log_prob
    
    def _get_batch_index(self, x: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
        """Randomly sample batches of sequences from the data."""
        seq_len = x.shape[0]
        assert sequence_length <= seq_len, \
            f"sequence_length ({sequence_length}) must be <= sequence length ({seq_len})"
        
        max_start_idx = seq_len - sequence_length
        start_indices = torch.randint(0, max_start_idx + 1, (batch_size,))
        offsets = torch.arange(sequence_length)
        indices = start_indices.unsqueeze(1) + offsets
        batch = x[indices]
        
        return batch
    
    def r2_on_test_data(self, shift: int = 20,sigma_threshold=None,reduce=True) -> float:
        """
        Evaluate model prediction on test data with a shift.
        
        Only uses predictions with variance smaller than sigma_threshold.
        """
        if sigma_threshold is None:sigma_threshold=self.sigma_threshold
        was_training = self.model.training
        self.model.eval()
        
        r2s = []
        sigmas = []
        mus = []
        shifts = [shift * i for i in range(100)]
        
        for s in shifts:
            ind = len(self.x_train) + s
            if ind + 1 >= self.dataset_size:
                continue
            
            subset = self.x_full[ind:ind + self.lags][None]
            prev = subset[:, :-self.future_steps]
            next_y = subset[:, self.future_steps:, self.prediction_indices]
            
            with torch.no_grad():
                mu, logstd = self.model(prev)
            
            sigma = logstd.exp()
            mask = sigma < self.sigma_threshold
            
            if mask.sum() > 0:
                r2 = float(r2_score(mu[mask], next_y[mask]))
                if r2 != float('inf') and not torch.isnan(torch.tensor(r2)):
                    sigmas.append(logstd.mean().cpu().detach())
                    r2s.append(r2)
                    mus.append(mu.cpu().detach())
        
        if was_training:
            self.model.train()
        
        r2s = torch.tensor(r2s)
        r2s[r2s.isnan() | r2s.isinf()] = 0
        
        r2 = r2s.mean().item() if len(r2s) > 0 else 0.0
        if reduce:
            return r2
        else:
            return r2s,mus,sigmas
    
    def _totensor(self,x):
        if not isinstance(x,torch.Tensor):
            x = torch.tensor(x)
        if x.device!=self.device or x.dtype!=self.dtype:
            x=x.to(device=self.device,dtype=self.dtype)
        return x
    
    def fit(self, data: torch.Tensor|np.ndarray,test_size : int, verbose: bool = True) -> 'TimeseriesRegression':
        """
        Fit the model on the provided data.
        
        Args:
            data: Tensor of shape [SEQLEN, DIM] with all features
            test_size: test size for dataset
            verbose: Whether to print training progress
            
        Returns:
            self
        """
        data = self._totensor(data)
        self.in_dim = data.shape[-1]
        self.test_size=test_size
        self.train_size = len(data)-test_size
        # Store full dataset
        self.x_full = data
        self.dataset_size = len(self.x_full)
        
        # Split train/test
        if self.train_size is None:
            self.train_size = self.dataset_size - self.test_size
        
        self.x_train = self.x_full[-(self.train_size + self.test_size):-self.test_size]
        self.x_test = self.x_full[-self.test_size:]
        
        self.model = TimeSeriesModel(
            self.in_dim, 
            self.out_dim, 
            self.hid, 
            layers=self.layers, 
            dropout=self.dropout
        ).to(device=self.device, dtype=self.dtype)
        
        # Optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), fused=True, lr=self.lr)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.epochs)
        
        # Training loop
        best_test_r2 = -1e10
        best_params = None
        
        for epoch in range(1, self.epochs + 1):
            self.opt.zero_grad(True)
            
            # Sample batch
            batch = self._get_batch_index(self.x_train, self.batch_size, self.lags)
            prev_x = batch[:, :-self.future_steps]
            next_y = batch[:, self.future_steps:, self.prediction_indices]
            
            # Forward pass
            train_mu, logstd = self.model(prev_x)
            
            # Negative log-likelihood loss
            logp = self._normal_log_prob(next_y, train_mu, logstd)
            loss = (-logp).mean()
            
            # Backward pass
            loss.backward()
            self.opt.step()
            self.sch.step()
            
            # Track metrics
            train_r2 = r2_score(train_mu, next_y)
            
            # Periodic evaluation
            if epoch % self.check_each == 0:
                test_r2 = self.r2_on_test_data(shift=20)
                
                if verbose:
                    print(f"Epoch {epoch}\tLoss {loss.item():.3f}\tTrain_R2 {train_r2:.3f}\tTest_R2 {test_r2:.3f}")
                
                self.training_history.append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'train_r2': train_r2.item() if isinstance(train_r2, torch.Tensor) else train_r2,
                    'test_r2': test_r2
                })
                
                # Checkpoint best model
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_params = deepcopy(self.model.state_dict())
                else:
                    # Revert to best params if no improvement
                    if best_params is not None:
                        self.model.load_state_dict(best_params)
        
        # Load best parameters
        if best_params is not None:
            self.model.load_state_dict(best_params)
        
        self.best_test_r2 = best_test_r2
        self.best_params = best_params
        self.is_fitted = True
        
        if verbose:
            print(f"\nTraining complete. Best Test R²: {best_test_r2:.3f}")
        
        return self
    
    def predict(
        self, 
        X: torch.Tensor|np.ndarray,
        return_emb = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make predictions on new data.
        
        Args:
            X: Input tensor of shape [SEQLEN, DIM] or [BATCH, SEQLEN, DIM]
            return_uncertainty: If True, also return log_std (uncertainty)
            
        Returns:
            mu: Predicted mean values
            logstd: Predicted log standard deviation (if return_uncertainty=True)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        was_training = self.model.training
        self.model.eval()
        
        # Ensure correct device/dtype
        X = self._totensor(X)
        
        # Handle different input shapes
        if X.dim() == 2:
            # [SEQLEN, DIM] -> add batch dimension
            X = X[None]
        
        # Split input
        
        with torch.no_grad():
            result = self.model(X,return_emb=return_emb)
        
        if was_training:
            self.model.train()
        
        return result
    
    def evaluate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor,
        sigma_threshold: Optional[float] = None
    ) -> float:
        """
        Evaluate R² score on given data.
        
        Args:
            X: Input features [SEQLEN, DIM]
            y: True targets [SEQLEN, OUT_DIM]
            sigma_threshold: Filter high-uncertainty predictions (default: self.sigma_threshold)
            
        Returns:
            R² score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation. Call fit() first.")
        
        sigma_threshold = sigma_threshold or self.sigma_threshold
        
        was_training = self.model.training
        self.model.eval()
        
        X = self._totensor(X)
        y = self._totensor(y)
        
        if X.dim() == 2:
            X = X[None]
        if y.dim() == 2:
            y = y[None]
        
        prev_x = X[:, :-self.future_steps]
        
        with torch.no_grad():
            mu, logstd = self.model(prev_x)
        
        sigma = logstd.exp()
        mask = sigma < sigma_threshold
        
        if mask.sum() == 0:
            r2 = 0.0
        else:
            r2 = float(r2_score(mu[mask], y[:, self.future_steps:, self.prediction_indices][mask]))
        
        if was_training:
            self.model.train()
        
        return r2
    
    def get_training_history(self) -> List[dict]:
        """Return training history (loss, metrics per checkpoint)."""
        return self.training_history
    
    def get_best_test_r2(self) -> float:
        """Return the best test R² achieved during training."""
        return self.best_test_r2
    
    def save(self, path: str):
        """Save model weights and configuration."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving.")
        
        torch.save({
            'state_dict': self.model.state_dict(),
            'best_params': self.best_params,
            'best_test_r2': self.best_test_r2,
            'training_history': self.training_history,
            'config': {
                'in_dim': self.in_dim,
                'out_dim': self.out_dim,
                'hid': self.hid,
                'layers': self.layers,
                'dropout': self.dropout,
                'epochs': self.epochs,
                'lr': self.lr,
                'batch_size': self.batch_size,
                'lags': self.lags,
                'future_steps': self.future_steps,
                'test_size': self.test_size,
                'prediction_indices': self.prediction_indices,
                'sigma_threshold': self.sigma_threshold,
                'device': self.device,
                'dtype': str(self.dtype)
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'TimeseriesRegression':
        """Load model from saved checkpoint."""
        checkpoint = torch.load(path, map_location=device or 'cpu')
        
        # Reconstruct config
        config = checkpoint['config']
        config['device'] = device or config['device']
        config['dtype'] = torch.bfloat16 if config['dtype'] == 'torch.bfloat16' else torch.float32
        
        model = cls(**config)
        model.model.load_state_dict(checkpoint['state_dict'])
        model.best_params = checkpoint['best_params']
        model.best_test_r2 = checkpoint['best_test_r2']
        model.training_history = checkpoint['training_history']
        model.is_fitted = True
        
        return model
