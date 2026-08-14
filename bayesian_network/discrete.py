import torch
import torch.nn as nn
from typing import List,Optional,Union
from kemsekov_torch.bayesian_network.common import (
    Prod,
    Residual,
    resolve_device,
    resolve_dtype,
    make_chain_bn,
    _StructureBase,
)
from kemsekov_torch.common_modules import zero_module


class Generative(nn.Module):
    def __init__(self, dim : int,hid_dim=32,bins=16,hid_residuals=2):
        super().__init__()
        #accept input dim + mask of same length
        self.expand = nn.Linear(dim*2,hid_dim)
        self.mlp = nn.Sequential(*[
            *[Residual((
                nn.RMSNorm(hid_dim),
                Prod((
                    nn.Linear(hid_dim,hid_dim,bias=False),
                    nn.ELU()
                )),
                nn.SiLU(),
                zero_module(nn.Linear(hid_dim,hid_dim)),
            )) for i in range(hid_residuals)],
            nn.RMSNorm(hid_dim),
            nn.SiLU(),
        ])
        # output log-probability table.
        self.out = nn.Linear(hid_dim,bins)
    def forward(self,x,mask):
        """
        Assume we inference P(X0|X1,X2) for x=[X0,X1,X2,X3], then
        
        1. x contains only known values
        x=[0,X1,X2,0]
        
        2. mask defines roles of dimensions
        if X_i is what we need to infer, then mask[i]=1
        if X_j is known condition, mask[j]=-1
        else mask=0
        mask=[1,-1,-1,0]
        
        """
        c = torch.concat([x,mask],-1)
        dim = x.shape[-1]
        c_slice = c[:,:dim]
        
        # hide unknown data
        c_slice[mask==1]=0
        c_slice[mask==0]=0
        
        x = self.expand(c)
        x = self.mlp(x)+x
        
        # result is log-probability table for X0 values
        # ln P(X0|X1,X2)
        probs = self.out(x)
        return probs

class LinearInterpolation:
    """
    Accepts a unique grid of shape [B, bins] per function,
    and points of shape [B, bins] defining B distinct functions.
    """
    def __init__(self, grid, points):
        """
        grid:   [B, bins] (sorted float tensor per batch function)
        points: [B, bins] (values evaluated at the grid points)
        """
        if grid.shape != points.shape:
            raise ValueError(f"grid shape {grid.shape} must match points shape {points.shape}")
        self.grid = grid
        self.points = points

    def __call__(self, y):
        """
        Interpolates points at continuous coordinates y.

        y:      [K, B] (K inference query sets, each evaluating all B functions)
        Returns: [K, B] tensor of interpolated values
        """
        
        if y.ndim==1:
            y=y.unsqueeze(-1)
        
        K, B = y.shape
        _, bins = self.grid.shape
        
        # 1. Clamp y to each specific function's grid boundaries
        # grid[:, 0] and grid[:, -1] have shape [B]
        # Adding unsqueeze(0) broadcasts them to [1, B] to match y [K, B]
        grid_min = self.grid[:, 0].unsqueeze(0)
        grid_max = self.grid[:, -1].unsqueeze(0)
        y_clamped = torch.clamp(y, grid_min, grid_max)
        
        # 2. Find indices via vectorized 2D broadcasting comparison
        # y_clamped:   [K, B]    -> unsqueeze(-1) -> [K, B, 1]
        # self.grid:   [B, bins] -> unsqueeze(0)  -> [1, B, bins]
        # Mask shape:  [K, B, bins]
        mask = self.grid.unsqueeze(0) <= y_clamped.unsqueeze(-1)
        
        # Summing along the bins dimension gives the count of elements <= y
        idx_L = torch.sum(mask, dim=-1) - 1
        idx_L = torch.clamp(idx_L, 0, bins - 2) # Shape: [K, B]
        idx_R = idx_L + 1
        
        # 3. Create batch indices for advanced indexing
        # We need an indexing helper for the B dimension that matches the [K, B] structure
        # batch_b maps the corresponding function index 0 to B-1 for every K query row
        batch_b = torch.arange(B, device=y.device).unsqueeze(0).expand(K, B)
        
        # 4. Gather grid and point values using [batch_b, index] shapes
        grid_L = self.grid[batch_b, idx_L]     # Shape: [K, B]
        grid_R = self.grid[batch_b, idx_R]     # Shape: [K, B]
        
        points_L = self.points[batch_b, idx_L] # Shape: [K, B]
        points_R = self.points[batch_b, idx_R] # Shape: [K, B]
        
        # 5. Calculate weights
        denom = grid_R - grid_L
        denom = torch.where(denom == 0, torch.ones_like(denom), denom) 
        
        weight_R = (y_clamped - grid_L) / denom
        weight_L = 1.0 - weight_R
        
        # 6. Linearly interpolate
        return weight_L * points_L + weight_R * points_R

class CubicInterpolation:
    """
    Accepts a unique grid of shape [B, bins] per function,
    and points of shape [B, bins] defining B distinct functions.
    Uses natural cubic spline interpolation.
    """
    def __init__(self, grid, points):
        """
        grid:   [B, bins] (sorted float tensor per batch function)
        points: [B, bins] (values evaluated at the grid points)
        """
        if grid.shape != points.shape:
            raise ValueError(f"grid shape {grid.shape} must match points shape {points.shape}")
        self.grid = grid
        self.points = points
        self._compute_spline_coefficients()

    def _compute_spline_coefficients(self):
        B, N = self.grid.shape
        # Fallback to linear if not enough points for cubic spline
        self.is_linear = N < 4
        if self.is_linear:
            return
            
        M = N - 2
        
        # h_i = x_{i+1} - x_i
        h = self.grid[:, 1:] - self.grid[:, :-1] # [B, N-1]
        h = torch.where(h == 0, torch.ones_like(h), h) # Protect from division by zero
        
        # Build the tridiagonal matrix A of shape [B, M, M] for the second derivatives
        A = torch.zeros((B, M, M), device=self.grid.device)
        i = torch.arange(M, device=self.grid.device)
        
        # Main diagonal: 2*(h_{i-1} + h_i)
        A[:, i, i] = 2 * (h[:, i] + h[:, i+1])
        
        # Off-diagonals: h_i
        if M > 1:
            i_off = torch.arange(M - 1, device=self.grid.device)
            A[:, i_off, i_off+1] = h[:, i_off+1]
            A[:, i_off+1, i_off] = h[:, i_off+1]
            
        # Right hand side vector d of shape [B, M]
        d = 3 * ((self.points[:, 2:] - self.points[:, 1:-1]) / h[:, 1:] - 
                 (self.points[:, 1:-1] - self.points[:, :-2]) / h[:, :-1])
                 
        # Solve A * c_inner = d natively. torch.linalg.solve handles batches and is fully differentiable.
        c_inner = torch.linalg.solve(A, d.unsqueeze(-1)).squeeze(-1)
                
        # c represents the second derivatives divided by 2. Natural spline boundary conditions: c_0 = 0, c_{N-1} = 0
        c = torch.zeros((B, N), device=self.grid.device)
        c[:, 1:-1] = c_inner
            
        # Compute polynomial coefficients for each interval [x_i, x_{i+1}]
        # S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3
        self.a_coeff = self.points[:, :-1] # [B, N-1]
        self.c_coeff = c[:, :-1]           # [B, N-1]
        c_next = c[:, 1:]                  # [B, N-1]
        
        self.d_coeff = (c_next - self.c_coeff) / (3 * h)
        self.b_coeff = (self.points[:, 1:] - self.points[:, :-1]) / h - h * (2 * self.c_coeff + c_next) / 3
        self.h = h

    def __call__(self, y):
        """
        Interpolates points at continuous coordinates y.

        y:      [K, B] (K inference query sets, each evaluating all B functions)
        Returns: [K, B] tensor of interpolated values
        """
        if y.ndim == 1:
            y = y.unsqueeze(-1)
            
        K, B = y.shape
        _, N = self.grid.shape
        
        # 1. Clamp y to each specific function's grid boundaries
        grid_min = self.grid[:, 0].unsqueeze(0)
        grid_max = self.grid[:, -1].unsqueeze(0)
        y_clamped = torch.clamp(y, grid_min, grid_max)
        
        # 2. Find indices via vectorized 2D broadcasting comparison
        mask = self.grid.unsqueeze(0) <= y_clamped.unsqueeze(-1)
        idx_L = torch.sum(mask, dim=-1) - 1
        idx_L = torch.clamp(idx_L, 0, N - 2) # Shape: [K, B]
        
        # 3. Create batch indices for advanced indexing
        batch_b = torch.arange(B, device=y.device).unsqueeze(0).expand(K, B)
        
        # Fallback to linear if bins < 4
        if getattr(self, 'is_linear', False):
            idx_R = idx_L + 1
            grid_L = self.grid[batch_b, idx_L]
            grid_R = self.grid[batch_b, idx_R]
            points_L = self.points[batch_b, idx_L]
            points_R = self.points[batch_b, idx_R]
            
            denom = grid_R - grid_L
            denom = torch.where(denom == 0, torch.ones_like(denom), denom) 
            
            weight_R = (y_clamped - grid_L) / denom
            weight_L = 1.0 - weight_R
            return weight_L * points_L + weight_R * points_R

        # 4. Gather precomputed cubic spline coefficients
        a = self.a_coeff[batch_b, idx_L]
        b = self.b_coeff[batch_b, idx_L]
        c = self.c_coeff[batch_b, idx_L]
        d = self.d_coeff[batch_b, idx_L]
        x_L = self.grid[batch_b, idx_L]
        
        # 5. Evaluate cubic spline polynomial
        dx = y_clamped - x_L
        return a + b * dx + c * (dx ** 2) + d * (dx ** 3)

Interpolation=LinearInterpolation


class Structure(_StructureBase):
    def __init__(self,dataset,bayesian_network="all",bins=32,hid_dim=64,hid_residuals=2,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Union[str, torch.dtype] = "fp32",
                 verbose=False):
        """
        bayesian_network:
            - 'all' (default): train on all possible condition combinations,
            - list[list[int]]: explicit structure [[target, parents...], ...],
            - None: use a sequential chain-rule structure (each variable
              depends on all variables with a higher index).
        device: 'cuda', 'cpu', 'mps' or torch.device. Defaults to the best available.
        dtype:  compute dtype for training and inference:
                - 'fp32' : full float32 (default)
                - 'fp16' : half precision via AMP autocast + grad scaling (CUDA)
                - 'bf16' : bfloat16 via AMP autocast (CUDA/CPU)
        """
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        self.verbose = verbose
        if not isinstance(dataset, torch.Tensor):
            dataset = torch.tensor(dataset, dtype=torch.float32)
        dataset = dataset.to(device=self.device, dtype=torch.float32)
        self.dim = dataset.shape[-1]
        if bayesian_network is None:
            bayesian_network = make_chain_bn(self.dim)
        self.bayesian_network = bayesian_network
        self.model = Generative(self.dim,hid_dim,bins=bins,hid_residuals=hid_residuals)
        self.model = self.model.to(device=self.device)
        self.raw_dataset = dataset
        self.set_bins(bins)
        # lazily populated cuda-graph caches (fit / forward per batch size)
        self._fit_graph = None
        self._fwd_graphs = {}
        if self.verbose:
            print(f"Structure on {self.device} (dtype={self.dtype})")
    
    def forward(self,batch,mask,log_softmax=False):
        modelled_variable=(mask == 1).long().argmax(dim=-1)
        pred = self.model(batch,mask)
        grids = self.grids[modelled_variable]
        if log_softmax: pred = pred.log_softmax(-1)
        return Interpolation(grids,pred)

    def conditional_dist(self,condition : torch.Tensor,variables: List[int],log_softmax=True):
        """
        Return conditional distribution over provided condition variables.
        You can condition by any variables(it may even not be learned).
        
        condition: 
            tensor of shape `[BATCH,D]`
            
        variables: 
            list of length `(D+1)`, where `variable_ind[0]` is index of variable that
            you want to get probability dist, and `variable_ind[1:]` is indices of `condition` dimensions
            relative to input dataset
        """
        
        if not isinstance(condition,torch.Tensor): condition=torch.tensor(condition,dtype=torch.float32)
        condition = condition.to(device=self.device,dtype=torch.float32)
        if condition.ndim==1:condition=condition.unsqueeze(0)
        
        inp = torch.zeros((condition.shape[0],self.dim),device=self.device,dtype=torch.float32)
        inp[:,variables[1:]]=condition
        infer_ind = variables[0]
        mask=torch.zeros_like(inp)
        mask[:,infer_ind]=1
        mask[:,variables[1:]]=-1
        B = condition.shape[0]
        fwd = self._get_fwd_graph(B)
        if fwd is not None:
            points, _ = fwd.replay(inp, mask)
        else:
            with self._amp():
                points = self.model.forward(inp,mask)
            points = points.float()
        grid = self.grids[infer_ind]
        grid = grid.float().expand_as(points)
        if log_softmax: points=points.log_softmax(-1)
        return Interpolation(grid,points)
