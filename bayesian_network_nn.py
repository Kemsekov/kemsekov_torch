import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Literal
from kemsekov_torch.common_modules import get_optim_groups

class Quantize:
    """
    This is small convenience tool for converting continuous data into discrete
    representation and inverse
    """
    def __init__(self,data : torch.Tensor,bins=32):
        # data of shape [batch,dim]
        if not isinstance(data,torch.Tensor):
            data = torch.tensor(data)
        self.center = data.mean(0)    
        self.scale = data.std(0)*2.5
        
        # protect from zero features
        self.scale[self.scale==0]=1 
        
        self.bins=bins
    def quantize(self,x: torch.Tensor,dimensions : List[int]):
        if not isinstance(x,torch.Tensor):x=torch.tensor(x)
        # x is some subset of data features, x=data[:,dimensions]
        centers = self.center[dimensions].unsqueeze(0)
        scales = self.scale[dimensions].unsqueeze(0)
        normalized = ((x-centers)/scales+1)/2
        quantized = torch.floor(normalized * self.bins).clamp(0, self.bins - 1).long()
        return quantized
    
    def dequantize(self,q:torch.Tensor,dimensions : List[int]):
        if not isinstance(q,torch.Tensor):q=torch.tensor(q)
        centers = self.center[dimensions].unsqueeze(0)
        scales = self.scale[dimensions].unsqueeze(0)
        symmetric = ((q+0.5)/self.bins)*2-1
        denorm = symmetric*scales+centers
        return denorm

class Prod(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.m=module
    def forward(self,x):
        return x*self.m(x)

class Generative(nn.Module):
    def __init__(self, dim : int,hid_dim=32,bins=16):
        super().__init__()
        #accept input dim + mask of same length
        self.expand = nn.Linear(dim*2,hid_dim)
        self.mlp = nn.Sequential(*[
            nn.RMSNorm(hid_dim),
            Prod(nn.Sequential(
                nn.Linear(hid_dim,hid_dim),
                nn.Tanh()
            )),
            nn.SiLU(),
            nn.Linear(hid_dim,hid_dim),
            nn.RMSNorm(hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim,hid_dim),
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

class Interpolation:
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

class Structure:
    def __init__(self,dataset,bayesian_network,bins,hid_dim):
        self.quantize = Quantize(dataset,bins=bins)
        self.bayesian_network=bayesian_network
        self.dim = dataset.shape[-1]
        self.model = Generative(self.dim,hid_dim,bins=bins)
        self.dataset=self.quantize.dequantize(self.quantize.quantize(dataset,list(range(self.dim))),list(range(self.dim)))
        self.bins = bins
        
        grids = []
        for infer_ind in range(self.dim):
            grid = self.quantize.dequantize(torch.arange(self.bins)[None,:],[infer_ind])
            grids.append(grid)
        self.grids=torch.concat(grids)
        
    def fit(self,epochs=2048,batch_size=256,lr=0.01,loss_function : Literal['cross_entropy','mle'] = 'cross_entropy'):
        opt = torch.optim.AdamW(get_optim_groups(self.model),lr=lr,fused=True)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
        dataset=self.dataset
        bayesian_network=self.bayesian_network
        
        is_bayesian_specified = bayesian_network is not None and len(bayesian_network)>0
        
        running = torch.arange(batch_size)
        for i in range(epochs):
            batch = dataset[torch.randperm(len(dataset))[:batch_size]]
            # now we must create random masks out of provided bayesian network
            # for P(X0|X1,X2) with X=[X0,X1,X2,X3]
            # mask=[1,-1,-1,0]
            mask = torch.zeros_like(batch)
            modelled_variable = torch.zeros(batch_size,dtype=torch.long)
            
            if is_bayesian_specified:
                size = batch_size//len(bayesian_network)+1
                for ind,imp in enumerate(bayesian_network):
                    part = batch_size*ind//len(bayesian_network)
                    mask_slice = mask[part:part+size]
                    mask_slice[:,imp[0]]=1
                    for cond_var in imp[1:]:
                        mask_slice[:,cond_var]=-1
            else:
                conditional_mask = torch.rand_like(mask)<0.5
                mask[conditional_mask]=-1
                mask[running,modelled_variable]=1
            opt.zero_grad(True)
            modelled_variable=(mask == 1).long().argmax(dim=-1)

            #why this version of loss does not work?
            #============================
            if loss_function=='mle':
                y=batch[running,modelled_variable]
                prob = self.forward(batch,mask,log_softmax=True)
                loss = (-prob(y.unsqueeze(0))).mean()
            #============================
            else:
                pred = self.model(batch,mask)
                expected_ind = self.quantize.quantize(batch,list(range(self.dim)))[running,modelled_variable]
                loss = F.cross_entropy(pred, expected_ind)
            #============================
            loss.backward()
            opt.step()
            sch.step()
            print(f"Loss:{loss:0.3f}")
    
    def forward(self,batch,mask,log_softmax=False):
        modelled_variable=(mask == 1).long().argmax(dim=-1)
        pred = self.model(batch,mask)
        grids = self.grids[modelled_variable]
        if log_softmax: pred = pred.log_softmax(-1)
        return Interpolation(grids,pred)
    
    @torch.no_grad()
    def generate(self, batch_size=128):
        dim = self.dim # Number of features
        device = next(self.model.parameters()).device
        
        # Store our generated continuous values
        samples = torch.zeros((batch_size, dim), device=device)
        bayesian_network=self.bayesian_network
        
        if bayesian_network is None:
            all=list(range(self.dim))
            bayesian_network=[all[-p-1:] for p in range(self.dim)]
        # --- Robust Topological Sort ---
        # Ensures we only sample a variable if all its conditions have already been sampled
        sorted_bn = []
        sampled_vars = set()
        remaining_bn = list(bayesian_network)
        
        while remaining_bn:
            for dependency in list(remaining_bn):
                target_var = dependency[0]
                cond_vars = set(dependency[1:])
                if cond_vars.issubset(sampled_vars):
                    sorted_bn.append(dependency)
                    sampled_vars.add(target_var)
                    remaining_bn.remove(dependency)
                    
        # --- Autoregressive Sampling Loop ---
        for dependency in sorted_bn:
            target_var = dependency[0]
            cond_vars = dependency[1:]
            
            # 1. Build the mask
            mask = torch.zeros((batch_size, dim), device=device)
            mask[:, target_var] = 1           # 1 for the variable we are predicting
            for c_var in cond_vars:
                mask[:, c_var] = -1           # -1 for known conditions
                
            # 2. Build the input tensor (plugging in previously sampled conditions)
            x_in = torch.zeros((batch_size, dim), device=device)
            for c_var in cond_vars:
                x_in[:, c_var] = samples[:, c_var]
                
            # 3. Predict probabilities
            # (Assuming model returns raw logits. Use .exp() if it returns log_softmax)
            logits = self.model(x_in, mask)
            probs = torch.softmax(logits, dim=-1)
            
            # 4. Sample bin indices from the categorical distribution
            sampled_bins = torch.multinomial(probs, num_samples=1).squeeze(-1) # Shape: [batch_size]
            
            # 5. Dequantize bins to continuous values and store them
            # unsqueeze to [batch_size, 1] to match your dequantize expectations
            sampled_vals = self.quantize.dequantize(sampled_bins.unsqueeze(-1), dimensions=[target_var]).squeeze(-1)
            samples[:, target_var] = sampled_vals
            
        return samples

    def conditional_dist(self,condition : torch.Tensor,variables: List[int],softmax=True):
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
        
        if not isinstance(condition,torch.Tensor): condition=torch.tensor(condition)
        if condition.ndim==1:condition=condition.unsqueeze(0)
        
        inp = torch.zeros((condition.shape[0],self.dim))
        inp[:,variables[1:]]=condition
        infer_ind = variables[0]
        mask=torch.zeros_like(inp)
        mask[:,infer_ind]=1
        mask[:,variables[1:]]=-1
        grid = self.quantize.dequantize(torch.arange(self.bins)[None,:],[infer_ind])
        points = self.model.forward(inp,mask)
        grid = grid.expand_as(points)
        if softmax: points=points.softmax(-1)
        return Interpolation(grid,points)
