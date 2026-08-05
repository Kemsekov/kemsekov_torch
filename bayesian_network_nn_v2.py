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
    def normalize(self,x,dimensions):
        centers = self.center[dimensions].unsqueeze(0)
        scales = self.scale[dimensions].unsqueeze(0)
        normalized = ((x-centers)/scales+1)/2
        return normalized
    
    def quantize(self,x: torch.Tensor,dimensions : List[int]):
        if not isinstance(x,torch.Tensor):x=torch.tensor(x)
        # x is some subset of data features, x=data[:,dimensions]
        normalized = self.normalize(x,dimensions)
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
        if isinstance(module,list) or isinstance(module,tuple):module = nn.Sequential(*module)
        self.m=module
    def forward(self,x):
        return x*self.m(x)
class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        if isinstance(module,list) or isinstance(module,tuple):module = nn.Sequential(*module)
        self.m=module
    def forward(self,x):
        return x+self.m(x)

class Generative(nn.Module):
    def __init__(self, dim : int,hid_dim=32,bins=16,dist_hid=16,hid_residuals=2,dist_residuals=2):
        super().__init__()
        #accept input dim + mask of same length
        self.expand = nn.Linear(dim*2,hid_dim)
        self.mlp = nn.Sequential(*[
            *[Residual((
                nn.RMSNorm(hid_dim),
                Prod((
                    nn.Linear(hid_dim,hid_dim),
                    nn.Tanh()
                )),
                nn.SiLU(),
                nn.Linear(hid_dim,hid_dim),
            )) for i in range(hid_residuals)],
            nn.RMSNorm(hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim,dist_hid),
        ])
        self.residual_linear = nn.Linear(hid_dim,dist_hid)
        # output log-probability table.
        self.out = nn.Linear(hid_dim,bins)
        
        self.time_emb = nn.Sequential(
            nn.Linear(1,dist_hid),
            nn.SiLU(),
            nn.Linear(dist_hid,dist_hid),
        )
        self.out_prob = nn.Sequential(
          *[Residual((
                nn.RMSNorm(dist_hid),
                Prod((
                    nn.Linear(dist_hid,dist_hid),
                    nn.Tanh()
                )),
                nn.SiLU(),
                nn.Linear(dist_hid,dist_hid),
            )) for i in range(dist_residuals)],
            nn.RMSNorm(dist_hid),
            nn.SiLU(),
        )
        self.final = nn.Linear(dist_hid,1)
        self.bins=bins
        
        self.scale=(-3,3)
        
    def forward(self,x,mask,return_hid=False):
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
        x = self.encode(x,mask)
        t = torch.linspace(self.scale[0],self.scale[1],self.bins,device=x.device,dtype=x.dtype).unsqueeze(-1) #[BINS,1]
        out = self.log_prob(x,t)
        return (out,x) if return_hid else out
    
    def encode(self,x,mask):
        """
        x: [Batch,dim]
        
        mask: [Batch,dim]
        
        return: [batch,dist_dim]
        """
        c = torch.concat([x,mask],-1)
        dim = x.shape[-1]
        c_slice = c[:,:dim]
        
        # hide unknown data
        c_slice[mask==1]=0
        c_slice[mask==0]=0
        
        x = self.expand(c)
        x = self.mlp(x)+self.residual_linear(x)
        return x
    
    def log_prob(self,x,t):
        """
        x: [batch,dim]
        
        t: [bins,1] or [batch,bins,1]
        
        returns [batch,bins]
        """
        t = self.time_emb(t) #[BINS,dist_hid]
        if t.ndim==2:t=t[None,:]
        
        xt = x[:,None]+t #[BATCH,BINS,dist_hid]
        probs = self.out_prob(xt) #[BATCH,bins,dist_hid]
        
        return self.final(probs)[:,:,0]
    
class Interpolation:
    """
    Accepts a unique grid of shape [B, bins] per function,
    and points of shape [B, bins] defining B distinct functions.
    """
    def __init__(self, grid, points,hid, model : Generative,centers,scales):
        """
        grid:   [B, bins] (sorted float tensor per batch function)
        points: [B, bins] (values evaluated at the grid points)
        hid:    [B, hid_dim] (curve parametrization hidden states)
        centers: [B]
        scales: [B]
        """
        if grid.shape != points.shape:
            raise ValueError(f"grid shape {grid.shape} must match points shape {points.shape}")
        self.grid = grid
        self.points = points
        self.points_log_softmax = points.log_softmax(-1)
        self.hid=hid
        self.model=model
        self.centers=centers
        self.scales=scales

    def exact(self,y):
        """
        Interpolates points at continuous coordinates y.

        y:      [K, B] (K inference query sets, each evaluating all B functions)
        Returns: [K, B] tensor of interpolated values
        """
        if y.ndim==1:
            y=y.unsqueeze(-1)
        
        #y is [K,B]
        yt = y.transpose(0,1)[:,:,None] #yt is [B,K,1]
        
        y_normalized = ((yt-self.centers[:,None,None])/self.scales[:,None,None]+1)/2
        # now y_normalized in [0;1] scale
        width = self.model.scale[1]-self.model.scale[0]
        y_normalized*=width
        y_normalized+=self.model.scale[0]
        
        out = self.model.log_prob(self.hid,y_normalized) #[B,K]
        # now we must concat
        out = torch.concat([self.points,out],-1) #[B,bins+K]
        out = out.log_softmax(-1) #use log softmax to mimic probability dist
        return out[:,-len(y):]
    
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
        
        points_L = self.points_log_softmax[batch_b, idx_L] # Shape: [K, B]
        points_R = self.points_log_softmax[batch_b, idx_R] # Shape: [K, B]
        
        # 5. Calculate weights
        denom = grid_R - grid_L
        denom = torch.where(denom == 0, torch.ones_like(denom), denom) 
        
        weight_R = (y_clamped - grid_L) / denom
        weight_L = 1.0 - weight_R
        
        # 6. Linearly interpolate
        return weight_L * points_L + weight_R * points_R

class Structure:
    def __init__(self,dataset,bayesian_network,bins=32,hid_dim=64,dist_hid=64):
        self.bayesian_network=bayesian_network
        self.dim = dataset.shape[-1]
        self.model = Generative(self.dim,hid_dim,bins=bins,dist_hid=dist_hid)
        self.raw_dataset=dataset
        self.set_bins(bins)
    
    def set_bins(self,bins):
        self.bins=bins
        self.model.bins=bins
        self.quantize = Quantize(self.raw_dataset,bins=bins)

        self.dataset=self.raw_dataset
        self.dataset=self.quantize.dequantize(self.quantize.quantize(self.raw_dataset,list(range(self.dim))),list(range(self.dim)))
        
        grids = []
        for infer_ind in range(self.dim):
            grid = self.quantize.dequantize(torch.arange(self.bins)[None,:],[infer_ind])
            grids.append(grid)
        self.grids=torch.concat(grids)
    
    def fit(self,epochs=2048,batch_size=256,lr=0.01,loss_function : Literal['cross_entropy','mle'] = 'cross_entropy',random_conditional_prob=0.4):
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
                conditional_mask = torch.rand_like(mask)<random_conditional_prob
                mask[conditional_mask]=-1
                mask[running,torch.randint(0,self.dim,(batch_size,))]=1
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
        pred,hid = self.model(batch,mask,return_hid=True)
        grids = self.grids[modelled_variable]
        # if log_softmax: pred = pred.log_softmax(-1)
        return Interpolation(
            grids,
            pred,
            hid,
            self.model,
            self.quantize.center[modelled_variable],
            self.quantize.scale[modelled_variable]
        )
    
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

    def conditional_dist(self,condition : torch.Tensor,variables: List[int]):
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
        grid = self.grids[infer_ind]
        points,hid = self.model.forward(inp,mask,return_hid=True)
        
        grid = grid.expand_as(points)
        return Interpolation(grid,points,hid,self.model,self.quantize.center[[infer_ind]],self.quantize.scale[[infer_ind]])
    def full_joint_log(self, data: torch.Tensor):
        """
        Computes log of the full joint probability of the provided data points
        using the chain rule defined by the bayesian_network.
        
        P(X) = prod P(X_i | Parents(X_i))
        
        data: tensor of shape [B, dim]
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 1:
            data = data.unsqueeze(0)
            
        device = data.device
        log_joint = torch.zeros(data.shape[0], device=device)
        
        bayesian_network = self.bayesian_network
        if bayesian_network is None:
            all_vars = list(range(self.dim))
            bayesian_network = [all_vars[-p-1:] for p in range(self.dim)]
            
        # --- Robust Topological Sort (Ensures valid chain rule order) ---
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
                    
        # --- Accumulate Conditional Log-Probabilities ---
        for dependency in sorted_bn:
            target_var = dependency[0]
            cond_vars = dependency[1:]
            
            # 1. Extract condition values from the data
            if len(cond_vars) > 0:
                cond_values = data[:, cond_vars]
            else:
                # Handle unconditional probabilities (e.g., P(Z))
                cond_values = torch.empty((data.shape[0], 0), device=device)
                
            # 2. Get the continuous conditional distribution 
            # We use log_softmax=True to match the geometric interpolation of your MLE loss!
            interp = self.conditional_dist(cond_values, dependency)
            
            # 3. Evaluate at the target variable's actual continuous values
            target_values = data[:, target_var]
            
            # interp expects [K, B]. target_values is [B]. Unsqueeze to [1, B]
            log_p_y = interp.exact(target_values.unsqueeze(0)).squeeze(0) # Shape: [B]
            
            # 4. Accumulate log-probabilities
            log_joint += log_p_y
            
        return log_joint
    def partial_joint_log(self, data: torch.Tensor, variables: List[int]):
        """
        Computes the joint probability over a specific subset of variables.
        
        data: tensor of shape [Batch, d] where d is the number of variables in the subset
        variables: List[int] containing the original indices of the variables in the subset
        return_log: if True, returns log-probabilities (prevents underflow)
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 1:
            data = data.unsqueeze(0)
            
        if data.shape[1] != len(variables):
            raise ValueError(f"data shape {data.shape} does not match number of variables {len(variables)}")
            
        device = data.device
        log_joint = torch.zeros(data.shape[0], device=device)
        
        is_bn_specified = self.bayesian_network is not None and len(self.bayesian_network) > 0
        
        if is_bn_specified:
            # 1. Build mapping from variable to its parents in the original BN
            parents_map = {}
            for dep in self.bayesian_network:
                parents_map[dep[0]] = set(dep[1:])
                
            sub_bn = []
            target_vars_set = set(variables)
            
            # 2. Check if the subset is computable from the provided BN (The Closure Check)
            for v in variables:
                if v not in parents_map:
                    raise ValueError(f"Variable {v} is not defined in the provided bayesian_network.")
                
                parents = parents_map[v]
                if not parents.issubset(target_vars_set):
                    missing = parents - target_vars_set
                    raise ValueError(
                        f"Cannot compute partial joint for subset {variables}. "
                        f"Variable {v} depends on parents {list(parents)} in the bayesian_network, "
                        f"but the following parents are missing from the subset: {list(missing)}. "
                        f"(Marginalizing them out requires numerical integration, which is not supported)."
                    )
                sub_bn.append([v] + list(parents))
                
            # 3. Topologically sort the sub-BN to ensure valid chain rule order
            sorted_bn = []
            sampled_vars = set()
            remaining_bn = list(sub_bn)
            
            while remaining_bn:
                for dependency in list(remaining_bn):
                    target_var = dependency[0]
                    cond_vars = set(dependency[1:])
                    if cond_vars.issubset(sampled_vars):
                        sorted_bn.append(dependency)
                        sampled_vars.add(target_var)
                        remaining_bn.remove(dependency)
                        
        else:
            # If no BN is specified, build an arbitrary autoregressive chain for the subset
            # e.g., variables = [0, 1, 2] -> [[2, 1, 0], [1, 0], [0]]
            sorted_bn = [variables[-p-1:] for p in range(len(variables))]
            
        # --- Accumulate Conditional Log-Probabilities ---
        # Map original variable indices to their column index in the `data` tensor
        var_to_col = {v: i for i, v in enumerate(variables)}
        
        for dependency in sorted_bn:
            target_var = dependency[0]
            cond_vars = dependency[1:]
            
            # 1. Extract condition values from the subset data
            if len(cond_vars) > 0:
                cond_cols = [var_to_col[c] for c in cond_vars]
                cond_values = data[:, cond_cols]
            else:
                cond_values = torch.empty((data.shape[0], 0), device=device)
                
            # 2. Get the continuous conditional distribution 
            # `dependency` contains the original indices, which `conditional_dist` expects
            interp = self.conditional_dist(cond_values, dependency)
            
            # 3. Evaluate at the target variable's actual continuous values
            target_col = var_to_col[target_var]
            target_values = data[:, target_col]
            
            # interp expects [K, B]. target_values is [B]. Unsqueeze to [1, B]
            log_p_y = interp.exact(target_values.unsqueeze(0)).squeeze(0)[:,0]
            # 4. Accumulate log-probabilities
            log_joint += log_p_y
            
        return log_joint