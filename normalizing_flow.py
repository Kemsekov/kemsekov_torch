from torch.distributions import Normal
from kemsekov_torch.residual import Residual
from kemsekov_torch.common_modules import mmd_rbf,Prod, AddConst
from kemsekov_torch.fm.cuda_graph import _CudaGraph
from typing import Callable, Generator, Literal, Optional
from copy import deepcopy
import torch
import torch.nn as nn
from invertible_nn import *

class LossNormalizer1d(nn.Module):
    def __init__(self, in_dim,hidden_dim=32) -> None:
        super().__init__()
        self.expand = nn.Linear(in_dim,hidden_dim)
        self.net = nn.Sequential(
            Residual([
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ]),
            nn.Softplus()
        )
    def forward(self,x : torch.Tensor):
        """
        Forward pass of the loss normalizer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_dim]
        Returns:
            torch.Tensor: Normalized loss weights of shape [batch_size, in_dim]
        """
        x = self.expand(x)
        return self.net(x)[:,0]
class NormalizingFlowScaler:
    """
    Data scaler for normalizing flow
    """
    def __init__(self) -> None:
        self.mean = 0
        self.std = 1
        
    def inverse(self,data):
        input_shape = list(data.shape)
        input_shape[-1]//=2
        last_dim = input_shape[-1]
        data = data.flatten(-1)[:,:last_dim]
        return (data*self.std[:,:last_dim]+self.mean[:,:last_dim]).reshape(input_shape)
        
    def transform(self,data : torch.Tensor):
        input_shape = list(data.shape)
        input_shape[-1]*=2
        data = torch.concat([data,data.log_softmax(-1)],-1).flatten(-1)
        data = (data-self.mean)/self.std
        return data.reshape(input_shape)
    
    def fit_transform(self,data : torch.Tensor):
        input_shape = list(data.shape)
        input_shape[-1]*=2
        data = torch.concat([data,data.log_softmax(-1)],-1).flatten(-1)
        self.mean = data.mean(0,keepdim=True)
        self.std = data.std(0,keepdim=True)+1e-6
        
        data = (data-self.mean)/self.std
        return data.reshape(input_shape)
class NormalizingFlow:
    """
    Wrapper around your InvertibleSequential + flow_nll_loss training loop.
    
    You must use this class alongside `NormalizingFlowScaler`

    Key features:
    - Model definition is fully determined in __init__ (input_dim is required, not inferred from data).
    - fit(...) trains on a tensor dataset and returns the best model (CPU, eval).
    - Works with flow_nll_loss that returns either:
        * loss
        * (loss, diagnostics_dict)
      (avoids "iteration over a 0-d tensor" unpacking error).
    - Optional gradient clipping via torch.nn.utils.clip_grad_norm_. [web:381]
    - Uses optimizer.zero_grad(set_to_none=True) for performance/memory. [web:399]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        layers: int = 3,
        dropout=0.05,
        device: Optional[str] = 'cpu',
        non_linearity : Union[SmoothSymmetricSqrt,InvertibleIdentity] = InvertibleIdentity,
    ):
        self.non_linearity=non_linearity
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.cuda_graphs = True
        self._train_graphs = {}  # fit() training steps, keyed by ('fit', B, has_prior)
        self._infer_graphs = {}  # forward/inverse inference, keyed by ('forward'|'inverse', shape)
        self._opt_graphs = {}    # LBFGS closures, keyed by ('optimize', B, cols) / ('conditional', B)
        self.model = self._build_model(dropout).to(self.device)
        self.best_trained_model = None
        self._data_mean = 0
        self._data_std = 1

    def to(self,device):
        self.device=device
        self._invalidate_graphs()
        self.model=self.model.to(device)

    def _graphs_available(self):
        """
        Whether CUDA-graph acceleration is enabled and usable on this model.
        """
        return (self.cuda_graphs and torch.cuda.is_available()
                and str(self.device).startswith('cuda'))

    def _can_graph(self):
        """
        Inference graphs require grad to be disabled; training-step graphs
        (fit/optimize/conditional_sample) work under grad via autograd.grad.
        """
        return self._graphs_available() and not torch.is_grad_enabled()

    def _invalidate_graphs(self):
        """
        Drops all captured graphs. Must be called whenever parameter tensors
        are replaced or moved (device/dtype casts), since captured graphs
        reference the old memory addresses.
        """
        self._train_graphs.clear()
        self._infer_graphs.clear()
        self._opt_graphs.clear()

    def _eager_forward(self, data : torch.Tensor):
        """
        Forward pass without CUDA-graph acceleration (used inside captured
        steps, where nested graph capture is illegal).
        """
        return self.model(data)

    def _eager_inverse(self, data : torch.Tensor):
        """
        Inverse pass without CUDA-graph acceleration (used inside captured
        steps, where nested graph capture is illegal).
        """
        return self.model.inverse(data)

    def _graph_infer(self, kind, shape, eager_fn):
        """
        Returns (caching on first call) a CUDA graph that runs ``eager_fn``
        (single-tensor-in, tuple-of-tensors-out) on a static input of the
        given shape, keyed by (kind, shape). Returns None if unavailable.
        """
        key = (kind, tuple(shape))
        cg = self._infer_graphs.get(key)
        if cg is None:
            xb = torch.empty(shape, device=self.device)
            def fn(xb):
                out = eager_fn(xb)
                return out if isinstance(out, tuple) else (out,)
            cg = _CudaGraph.capture(fn, [xb])
            if cg is None:
                self.cuda_graphs = False
                return None
            self._infer_graphs[key] = cg
        return cg
    
    def _build_model(self,dropout_p) -> nn.Module:
        if self.input_dim % 2 != 0:
            raise ValueError(
                f"input_dim must be even for InvertibleScaleAndTranslate(input.chunk(2)). Got {self.input_dim}."
            )

        norm = nn.RMSNorm
        # norm = nn.BatchNorm1d
        # act = nn.ReLU
        act = nn.GELU
        # act = nn.SiLU
        dropout = lambda: nn.Dropout(p=dropout_p)
        input_dim = self.input_dim
        half = input_dim // 2
        blocks = []
        for i in range(self.layers):
            steps = [
                nn.Linear(half, self.hidden_dim),
                norm(self.hidden_dim),
                # dropout(),
                Residual([
                    # norm(self.hidden_dim),
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                ],init_at_zero=True),
                # Residual([
                #     # norm(self.hidden_dim),
                #     act(),
                #     # dropout(),
                #     nn.Linear(self.hidden_dim, self.hidden_dim),
                # ],init_at_zero=True),

                Prod(nn.Sequential(
                    act(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    norm(self.hidden_dim),
                    # SmoothSymmetricSqrt()
                    nn.Tanh(),
                    # AddConst(1.0)
                )),
                act(),
                nn.Linear(self.hidden_dim, input_dim),
            ]
            if i==self.layers-1 and "Norm" in str(steps[-1]):
                steps=steps[:-1]
            blocks.append(
                InvertibleScaleAndTranslate(
                    model=nn.Sequential(*steps),
                    dimension_split=-1,
                    non_linearity=self.non_linearity,
                )
            )
        blocks[-1].non_linearity = InvertibleIdentity()
        return InvertibleSequential(*blocks)
    
    def sample(self,count : int) -> torch.Tensor:
        """
        Generates samples drawn from trained distribution
        """
        if self._can_graph():
            noise = torch.randn((count,self.input_dim),device=self.device)
            cg = self._graph_infer('inverse', noise.shape, self._eager_inverse)
            if cg is not None:
                cg.inputs[0].copy_(noise)
                cg.replay()
                return cg.outputs[0].clone()
        return self.model.inverse(torch.randn((count,self.input_dim),device=self.device))
    
    def log_prob(self, data : torch.Tensor) -> torch.Tensor:
        if self._can_graph():
            data = data.to(self.device)
            def fn(xb):
                z, jacobians = self._eager_forward(xb)
                # manual N(0,1) log prob, bit-identical to Normal(0,1).log_prob,
                # but without any tensor creation inside the capture
                log_pz = (-(z**2) / 2 - 0.9189385332046727).flatten(-1).sum(dim=-1)
                log_det = 0.0
                for jd in jacobians:
                    log_det = log_det + torch.log(torch.abs(jd) + 1e-8).flatten(-1).sum(dim=-1)
                return (log_pz + log_det,)
            cg = self._graph_infer('log_prob', data.shape, fn)
            if cg is not None:
                cg.inputs[0].copy_(data)
                cg.replay()
                return cg.outputs[0].clone().to(data.device)
        model = self.model
        z, jacobians = model(data.to(self.device))
        
        # log p(z) under standard normal
        log_pz = Normal(0, 1).log_prob(z).flatten(-1).sum(dim=-1)
        
        # log |det J|
        log_det = 0.0
        for jd in jacobians:
            log_abs_jd = torch.log(torch.abs(jd) + 1e-8)
            log_det += log_abs_jd.flatten(-1).sum(dim=-1)
        
        # log p(x) = log p(z) + log |det J|
        log_px = log_pz + log_det
        
        return log_px.to(data.device)

    def interpolate(self,dataA : torch.Tensor,dataB : torch.Tensor, N : int):
        """
        Generate N interpolated samples between dataA and dataB via latent space linear interpolation.
        
        Args:
            dataA: Starting data point tensor
            dataB: Ending data point tensor  
            N: Number of interpolation steps to generate
        
        Yields:
            torch.Tensor: Interpolated sample at each step from dataA to dataB
        """
        m = self.model
        dataA = dataA.to(self.device)
        dataB = dataB.to(self.device)
        if self._can_graph():
            def fwd(xb):
                return (self._eager_forward(xb)[0],)
            cg_f = self._graph_infer('forward', dataA.shape, fwd)
            cg_i = self._graph_infer('inverse', dataA.shape, self._eager_inverse)
            if cg_f is not None and cg_i is not None:
                cg_f.inputs[0].copy_(dataA)
                cg_f.replay()
                latentsA = cg_f.outputs[0].clone()
                cg_f.inputs[0].copy_(dataB)
                cg_f.replay()
                latentsB = cg_f.outputs[0].clone()
                time = torch.linspace(0,1,N)
                for i in range(N):
                    t = time[i]
                    interpolated = (1-t)*latentsA+t*latentsB
                    cg_i.inputs[0].copy_(interpolated)
                    cg_i.replay()
                    yield cg_i.outputs[0].clone().to(dataA.device)
                return
        latentsA = m(dataA)[0]
        latentsB = m(dataB)[0]
        time = torch.linspace(0,1,N)
        for i in range(N):
            t = time[i]
            interpolated = (1-t)*latentsA+t*latentsB
            yield m.inverse(interpolated)

    def optimize(self, data: torch.Tensor, lr: float = 1.0, epochs: int = 1, 
             columns_to_optimize: list[int] = None):
        """
        Optimize only specific columns of data to maximize log probability.
        
        Args:
            data: Input tensor of shape [batch_size, input_dim]
            columns_to_optimize: List of column indices to optimize (0-based). 
                                If None or empty, all columns will be optimized.
            
        Returns:
            Optimized data tensor, final loss
        """
        batch_size, input_dim = data.shape
        
        # Handle default case - optimize all columns if none specified
        if columns_to_optimize is None or len(columns_to_optimize) == 0:
            columns_to_optimize = list(range(input_dim))
        
        # Validate column indices
        columns_to_optimize = [c for c in columns_to_optimize if 0 <= c < input_dim]
        if not columns_to_optimize:
            return data.clone(), -self.log_prob(data).sum().detach()
        
        # Identify fixed columns as those not in columns_to_optimize
        all_columns = list(range(input_dim))
        fixed_columns = [c for c in all_columns if c not in columns_to_optimize]
        
        # Create optimizable parameters for only the specified columns
        optimizable_data = data[:, columns_to_optimize].clone().detach().requires_grad_(True)
        
        # Fixed data doesn't need gradients
        if fixed_columns:
            fixed_data = data[:, fixed_columns].clone().detach()
        else:
            fixed_data = None
        
        # Define optimizer on only the optimizable part
        optimizer = torch.optim.LBFGS([optimizable_data], lr=lr, max_iter=20)
        
        class IterationData:
            best_loss = 1e8
            best_optimizable_data = optimizable_data.clone().detach()
        
        iteration = IterationData()
        self._iteration = iteration
        # Reconstruct full tensor by combining optimizable and fixed parts
        self._current_data = torch.zeros_like(data)

        use_graphs = self._graphs_available()
        if use_graphs:
            # capture the whole LBFGS closure (forward + log_prob + backward
            # w.r.t. the full batch) as a CUDA graph; the LBFGS step itself
            # stays eager and reads p.grad from the shared grad buffer
            og = self._get_optimize_graph(batch_size, tuple(columns_to_optimize), self.device)
            if og is None:
                use_graphs = False
        def closure():
            optimizer.zero_grad()
            if use_graphs:
                # assemble the full batch eagerly: in-place column writes are
                # illegal inside a capture, so the original eager code runs
                # here and only the forward+loss+backward is replayed
                current_data = self._current_data.detach()
                current_data[:, columns_to_optimize] = optimizable_data
                if fixed_columns:
                    current_data[:, fixed_columns] = fixed_data
                # copy under no_grad: the captured graph's own backward
                # provides the gradient w.r.t. the static input buffer
                with torch.no_grad():
                    og.inputs[0].copy_(current_data.to(self.device))
                og.replay()
                # clone: graph outputs are shared buffers overwritten by the
                # next replay, and best-loss tracking must keep a stable value
                loss = og.outputs[0].detach().clone()
                g = og.outputs[1]
                optimizable_data.grad = g[:, columns_to_optimize].to(optimizable_data.device)
            else:
                iteration = self._iteration
                current_data = self._current_data.detach()
                
                # Fill in the optimizable columns
                current_data[:, columns_to_optimize] = optimizable_data
                
                # Fill in fixed columns if any exist
                if fixed_columns:
                    current_data[:, fixed_columns] = fixed_data
                
                # Compute loss on the full tensor
                loss = -self.log_prob(current_data).sum()
                
                loss.backward()
            iteration = self._iteration
            if loss<iteration.best_loss:
                iteration.best_loss=loss
                iteration.best_optimizable_data=optimizable_data.detach().clone()
            return loss
        
        # Run optimization
        for i in range(epochs):
            loss = optimizer.step(closure)
        
        # Create final result by combining optimized and fixed parts
        result = torch.zeros_like(data)
        result[:, columns_to_optimize] = iteration.best_optimizable_data
        
        # Add back fixed columns if any exist
        if fixed_columns:
            result[:, fixed_columns] = fixed_data
        
        return result, iteration.best_loss

    def fit(
        self,
        data: torch.Tensor,
        batch_size: int = 512,
        epochs: int = 30,
        data_renoise_start=0.1,
        data_renoise_end=0.01,
        lr: float = 1e-2,
        grad_clip_max_norm: Optional[float] = 1,
        debug: bool = False,
        loss_normalizer_weight = 0.1,
        data_prior : Optional[torch.Tensor] = None,
        scheduler : Literal['exponential','cosine'] = 'cosine',
    ) -> nn.Module:
        """
        Train on `data` and return best model.

        Args:
            data: Tensor of shape [N, input_dim].
            batch_size: Batch size.
            epochs: Epoch count.
            data_renoise_start: dataset renoise factor. How much renoise training data at the first epochs.
            data_renoise_end: lowest dataset renoise factor.
            lr: AdamW learning rate.
            grad_clip_max_norm: If not None, clip global grad norm to this value. [web:381]
            debug: If True, prints when best loss improves.

        Returns:
            trained_model: Best model on CPU in eval() mode.
        """
        if data.ndim != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"Expected data shape [N, {self.input_dim}], got {tuple(data.shape)}")

        batch_size = min(batch_size,data.shape[0])
        data = data.to(self.device)

        if data_prior is not None:
            data_prior = data_prior.to(self.device)
        
        data_renoise_start *= data.std(0).median()
        data_renoise_end *= data.std(0).median()

        # loss_normalizer is recreated on every fit() call, so any captured
        # training graphs referencing the previous one's parameters are stale
        self._train_graphs.clear()

        self.model.train()
        loss_normalizer = LossNormalizer1d(self.input_dim,hidden_dim=self.hidden_dim).to(self.device)
        
        optim = torch.optim.AdamW(list(self.model.parameters())+list(loss_normalizer.parameters()), lr=lr,fused=True)
        
        best_loss = float("inf")
        best_trained_model = deepcopy(self.model).to(self.device)
        improved = False
        n = data.shape[0]
        slices = list(range(0, n, batch_size))
        
        total_steps = len(slices)*epochs
        
        if scheduler=='cosine':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim,total_steps)
        if scheduler=='exponential':
            sch = torch.optim.lr_scheduler.ExponentialLR(optim,(0.15)**(1/total_steps))
        use_train_graphs = self._graphs_available()
        try:
            for epoch in range(epochs):
                if debug and improved:
                    print(f"Epoch {(str(epoch)+"   ")[:3]}: best_loss={str(best_loss)[:5]} renoise_level={str(renoise_level.item())[:5]}")
                improved = False

                # shuffle each epoch
                perm = torch.randperm(n, device=self.device)
                data_shuf = data[perm]
                if data_prior is not None:
                    prior_shuf = data_prior[perm]

                losses = []
                part = (epoch+1)/epochs
                renoise_level = (data_renoise_start*(1-part)+data_renoise_end*part)
                for start in slices:
                    batch = data_shuf[start : start + batch_size]
                    B = batch.shape[0]
                    noise = torch.randn_like(batch)*renoise_level

                    if use_train_graphs:
                        # CUDA-graph training step: copy inputs into static
                        # buffers, replay the captured forward+loss+backward+
                        # clip graph, expose shared grad buffers as p.grad,
                        # then step the optimizer eagerly
                        tg = self._get_fit_graph(B, self.model, loss_normalizer, data_prior,
                                                 grad_clip_max_norm, loss_normalizer_weight,
                                                 self.device)
                        if tg is not None:
                            tg.inputs[0].copy_(batch)
                            tg.inputs[1].copy_(noise)
                            if tg.inputs[2] is not None:
                                tg.inputs[2].copy_(prior_shuf[start : start + B])
                            tg.replay()
                            for p, buf in tg.grad_buffers.items():
                                p.grad = buf
                            for p in tg.unused_params:
                                p.grad = None
                            loss = tg.outputs[0].detach()
                            nil = tg.outputs[1].detach()
                            optim.step()
                            sch.step()
                            losses.append(nil.mean())
                            continue
                        use_train_graphs = False

                    if renoise_level>0:
                        batch=batch+noise
                    
                    optim.zero_grad(set_to_none=True)
                    z,jac = self.model(batch)
                    loss_weight = loss_normalizer(z) # log(1/nil)=-log(nil)
                    
                    nil,log_det = flow_nll_loss(z,jac, batch, sum_dim=[-1])
                    
                    nil+=8
                    
                    model_loss = (loss_weight.detach().exp()*nil).mean()
                    
                    with torch.no_grad():
                        expected_loss = -nil.clamp(1e-7).log().detach()
                        
                    normalizer_loss = torch.nn.functional.mse_loss(expected_loss,loss_weight)
                    loss = model_loss+normalizer_loss*loss_normalizer_weight
                    if data_prior is not None:
                        prior_batch = prior_shuf[start : start + batch_size]
                        loss+=(z-prior_batch).pow(2).mean()

                    loss.backward()
                    
                    if grad_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=grad_clip_max_norm,
                            norm_type=2.0,
                        )

                    optim.step()
                    sch.step()
                    losses.append(nil.mean())
                mean_loss = sum(losses)/len(losses)
                if mean_loss < best_loss:
                    best_loss = mean_loss.item()
                    best_trained_model = deepcopy(self.model)
                    improved = True
        except KeyboardInterrupt:
            if debug: print("Stop training")
        if debug and improved:
            print(f"Last Epoch {epoch}: best_loss={best_loss:0.3f}")
        self.model.eval()
        with torch.no_grad():
            for a,b in zip(self.model.parameters(),best_trained_model.parameters()):
                a.copy_(b.to(a.device))

    def _get_fit_graph(self, B, model, loss_normalizer, data_prior, grad_clip_max_norm,
                       loss_normalizer_weight, device):
        """
        Returns (capturing on first call) a CUDA graph of one fit() training
        step for batch size B: renoise + forward + loss + gradients + clip.
        Gradients are computed with torch.autograd.grad and copied into shared
        eager buffers (loss.backward() allocates per-capture .grad tensors
        that would be reallocated on each new capture), then exposed as p.grad
        by the caller.

        The optimizer step, scheduler step and all RNG draws stay eager: LR is
        scheduled by the caller and the RNG must not be frozen by the graph.
        Returns None (and disables graphs) if the capture fails.
        """
        key = ('fit', B, data_prior is not None)
        tg = self._train_graphs.get(key)
        if tg is not None:
            return tg
        params = list(model.parameters()) + list(loss_normalizer.parameters())
        used_ids = set()
        bx = torch.empty(B, self.input_dim, device=device)
        bn = torch.empty_like(bx)  # eager-drawn renoise noise
        bp = torch.empty_like(bx) if data_prior is not None else None
        def fn(bx, bn, bp):
            batch = bx + bn
            z, jac = self._eager_forward(batch)
            loss_weight = loss_normalizer(z)  # log(1/nil)=-log(nil)
            nil, log_det = flow_nll_loss(z, jac, batch, sum_dim=[-1])
            # NOTE: eager does `nil += 8`; in-place ops on tensors read by the
            # replayed backward are unsafe, so the same math is done out-of-place
            nil = nil + 8
            model_loss = (loss_weight.detach().exp() * nil).mean()
            expected_loss = -nil.clamp(1e-7).log().detach()
            normalizer_loss = torch.nn.functional.mse_loss(expected_loss, loss_weight)
            loss = model_loss + normalizer_loss * loss_normalizer_weight
            if bp is not None:
                loss = loss + (z - bp).pow(2).mean()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            used = [(p, g) for p, g in zip(params, grads) if g is not None]
            for p, _ in used:
                used_ids.add(id(p))
            if grad_clip_max_norm is not None:
                # capture-safe clip (bit-identical arithmetic to the eager
                # clip_grad_norm_): one fused norm + one fused scale kernel
                norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([g for _, g in used], 2.0)), 2.0)
                scale = torch.clamp(grad_clip_max_norm / (norm + 1e-6), max=1.0)
                torch._foreach_mul_([g for _, g in used], scale)
            fn.grads = used
            return loss, nil
        tg = _CudaGraph.capture(fn, [bx, bn, bp])
        if tg is None:
            self.cuda_graphs = False
            return None
        tg.grad_buffers = dict(fn.grads)
        tg.unused_params = [p for p in params if id(p) not in used_ids]
        self._train_graphs[key] = tg
        return tg

    def _get_optimize_graph(self, B, columns, device):
        """
        Returns (capturing on first call) a CUDA graph of one optimize()
        LBFGS closure: forward + log_prob + gradient w.r.t. the full batch.
        The caller assembles the full batch eagerly (in-place column writes
        are illegal inside a capture) and the LBFGS step stays eager, reading
        p.grad from the shared grad buffer sliced to the optimized columns.
        """
        key = ('optimize', B, columns)
        og = self._opt_graphs.get(key)
        if og is not None:
            return og
        bx = torch.empty(B, self.input_dim, device=device)
        bx.requires_grad_(True)
        def fn(bx):
            z, jacobians = self._eager_forward(bx)
            # manual N(0,1) log prob, bit-identical to Normal(0,1).log_prob,
            # but without any tensor creation inside the capture
            log_pz = (-(z**2) / 2 - 0.9189385332046727).flatten(-1).sum(dim=-1)
            log_det = 0.0
            for jd in jacobians:
                log_det = log_det + torch.log(torch.abs(jd) + 1e-8).flatten(-1).sum(dim=-1)
            loss = -(log_pz + log_det).sum()
            grad = torch.autograd.grad(loss, bx)[0]
            fn.grad = grad
            return loss, grad
        og = _CudaGraph.capture(fn, [bx])
        if og is None:
            self.cuda_graphs = False
            return None
        self._opt_graphs[key] = og
        return og

    def _get_conditional_graph(self, num_samples, constraint, original_prior,
                               mode_closeness_weight, device):
        """
        Returns (capturing on first call) a CUDA graph of one
        conditional_sample() LBFGS closure: x = M_inv(z), prior loss +
        constraint loss, gradient w.r.t. z. The LBFGS step and the Langevin
        noise stay eager. Note: the constraint callable is executed at capture
        time and its kernels are replayed; python-level control flow inside it
        is only evaluated once.
        """
        key = ('conditional', num_samples)
        cg = self._opt_graphs.get(key)
        if cg is not None:
            return cg
        # torch.empty, not randn: capturing must not consume RNG, otherwise
        # the caller's (eager) Langevin-noise stream would diverge from the
        # non-graph path
        zb = torch.empty(num_samples, self.input_dim, device=device, requires_grad=True)
        def fn(zb):
            # Forward pass: x = M_inv(z)
            x = self._eager_inverse(zb)

            # Compute prior loss: L_prior = ||z||² (keep z in N(0,I)) must match original generated prior
            L_prior = (zb * zb).mean()
            L_prior = (L_prior - original_prior) ** 2 + mode_closeness_weight * L_prior

            # Compute constraint loss: L_constraint = constraint(x)
            L_constraint = constraint(x)

            # Total loss: L_total = L_prior + λ * L_constraint
            L_total = L_prior + L_constraint

            grad = torch.autograd.grad(L_total, zb)[0]
            fn.grad = grad
            return L_total, grad
        cg = _CudaGraph.capture(fn, [zb])
        if cg is None:
            self.cuda_graphs = False
            return None
        self._opt_graphs[key] = cg
        return cg
        
    def to_prior(self,data : torch.Tensor) -> torch.Tensor:
        """
        Converts data tensor to latent space (standard normal dist)
        """
        if self._can_graph():
            data = data.to(self.device)
            def fwd(xb):
                return (self._eager_forward(xb)[0],)
            cg = self._graph_infer('forward', data.shape, fwd)
            if cg is not None:
                cg.inputs[0].copy_(data)
                cg.replay()
                return cg.outputs[0].clone().to(data.device)
        return self.model(data.to(self.device))[0]
    
    def to_target(self,latent_prior : torch.Tensor) -> torch.Tensor:
        """
        Converts data tensor to target posterior space(dataset distribution)
        """
        if self._can_graph():
            latent_prior = latent_prior.to(self.device)
            cg = self._graph_infer('inverse', latent_prior.shape, self._eager_inverse)
            if cg is not None:
                cg.inputs[0].copy_(latent_prior)
                cg.replay()
                return cg.outputs[0].clone().to(latent_prior.device)
        return self.model.inverse(latent_prior.to(self.device))

    def conditional_sample(
        self,
        constraint : Callable[[torch.Tensor],torch.Tensor],
        num_samples: int,
        noise_scale: float = 0.0,
        steps: int = 2,
        lr: float = 1,
        mode_closeness_weight = 1.0
    ) -> torch.Tensor:
        """
        Sample from p(X | X[c_i] = v_i) using constrained latent space optimization with Langevin dynamics.

        Args:
            constraint: Constraint loss function. Accepts generated target in (num_samples,dim) shape and returns loss (scalar tensor) that defines condition for sampling.
            num_samples: Number of samples to generate
            noise_scale: Scale of noise added during Langevin dynamics (default 0.00). Increasing this value will result in samples more spread from condition. Values around [0 to 0.05] are generally good enough.
            steps: Number of optimization steps (default 2)
            lr: Learning rate for the optimization (default 1)
            mode_closeness_weight: Weight for trying to sample closer to distribution mode. Increasing this value make samples cluster more around closest distribution mode, potentially leading to mode collapse (all samples are the same). Values [0 to 2] are generally good enough.

        Returns:
            torch.Tensor: Samples of shape [num_samples, input_dim] satisfying the conditions
        """

        model = self.model
        model.eval()

        # Initialize z from standard normal distribution
        z = torch.randn(num_samples, self.input_dim, device=self.device, requires_grad=True)

        original_prior = (z * z).mean().detach()

        # Create optimizer for the latent variable z
        optimizer = torch.optim.LBFGS([z], lr=lr)

        class Iteration:
            best_sample = z.clone().detach()
            best_loss = 1e8
        self._iteration = Iteration()

        use_graphs = self._graphs_available()
        if use_graphs:
            cg = self._get_conditional_graph(num_samples, constraint, original_prior,
                                             mode_closeness_weight, self.device)
            if cg is None:
                use_graphs = False
        
        def closure():
            optimizer.zero_grad()

            if use_graphs:
                with torch.no_grad():
                    cg.inputs[0].copy_(z)
                cg.replay()
                # clone: graph outputs are shared buffers overwritten by the
                # next replay, and best-loss tracking must keep a stable value
                L_total = cg.outputs[0].detach().clone()
                z.grad = cg.outputs[1]
            else:
                # Forward pass: x = M_inv(z)
                x = model.inverse(z)

                # Compute prior loss: L_prior = ||z||² (keep z in N(0,I)) must match original generated prior
                L_prior = (z * z).mean()
                L_prior = (L_prior-original_prior)**2+mode_closeness_weight*L_prior

                # Compute constraint loss: L_constraint = constraint(x)
                L_constraint = constraint(x)

                # Total loss: L_total = L_prior + λ * L_constraint
                L_total = L_prior + L_constraint

                L_total.backward()

            it = self._iteration
            if L_total<it.best_loss:
                it.best_loss = L_total
                it.best_sample = z.clone().detach()
            
            with torch.no_grad():
                z.data += (noise_scale) * torch.randn_like(z)
            return L_total
        
        for t in range(steps):
            # Perform optimizer step
            optimizer.step(closure)


        with torch.no_grad():
            final_x = model.inverse(self._iteration.best_sample)

        return final_x