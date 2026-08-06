import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import List, Literal, Optional, Union
from kemsekov_torch.common_modules import get_optim_groups


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Pick the best available device when none is given: cuda > mps > cpu."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def resolve_dtype(dtype: Union[str, torch.dtype] = "fp32") -> torch.dtype:
    """Normalize a dtype name ('fp32'/'fp16'/'bf16') or torch dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype {dtype!r}, expected one of {list(mapping)}")
    return mapping[dtype]


class Quantize:
    """
    This is small convenience tool for converting continuous data into discrete
    representation and inverse
    """
    def __init__(self, data: torch.Tensor, bins=32):
        # data of shape [batch,dim]
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.center = data.mean(0)
        self.scale = data.std(0) * 2.5

        # protect from zero features
        self.scale[self.scale == 0] = 1

        self.bins = bins

    def normalize(self, x, dimensions):
        centers = self.center[dimensions].unsqueeze(0)
        scales = self.scale[dimensions].unsqueeze(0)
        normalized = ((x - centers) / scales + 1) / 2
        return normalized

    def quantize(self, x: torch.Tensor, dimensions: List[int]):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        # x is some subset of data features, x=data[:,dimensions]
        normalized = self.normalize(x, dimensions)
        quantized = torch.floor(normalized * self.bins).clamp(0, self.bins - 1).long()
        return quantized

    def dequantize(self, q: torch.Tensor, dimensions: List[int]):
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q)
        centers = self.center[dimensions].unsqueeze(0)
        scales = self.scale[dimensions].unsqueeze(0)
        symmetric = ((q + 0.5) / self.bins) * 2 - 1
        denorm = symmetric * scales + centers
        return denorm


class Prod(nn.Module):
    def __init__(self, module):
        super().__init__()
        if isinstance(module, list) or isinstance(module, tuple):
            module = nn.Sequential(*module)
        self.m = module

    def forward(self, x):
        return x * self.m(x)


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        if isinstance(module, list) or isinstance(module, tuple):
            module = nn.Sequential(*module)
        self.m = module

    def forward(self, x):
        return x + self.m(x)


class Generative(nn.Module):
    def __init__(self, dim: int, hid_dim=32, bins=16, dist_hid=16, hid_residuals=2, dist_residuals=2):
        super().__init__()
        #accept input dim + mask of same length
        self.expand = nn.Linear(dim * 2, hid_dim)
        self.mlp = nn.Sequential(*[
            *[Residual((
                nn.RMSNorm(hid_dim),
                Prod((
                    nn.Linear(hid_dim, hid_dim),
                    nn.Tanh()
                )),
                nn.SiLU(),
                nn.Linear(hid_dim, hid_dim),
            )) for i in range(hid_residuals)],
            nn.RMSNorm(hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, dist_hid),
        ])
        self.residual_linear = nn.Linear(hid_dim, dist_hid)
        # output log-probability table.
        self.out = nn.Linear(hid_dim, bins)

        self.time_emb = nn.Sequential(
            nn.Linear(1, dist_hid),
            nn.SiLU(),
            nn.Linear(dist_hid, dist_hid),
        )
        self.out_prob = nn.Sequential(
            *[Residual((
                nn.RMSNorm(dist_hid),
                Prod((
                    nn.Linear(dist_hid, dist_hid),
                    nn.Tanh()
                )),
                nn.SiLU(),
                nn.Linear(dist_hid, dist_hid),
            )) for i in range(dist_residuals)],
            nn.RMSNorm(dist_hid),
            nn.SiLU(),
        )
        self.final = nn.Linear(dist_hid, 1)
        self.bins = bins

        self.scale = (-3, 3)

        # constant bin grid, cached lazily per (device, dtype) -- values identical
        # to torch.linspace(scale[0], scale[1], bins, device=...), just computed once.
        self._t_cache = {}

    def _t(self, x):
        key = (x.device, x.dtype)
        t = self._t_cache.get(key)
        if t is None:
            t = torch.linspace(self.scale[0], self.scale[1], self.bins, device=x.device)
            t = t.to(dtype=x.dtype).unsqueeze(-1)  # [BINS,1]
            self._t_cache[key] = t
        return t

    def forward(self, x, mask, return_hid=False):
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
        x = self.encode(x, mask)
        # build linspace in fp32 for precision, then cast to activation dtype
        t = self._t(x)
        out = self.log_prob(x, t)
        return (out, x) if return_hid else out

    def encode(self, x, mask):
        """
        x: [Batch,dim]

        mask: [Batch,dim]

        return: [batch,dist_dim]
        """
        c = torch.concat([x, mask], -1)
        dim = x.shape[-1]
        c_slice = c[:, :dim]

        # hide unknown data
        c_slice[mask == 1] = 0
        c_slice[mask == 0] = 0

        x = self.expand(c)
        x = self.mlp(x) + self.residual_linear(x)
        return x

    def log_prob(self, x, t):
        """
        x: [batch,dim]

        t: [bins,1] or [batch,bins,1]

        returns [batch,bins]
        """
        t = self.time_emb(t)  # [BINS,dist_hid]
        if t.ndim == 2:
            t = t[None, :]

        xt = x[:, None] + t  # [BATCH,BINS,dist_hid]
        probs = self.out_prob(xt)  # [BATCH,bins,dist_hid]

        return self.final(probs)[:, :, 0]


class Interpolation:
    """
    Accepts a unique grid of shape [B, bins] per function,
    and points of shape [B, bins] defining B distinct functions.
    """
    def __init__(self, grid, points, hid, model: Generative, centers, scales):
        """
        grid:   [B, bins] (sorted float tensor per batch function)
        points: [B, bins] (values evaluated at the grid points)
        hid:    [B, hid_dim] (curve parametrization hidden states)
        centers: [B]
        scales: [B]
        """
        if grid.shape != points.shape:
            raise ValueError(f"grid shape {grid.shape} must match points shape {points.shape}")
        # keep grid and points in the same dtype so interpolation math is consistent
        if grid.dtype != points.dtype:
            grid = grid.to(dtype=points.dtype)
        self.grid = grid
        self.points = points
        self.points_log_softmax = points.log_softmax(-1)
        self.hid = hid
        self.model = model
        self.centers = centers
        self.scales = scales

    def exact(self, y):
        """
        Interpolates points at continuous coordinates y.

        y:      [K, B] (K inference query sets, each evaluating all B functions)
        Returns: [K, B] tensor of interpolated values
        """
        y = y.to(device=self.grid.device, dtype=self.points.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        #y is [K,B]
        yt = y.transpose(0, 1)[:, :, None]  # yt is [B,K,1]

        y_normalized = ((yt - self.centers[:, None, None]) / self.scales[:, None, None] + 1) / 2
        # now y_normalized in [0;1] scale
        width = self.model.scale[1] - self.model.scale[0]
        y_normalized *= width
        y_normalized += self.model.scale[0]

        out = self.model.log_prob(self.hid, y_normalized)  # [B,K]
        # match dtype of stored points before concatenating
        if out.dtype != self.points.dtype:
            out = out.to(dtype=self.points.dtype)
        # now we must concat
        out = torch.concat([self.points, out], -1)  # [B,bins+K]
        out = out.log_softmax(-1)  # use log softmax to mimic probability dist
        return out[:, -len(y):]

    def __call__(self, y):
        """
        Interpolates points at continuous coordinates y.

        y:      [K, B] (K inference query sets, each evaluating all B functions)
        Returns: [K, B] tensor of interpolated values
        """

        y = y.to(device=self.grid.device, dtype=self.points.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        K, B = y.shape
        _, bins = self.grid.shape
        # 1. Clamp y to each specific function's grid boundaries
        # grid[:, 0] and grid[:, -1] have shape [B]
        # Adding unsqueeze(0) broadcasts them to [1, B] to match y [K, B]
        grid_min = self.grid[:, 0].unsqueeze(0)
        grid_max = self.grid[:, -1].unsqueeze(0)
        y_clamped = torch.clamp(y, grid_min, grid_max)

        # 2. Find indices via a single batched searchsorted.
        #    count of grid elements <= y  ==  sum(grid <= y) in the original
        #    implementation, but without materializing the [K, B, bins] mask.
        #    (integer result -> bit-identical to the mask-sum version)
        idx_L = torch.searchsorted(self.grid, y_clamped.t(), right=True).t() - 1
        idx_L = torch.clamp(idx_L, 0, bins - 2)  # Shape: [K, B]
        idx_R = idx_L + 1

        # 3. Create batch indices for advanced indexing
        # We need an indexing helper for the B dimension that matches the [K, B] structure
        # batch_b maps the corresponding function index 0 to B-1 for every K query row
        batch_b = torch.arange(B, device=y.device).unsqueeze(0).expand(K, B)

        # 4. Gather grid and point values using [batch_b, index] shapes
        grid_L = self.grid[batch_b, idx_L]     # Shape: [K, B]
        grid_R = self.grid[batch_b, idx_R]     # Shape: [K, B]

        points_L = self.points_log_softmax[batch_b, idx_L]  # Shape: [K, B]
        points_R = self.points_log_softmax[batch_b, idx_R]  # Shape: [K, B]

        # 5. Calculate weights
        denom = grid_R - grid_L
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        weight_R = (y_clamped - grid_L) / denom
        weight_L = 1.0 - weight_R

        # 6. Linearly interpolate
        return weight_L * points_L + weight_R * points_R


class _FitGraph:
    """CUDA-graph capture of the full training step (forward + loss + backward).

    Numerics are bit-identical to the eager training step: the same kernels are
    executed in the same order on the same data, only the CPU launch overhead is
    eliminated.  RNG-consuming ops (randperm/rand_like/randint) stay outside.
    """

    def __init__(self, structure, opt, scaler, batch_size):
        self.structure = structure
        self.opt = opt
        self.scaler = scaler
        dev = structure.device
        dim = structure.dim
        self.batch = torch.zeros(batch_size, dim, device=dev)
        self.mask = torch.zeros(batch_size, dim, device=dev)
        self.expected = torch.zeros(batch_size, dtype=torch.long, device=dev)
        self.loss_buf = torch.empty((), device=dev)
        self._capture()

    def _step(self):
        self.opt.zero_grad(set_to_none=False)
        with self.structure._amp():
            pred = self.structure.model(self.batch, self.mask)
            loss = F.cross_entropy(pred, self.expected)
        self.scaler.scale(loss).backward()
        self.loss_buf.copy_(loss.detach())

    def _capture(self):
        self._step()  # warmup: allocates grads / scratch so capture is stable
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        torch.cuda.synchronize()
        self.g = g

    def replay(self, batch, mask, expected):
        self.batch.copy_(batch)
        self.mask.copy_(mask)
        self.expected.copy_(expected)
        self.g.replay()
        return self.loss_buf


class _ForwardGraph:
    """CUDA graph for model(inp, mask, return_hid=True) under no_grad + amp.

    Points/hid come out in fp32 (matches the .float() casts done by the eager
    inference paths) while the internal compute keeps the configured dtype.
    """

    def __init__(self, structure, B):
        dev = structure.device
        dim = structure.dim
        self.structure = structure
        self.inp = torch.zeros(B, dim, device=dev)
        self.mask = torch.zeros(B, dim, device=dev)
        self.points = torch.zeros(B, structure.model.bins, device=dev, dtype=torch.float32)
        self.hid = torch.zeros(B, structure.model.mlp[-1].out_features, device=dev, dtype=torch.float32)
        self._capture()

    def _step(self):
        with torch.no_grad():
            with self.structure._amp():
                out, hid = self.structure.model.forward(self.inp, self.mask, return_hid=True)
        self.points.copy_(out)
        self.hid.copy_(hid)

    def _capture(self):
        self._step()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        torch.cuda.synchronize()
        self.g = g

    def replay(self, inp, mask):
        self.inp.copy_(inp)
        self.mask.copy_(mask)
        self.g.replay()
        return self.points, self.hid


class _ExactGraph:
    """CUDA graph for the log_prob evaluation inside Interpolation.exact().

    Captures normalize(y) + model.log_prob(hid, t) + fp32 cast; the [B, bins+K]
    concat + log_softmax stays eager (identical results either way).
    """

    def __init__(self, structure, B, K, points_dtype):
        dev = structure.device
        model = structure.model
        self.structure = structure
        self.y = torch.zeros(K, B, device=dev, dtype=points_dtype)
        self.centers = torch.zeros(B, device=dev, dtype=torch.float32)
        self.scales = torch.ones(B, device=dev, dtype=torch.float32)
        self.hid = torch.zeros(B, model.mlp[-1].out_features, device=dev, dtype=torch.float32)
        self.out = torch.zeros(B, K, device=dev, dtype=torch.float32)
        self.width = model.scale[1] - model.scale[0]
        self.scale0 = model.scale[0]
        self._capture()

    def _step(self):
        yt = self.y.transpose(0, 1)[:, :, None]  # [B,K,1]
        yn = ((yt - self.centers[:, None, None]) / self.scales[:, None, None] + 1) / 2
        yn = yn * self.width + self.scale0
        with torch.no_grad():
            with self.structure._amp():
                out = self.structure.model.log_prob(self.hid, yn)  # [B,K]
        self.out.copy_(out)

    def _capture(self):
        self._step()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        torch.cuda.synchronize()
        self.g = g

    def replay(self, y, centers, scales, hid):
        self.y.copy_(y)
        self.centers.copy_(centers)
        self.scales.copy_(scales)
        self.hid.copy_(hid)
        self.g.replay()
        return self.out


class Structure:
    def __init__(self, dataset, bayesian_network, bins=32, hid_dim=64, dist_hid=64,
                 hid_residuals=2, dist_residuals=2,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Union[str, torch.dtype] = "fp32",
                 verbose=False):
        """
        device: 'cuda', 'cpu', 'mps' or torch.device. Defaults to the best available.
        dtype:  compute dtype for training and inference:
                - 'fp32' : full float32 (default)
                - 'fp16' : half precision via AMP autocast + grad scaling (CUDA)
                - 'bf16' : bfloat16 via AMP autocast (CUDA/CPU)
        """
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        self.bayesian_network = bayesian_network
        self.verbose = verbose
        if not isinstance(dataset, torch.Tensor):
            dataset = torch.tensor(dataset, dtype=torch.float32)
        dataset = dataset.to(device=self.device, dtype=torch.float32)
        self.dim = dataset.shape[-1]
        self.model = Generative(self.dim, hid_dim, bins=bins, dist_hid=dist_hid,
                                hid_residuals=hid_residuals, dist_residuals=dist_residuals)
        self.model = self.model.to(device=self.device)
        self.raw_dataset = dataset
        self.set_bins(bins)
        # lazily populated cuda-graph caches (fit / forward / exact per shape)
        self._fit_graph = None
        self._fwd_graphs = {}
        self._exact_graphs = {}
        if self.verbose:
            print(f"Structure on {self.device} (dtype={self.dtype})")

    # ------------------------------------------------------------------ AMP ---
    def _amp(self):
        """Context manager enabling autocast for the configured compute dtype."""
        if self.dtype == torch.float32:
            return nullcontext()
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=self.dtype)
        if self.device.type == "cpu" and self.dtype == torch.bfloat16:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        if self.device.type == "mps" and self.dtype == torch.float16:
            return torch.autocast("mps", dtype=torch.float16)
        # fp16 on CPU / unsupported combos: fall back to fp32 compute
        return nullcontext()

    def _scaler(self):
        """GradScaler for fp16 CUDA training; a no-op otherwise."""
        use_scaler = self.device.type == "cuda" and self.dtype == torch.float16
        return torch.amp.GradScaler(self.device.type, enabled=use_scaler)

    def to(self, device=None, dtype=None):
        """Move the structure (model, quantizer stats, grids, dataset) to device/dtype."""
        self._fit_graph = None
        self._fwd_graphs = {}
        self._exact_graphs = {}
        self.model._t_cache.clear()
        if device is not None:
            self.device = resolve_device(device)
            self.model = self.model.to(self.device)
            self.raw_dataset = self.raw_dataset.to(self.device)
            self.dataset = self.dataset.to(self.device)
            self.dataset_q = self.dataset_q.to(self.device)
            self.grids = self.grids.to(self.device)
            self.quantize.center = self.quantize.center.to(self.device)
            self.quantize.scale = self.quantize.scale.to(self.device)
        if dtype is not None:
            self.dtype = resolve_dtype(dtype)
        return self

    # ---------------------------------------------------------------- data ---
    def set_bins(self, bins):
        self.bins = bins
        self.model.bins = bins
        self.model._t_cache.clear()
        self._fit_graph = None
        self._fwd_graphs = {}
        self._exact_graphs = {}
        self.quantize = Quantize(self.raw_dataset, bins=bins)

        self.dataset = self.raw_dataset
        self.dataset = self.quantize.dequantize(
            self.quantize.quantize(self.raw_dataset, list(range(self.dim))),
            list(range(self.dim)),
        )

        grids = []
        for infer_ind in range(self.dim):
            grid = self.quantize.dequantize(torch.arange(self.bins, device=self.device)[None, :], [infer_ind])
            grids.append(grid)
        self.grids = torch.concat(grids)

        # precomputed quantized dataset: quantize() is a row-wise elementwise op,
        # so quantize(batch)[r] == dataset_q[perm[r]] bit-exactly, but computing
        # it once removes a full quantize pass from every training epoch.
        self.dataset_q = self.quantize.quantize(self.raw_dataset, list(range(self.dim)))

    # ------------------------------------------------------------- training --
    def fit(self, epochs=2048, batch_size=256, lr=0.01,
            loss_function: Literal['cross_entropy', 'mle'] = 'cross_entropy',
            random_conditional_prob=0.4, verbose=None):
        if verbose is None:
            verbose = self.verbose
        opt = torch.optim.AdamW(get_optim_groups(self.model), lr=lr,
                                fused=(self.device.type in ("cuda", "cpu")))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        scaler = self._scaler()
        dataset = self.dataset
        bayesian_network = self.bayesian_network

        is_bayesian_specified = bayesian_network is not None and len(bayesian_network) > 0

        running = torch.arange(batch_size, device=self.device)

        use_graph = (self.device.type == "cuda" and loss_function == 'cross_entropy'
                     and torch.cuda.is_available())
        # fp16's GradScaler is recreated per fit() call and its _scale tensor is
        # baked into the captured graph, so fp16 graphs must be re-captured.
        cacheable = self.dtype != torch.float16
        if use_graph:
            cached = self._fit_graph
            if cacheable and cached is not None and cached[0] == batch_size:
                fit_graph = cached[1]
            else:
                try:
                    fit_graph = _FitGraph(self, opt, scaler, batch_size)
                except Exception:
                    fit_graph = None
                if cacheable and fit_graph is not None:
                    self._fit_graph = (batch_size, fit_graph)
        else:
            fit_graph = None

        for i in range(epochs):
            perm = torch.randperm(len(dataset), device=self.device)[:batch_size]
            batch = dataset[perm]
            # now we must create random masks out of provided bayesian network
            # for P(X0|X1,X2) with X=[X0,X1,X2,X3]
            # mask=[1,-1,-1,0]
            mask = torch.zeros_like(batch)
            modelled_variable = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            if is_bayesian_specified:
                size = batch_size // len(bayesian_network) + 1
                for ind, imp in enumerate(bayesian_network):
                    part = batch_size * ind // len(bayesian_network)
                    mask_slice = mask[part:part + size]
                    mask_slice[:, imp[0]] = 1
                    for cond_var in imp[1:]:
                        mask_slice[:, cond_var] = -1
            else:
                conditional_mask = torch.rand_like(mask) < random_conditional_prob
                mask[conditional_mask] = -1
                mask[running, torch.randint(0, self.dim, (batch_size,), device=self.device)] = 1
            modelled_variable = (mask == 1).long().argmax(dim=-1)

            if fit_graph is not None:
                # quantize() is row-wise elementwise: quantize(batch)[running] ==
                # dataset_q[perm][running] bit-exactly.
                expected_ind = self.dataset_q[perm, modelled_variable]
                loss = fit_graph.replay(batch, mask, expected_ind)
                scaler.step(opt)
                scaler.update()
                sch.step()
                if verbose:
                    print(f"Loss:{loss.item():0.3f}")
            else:
                opt.zero_grad(True)
                with self._amp():
                    # ============================
                    if loss_function == 'mle':
                        y = batch[running, modelled_variable]
                        prob = self.forward(batch, mask, log_softmax=True)
                        loss = (-prob(y.unsqueeze(0))).mean()
                    # ============================
                    else:
                        pred = self.model(batch, mask)
                        expected_ind = self.quantize.quantize(batch, list(range(self.dim)))[running, modelled_variable]
                        loss = F.cross_entropy(pred, expected_ind)
                # ============================
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sch.step()
                if verbose:
                    print(f"Loss:{loss:0.3f}")

    # ------------------------------------------------------------ inference --
    def forward(self, batch, mask, log_softmax=False):
        modelled_variable = (mask == 1).long().argmax(dim=-1)
        pred, hid = self.model(batch, mask, return_hid=True)
        grids = self.grids[modelled_variable]
        grids = grids.to(dtype=pred.dtype) if grids.dtype != pred.dtype else grids
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
        dim = self.dim  # Number of features
        device = self.device

        # Store our generated continuous values
        samples = torch.zeros((batch_size, dim), device=device)
        bayesian_network = self.bayesian_network

        if bayesian_network is None:
            all = list(range(self.dim))
            bayesian_network = [all[-p - 1:] for p in range(self.dim)]
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

        fwd = self._fwd_graphs.get(batch_size)
        if fwd is None and device.type == "cuda" and torch.cuda.is_available():
            try:
                fwd = _ForwardGraph(self, batch_size)
            except Exception:
                fwd = None
            self._fwd_graphs[batch_size] = fwd

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
            if fwd is not None:
                points, _ = fwd.replay(x_in, mask)
                logits = points
            else:
                with self._amp():
                    # (Assuming model returns raw logits. Use .exp() if it returns log_softmax)
                    logits = self.model(x_in, mask)
            probs = torch.softmax(logits.float(), dim=-1)

            # 4. Sample bin indices from the categorical distribution
            sampled_bins = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Shape: [batch_size]

            # 5. Dequantize bins to continuous values and store them
            # unsqueeze to [batch_size, 1] to match your dequantize expectations
            sampled_vals = self.quantize.dequantize(sampled_bins.unsqueeze(-1), dimensions=[target_var]).squeeze(-1)
            samples[:, target_var] = sampled_vals.float()

        return samples

    def conditional_dist(self, condition: torch.Tensor, variables: List[int]):
        """
        Return conditional distribution over provided condition variables.
        You can condition by any variables(it may even not be learned).

        condition:
            tensor of shape `[BATCH,D]`

        variables:
            list of length `(D+1)`, where `variable_ind[0]` is index of variable that
            you want to get probability dist, and `variable_ind[1:]` is indices of `condition`
            dimensions relative to input dataset
        """

        if not isinstance(condition, torch.Tensor):
            condition = torch.tensor(condition, dtype=torch.float32)
        condition = condition.to(device=self.device, dtype=torch.float32)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)

        inp = torch.zeros((condition.shape[0], self.dim), device=self.device, dtype=torch.float32)
        inp[:, variables[1:]] = condition
        infer_ind = variables[0]
        mask = torch.zeros_like(inp)
        mask[:, infer_ind] = 1
        mask[:, variables[1:]] = -1
        grid = self.grids[infer_ind]
        B = condition.shape[0]
        fwd = self._fwd_graphs.get(B)
        if fwd is None and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                fwd = _ForwardGraph(self, B)
            except Exception:
                fwd = None
            self._fwd_graphs[B] = fwd
        if fwd is not None:
            points, hid = fwd.replay(inp, mask)
        else:
            with self._amp():
                points, hid = self.model.forward(inp, mask, return_hid=True)
            points = points.float()
            hid = hid.float()
        grid = grid.float().expand_as(points)
        return Interpolation(grid, points, hid, self.model,
                             self.quantize.center[[infer_ind]].float(),
                             self.quantize.scale[[infer_ind]].float())

    def _exact(self, interp, y):
        """Evaluate interp.exact(y), using a captured CUDA graph when available."""
        B, K = y.shape[1], y.shape[0]
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return interp.exact(y)
        g = self._exact_graphs.get((B, K))
        if g is None:
            try:
                g = _ExactGraph(self, B, K, interp.points.dtype)
            except Exception:
                g = None
            self._exact_graphs[(B, K)] = g
        if g is None:
            return interp.exact(y)
        y = y.to(device=self.device, dtype=interp.points.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        out = g.replay(y, interp.centers, interp.scales, interp.hid)  # [B,K] fp32
        # eager concat + log_softmax, identical to Interpolation.exact()
        out = torch.concat([interp.points, out], -1)  # [B,bins+K]
        out = out.log_softmax(-1)
        return out[:, -K:]  # [B, K], same shape convention as Interpolation.exact

    def full_joint_log(self, data: torch.Tensor):
        """
        Computes log of the full joint probability of the provided data points
        using the chain rule defined by the bayesian_network.

        P(X) = prod P(X_i | Parents(X_i))

        data: tensor of shape [B, dim]
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to(device=self.device, dtype=torch.float32)
        if data.ndim == 1:
            data = data.unsqueeze(0)

        device = self.device
        log_joint = torch.zeros(data.shape[0], device=device)

        bayesian_network = self.bayesian_network
        if bayesian_network is None:
            all_vars = list(range(self.dim))
            bayesian_network = [all_vars[-p - 1:] for p in range(self.dim)]

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
            log_p_y = self._exact(interp, target_values.unsqueeze(0)).squeeze(0)[:, 0]  # Shape: [B]

            # 4. Accumulate log-probabilities
            log_joint += log_p_y.float()

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
        data = data.to(device=self.device, dtype=torch.float32)
        if data.ndim == 1:
            data = data.unsqueeze(0)

        if data.shape[1] != len(variables):
            raise ValueError(f"data shape {data.shape} does not match number of variables {len(variables)}")

        device = self.device
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
            sorted_bn = [variables[-p - 1:] for p in range(len(variables))]

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
            log_p_y = self._exact(interp, target_values.unsqueeze(0)).squeeze(0)[:, 0]
            log_joint += log_p_y.float()

        return log_joint
