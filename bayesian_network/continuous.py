import torch
import torch.nn as nn
from typing import List, Optional, Union
from kemsekov_torch.bayesian_network.common import (
    Prod,
    Residual,
    resolve_device,
    resolve_dtype,
    make_chain_bn,
    _ForwardGraph,
    _StructureBase,
)


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
        torch.cuda.synchronize()  # drain any pending work so capture is stable
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


class Structure(_StructureBase):
    def __init__(self, dataset, bayesian_network="all", bins=32, hid_dim=64, dist_hid=64,
                 hid_residuals=2, dist_residuals=2,
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

    # continuous-specific graph cache bookkeeping ---------------------------
    def set_bins(self, bins):
        self._exact_graphs = {}
        super().set_bins(bins)

    def to(self, device=None, dtype=None):
        self._exact_graphs = {}
        self.model._t_cache.clear()
        return super().to(device, dtype)

    def _make_fwd_graph(self, B):
        # the continuous model can also return the curve parametrization hid
        return _ForwardGraph(self, B, return_hid=True)

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
        fwd = self._get_fwd_graph(B)
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
        return out[:, -len(y):]  # [B, K], same shape convention as Interpolation.exact

    def _evaluate(self, interp, y):
        """Evaluate interp.exact(y) (CUDA-graph backed), returning a [B] log-prob."""
        return self._exact(interp, y.unsqueeze(0)).squeeze(0)[:, 0]
