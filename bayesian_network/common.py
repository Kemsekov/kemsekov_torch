"""Shared building blocks for the bayesian_network implementations.

Both `continuous.py` and `discrete.py` build on the same infrastructure:

* device / dtype resolution,
* `Quantize`, `Prod`, `Residual` helper modules,
* CUDA-graph captured training (`_FitGraph`) and inference (`_ForwardGraph`) steps,
* the AMP-aware `_StructureBase` mixin holding `fit`, `generate`, `set_bins`,
  `full_joint_log`, `partial_joint_log`, `_amp`, `_scaler` and `to`.

The per-implementation models (`Generative`) and interpolation classes remain
in their respective files; subclasses only need to provide `conditional_dist`,
`forward`, `_make_fwd_graph` (optionally with hidden states) and `_evaluate`.
"""
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


def topological_sort(bayesian_network):
    """Order a bayesian network so every condition precedes its target variable."""
    sorted_bn = []
    sampled_vars = set()
    remaining_bn = list(bayesian_network)

    while remaining_bn:
        progressed = False
        for dependency in list(remaining_bn):
            target_var = dependency[0]
            cond_vars = set(dependency[1:])
            if cond_vars.issubset(sampled_vars):
                sorted_bn.append(dependency)
                sampled_vars.add(target_var)
                remaining_bn.remove(dependency)
                progressed = True
        if not progressed:
            raise ValueError("bayesian_network contains cyclic dependencies")
    return sorted_bn


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


def model_hid_dim(model):
    """Width of the hidden state produced by the model's encode path.

    Works for both the continuous model (whose mlp ends in a Linear with
    `dist_hid` outputs) and the discrete model (whose mlp ends in an
    activation, so the hidden width equals `expand.out_features`).
    """
    last = model.mlp[-1]
    if isinstance(last, nn.Linear):
        return last.out_features
    return model.expand.out_features


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
        torch.cuda.synchronize()  # drain any pending work so capture is stable
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
    """CUDA graph for model(inp, mask) under no_grad + amp.

    Points/hid come out in fp32 (matches the .float() casts done by the eager
    inference paths) while the internal compute keeps the configured dtype.
    When `return_hid=True` the model must support `forward(x, mask,
    return_hid=True)` and return `(out, hid)`.
    """

    def __init__(self, structure, B, return_hid=False):
        dev = structure.device
        dim = structure.dim
        model = structure.model
        self.structure = structure
        self.return_hid = return_hid
        self.inp = torch.zeros(B, dim, device=dev)
        self.mask = torch.zeros(B, dim, device=dev)
        self.points = torch.zeros(B, model.out.out_features, device=dev, dtype=torch.float32)
        self.hid = None
        if return_hid:
            self.hid = torch.zeros(B, model_hid_dim(model), device=dev, dtype=torch.float32)
        self._capture()

    def _step(self):
        with torch.no_grad():
            with self.structure._amp():
                if self.return_hid:
                    out, hid = self.structure.model.forward(self.inp, self.mask, return_hid=True)
                else:
                    out = self.structure.model(self.inp, self.mask)
        self.points.copy_(out)
        if self.return_hid:
            self.hid.copy_(hid)

    def _capture(self):
        torch.cuda.synchronize()  # drain any pending work so capture is stable
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


class _StructureBase:
    """Shared structure: AMP helpers, CUDA-graph cached fit/generate and the
    joint-log-probability computation over a bayesian network.

    Subclasses provide `__init__` (building `self.model`, `self.device`,
    `self.dtype`, `self.raw_dataset`, `self.bayesian_network`, `self.dim` and
    the `_fit_graph` / `_fwd_graphs` caches, then calling `self.set_bins`),
    `conditional_dist`, `forward` and optionally `_make_fwd_graph` / `_evaluate`.
    """

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
        t_cache = getattr(self.model, "_t_cache", None)
        if t_cache is not None:
            t_cache.clear()
        self._fit_graph = None
        self._fwd_graphs = {}
        self.quantize = Quantize(self.raw_dataset, bins=bins)

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

    # ---------------------------------------------------------- cuda graphs ---
    def _make_fwd_graph(self, B):
        """Build a forward CUDA graph for batch size B (override to add hid)."""
        return _ForwardGraph(self, B)

    def _get_fwd_graph(self, B):
        fwd = self._fwd_graphs.get(B)
        if fwd is None and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                fwd = self._make_fwd_graph(B)
            except Exception:
                fwd = None
            self._fwd_graphs[B] = fwd
        return fwd

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
                    if loss_function == 'mle':
                        y = batch[running, modelled_variable]
                        prob = self.forward(batch, mask, log_softmax=True)
                        loss = (-prob(y.unsqueeze(0))).mean()
                    else:
                        pred = self.model(batch, mask)
                        expected_ind = self.quantize.quantize(batch, list(range(self.dim)))[running, modelled_variable]
                        loss = F.cross_entropy(pred, expected_ind)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sch.step()
                if verbose:
                    print(f"Loss:{loss:0.3f}")

    # ------------------------------------------------------------ inference --
    @torch.no_grad()
    def generate(self, batch_size=128):
        dim = self.dim  # Number of features
        device = self.device

        # Store our generated continuous values
        samples = torch.zeros((batch_size, dim), device=device)
        bayesian_network = self.bayesian_network

        if bayesian_network is None:
            all_vars = list(range(self.dim))
            bayesian_network = [all_vars[-p - 1:] for p in range(self.dim)]
        # --- Robust Topological Sort ---
        # Ensures we only sample a variable if all its conditions have already been sampled
        sorted_bn = topological_sort(bayesian_network)

        fwd = self._get_fwd_graph(batch_size)

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
                logits, _ = fwd.replay(x_in, mask)
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

    # ---------------------------------------------------------- joint logs ---
    def _evaluate(self, interp, y):
        """Evaluate the interpolation at values y, returning a [B] log-prob."""
        return interp(y.unsqueeze(0)).squeeze(0)

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
        sorted_bn = topological_sort(bayesian_network)

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

            log_p_y = self._evaluate(interp, target_values)  # Shape: [B]

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
            sorted_bn = topological_sort(sub_bn)

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

            log_p_y = self._evaluate(interp, target_values)

            # 4. Accumulate log-probabilities
            log_joint += log_p_y.float()

        return log_joint
