# Gated Delta Rule-2

Recurrent linear-attention layers implementing the **Gated Delta Rule-2**
operator from *Gated DeltaNet-2: Decoupling Erase and Write in Linear
Attention* (arXiv:2605.22791). It combines a channel-wise decay with a
delta-rule memory edit whose erase and write directions are decoupled by two
independent gates:

```
decayed:      S̄_t    = diag(α_t) S_{t-1}
read:         r_t    = S̄_tᵀ e_t                    e_t = b_t ⊙ k_t        (erase gate)
write:        diff_t = z_t − r_t                    z_t = w_t ⊙ v_t        (write gate)
state:        S_t    = S̄_t + k_t diff_tᵀ
output:       o_t    = S_tᵀ q_t
```

Equivalently `S_t = (I − k_t e_tᵀ) diag(α_t) S_{t−1} + k_t z_tᵀ`.
`q_t` and `k_t` are L2-normalized per head before use (per the paper's block
design), which also keeps the delta rule well conditioned.

The three module files provide the same operator with different mixing
strategies and identical parameters/buffers (state dicts are interchangeable):

| class | file | mixing | when to use |
|---|---|---|---|
| `GatedDelta2` | `serial.py` | python loop over tokens | numerical reference, short sequences |
| `GatedDelta2Scan` | `chunked.py` + `scan.py` | chunked WY parallel scan | training / long sequences (**recommended**) |

## Package layout

```
gated_delta_2/
├── base.py          # shared layer definitions, head movement, L2-norm q/k projection
├── serial.py        # GatedDelta2        -- token-by-token reference loop
├── scan.py          # reference scan kernel: chunk prep (WY), unit-lower solves,
│                    #   affine prefix scan (Blelloch / sequential), manual backward
├── fast.py          # optimized chunked WY scan (single inverted WY factor, lean cache)
├── recurrent.py     # reference fused per-token recurrence (plain + two-level split)
├── recurrent_fast.py# optimized fused per-token recurrence (atomic/buffered backward)
├── chunked.py       # GatedDelta2Scan    -- dispatch + chunked parallel-scan module
├── tuning.py        # runtime backend autotuner (cached under ~/.cache)
├── validate.py      # serial-agreement self-check (fwd + grad) for every backend
└── __init__.py      # public exports
```

All modules consume the same per-head inputs produced by
`base.GatedDelta2Base._project` (projections → gates → head split → L2 norm →
`e_t = b_t ⊙ k_t`, `z_t = w_t ⊙ v_t`, `α_t = exp(g_t)`); only the sequence
mixing differs, so the three classes are exact drop-ins of each other.

## Quickstart

```python
import torch
from gated_delta_2 import GatedDelta2Scan

gd = GatedDelta2Scan(32, 64, 20, heads=2)
x = torch.randn((7, 100, 32))
print(gd(x).shape)  # torch.Size([7, 100, 20])
```

## Mixed precision (fp16 / bf16)

The module is safe to run in float16 or bfloat16
(`model.half()` / `model.bfloat16()` — parameters and buffers stay in the
model precision, and forward+backward work for both `GatedDelta2` and
`GatedDelta2Scan`):

* The linear projections and gates run in the model precision.
* The mixing-critical tensors (L2-normalized `q`/`k`, the decay factors
  `α_t = exp(g_t)`, and the gated `e_t`, `z_t`) are computed in float32 to
  keep the recurrence well conditioned and to avoid fp16 underflow of the
  decay exponentials.
* The serial loop and the scan kernel both consume these float32 tensors, so
  the two implementations keep their ~1e-6 agreement; only the final outputs
  are cast back to the model precision.
* The mixing kernels run with autocast disabled internally, so they are safe
  inside `torch.autocast` / `accelerate` mixed-precision (e.g.
  `mixed_precision='bf16'`) and `torch.compile` — only the linear projections
  and gates are autocast to the model precision.

## Grouped-query heads (GQA)

All classes take `kv_heads` (default `heads`; `heads` must be divisible by
`kv_heads`). Each query head `h` reads the key/value head
`h // (heads // kv_heads)`, the same grouping as
`F.scaled_dot_product_attention(..., enable_gqa=True)`. The erase and decay
gates ride with K, the write gate with V, so all query heads of a group share
one state and one value path; only their queries differ. The projections
shrink with the kv head count (`QK` holds `heads` Q heads + `kv_heads` K
heads; `V`, `erase_gate`, `write_gate`, `decay_gate` scale with `kv_heads`),
while the output projection still scales with `heads`. The sequence-mixing
kernels are unchanged: `_project` expands the kv tensors to query heads before
mixing (`base._expand_kv_heads`), so serial, scan and Triton paths all support
GQA.

## Implementation paths (`GatedDelta2Scan`)

The module picks the mixing implementation automatically:

* **Fused recurrent Triton kernels** (`recurrent_fast.py`, used on CUDA when
  the mixing tensors are float32 and `QK_dim`, `V_dim <= 256`): the exact
  per-token recurrence with the state held in registers (FLA
  `fused_recurrent` style), one block per (batch, head) tiled over the value
  dim. This is numerically the same recurrence as the serial reference, needs
  no WY solve / chunk scan and handles any decay strength exactly (hard
  resets included). Forward + checkpointed manual backward.  These kernels
  are the *preferred* path: the dispatch uses them whenever the constraints
  above are met (and the autotuner always benchmarks them).
* **Optimized chunked WY scan** (`fast.py`): the WY-parallel formulation with
  float64 decay normalization, used on CPU, for float64 models (gradcheck),
  for wide dims and for long sequences where it measures faster.
* **Chunked differentiable scan** (`scan.py::_scan_fwd`): branch-free
  autograd variant for `torch.compile` and for regimes with moderate decay.
* **Sequential float64 fallback** (`Delta2ScanFn`): when the chunked WY form
  cannot represent the cumulative decay (extreme decay / hard resets); exact
  and rarely used.

`validate.py` exercises all precisions (float32/float16/bfloat16) and every
backend against the serial reference; the serial agreement holds across every
path.

## How the parallel scan works (`scan.py`)

1. The sequence is split into chunks of size `C` (default 64). Inside a
   chunk the channel-wise decay is absorbed into the rank-one factors by a
   decay-normalized basis (`γ_r = Π α_i` computed in log space):
   `k̄_r = γ_r⁻¹⊙k_r`, `ē_r = γ_r⊙e_r`, `q̃_r = γ_r⊙q_r`, so the recurrence
   becomes a pure asymmetric delta rule.
2. Per chunk the WY matrices `T = tril(ĒK̄ᵀ, −1)` and the unit-lower-triangular
   solve `(I+T)[Y|U] = [Ē|Z]` collapse all token interactions of the chunk
   into one affine map on the chunk-start state
   `S' = diag(γ_C) S + K_tailᵀ(U − Y S)` and one output block
   `O = Q̃S + Aqk(U − YS)`.
3. Chunk-start states are propagated with an **affine parallel scan** over the
   ~`L/C` chunk maps (`scan` mode: work-efficient Blelloch scan, depth
   `O(log L)`; `seq` mode: plain sequential loop for few chunks; `auto`
   chooses per call).
4. The **backward pass is manual** (no autograd through the scan): a reversed
   affine scan computes the chunk-state adjoints and per-chunk VJPs push
   gradients through the WY inverse (`dT = −tril(Aᵀ dA Aᵀ, −1)`), the
   triangular solves, and the elementwise gates/decays (reverse cumsum).

## Options (`GatedDelta2Scan`)

| argument | values | meaning |
|---|---|---|
| `chunk` | int (default 64) | chunk size; 32–128 typical, 64 is a good default |
| `scan_mode` | `"auto"` / `"scan"` / `"seq"` | state propagation over chunks |
| `prec` | `"fp32"` / `"mixed"` / `"fp64"` | scan-internal precision |
| `erase_gate_scale` | float (default 1.0) | erase gate range multiplier |

`prec` details:

* `"fp32"` (default): pure float32 pipeline. With L2-normalized keys this
  matches the fp32 serial reference to ~1e-6 relative. If cumulative decay in
  a chunk would leave the float32-representable range, that call automatically
  re-runs in float64.
* `"mixed"`: float64 only for the decay cumsum/exp, everything else float32
  (previous default).
* `"fp64"`: full float64 internals (also used automatically when the module
  or inputs are float64, e.g. for `torch.autograd.gradcheck`).

If the decay is so strong that even float64 cannot represent the
decay-normalized factors of a chunk (cumulative log-decay below ~-700, e.g.
hard resets from underflowed `α_t`), the scan falls back to an exact
per-token float64 recurrence with a matching manual adjoint, so `GatedDelta2`
and `GatedDelta2Scan` stay in agreement (outputs and gradients) at any decay
strength.

## Validation

Quick check that a parallel scan matches the serial reference (outputs and
gradients through the whole model).  Note that the default init zeroes the
output projection, which makes the mixing path irrelevant to the loss and the
mixing gradients exactly zero — randomize the parameters (or run
`python -m gated_delta_2.validate`, which does) to exercise the backward:

```python
import torch
from gated_delta_2 import GatedDelta2, GatedDelta2Scan

torch.manual_seed(0)
m1 = GatedDelta2(32, 64, 20, heads=2)
with torch.no_grad():
    for name, p in m1.named_parameters():
        p.copy_(torch.randn_like(p) * 0.15)
        if name == "decay":
            p.abs_()
torch.manual_seed(0)
m2 = GatedDelta2Scan(32, 64, 20, heads=2)
for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
    p2.data.copy_(p1.data)

x = torch.randn(7, 100, 32, requires_grad=True)
y1, y2 = m1(x), m2(x)
print((y1 - y2).abs().max().item() / y1.abs().max().item())   # ~1e-7
g1 = torch.autograd.grad(y1.square().mean(), m1.parameters())
g2 = torch.autograd.grad(y2.square().mean(), m2.parameters())
print(max(((a - b).abs().max() / a.abs().max()).item() for a, b in zip(g1, g2)))
```

`python -m gated_delta_2.validate` sweeps the dtypes, GQA configurations and
every backend (`auto`, `triton`, `split`, `scan`, `autograd`, plus the chunked
scan with Triton disabled) and checks the forward, input-gradient and
parameter-gradient agreement against `GatedDelta2`.

Measured on CPU (torch 2.14, 6 threads, dim=32, QK=64, V=20, heads=2, B=7,
fp32 default): forward+backward of `GatedDelta2Scan` vs `GatedDelta2` is
~5× faster at L=100, ~13× at L=512, ~35× at L=1024, ~70× at L=2048.
Agreement (fp32): output, input-gradient and parameter-gradient relative
errors ~1e-6 to ~1e-5 across L = 100 … 8192.

## References

* A. Hatamizadeh, Y. Choi, J. Kautz: *Gated DeltaNet-2: Decoupling Erase and
  Write in Linear Attention*, arXiv:2605.22791.
* Delta-rule parallelization background: Yang et al., *Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length* (DeltaNet).

## Dispatch (`chunked.py`)

`GatedDelta2Scan` routes each call as follows:

* **Fused per-token Triton recurrence** (`recurrent_fast.py`) whenever its
  kernel constraints are met (CUDA, float32 mixing tensors,
  `QK_dim, V_dim <= 256`).  This is the exact recurrence of the serial
  reference (no WY solve, no decay normalization), so it handles any sequence
  length and any decay strength (hard resets included).  It is the preferred
  path and is never excluded from autotuning.
* **Optimized chunked WY scan** (`fast.py`) otherwise: CPU, float64 models,
  wide dims, or shapes where it measures faster (`scan`).
* **Autotuning** (`impl="auto"`, default): the first time a
  `(device, dtype, shape, training)` combination is seen, the applicable
  candidates (`triton`, `scan`, and `autograd` when the decay is moderate)
  are benchmarked on private cloned inputs, checked against each other
  (`5e-3` relative, finite) and the winner is cached in
  `~/.cache/gd2_tune.json`.  On CPU only `scan`/`autograd` are available, so
  all backends are equal there.
* **Static heuristic** (`GD2_AUTOTUNE=0`, `torch.compile`, or tuning failure):
  the fused recurrent kernel when applicable, else the chunked scan.
* `impl="triton" | "split" | "scan" | "autograd"` forces a backend.  The
  reference two-level recurrent kernels (`recurrent.py`, `"split"`) are kept
  as an explicit escape hatch and are never autotuned (slow first-use JIT);
  the fused recurrent path supersedes them whenever it is applicable.

## Optimized backends (`fast.py`, `recurrent_fast.py`, `tuning.py`)

`GatedDelta2Scan` now chooses its mixing backend at runtime; the sequence
mixing continues to match the serial reference (`GatedDelta2`) up to float
rounding on CPU and CUDA for float32/float16/bfloat16 and any decay strength.

* `fast.py` -- chunked WY scan that inverts the unit-lower WY factor once
  (`Tinv = (I + tril(E K^T, -1))^-1`) and reuses it as batched matmuls in the
  forward and backward passes, caches a leaner set of intermediates
  (`Kb/Eb/Qb/Kt` are recomputed), and can drop `Tinv/Y/U` as well when the
  `budget` (default 256 MB, `GD2_SCAN_BUDGET_MB`) is exceeded.
* `recurrent_fast.py` -- fused per-token Triton recurrence whose backward
  accumulates key-side gradients either with per-value-tile buffers plus
  reduction kernels or with relaxed atomic adds into `(B, L, DK)`, switching
  at `GD2_ATOMIC_MB` (default 64 MB) to keep the scratch small.
* `tuning.py` -- first time a `(device, dtype, shape, training)` combination
  is seen, the applicable backends are benchmarked on private cloned inputs
  (one warmup + timed runs; forward-only in inference, forward+backward in
  training), checked against the reference scan (`5e-3` relative, finite),
  and the winner is cached in `~/.cache/gd2_tune.json`.  On CUDA the peak
  memory of each candidate is measured as well and, among candidates within
  10% of the fastest, the least memory-hungry one wins.

`impl` selects a specific backend instead of autotuning:

| impl | backend |
|---|---|
| `"auto"` (default) | autotuned |
| `"triton"` | fused per-token Triton recurrence |
| `"split"` | two-level split recurrent kernels |
| `"scan"` | optimized chunked WY scan (`fast.py`) |
| `"autograd"` | branch-free differentiable scan, made for `torch.compile` |

Under `torch.compile` the tuner is bypassed and the static heuristic picks
the preferred custom kernel (fused recurrent Triton when applicable, else the
chunked scan); the kernels are opaque to Dynamo, so the compiler still fuses
the projections.  The differentiable scan is only selected when every chunk's
cumulative log-decay stays well inside the fp32 range: its normalization is
fp32 and the compiled backward overflows when the decay-normalized factors
approach the fp32 limit, even if the forward is finite (the safe chunked scan
falls back to fp64 / sequential internally).  Set `GD2_AUTOTUNE=0` to disable
runtime tuning and use the static heuristic.
