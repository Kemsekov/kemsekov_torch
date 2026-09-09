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
├── base.py        # shared layer definitions, head movement, L2-norm q/k projection
├── serial.py      # GatedDelta2        -- token-by-token reference loop
├── scan.py        # scan kernel: chunk prep (WY), unit-lower solves,
│                  #   affine prefix scan (Blelloch / sequential), manual backward
├── chunked.py     # GatedDelta2Scan    -- chunked parallel-scan module
└── __init__.py    # public exports
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

## Implementation paths (`GatedDelta2Scan`)

The module picks the mixing implementation automatically:

* **Fused recurrent Triton kernels** (`recurrent.py`, used on CUDA when the
  mixing tensors are float32 and `QK_dim`, `V_dim <= 256`): the exact
  per-token recurrence with the state held in registers (FLA
  `fused_recurrent` style), one block per (batch, head) tiled over the value
  dim. This is numerically the same recurrence as the serial reference, needs
  no WY solve / chunk scan and handles any decay strength exactly (hard
  resets included). Forward + checkpointed manual backward.
* **Chunked differentiable scan** (`scan.py::_scan_fwd`): the WY-parallel
  formulation with float64 decay normalization and autograd backward —
  used on CPU, for float64 models (gradcheck), or when the dims exceed the
  Triton path.
* **Sequential float64 fallback** (`Delta2ScanFn`): only when the cumulative
  decay of a chunk leaves the float64 range (extreme decay); exact and
  rarely used.

`validate.py` exercises all precisions (float32/float16/bfloat16); the
serial-vs-scan agreement holds across every path.

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
gradients through the whole model):

```python
import torch
from gated_delta_2 import GatedDelta2, GatedDelta2Scan

torch.manual_seed(0)
m1 = GatedDelta2(32, 64, 20, heads=2)
torch.manual_seed(0)
m2 = GatedDelta2Scan(32, 64, 20, heads=2)
for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
    p2.data.copy_(p1.data)

x = torch.randn(7, 100, 32, requires_grad=True)
y1, y2 = m1(x), m2(x)
print((y1 - y2).abs().max().item() / y1.abs().max().item())   # ~1e-6
g1 = torch.autograd.grad(y1.square().mean(), m1.parameters())
g2 = torch.autograd.grad(y2.square().mean(), m2.parameters())
print(max(((a - b).abs().max() / a.abs().max()).item() for a, b in zip(g1, g2)))
```

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

## Auto-selected backend (this package)

`GatedDelta2Scan` here measures nothing at runtime: it *routes* each call to
the fastest of the two CUDA kernels by a rule tuned offline (optuna over a
432-config benchmark grid, objective = mean over the grid of
`t_chosen / t_gd21`, so every shape contributes equally):

* plain per-row fused kernels (the `gated_delta_2` implementation) when
  `L < 1024`, or `DK/DV < 16`, or the state tiles are wide
  (`DK >= 64 and DV >= 64`, or `rows = batch*heads >= 32` together with
  `DK >= 64` or `DV >= 64`);
* the two-level split-recurrent kernels (the `gated_delta_21`
  implementation) for long sequences (`L >= 1024`) with narrow tiles.

On CPU and for shapes that cannot use Triton the module falls back to the
same chunked scan as the other two packages (identical numerics), so all
three are equal there; the CPU default `chunk=64` was verified optimal
against 16..512 on the CPU grid.
