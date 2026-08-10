# Fused Triton SelfAttention / CrossAttention — optimization report

## Target
`kemsekov_torch.attention.SelfAttention` and `CrossAttention` (fp32, 2D conv-layout
vision attention, `add_rotary_embedding=True`, `xsa=True`, `add_absolute_pos=True`),
forward + backward, on RTX 5080 (Blackwell sm_120), torch 2.13 + triton 3.7.1.

## What torch.compile did (baseline)
`torch.compile(mode="max-autotune")` gave **~1.0x vs eager** (4.96 ms vs 4.93 ms for
SelfAttention fwd+bwd at L=1024):
- `_scaled_dot_product_efficient_attention` (fwd AND bwd) stayed **extern** — the
  cutlass **sm80** fp32 FMA kernels (no tensor cores) = ~50% of GPU time
  (0.36 ms fwd + 1.27 ms bwd per call).
- ~10 small kernels fwd + ~14 bwd around it: qkv transpose-copies (2 full qkv
  copies for the rotary + layout), xsa vector_norm/mul/sub kernels, output
  transpose-copy, rotary setitem CopySlices in the backward.
- `1x1` convs (to_qkv/to_out) stayed cuBLAS extern.

## What we hand-wrote (`kemsekov_torch/fast_attention/__init__.py`, `FastSelfAttention` /
`FastCrossAttention` — subclasses with identical `__init__` + `state_dict`)
Four triton kernels replace the whole SDPA region (tf32 tensor-core dots, fp32
accumulation — within the 1e-2 tolerance gate):
1. `_rot_kernel` — RoPE written directly into a `[B,{3|2},H,L,D]` buffer (no
   permute/transpose copies); also the inverse (backward). k-only for cross.
2. `_attn_fwd_kernel` — flash-attention fwd (online softmax) + XSA projection,
   stores A (bwd), LSE, and A_x (out-proj input) directly in the `[B,L,inner]`
   layout.
3. `_outproj_kernel` — `A_x @ Wᵀ + bias + residual`, writes the final
   `[B, dim, H, W]` in-place (kills the attn materialization + transpose copy).
4. `_bwd_prep_kernel` (large L) / single `_attn_bwd_kernel_single` (small L) —
   `dY@Wᵀ` (K-chunked), XSA backward (incl. exact dv̂ projection term),
   `dW_out`/`dbias` accumulation (staged per-(b,h) atomics + `torch.sum`), and
   the two-pass flash backward (LSE-recomputed softmax, dk/dv atomics, dq local).

Also fixed `AbsoluteRelativePositionalEmbedding` in `kemsekov_torch/attention.py`
(buffer mutations `cached_grid`/`max_dim_size` moved behind a
`@torch.compiler.disable` helper) so the eager module compiles/cudagraphs at all.

## Pitfalls found along the way (Blackwell + triton 3.7)
- `tl.trans` of a loaded tile is **miscompiled** when the tile's first dim has a
  stride > 1 (rows silently lost) — loads must be done in the natural
  `[rows, D]` orientation (hence the `[B,H,L,D]` qkv layout from the rot kernel).
- fp32 dot operands exceed the 101 KB smem limit with >2 dots live → split the
  backward into prep + flash kernels and use BN=32.
- Python-float scalars can bind as **fp64** under dynamo (scale, log2e) and
  poison the fp32 dots → explicit `.to(tl.float32)` casts.
- The un-rotated v-part of `dqkv_grad` was `torch.empty` garbage (masked by
  zeroed allocator memory in isolation) → explicit transpose-copy.
- Inductor may hand the custom op non-contiguous (channels-last) conv outputs →
  `.contiguous()` in the autograd Function.

## Correctness (gate: rtol=1e-2, atol=1e-2 vs eager, fp32)
All checks pass at spatial 8/16/32/64 (L=64..4096), forward AND all parameter
gradients (to_qkv/to_q/to_kv/to_out/norms/abs_emb):
- forward max_abs ~1e-4, input grads ~1e-10, weight grads ~1e-5..1e-4
  (tf32-level; bias grad ~6e-3, same as eager's own run-to-run scale).

## Measured effect (fwd+bwd, B=4, dim=512, heads=8, head_dim=64, µs)

| L | module | eager | torch.compile | fused | vs eager | vs compile |
|---|--------|-------|---------------|-------|----------|------------|
| 64 | SelfAtt | 1303 | 897 | 778 | **1.68x** | 1.15x |
| 1024 | SelfAtt | 4956 | 4932 | 2404 | **2.06x** | **2.05x** |
| 4096 | SelfAtt | 48402 | 49004 | 21041 | **2.30x** | **2.33x** |
| 64 | CrossAtt | 1050 | 864 | 674 | **1.56x** | 1.28x |
| 1024 | CrossAtt | 4758 | 4837 | 2472 | **1.92x** | **1.96x** |
| 4096 | CrossAtt | 46381 | 49124 | 21221 | **2.19x** | **2.31x** |

The main wins: SDPA fwd/bwd on tensor cores (tf32) instead of the sm80 FMA
cutlass kernels, plus elimination of ~4 full-tensor copies and the attn/outproj
materializations.

## Caveats
- Wrapping the Fast* module in `torch.compile` runs (default mode) but is
  flaky after a few iterations (AOTAutograd's memory planner cannot see the
  autograd Function's internal buffers) — the module is meant to be used
  standalone; it needs no compile.
- Dropout>0 (training) and `linear=True` cross-attention fall back to the
  original eager forward.
- CA with `xsa=True` requires Lq == Lk (same as the eager module itself).
