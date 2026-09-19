"""Quick self-validation of the package (CPU-friendly, low memory).

Verifies that the parallel-scan module matches the serial reference on CPU and
(when available) CUDA for float32, float16 and bfloat16 -- forward outputs,
input gradients and parameter gradients -- both dense (``heads == kv_heads``)
and with grouped-query heads (``kv_heads < heads``), including the CUDA
recurrent Triton path and the chunked-scan path. Also checks that state dicts
are interchangeable. Run with:

    python -m gated_delta_2.validate

The full cross-length comparison suite lives in the repository tests
(see README); run heavy cases one at a time and watch memory usage.
"""
import torch

from . import GatedDelta2, GatedDelta2Scan
from . import recurrent

# tolerated forward / gradient relative error per dtype
TOL = {
    torch.float32: (1e-3, 1e-2),
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (5e-3, 5e-2),
}

# (heads, kv_heads): dense reference, GQA groups of 2/3, single kv head
GQA_CONFIGS = [(2, 2), (4, 2), (6, 3), (4, 1)]


def _max_rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-9)


def _check_dtype(dtype, device="cpu", heads=2, kv_heads=None, force_scan=False):
    torch.manual_seed(0)
    B, L, dim, qk, vd = 3, 64, 32, 64, 20
    x = torch.randn(B, L, dim, requires_grad=True).to(device=device, dtype=dtype)

    serial = GatedDelta2(dim, qk, vd, heads=heads, kv_heads=kv_heads).to(
        device=device, dtype=dtype
    )
    chunked = GatedDelta2Scan(dim, qk, vd, heads=heads, kv_heads=kv_heads).to(
        device=device, dtype=dtype
    )
    for (_, p1), (_, p2) in zip(serial.named_parameters(), chunked.named_parameters()):
        assert p1.shape == p2.shape
        p2.data.copy_(p1.data)

    # force_scan disables the fused Triton kernels so the chunked scan path is
    # exercised even on CUDA
    had_triton = recurrent._HAS_TRITON
    if force_scan:
        recurrent._HAS_TRITON = False
    try:
        y_ref = serial(x)
        y = chunked(x)
        rel = _max_rel(y_ref, y)
        g_ref = torch.autograd.grad(
            y_ref.square().mean(), [x, *serial.parameters()], retain_graph=True
        )
        g = torch.autograd.grad(y.square().mean(), [x, *chunked.parameters()])
    finally:
        recurrent._HAS_TRITON = had_triton
    rel_g = max(_max_rel(a, b) for a, b in zip(g_ref, g))
    tol_fwd, tol_grad = TOL[dtype]
    print(f"  {str(device):4s} {str(dtype):14s} heads={heads} kv_heads={kv_heads}"
          f"{' forced-scan' if force_scan else '':12s}: "
          f"fwd rel={rel:.2e}  grad rel={rel_g:.2e}")
    return rel < tol_fwd and rel_g < tol_grad


def _check_state_dict(heads=4, kv_heads=2):
    torch.manual_seed(1)
    B, L, dim, qk, vd = 3, 64, 32, 64, 20
    serial = GatedDelta2(dim, qk, vd, heads=heads, kv_heads=kv_heads)
    m2 = GatedDelta2Scan(dim, qk, vd, heads=heads, kv_heads=kv_heads)
    assert set(serial.state_dict()) == set(m2.state_dict())
    m2.load_state_dict(serial.state_dict())
    torch.manual_seed(1)
    m3 = GatedDelta2Scan(dim, qk, vd, heads=heads, kv_heads=kv_heads)
    with torch.no_grad():
        for (_, p1), (_, p3) in zip(serial.named_parameters(), m3.named_parameters()):
            p3.data.copy_(p1.data)
        torch.manual_seed(2)
        x2 = torch.randn(B, L, dim)
        d_sd = (m2(x2) - m3(x2)).abs().max().item()
        d_ref = _max_rel(serial(x2), m3(x2))
    print(f"  state-dict load == manual copy: {d_sd:.2e}  "
          f"chunked vs serial rel: {d_ref:.2e}")
    return d_sd == 0.0 and d_ref < 1e-3


def _repeat_head_blocks(p, per_head, groups):
    """Repeat each contiguous ``per_head``-row block ``groups`` times (the row
    layout produced by ``_expand_kv_heads``)."""
    return (
        p.reshape(-1, per_head, *p.shape[1:])
        .repeat_interleave(groups, dim=0)
        .reshape(-1, *p.shape[1:])
    )


def _check_gqa_grouping(device="cpu"):
    """A kv_heads model must equal a dense (kv_heads == heads) model whose
    key/value/gate weights are each group's shared weights repeated, i.e. the
    grouping is query head ``h`` -> kv head ``h // (heads // kv_heads)``."""
    torch.manual_seed(3)
    dim, qk, vd, heads, kv_heads, L = 32, 16, 12, 4, 2, 32
    grouped = GatedDelta2Scan(dim, qk, vd, heads=heads, kv_heads=kv_heads).to(device)
    dense = GatedDelta2Scan(dim, qk, vd, heads=heads, kv_heads=heads).to(device)
    g = heads // kv_heads
    per_head = {
        "erase_gate.weight": qk,
        "write_gate.weight": vd,
        "decay_gate.weight": qk,
        "decay_gate.bias": qk,
        "V.weight": vd,
    }
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(
            grouped.named_parameters(), dense.named_parameters()
        ):
            assert n1 == n2
            if n1 == "QK.weight":
                QG, KG = p1.split([qk * heads, qk * kv_heads], dim=0)
                KD = _repeat_head_blocks(KG, qk, g)
                p2.copy_(torch.cat([QG, KD], dim=0))
            elif n1 in per_head:
                p2.copy_(_repeat_head_blocks(p1, per_head[n1], g))
            else:
                assert p1.shape == p2.shape, n1
                p2.copy_(p1)
        for (n1, b1), (n2, b2) in zip(
            grouped.named_buffers(), dense.named_buffers()
        ):
            assert n1 == n2
            b2.copy_(b1)
    with torch.no_grad():
        x = torch.randn(2, L, dim, device=device)
        d = _max_rel(grouped(x), dense(x))
    print(f"  GQA grouping vs repeated dense weights: rel={d:.2e}")
    return d < 1e-5


def run():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ok = True
    for device in devices:
        for heads, kv_heads in GQA_CONFIGS:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                ok = _check_dtype(dtype, device, heads, kv_heads) and ok
            if device == "cuda":
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    ok = _check_dtype(
                        dtype, device, heads, kv_heads, force_scan=True
                    ) and ok
    ok = _check_gqa_grouping("cuda" if devices[-1] == "cuda" else "cpu") and ok
    ok = _check_state_dict() and ok
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
