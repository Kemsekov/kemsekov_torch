"""Quick self-validation of the package (CPU-friendly, low memory).

Verifies that the parallel-scan module matches the serial reference for a
small case (forward outputs and parameter gradients) in float32, float16 and
bfloat16, and that state dicts are interchangeable. Run with:

    python -m gated_delta_2.validate

The full cross-length comparison suite lives in the repository tests
(see README); run heavy cases one at a time and watch memory usage.
"""
import torch

from . import GatedDelta2, GatedDelta2Scan

# tolerated forward / parameter-gradient relative error per dtype
TOL = {
    torch.float32: (1e-3, 1e-2),
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (5e-3, 5e-2),
}


def _max_rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-9)


def _check_dtype(dtype):
    torch.manual_seed(0)
    B, L, dim, qk, vd, heads = 3, 64, 32, 64, 20, 2
    x = torch.randn(B, L, dim, requires_grad=True).to(dtype)

    serial = GatedDelta2(dim, qk, vd, heads=heads).to(dtype)
    chunked = GatedDelta2Scan(dim, qk, vd, heads=heads).to(dtype)
    for (_, p1), (_, p2) in zip(serial.named_parameters(), chunked.named_parameters()):
        assert p1.shape == p2.shape
        p2.data.copy_(p1.data)

    y_ref = serial(x)
    y = chunked(x)
    rel = _max_rel(y_ref, y)
    g_ref = torch.autograd.grad(
        y_ref.square().mean(), serial.parameters(), retain_graph=True
    )
    g = torch.autograd.grad(y.square().mean(), chunked.parameters())
    rel_g = max(_max_rel(a, b) for a, b in zip(g_ref, g))
    tol_fwd, tol_grad = TOL[dtype]
    print(f"{str(dtype):14s}: fwd rel={rel:.2e}  grad rel={rel_g:.2e}")
    return rel < tol_fwd and rel_g < tol_grad


def run():
    ok = all(_check_dtype(dt) for dt in (torch.float32, torch.float16, torch.bfloat16))

    torch.manual_seed(1)
    B, L, dim, qk, vd, heads = 3, 64, 32, 64, 20, 2
    serial = GatedDelta2(dim, qk, vd, heads=heads)
    m2 = GatedDelta2Scan(dim, qk, vd, heads=heads)
    m2.load_state_dict(serial.state_dict())
    torch.manual_seed(1)
    m3 = GatedDelta2Scan(dim, qk, vd, heads=heads)
    with torch.no_grad():
        for (_, p1), (_, p3) in zip(serial.named_parameters(), m3.named_parameters()):
            p3.data.copy_(p1.data)
        torch.manual_seed(2)
        x2 = torch.randn(B, L, dim)
        d_sd = (m2(x2) - m3(x2)).abs().max().item()
        d_ref = (serial(x2) - m3(x2)).abs().max().item() / max(
            serial(x2).abs().max().item(), 1e-9
        )
    print(f"state-dict load == manual copy: {d_sd:.2e}  "
          f"chunked vs serial rel: {d_ref:.2e}")
    ok = ok and d_sd == 0.0 and d_ref < 1e-3
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
