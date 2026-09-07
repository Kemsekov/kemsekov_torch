"""Quick self-validation of the package (CPU-friendly, low memory).

Verifies that the parallel-scan modules match the serial reference for a
small case (forward outputs and parameter gradients) and that state dicts
are interchangeable. Run with:

    python -m gated_delta_2.validate

The full cross-length comparison suite lives in the repository tests
(see README); run heavy cases one at a time and watch memory usage.
"""
import torch

from . import GatedDelta2, GatedDelta2Scan, GatedDelta2ScanDense


def _max_rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-9)


def run():
    torch.manual_seed(0)
    B, L, dim, qk, vd, heads = 3, 64, 32, 64, 20, 2
    x = torch.randn(B, L, dim, requires_grad=True)

    serial = GatedDelta2(dim, qk, vd, heads=heads)
    scans = {
        "chunked": GatedDelta2Scan(dim, qk, vd, heads=heads),
        "dense": GatedDelta2ScanDense(dim, qk, vd, heads=heads),
    }
    for mm in scans.values():
        for (_, p1), (_, p2) in zip(serial.named_parameters(), mm.named_parameters()):
            assert p1.shape == p2.shape
            p2.data.copy_(p1.data)

    y_ref = serial(x)
    ok = True
    for name, mm in scans.items():
        y = mm(x)
        rel = _max_rel(y_ref, y)
        g_ref = torch.autograd.grad(
            y_ref.square().mean(), serial.parameters(), retain_graph=True
        )
        g = torch.autograd.grad(y.square().mean(), mm.parameters())
        rel_g = max(_max_rel(a, b) for a, b in zip(g_ref, g))
        print(f"{name:8s}: fwd rel={rel:.2e}  grad rel={rel_g:.2e}")
        ok = ok and rel < 1e-3 and rel_g < 1e-2

    torch.manual_seed(1)
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
