"""Optimized chunked WY scan for the Gated Delta Rule-2.

This module keeps the exact same chunked formulation as ``scan.py`` but

* forms the inverse of the unit-lower WY factor ``T = I + tril(E K^T, -1)``
  once in the forward pass and reuses it as plain batched matmuls in both the
  forward and the backward pass (instead of calling a batched triangular solve
  three times);
* caches a strictly smaller set of intermediates: ``Kb``, ``Eb``, ``Qb`` and
  ``Kt`` are recomputed from ``g/gi`` and the raw inputs inside the backward
  pass (they are pure elementwise products);
* exposes a ``budget`` knob: when the estimated size of the cached
  intermediates exceeds the budget the forward pass drops ``Y`` and ``U`` as
  well and recomputes the whole chunk preparation inside the backward pass.

Both paths produce the same recurrence as ``serial.GatedDelta2`` up to float
rounding.
"""

import os

import torch

from .scan import (
    _chunk_backward,
    _chunk_forward,
    _chunk_prep,
    _gamma_prep,
    _seq_backward,
    _seq_forward,
    _states_fwd,
    CACHE_BYTES_LIMIT,
)

# bytes of cached intermediates per forward call before we stop caching the
# WY auxiliaries and recompute them in the backward pass instead
BUDGET = int(float(os.environ.get("GD2_SCAN_BUDGET_MB", "256")) * 2**20)


def _unit_tril_inv(Tmat):
    """Inverse of a batched unit-lower-triangular matrix."""
    C = Tmat.shape[-1]
    dt = Tmat.dtype
    if dt == torch.float32 and C > 64:
        Tmat = Tmat.double()
    eye = torch.eye(C, dtype=Tmat.dtype, device=Tmat.device)
    inv = torch.linalg.solve_triangular(
        Tmat, eye, upper=False, unitriangular=True
    )
    return inv.to(dt)


def _pad_inputs(a, k, e, q, z, C):
    B, L, dk = k.shape
    dv = z.shape[-1]
    nch = (L + C - 1) // C
    Lp = nch * C
    if Lp > L:
        pad = Lp - L
        a = torch.cat([a, a.new_ones(B, pad, dk)], dim=1)
        k = torch.cat([k, k.new_zeros(B, pad, dk)], dim=1)
        e = torch.cat([e, e.new_zeros(B, pad, dk)], dim=1)
        q = torch.cat([q, q.new_zeros(B, pad, dk)], dim=1)
        z = torch.cat([z, z.new_zeros(B, pad, dv)], dim=1)
    return a, k, e, q, z, nch, Lp


def _prep_fast(a, k, e, q, z, C, dt, prec, budget=BUDGET):
    """Optimized analogue of ``scan._chunk_prep``.

    Returns ``None`` when the decay-normalized factors cannot be represented
    (the caller must fall back to the exact sequential path), a dict with
    ``fast=False`` when the fp64 recursion of the reference prep had to run,
    and the optimized dict otherwise.
    """
    a, k, e, q, z, nch, Lp = _pad_inputs(a, k, e, q, z, C)
    B = k.shape[0]
    dk = k.shape[-1]
    dv = z.shape[-1]
    tiny = torch.finfo(a.dtype).tiny
    a4 = a.clamp_min(tiny).view(B, nch, C, dk)
    k4 = k.view(B, nch, C, dk)
    e4 = e.view(B, nch, C, dk)
    q4 = q.view(B, nch, C, dk)
    z4 = z.view(B, nch, C, dv)
    gp = _gamma_prep(a4, dt, prec)
    if gp is None:
        if dt == torch.float64:
            return None
        P = _chunk_prep(a, k, e, q, z, C, dt, prec)
        if P is None:
            return None
        return {"fast": False, "P": P}
    g, gi, amask = gp
    Kb = k4 * gi
    Eb = e4 * g
    Qb = q4 * g
    eyeC = torch.eye(C, dtype=dt, device=a.device)
    EK = torch.matmul(torch.cat([Eb, Qb], dim=-2), Kb.transpose(-1, -2))
    Tmat = torch.tril(EK[..., :C, :], -1) + eyeC
    Aqk = torch.tril(EK[..., C:, :])
    Tinv = _unit_tril_inv(Tmat)
    YU = torch.matmul(Tinv, torch.cat([Eb, z4], dim=-1))
    Y, U = YU.split([dk, dv], dim=-1)
    gC = g[..., -1, :]
    Kt = Kb * gC.unsqueeze(2)
    MU = torch.matmul(Kt.transpose(-1, -2), torch.cat([Y, U], dim=-1))
    M = torch.diag_embed(gC) - MU[..., :dk]
    cb = MU[..., dk:]
    ia = 1.0 / a4
    budgeted = (
        sum(
            t.numel() * t.element_size()
            for t in (g, gi, ia, Tinv, Y, U, Aqk, gC, M, cb)
        )
        <= budget
    )
    return dict(
        fast=True,
        budgeted=budgeted,
        a=a, k=k, e=e, q=q, z=z,
        g=g, gi=gi, ia=ia, Kb=Kb, Eb=Eb, Qb=Qb, Tinv=Tinv, Tmat=Tmat,
        Y=Y, U=U, Aqk=Aqk, gC=gC, Kt=Kt, M=M, cb=cb, amask=amask,
        nch=nch, Lp=Lp,
    )


def _recompute_wy(P, C):
    """Recompute ``Tinv``, ``Y`` and ``U`` from the cached gates (used by the
    low-memory scan mode)."""
    g = P["g"]
    gi = P["gi"]
    C = P["C"]
    B = g.shape[0]
    nch = g.shape[1]
    dk = g.shape[-1]
    k4 = P["k"].view(B, nch, C, dk)
    e4 = P["e"].view(B, nch, C, dk)
    z4 = P["z"].view(B, nch, C, P["z"].shape[-1])
    dv = z4.shape[-1]
    Eb = e4 * g
    Kb = k4 * gi
    eyeC = torch.eye(C, dtype=g.dtype, device=g.device)
    EK = torch.matmul(Eb, Kb.transpose(-1, -2))
    Tmat = torch.tril(EK, -1) + eyeC
    P["Tinv"] = _unit_tril_inv(Tmat)
    YU = torch.matmul(P["Tinv"], torch.cat([Eb, z4], dim=-1))
    P["Y"], P["U"] = YU.split([dk, dv], dim=-1)
    return P


def _chunk_forward_fast(P, B, L, dk, dv, scan_mode):
    S_after = _states_fwd(P["M"], P["cb"], scan_mode)
    S_in = torch.cat(
        [S_after.new_zeros(B, 1, dk, dv), S_after[:, :-1]], dim=1
    )
    YS = torch.matmul(P["Y"], S_in)
    O = torch.matmul(P["Qb"], S_in) + torch.matmul(P["Aqk"], P["U"] - YS)
    return O.reshape(B, P["Lp"], dv)[:, :L], S_in


def _chunk_backward_fast(go, S_in, P, B, L, dk, dv, scan_mode):
    nch = P["nch"]
    Lp = P["Lp"]
    g = P["g"]
    gi = P["gi"]
    gC = P["gC"]
    Tinv = P["Tinv"]
    Y = P["Y"]
    U = P["U"]
    Aqk = P["Aqk"]
    M = P["M"]
    if go.shape[1] < Lp:
        go = torch.cat([go, go.new_zeros(B, Lp - go.shape[1], dv)], dim=1)
    gc = go.view(B, nch, -1, dv)
    k4 = P["k"].view(B, nch, -1, dk)
    e4 = P["e"].view(B, nch, -1, dk)
    q4 = P["q"].view(B, nch, -1, dk)
    z4 = P["z"].view(B, nch, -1, dv)
    Kb = k4 * gi
    Eb = e4 * g
    Qb = q4 * g
    Kt = Kb * gC.unsqueeze(2)

    gQ = torch.matmul(Aqk.transpose(-1, -2), gc)
    loc = torch.matmul(Qb.transpose(-1, -2), gc)
    loc.sub_(torch.matmul(Y.transpose(-1, -2), gQ))
    Mf = M.flip(1).transpose(-1, -2)
    locf = loc.flip(1)
    R = _states_fwd(Mf, locf, scan_mode).flip(1)
    RA = R.new_zeros(B, nch, dk, dv)
    RA[:, :-1] = R[:, 1:]
    S = S_in
    YS = torch.matmul(Y, S)
    WY = U - YS
    ST = S.transpose(-1, -2)
    RST = torch.matmul(RA, ST)
    dY = torch.matmul(gQ, ST)
    dY.neg_()
    dY.sub_(torch.matmul(Kt, RST))
    dU = torch.matmul(Kt, RA)
    dU.add_(gQ)
    dKt = torch.matmul(WY, RA.transpose(-1, -2))
    dgC = torch.diagonal(RST, dim1=-2, dim2=-1) + (dKt * Kb).sum(dim=2)
    dAqk = torch.tril(torch.matmul(gc, WY.transpose(-1, -2)))
    dQb = torch.matmul(gc, ST)
    dQb.add_(torch.matmul(dAqk, Kb))
    dKb = torch.matmul(dAqk.transpose(-1, -2), Qb)
    dKb.addcmul_(dKt, gC.unsqueeze(2))
    dA = torch.matmul(dY, Eb.transpose(-1, -2))
    dA.add_(torch.matmul(dU, z4.transpose(-1, -2)))
    Cd = Y.shape[-2]
    X = torch.matmul(Tinv.transpose(-1, -2), torch.cat([dA, dY, dU], dim=-1))
    v = X[..., :Cd]
    dEb = X[..., Cd : Cd + dk]
    dZ = X[..., Cd + dk :]
    w = torch.matmul(v, Tinv.transpose(-1, -2))
    dT = -torch.tril(w, -1)
    dEb.add_(torch.matmul(dT, Kb))
    dKb.add_(torch.matmul(dT.transpose(-1, -2), Eb))
    dlogg = torch.mul(dQb, Qb)
    dlogg.addcmul_(dEb, Eb)
    dlogg.addcmul_(dKb, Kb, value=-1)
    dlogg = torch.cat(
        [
            dlogg[:, :, :-1, :],
            dlogg[:, :, -1:, :] + (dgC * gC).unsqueeze(2),
        ],
        dim=2,
    )
    if P["amask"] is not None:
        dlogg = dlogg * P["amask"]
    dlogg = torch.flip(
        torch.cumsum(torch.flip(dlogg, dims=(2,)), dim=2), dims=(2,)
    )
    dalpha = (P["ia"] * dlogg).reshape(B, Lp, dk)[:, :L]
    dK_g = (dKb * gi).reshape(B, Lp, dk)[:, :L]
    dE_g = (dEb * g).reshape(B, Lp, dk)[:, :L]
    dQ_g = (dQb * g).reshape(B, Lp, dk)[:, :L]
    dZ_g = dZ.reshape(B, Lp, dv)[:, :L]
    return dK_g, dQ_g, dalpha, dE_g, dZ_g


class FastScanFn(torch.autograd.Function):
    """Chunked WY scan with inverted WY factor and a lean cache."""

    @staticmethod
    def forward(ctx, a, k, e, q, z, C, scan_mode, prec, budget):
        with torch.amp.autocast(a.device.type, enabled=False):
            return FastScanFn._forward(ctx, a, k, e, q, z, C, scan_mode, prec, budget)

    @staticmethod
    def _forward(ctx, a, k, e, q, z, C, scan_mode, prec, budget):
        ctx.shapes = [a.shape, k.shape, e.shape, q.shape, z.shape]
        if scan_mode == "auto":
            nch = (z.shape[-2] + C - 1) // C
            scan_mode = "seq" if nch <= 256 else "scan"
        ctx.dtype = a.dtype
        ctx.prec = prec
        dt = (
            torch.float64
            if (a.dtype == torch.float64 or prec == "fp64")
            else torch.float32
        )
        a = a.to(dt).squeeze(-1)
        k = k.to(dt).squeeze(-1)
        e = e.to(dt).squeeze(-1)
        q = q.to(dt).squeeze(-1)
        z = z.to(dt).squeeze(-1)
        P = _prep_fast(a, k, e, q, z, C, dt, prec, budget)
        if P is None:
            ctx.seq = True
            ctx.save_for_backward(a, k, e, q, z)
            out = _seq_forward(
                a.double(), k.double(), e.double(), q.double(), z.double()
            )
            return out.to(ctx.dtype)
        ctx.seq = False
        B, L, dk = k.shape
        dv = z.shape[-1]
        if not P["fast"]:
            # reference fp64-recursed preparation: reuse its backward
            P = P["P"]
            out, S_in, P = _chunk_forward(a, k, e, q, z, C, dt, scan_mode, prec, P=P)
            ctx.ref_prep = True
            ctx.C = C
            ctx.nch = P["nch"]
            ctx.Lp = P["Lp"]
            ctx.dt = dt
            ctx.scan_mode = scan_mode
            ctx.P_keys = [
                key
                for key in (
                    "a", "k", "e", "q", "z", "g", "gi", "ia", "Kb", "Eb",
                    "Qb", "Tmat", "Y", "U", "Aqk", "gC", "Kt", "M", "cb",
                    "amask",
                )
                if P[key] is not None
            ]
            saved = [a, k, e, q, z, S_in] + [P[key] for key in ctx.P_keys]
            mem = sum(t.numel() * t.element_size() for t in saved)
            if mem <= CACHE_BYTES_LIMIT:
                ctx.save_for_backward(*saved)
                ctx.cached = True
            else:
                ctx.save_for_backward(a, k, e, q, z, S_in)
                ctx.cached = False
            return out.to(ctx.dtype)
        ctx.ref_prep = False
        out, S_in = _chunk_forward_fast(P, B, L, dk, dv, scan_mode)
        ctx.C = C
        ctx.nch = P["nch"]
        ctx.Lp = P["Lp"]
        ctx.L = L
        ctx.dt = dt
        ctx.scan_mode = scan_mode
        ctx.budgeted = bool(P["budgeted"])
        base_keys = ["g", "gi", "ia", "Aqk", "gC", "M", "cb"]
        if ctx.budgeted:
            base_keys += ["Tinv", "Y", "U"]
        ctx.fast_keys = [key for key in base_keys if P[key] is not None]
        ctx.save_for_backward(
            P["a"], P["k"], P["e"], P["q"], P["z"], S_in,
            *(P[key] for key in ctx.fast_keys),
        )
        return out.to(ctx.dtype)

    @staticmethod
    def backward(ctx, go):
        with torch.amp.autocast(go.device.type, enabled=False):
            return FastScanFn._backward(ctx, go)

    @staticmethod
    def _backward(ctx, go):
        saved = ctx.saved_tensors
        sa, sk, se, sq, sz = ctx.shapes
        if ctx.seq:
            a, k, e, q, z = saved
            da, dk, de_, dq, dz = _seq_backward(
                go.double(), a.double(), k.double(), e.double(), q.double(),
                z.double(),
            )
        elif ctx.ref_prep:
            a, k, e, q, z, S_in = saved[:6]
            if ctx.cached:
                P = dict(zip(ctx.P_keys, saved[6:]))
                P["nch"] = ctx.nch
                P["Lp"] = ctx.Lp
                P["amask"] = P.get("amask")
            else:
                P = _chunk_prep(a, k, e, q, z, ctx.C, ctx.dt, ctx.prec)
            wdt = P["M"].dtype
            dk_, dq_, da, de_, dz = _chunk_backward(
                go.to(wdt), S_in, P, a, k, e, q, z, ctx.C, wdt, ctx.scan_mode
            )
            dk, dq = dk_, dq_
        else:
            a, k, e, q, z, S_in = saved[:6]
            B, Lp, dk_ = k.shape
            L = ctx.L
            dv = z.shape[-1]
            P = {
                "a": a, "k": k, "e": e, "q": q, "z": z,
                "C": ctx.C,
            }
            P.update(dict(zip(ctx.fast_keys, saved[6:])))
            if not ctx.budgeted:
                _recompute_wy(P, ctx.C)
            P["nch"] = ctx.nch
            P["Lp"] = ctx.Lp
            P["amask"] = P.get("amask")
            dk, dq, da, de_, dz = _chunk_backward_fast(
                go.to(ctx.dt), S_in, P, B, L, dk_, dv, ctx.scan_mode
            )
        return (
            da.reshape(sa).to(ctx.dtype),
            dk.reshape(sk).to(ctx.dtype),
            de_.reshape(se).to(ctx.dtype),
            dq.reshape(sq).to(ctx.dtype),
            dz.reshape(sz).to(ctx.dtype),
            None,
            None,
            None,
            None,
        )
