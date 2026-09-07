import torch
import torch.nn.functional as F

LOG_MIN = {torch.float32: -85.0, torch.float64: -700.0}
CACHE_BYTES_LIMIT = 1 << 31


def _tri_solve(A, RHS, upper=False):
    dt = A.dtype
    if dt == torch.float32 and A.shape[-1] > 64:
        A = A.double()
        RHS = RHS.double()
    X = torch.linalg.solve_triangular(A, RHS, upper=upper, unitriangular=True)
    return X.to(dt)


def _states_seq(M, b):
    B, n, d, _ = M.shape
    v = b.shape[-1]
    acc = M.new_zeros(B, d, v)
    out = []
    for i in range(n):
        acc = torch.matmul(M[:, i], acc) + b[:, i]
        out.append(acc)
    return torch.stack(out, 1)


def _affine_prefix_scan(M, b):
    B, n, d, _ = M.shape
    v = b.shape[-1]
    n2 = 1 << (n - 1).bit_length() if n > 1 else 1
    eye = torch.eye(d, dtype=M.dtype, device=M.device)
    if n2 == n:
        MA = M.clone()
        bb = b.clone()
    else:
        MA = eye.expand(B, n2, d, d).clone()
        bb = M.new_zeros(B, n2, d, v)
        MA[:, :n] = M
        bb[:, :n] = b
    levels = n2.bit_length() - 1
    for lvl in range(levels):
        half = 1 << lvl
        seg = half << 1
        hi = torch.arange(seg - 1, n2, seg, device=M.device)
        lo = hi - half
        a_lo = MA[:, lo]
        b_lo = bb[:, lo]
        a_hi = MA[:, hi]
        b_hi = bb[:, hi]
        MA[:, hi] = a_hi @ a_lo
        bb[:, hi] = a_hi @ b_lo + b_hi
    MA[:, n2 - 1] = eye
    bb[:, n2 - 1] = 0
    for lvl in range(levels - 1, -1, -1):
        half = 1 << lvl
        seg = half << 1
        hi = torch.arange(seg - 1, n2, seg, device=M.device)
        lo = hi - half
        tA = MA[:, lo].clone()
        tb = bb[:, lo].clone()
        eA = MA[:, hi]
        eb = bb[:, hi]
        MA[:, lo] = eA
        bb[:, lo] = eb
        MA[:, hi] = tA @ eA
        bb[:, hi] = tA @ eb + tb
    return M @ bb[:, :n] + b


def _gamma_prep(a, dt, prec):
    log_min = LOG_MIN[dt]
    if prec == "fp32":
        Lc = a.log().cumsum(dim=2)
        if (Lc < log_min).any():
            return None
        g = torch.exp(Lc)
        gi = torch.exp(-Lc)
        return g, gi, None
    la = a.log()
    Lc = la.double().cumsum(dim=2)
    if dt == torch.float32 and (Lc < log_min).any():
        return None
    amask = Lc >= log_min
    Lc = Lc.clamp_min(log_min)
    if dt == torch.float32:
        g = torch.exp(Lc.float())
        gi = 1.0 / g
    else:
        g = torch.exp(Lc)
        gi = torch.exp(-Lc)
    return g, gi, amask


def _chunk_prep(a, k, e, q, z, C, dt, prec):
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
    a = a.clamp_min(torch.finfo(a.dtype).tiny)
    a = a.view(B, nch, C, dk)
    k = k.view(B, nch, C, dk)
    e = e.view(B, nch, C, dk)
    q = q.view(B, nch, C, dk)
    z = z.view(B, nch, C, dv)
    gp = _gamma_prep(a, dt, prec)
    if gp is None:
        return _chunk_prep(
            a.double().view(B, nch * C, dk),
            k.double().view(B, nch * C, dk),
            e.double().view(B, nch * C, dk),
            q.double().view(B, nch * C, dk),
            z.double().view(B, nch * C, dv),
            C, torch.float64, "fp64",
        )
    g, gi, amask = gp
    Kb = k * gi
    Eb = e * g
    Qb = q * g
    eyeC = torch.eye(C, dtype=dt, device=a.device)
    Tmat = torch.tril(torch.matmul(Eb, Kb.transpose(-1, -2)), -1) + eyeC
    YU = _tri_solve(Tmat, torch.cat([Eb, z], dim=-1))
    Y, U = YU.split([dk, dv], dim=-1)
    Aqk = torch.tril(torch.matmul(Qb, Kb.transpose(-1, -2)))
    gC = g[:, :, -1, :]
    Kt = Kb * gC.unsqueeze(2)
    M = torch.diag_embed(gC) - torch.matmul(Kt.transpose(-1, -2), Y)
    cb = torch.matmul(Kt.transpose(-1, -2), U)
    ia = 1.0 / a
    return dict(
        a=a, k=k, e=e, q=q, z=z, g=g, gi=gi, ia=ia, Kb=Kb, Eb=Eb, Qb=Qb,
        Tmat=Tmat, Y=Y, U=U, Aqk=Aqk, gC=gC, Kt=Kt, M=M, cb=cb, amask=amask,
        nch=nch, Lp=Lp,
    )


def _states_fwd(M, cb, mode):
    if mode == "seq":
        return _states_seq(M, cb)
    return _affine_prefix_scan(M, cb)


def _chunk_forward(a, k, e, q, z, C, dt, scan_mode, prec):
    B, L, dk = k.shape
    dv = z.shape[-1]
    P = _chunk_prep(a, k, e, q, z, C, dt, prec)
    S_after = _states_fwd(P["M"], P["cb"], scan_mode)
    S_in = torch.cat(
        [S_after.new_zeros(B, 1, dk, dv), S_after[:, :-1]], dim=1
    )
    YS = torch.matmul(P["Y"], S_in)
    O = torch.matmul(P["Qb"], S_in) + torch.matmul(P["Aqk"], P["U"] - YS)
    out = O.reshape(B, P["Lp"], dv)[:, :L]
    return out, S_in, P


def _chunk_backward(go, S_in, P, a, k, e, q, z, C, dt, scan_mode):
    B, L, dkd = k.shape
    dvd = z.shape[-1]
    nch = P["nch"]
    gC = P["gC"]
    Kb = P["Kb"]
    Eb = P["Eb"]
    Qb = P["Qb"]
    Tmat = P["Tmat"]
    Y = P["Y"]
    U = P["U"]
    Aqk = P["Aqk"]
    Kt = P["Kt"]
    M = P["M"]
    if go.shape[1] < P["Lp"]:
        go = torch.cat([go, go.new_zeros(B, P["Lp"] - L, dvd)], dim=1)
    gc = go.view(B, nch, C, dvd)
    gQ = torch.matmul(Aqk.transpose(-1, -2), gc)
    loc = torch.matmul(Qb.transpose(-1, -2), gc) - torch.matmul(
        Y.transpose(-1, -2), gQ
    )
    Mf = M.flip(1).transpose(-1, -2)
    locf = loc.flip(1)
    if scan_mode == "seq":
        R = _states_seq(Mf, locf).flip(1)
    else:
        R = _affine_prefix_scan(Mf, locf).flip(1)
    RA = torch.cat([R[:, 1:], R.new_zeros(B, 1, dkd, dvd)], dim=1)
    S = S_in
    YS = torch.matmul(Y, S)
    WY = U - YS
    ST = S.transpose(-1, -2)
    RST = torch.matmul(RA, ST)
    dY = -torch.matmul(gQ, ST) - torch.matmul(Kt, RST)
    dU = gQ + torch.matmul(Kt, RA)
    dKt = torch.matmul(WY, RA.transpose(-1, -2))
    dgC = torch.diagonal(RST, dim1=-2, dim2=-1) + (dKt * Kb).sum(dim=2)
    dAqk = torch.tril(torch.matmul(gc, WY.transpose(-1, -2)))
    dQb = torch.matmul(gc, ST) + torch.matmul(dAqk, Kb)
    dKb = torch.matmul(dAqk.transpose(-1, -2), Qb) + dKt * gC.unsqueeze(2)
    dA = torch.matmul(dY, Eb.transpose(-1, -2)) + torch.matmul(
        dU, P["z"].transpose(-1, -2)
    )
    v = _tri_solve(Tmat.transpose(-1, -2), dA, upper=True)
    w = _tri_solve(Tmat, v.transpose(-1, -2)).transpose(-1, -2)
    dT = -torch.tril(w, -1)
    dEb, dZ = _tri_solve(
        Tmat.transpose(-1, -2), torch.cat([dY, dU], dim=-1), upper=True
    ).split([dkd, dvd], dim=-1)
    dEb = torch.matmul(dT, Kb) + dEb
    dKb = dKb + torch.matmul(dT.transpose(-1, -2), Eb)
    dlogg = dQb * Qb + dEb * Eb - dKb * Kb
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
    dalpha = (P["ia"] * dlogg).reshape(B, P["Lp"], dkd)[:, :L]
    dK_g = (dKb * P["gi"]).reshape(B, P["Lp"], dkd)[:, :L]
    dE_g = (dEb * P["g"]).reshape(B, P["Lp"], dkd)[:, :L]
    dQ_g = (dQb * P["g"]).reshape(B, P["Lp"], dkd)[:, :L]
    dZ_g = dZ.reshape(B, P["Lp"], dvd)[:, :L]
    return dK_g, dQ_g, dalpha, dE_g, dZ_g


class Delta2ScanFn(torch.autograd.Function):
    _SCAN_MODES = ("scan", "seq", "auto")
    _PRECS = ("fp32", "mixed", "fp64")

    @staticmethod
    def forward(ctx, a, k, e, q, z, C, scan_mode, prec):
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
        out, S_in, P = _chunk_forward(a, k, e, q, z, C, dt, scan_mode, prec)
        ctx.C = C
        ctx.nch = P["nch"]
        ctx.Lp = P["Lp"]
        ctx.dt = dt
        ctx.scan_mode = scan_mode
        ctx.P_keys = [
            key
            for key in (
                "a", "k", "e", "q", "z", "g", "gi", "ia", "Kb", "Eb", "Qb",
                "Tmat", "Y", "U", "Aqk", "gC", "Kt", "M", "cb", "amask",
            )
            if P[key] is not None
        ]
        mem = sum(
            P[key].numel() * P[key].element_size() for key in ctx.P_keys
        ) + S_in.numel() * S_in.element_size()
        if mem <= CACHE_BYTES_LIMIT:
            ctx.save_for_backward(
                a, k, e, q, z, S_in, *(P[key] for key in ctx.P_keys)
            )
            ctx.cached = True
        else:
            ctx.save_for_backward(a, k, e, q, z, S_in)
            ctx.cached = False
        return out.to(ctx.dtype)

    @staticmethod
    def backward(ctx, go):
        saved = ctx.saved_tensors
        a, k, e, q, z, S_in = saved[:6]
        if ctx.cached:
            P = dict(zip(ctx.P_keys, saved[6:]))
            P["nch"] = ctx.nch
            P["Lp"] = ctx.Lp
            P["amask"] = P.get("amask")
        else:
            P = _chunk_prep(a, k, e, q, z, ctx.C, ctx.dt, ctx.prec)
        wdt = P["M"].dtype
        dK, dQ, da, de, dz = _chunk_backward(
            go.to(wdt), S_in, P, a, k, e, q, z, ctx.C, wdt, ctx.scan_mode
        )
        sa, sk, se, sq, sz = ctx.shapes
        return (
            da.reshape(sa).to(ctx.dtype),
            dK.reshape(sk).to(ctx.dtype),
            de.reshape(se).to(ctx.dtype),
            dQ.reshape(sq).to(ctx.dtype),
            dz.reshape(sz).to(ctx.dtype),
            None,
            None,
            None,
        )
