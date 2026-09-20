"""Runtime backend autotuner for the Gated Delta Rule-2 mixing backends.

The best mixing backend depends on the device, the torch/triton versions, the
dtype, the head dims and the sequence length.  Rather than hard-coding a table,
the module *measures* the candidates the first time it sees a shape (fused
Triton recurrence, chunked WY scan, differentiable autograd scan, ...),
validates that every candidate agrees numerically with the reference chunked
scan, and caches the winner on disk (``~/.cache/gd2_tune.json``).

Set ``GD2_AUTOTUNE=0`` to disable tuning and use the static heuristic.
"""

import json
import os
import time

import torch

_TUNE_FILE = os.environ.get(
    "GD2_TUNE_FILE",
    os.path.join(os.path.expanduser("~"), ".cache", "gd2_tune.json"),
)
_ENABLED = os.environ.get("GD2_AUTOTUNE", "1") != "0"

_cache = None


def enabled():
    return _ENABLED


def load():
    global _cache
    if _cache is None:
        try:
            with open(_TUNE_FILE) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def save():
    try:
        os.makedirs(os.path.dirname(_TUNE_FILE), exist_ok=True)
        tmp = _TUNE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp, _TUNE_FILE)
    except Exception:
        pass


def lookup(key):
    return load().get(key, {}).get("impl")


def store(key, impl, scores, mems=None):
    entry = {"impl": impl, "scores": scores}
    if mems:
        entry["mem"] = mems
    load()[key] = entry
    save()


def rel_err(a, b):
    denom = b.abs().max().item()
    return (a.float() - b.float()).abs().max().item() / max(denom, 1e-9)


def _time(fn, reps, warm=1, sync=None):
    for _ in range(warm):
        fn()
    if sync is not None:
        sync()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        best = min(best, time.perf_counter() - t0)
    return best


def tune(cands, training, tol, reps=2, sync=None, mem=False, time_margin=0.10):
    """Time and validate candidate backends.

    ``cands`` is a list of ``(name, run)`` where ``run(training)`` executes the
    full forward (``training=False``) or forward+backward (``training=True``)
    on private, cloned inputs and returns the (detached) output tensor.
    Candidates whose output disagrees with the reference or that raise are
    excluded.  When ``mem`` is true the peak CUDA memory of each candidate is
    measured as well; among the candidates within ``time_margin`` of the
    fastest one the least memory-hungry wins.
    """
    scores = {}
    mems = {}
    ref = None
    for name, run in cands:
        try:
            if training:
                out = run(True)
            else:
                with torch.no_grad():
                    out = run(False)
            if ref is None:
                ref = out.clone()
            elif rel_err(out, ref) > tol:
                continue
            if not torch.isfinite(out).all():
                continue
            if mem:
                if sync is not None:
                    sync()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                scores[name] = _time(
                    lambda r=run: r(training), reps, sync=sync
                )
                mems[name] = max(
                    0.0, float(torch.cuda.max_memory_allocated() - base)
                )
            else:
                scores[name] = _time(
                    lambda r=run: r(training), reps, sync=sync
                )
        except Exception:
            continue
    if not scores:
        return None, {}, {}
    best_t = min(scores.values())
    eligible = [n for n in scores if scores[n] <= best_t * (1.0 + time_margin)]
    if mems:
        best = min(eligible, key=lambda n: (mems[n], scores[n]))
    else:
        best = min(eligible, key=lambda n: scores[n])
    return best, scores, mems
