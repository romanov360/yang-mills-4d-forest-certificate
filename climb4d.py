#!/usr/bin/env python3
"""Hill-climb lambda_max of the d = 4 period-2 forest-gauge-fixed SU(2) Hessian over fields on the L-torus.

Every eigenvalue is a lower bound for h_F (periodic fields).  usage: climb4d.py L seed steps
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import torus3d as t  # noqa: E402
import forest4d as f4  # noqa: E402

h = t.h
L, seed, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(seed)
T = t.Torus(L, d=4, pattern=f4.PAT4, P=2)
free = T.free
ident = np.array([np.eye(2, dtype=complex)] * T.nE)
print("identity lambda_max", t.lam(T, ident), "(-lambda_min)", -np.linalg.eigvalsh(t.free_hessian(T, ident))[0], flush=True)
bestQ = None
for which in (0, 1):
    best_all = -1
    for start in range(3):
        Q = h.random_su2(rng, T.nE) if start else ident.copy()
        if start == 2:
            Q = ident.copy(); Q[:, 0, 0] = -1; Q[:, 1, 1] = -1  # -I
        for e in T.fixed:
            Q[e] = np.eye(2)
        ev = lambda Q: (lambda v: (v[-1], -v[0]))(np.linalg.eigvalsh(t.free_hessian(T, Q)))[which]
        best, scale = ev(Q), 1.0
        for k in range(steps):
            e = free[rng.integers(len(free))]
            old = Q[e].copy()
            Q[e] = h.expm_su2(sum(c * b for c, b in zip(rng.normal(size=3) * scale, h.BASIS))) @ Q[e]
            v = ev(Q)
            if v >= best:
                best = v
            else:
                Q[e] = old
            if k % 300 == 299:
                scale *= 0.7
        if best > best_all:
            np.save(f"bestQ_L{L}_s{seed}_{which}.npy", Q)
        best_all = max(best_all, best)
        print(f"{['lambda_max', '-lambda_min'][which]} start {start}: {best:.6f}", flush=True)
    print(f"L={L}: sup {['lambda_max', '-lambda_min'][which]} = {best_all:.6f}", flush=True)
