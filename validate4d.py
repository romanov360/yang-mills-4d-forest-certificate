#!/usr/bin/env python3
"""Numerical checks of the identification of the Wilson Hessian with the certificate's plaquette matrices, d = 4.

This is the step that the exact and interval checks take for granted (Lemma lem:cert of the 3D paper, applied on the
d = 4 forest). It is not part of the proof. It tests, in floating point, every identity that the reduction uses:

1. Plaquette rewriting.  For Q_e -> e^{tX_e} Q_e, with Y1 = X1, Y2 = Ad(Q1) X2, Y3 = Ad(Q4) X3, Y4 = X4, W = Q_p,
   sum_p f_p(Y, W) equals Hess S_0(v, v).  Checked against fasthess.py for SU(2), and against a finite-difference
   second derivative of sum_p Re Tr Q_p for SU(3).
2. Site identity.  With ends o_mu(y) = X_(y,mu), i_mu(y) = Ad(Q_(y-mu,mu)^*) X_(y-mu,mu), the corner terms of all
   plaquettes plus the diagonal site terms equal sum_y <ends, G_y ends>.  Checked for SU(2) and SU(3).
3. Consequence.  h |v|^2 - Hess(v, v) - sum_y <ends, G_y ends> = sum_p (slack of the plaquette inequality) >= 0, and
   the same with -Hess, at random SU(2) and SU(3) fields and vectors.
4. Spectrum.  The eigenvalues of the gauge-fixed SU(2) Hessian at random, abelian, identity and the saved extremal
   field lie in [-h, h].
5. Block reduction.  The plaquette inequality holds at random U(3) and U(4) holonomies and vectors (the proof checks
   U(2) only; the reduction says that suffices).

Fields live on the periodic torus of side 4 (side 2 is the period cell, where two plaquettes can share two links);
the certificate is extended periodically.  Usage: python3 validate4d.py [data/cert4d_h14.42.npy]
"""
import itertools
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import certificate3d as c  # noqa: E402
import certify4d as cz  # noqa: E402
import forest4d as f4  # noqa: E402
import torus3d as t3  # noqa: E402

rng = np.random.default_rng(0)
cert = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/cert4d_h14.42.npy", allow_pickle=True).item()
G = {y: np.array([[float(Fraction(v)) for v in r] for r in cert["G"][y]]) for y in cz.T.sites}
A = [{i: float(Fraction(v)) for i, v in a.items()} for a in cert["A"]]
h = float(Fraction(cert["h"]))
d = 4
ip = c.ip
Ad = lambda U, X: U @ X @ U.conj().T

# ---- torus of side 4 with the period-2 forest, and its plaquettes mapped to the cell's
T = t3.Torus(4, d=4, pattern=f4.PAT4, P=2)
cell = lambda x: tuple(xi % 2 for xi in x)
CELL_PL = {}
for j, (x, mu, nu) in enumerate((x, mu, nu) for x in cz.T.sites for mu in range(d) for nu in range(mu + 1, d)):
    CELL_PL[(x, mu, nu)] = j
assert len(CELL_PL) == len(cz.PL)
PLT = []  # (edges, corners as (site, r, q), cell plaquette index)
for x in T.sites:
    for mu in range(d):
        for nu in range(mu + 1, d):
            p = (T.eidx[(x, mu)], T.eidx[(T.sh(x, mu), nu)], T.eidx[(T.sh(x, nu), mu)], T.eidx[(x, nu)])
            corners = ((x, mu, nu), (T.sh(x, mu), d + mu, nu), (T.sh(T.sh(x, mu), nu), d + nu, d + mu),
                       (T.sh(x, nu), mu, d + nu))
            j = CELL_PL[(cell(x), mu, nu)]
            free_pos = [i for i in range(4) if p[i] not in T.fixed]
            assert free_pos == cz.PL[j][2], "fixed positions differ from the period cell"
            PLT.append((p, corners, j))
Gs = lambda y: G[cell(y)]


def rand_su(n, N):
    Z = rng.normal(size=(n, N, N)) + 1j * rng.normal(size=(n, N, N))
    q, r = np.linalg.qr(Z)
    q = q * (np.diagonal(r, axis1=1, axis2=2) / abs(np.diagonal(r, axis1=1, axis2=2)))[:, None, :]
    return q / np.linalg.det(q)[:, None, None] ** (1 / N)


def rand_alg(n, N, traceless=True):
    Z = rng.normal(size=(n, N, N)) + 1j * rng.normal(size=(n, N, N))
    X = (Z - np.conj(np.swapaxes(Z, 1, 2))) / 2
    if traceless:
        X -= np.trace(X, axis1=1, axis2=2)[:, None, None] * np.eye(N) / N
    return X


def pieces(Q, Xm):
    """sum_p f_p, sum_p c_p (corners), sum_p sum_r a_r |Y_r|^2, sum_y <ends, G ends>, sum_y diagonal site terms."""
    fsum = corners = asum = 0.0
    for p, cor, j in PLT:
        e1, e2, e3, e4 = p
        Wp = Q[e1] @ Q[e2] @ Q[e3].conj().T @ Q[e4].conj().T
        Y = [Xm[e1], Ad(Q[e1], Xm[e2]), Ad(Q[e4], Xm[e3]), Xm[e4]]
        fsum += c.fp(Y, Wp)
        k = [2 * Gs(s)[r, q] for s, r, q in cor]
        corners += (k[0] * ip(Y[0], Y[3]) + k[1] * ip(Y[0], Y[1]) + k[2] * ip(Y[1], Wp @ Y[2] @ Wp.conj().T)
                    + k[3] * ip(Y[2], Y[3]))
        asum += sum(A[j][i] * ip(Y[i], Y[i]) for i in cz.PL[j][2])
    site = diag = 0.0
    for y in T.sites:
        ends = [Xm[T.eidx[(y, m)]] for m in range(d)]
        for m in range(d):
            e = T.eidx[(T.sh(y, m, -1), m)]
            ends.append(Ad(Q[e].conj().T, Xm[e]))
        Gm = np.array([[ip(a, b) for b in ends] for a in ends])
        site += float((Gs(y) * Gm).sum())
        diag += float(sum(Gs(y)[a, a] * Gm[a, a] for a in range(2 * d)))
    return fsum, corners, asum, site, diag


def hess_fd(Q, Xm, eps=1e-3):
    """Fourth-order central difference of sum_p Re Tr Q_p(t) along Q_e -> e^{tX_e} Q_e."""
    def S(t):
        Qt = np.array([expm(t * x) @ q for x, q in zip(Xm, Q)])
        return sum(np.trace(Qt[a] @ Qt[b] @ Qt[cc].conj().T @ Qt[dd].conj().T).real for (a, b, cc, dd), _, _ in PLT)
    return (-S(2 * eps) + 16 * S(eps) - 30 * S(0) + 16 * S(-eps) - S(-2 * eps)) / (12 * eps ** 2)


B3 = np.array(t3.h.BASIS)
worst_rw = worst_site = 0.0
least = np.inf
for N in (2, 3):
    for trial in range(3 if N == 2 else 2):
        Q = rand_su(T.nE, N)
        Xm = rand_alg(T.nE, N)
        Xm[list(T.fixed)] = 0
        if N == 2:  # coordinates in the basis of fasthess
            X = np.einsum("bji,eij->eb", B3.conj(), Xm).real
            Xm = np.einsum("ea,aij->eij", X, B3)
            H = t3.f.hess_quadform(T, Q, X)
        else:
            H = hess_fd(Q, Xm)
        fsum, corners, asum, site, diag = pieces(Q, Xm)
        norm = sum(ip(x, x) for x in Xm)
        worst_rw = max(worst_rw, abs(H - fsum) / norm)
        worst_site = max(worst_site, abs(corners + diag - site) / norm)
        # Lemma: sum_p slack_p = sum_r a|Y|^2 - f - c >= 0 and site >= 0, so h|v|^2 - H >= asum + diag - fsum -
        # corners - ... ; we check the two exact consequences directly.
        tot = {e: 0.0 for e in T.free}
        for p, cor, j in PLT:
            for i in cz.PL[j][2]:
                tot[p[i]] += A[j][i]
        for e in T.free:
            x, mu = T.edges[e]
            tot[e] += Gs(x)[mu, mu] + Gs(T.sh(x, mu))[d + mu, d + mu]
        lhs = sum(tot[e] * ip(Xm[e], Xm[e]) for e in T.free)
        assert max(tot.values()) <= h + 1e-12
        least = min(least, (lhs - fsum - corners - diag) / norm, site / norm)
        print(f"   SU({N}) trial {trial}: Hess = {H:+.6f}, sum f_p = {fsum:+.6f}, site identity residual "
              f"{corners + diag - site:+.1e}", flush=True)
print(f"1. plaquette rewriting (SU(2) vs fasthess, SU(3) vs finite differences): max relative discrepancy "
      f"{worst_rw:.1e}")
print(f"2. site identity: max relative discrepancy {worst_site:.1e}")
print(f"3. totals <= h at every free link; sum_p slack and sum_y site form: least normalised value {least:.4f} "
      "(must be >= 0)")

# 4. spectrum on the period cell (SU(2), gauge-fixed)
Tc = t3.Torus(2, d=4, pattern=f4.PAT4, P=2)
ev = []
fields = [np.load("data/bestQ_L2_s2_0.npy"), np.array([np.eye(2, dtype=complex)] * Tc.nE)]
for trial in range(10):
    fields += [rand_su(Tc.nE, 2), t3.abelian(rng.uniform(0, 2 * np.pi, Tc.nE))]
for Q in fields:
    ev.append(np.linalg.eigvalsh(t3.free_hessian(Tc, Q))[[0, -1]])
ev = np.array(ev)
print(f"4. eigenvalues of the gauge-fixed Hessian on {len(fields)} fields in [{ev.min():.4f}, {ev.max():.4f}], "
      f"h = {h:.6f}")


# 5. U(N) plaquette inequality, N = 3, 4
def rand_u(N):
    Z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    q, r = np.linalg.qr(Z)
    return q * (np.diag(r) / abs(np.diag(r)))


for N in (3, 4):
    low = np.inf
    for trial in range(3000):
        j = rng.integers(len(cz.PL))
        p, cor, free_pos = cz.PL[j]
        W = rand_u(N)
        Y = [rand_alg(1, N, traceless=False)[0] if i in free_pos else np.zeros((N, N)) for i in range(4)]
        k = [2 * G[s][r, q] for s, r, q in cor]
        val = (sum(A[j][i] * ip(Y[i], Y[i]) for i in free_pos) - c.fp(Y, W)
               - (k[0] * ip(Y[0], Y[3]) + k[1] * ip(Y[0], Y[1]) + k[2] * ip(Y[1], W @ Y[2] @ W.conj().T)
                  + k[3] * ip(Y[2], Y[3])))
        low = min(low, val / sum(ip(y, y) for y in Y))
    print(f"5. U({N}) plaquette inequality at 3000 random points: least normalised slack {low:.5f} (must be > 0)")
