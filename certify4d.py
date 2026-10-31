#!/usr/bin/env python3
"""Site-plus-plaquette certificate for the d = 4 forest-gauge Hessian (the method of the d = 3 paper, Lemma 5.1).

G_y: real symmetric PSD 8 x 8 matrices on the ends o_0..o_3, i_0..i_3 at each of the 16 sites of the period cell,
with G_y[o_mu, i_mu] = 0.  a_{p,r} >= 0 on the three free links of each of the 96 plaquettes.  Each plaquette
inequality f_p + c_p <= sum_r a_{p,r}|Y_r|^2 is, after the U(2) reduction, a 12 x 12 LMI M_p(phi) >= 0 for all phases.
h = max over free links of (sum_{p ni e} a_{p,e} + G_tail[o,o] + G_head[i,i]).

Usage: python3 certify4d.py search h0          (cutting-plane SDP, prints the best uniform margin)
"""
import sys
from fractions import Fraction

import numpy as np

import certificate3d as c
import forest4d as f4

T = f4.Cell()
d = T.d


def cm(i, j):
    M = np.zeros((16, 16))
    M[4 * i:4 * i + 4, 4 * j:4 * j + 4] = 0.5 * np.eye(4)
    return M + M.T


C14, C12, C34 = cm(0, 3), cm(0, 1), cm(2, 3)


def c23(w):
    return c.qmat(lambda Y, W: c.ip(Y[1], W @ Y[2] @ W.conj().T), w)


_E = lambda a, b: np.diag([a, b]).astype(complex)
F_BASIS = [c.qmat(c.fp, _E(1, 0)), c.qmat(c.fp, _E(1j, 0)), c.qmat(c.fp, _E(0, 1)), c.qmat(c.fp, _E(0, 1j))]
_C0, _Cpi, _Chalf = c23(_E(1, 1)), c23(_E(1, -1)), c23(_E(1j, 1))
C_BASIS = [(_C0 + _Cpi) / 2, (_C0 - _Cpi) / 2, _Chalf - (_C0 + _Cpi) / 2]


def F_of(p1, p2):
    return (np.cos(p1)[:, None, None] * F_BASIS[0] + np.sin(p1)[:, None, None] * F_BASIS[1]
            + np.cos(p2)[:, None, None] * F_BASIS[2] + np.sin(p2)[:, None, None] * F_BASIS[3])


def C_of(p1, p2):
    dd = p1 - p2
    return C_BASIS[0][None] + np.cos(dd)[:, None, None] * C_BASIS[1] + np.sin(dd)[:, None, None] * C_BASIS[2]


def plaquettes():
    out = []
    for x in T.sites:
        for mu in range(d):
            for nu in range(mu + 1, d):
                p = (T.eidx[(x, mu)], T.eidx[(T.sh(x, mu), nu)], T.eidx[(T.sh(x, nu), mu)], T.eidx[(x, nu)])
                corners = ((x, mu, nu), (T.sh(x, mu), d + mu, nu), (T.sh(T.sh(x, mu), nu), d + nu, d + mu),
                           (T.sh(x, nu), mu, d + nu))
                free_pos = [i for i in range(4) if p[i] not in T.fixed]
                out.append((p, corners, free_pos))
    return out


PL = plaquettes()


def plaq_matrix(F, C23, k, a, free_pos):
    k14, k12, k23, k34 = k
    idx = [4 * i + q for i in free_pos for q in range(4)]
    M = -F - (k14 * C14 + k12 * C12 + k23 * C23 + k34 * C34)
    M = sum(a[i] * np.kron(np.diag([1.0 if j == i else 0.0 for j in range(4)]), np.eye(4)) for i in free_pos) + M
    M = 0.5 * (M + M.T)
    return M[np.ix_(idx, idx)]


def link_totals(G, A, add=lambda u, v: u + v, zero=0):
    coll = {e: zero for e in T.free}
    for (x, mu) in T.edges:
        e = T.eidx[(x, mu)]
        if e not in T.fixed:
            coll[e] = add(add(coll[e], G[x][mu, mu] if not isinstance(G[x], list) else G[x][mu][mu]),
                          G[T.sh(x, mu)][d + mu, d + mu] if not isinstance(G[x], list) else G[T.sh(x, mu)][d + mu][d + mu])
    for (p, corners, free_pos), a in zip(PL, A):
        for i in free_pos:
            coll[p[i]] = add(coll[p[i]], a[i])
    return coll


def search(h0, pts, gmargin=1e-4, sym=None):
    import cvxpy as cp  # needed only for the SDP; the exact checks run without it
    G = {y: cp.Variable((2 * d, 2 * d), symmetric=True) for y in T.sites}
    eps = cp.Variable()
    cons = []
    for y in T.sites:
        cons.append(G[y] >> gmargin * np.eye(2 * d))
        for mu in range(d):
            cons.append(G[y][mu, d + mu] == 0)
    A = [{i: cp.Variable(nonneg=True) for i in free_pos} for (_, _, free_pos) in PL]
    for (p, corners, free_pos), P, a in zip(PL, pts, A):
        k = [2 * G[s][r, q] for s, r, q in corners]
        for Fk, Ck in zip(F_of(P[:, 0], P[:, 1]), C_of(P[:, 0], P[:, 1])):
            cons.append(plaq_matrix(Fk, Ck, k, a, free_pos) >> eps * np.eye(4 * len(free_pos)))
    for e, tot in link_totals(G, A).items():
        cons.append(tot <= h0)
    cp.Problem(cp.Maximize(eps), cons).solve(solver="CLARABEL")
    return {y: G[y].value for y in T.sites}, [{i: v.value for i, v in a.items()} for a in A], eps.value


def numeric_matrices(G, A, j, P):
    p, corners, free_pos = PL[j]
    k = [2 * float(G[s][r][q]) for s, r, q in corners]
    idx = [4 * i + q for i in free_pos for q in range(4)]
    Dm = np.zeros((16, 16))
    for i in free_pos:
        Dm[4 * i:4 * i + 4, 4 * i:4 * i + 4] = float(A[j][i]) * np.eye(4)
    M = Dm[None] - F_of(P[:, 0], P[:, 1]) - (k[0] * C14 + k[1] * C12 + k[3] * C34)[None] - k[2] * C_of(P[:, 0], P[:, 1])
    M = 0.5 * (M + np.swapaxes(M, 1, 2))
    return M[:, idx][:, :, idx], abs(k[2])


def grid_pts(m):
    ph = 2 * np.pi * np.arange(m) / m
    return np.array([(x, y) for x in ph for y in ph])


def cutting_plane(h0, m0=4, mfine=64, rounds=10, add=6):
    pts = [grid_pts(m0) for _ in PL]
    fine = grid_pts(mfine)
    for r in range(rounds):
        Gf, Af, eps = search(h0, pts)
        worst = np.inf
        for j in range(len(PL)):
            M, _ = numeric_matrices(Gf, Af, j, fine)
            lam = np.linalg.eigvalsh(M)[:, 0]
            worst = min(worst, lam.min())
            pts[j] = np.vstack([pts[j], fine[np.argsort(lam)[:add]]])
        print(f"  h0={h0} round {r}: eps on cuts = {eps:.5f}, min on grid {mfine} = {worst:.5f}", flush=True)
        if worst > 0.8 * eps:
            break
    return Gf, Af, eps, worst


if __name__ == "__main__":
    h0 = float(sys.argv[2])
    Gf, Af, eps, worst = cutting_plane(h0)
    np.save(f"float_h{h0}.npy", {"G": Gf, "A": Af, "eps": eps, "worst": worst}, allow_pickle=True)


DEN = 10 ** 6


def rat(v):
    return Fraction(round(float(v) * DEN), DEN)


def ldl_psd(M):
    """Exact test that a symmetric rational matrix is positive definite (all LDL^T pivots > 0)."""
    M = [row[:] for row in M]
    n = len(M)
    for j in range(n):
        if M[j][j] <= 0:
            return False
        for i in range(j + 1, n):
            f = M[i][j] / M[j][j]
            for k in range(j, n):
                M[i][k] -= f * M[j][k]
    return True


def round_and_check(Gf, Af):
    G = {}
    for y in T.sites:
        g = [[rat(0.5 * (Gf[y][i, j] + Gf[y][j, i])) for j in range(2 * d)] for i in range(2 * d)]
        for mu in range(d):
            g[mu][d + mu] = g[d + mu][mu] = Fraction(0)
        assert ldl_psd(g), f"G at {y} not PD after rounding"
        G[y] = g
    A = [{i: max(Fraction(0), rat(v)) for i, v in a.items()} for a in Af]
    return G, A, max(link_totals(G, A, zero=Fraction(0)).values())


def round_file(fn):
    s = np.load(fn, allow_pickle=True).item()
    G, A, h = round_and_check(s["G"], s["A"])
    out = fn.replace("float_", "cert4d_")
    np.save(out, {"G": {y: [[str(v) for v in r] for r in G[y]] for y in T.sites},
                  "A": [{i: str(v) for i, v in a.items()} for a in A], "h": str(h)}, allow_pickle=True)
    print(f"{out}: exact per-link maximum h = {h} = {float(h):.6f}")
