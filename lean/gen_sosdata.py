#!/usr/bin/env python3
"""Write lean/SosData.lean: an exact sum-of-squares certificate that every plaquette matrix M_p(phi) is positive
semidefinite for all phases. It replaces the interval cover of cert_interval4d.py.

With c_a = cos phi_a, s_a = sin phi_a, the matrix 4 * 10^6 * M_p is a polynomial matrix in (c1, s1, c2, s2) of degree
2. It splits into two 6 x 6 blocks: the coordinates along iI, i sigma_3 (block D) and along i sigma_1, i sigma_2
(block O), each ordered by free position, then basis direction. For each block we find, by SDP, an integer symmetric
30 x 30 matrix P, indexed by (monomial a, coordinate i) with the monomials z = (1, c1, s1, c2, s2), such that

    z^T P z  =  2^K * 4 * 10^6 * M_block      modulo  c1^2 + s1^2 = 1,  c2^2 + s2^2 = 1,

and P is positive definite. On the torus the two sides agree, so M_p(phi) is positive semidefinite for every phi.
Lean (Sos4.lean) recomputes 4 * 10^6 * M_p from the certificate and the definition of the forms, checks the identity,
and checks that P is positive definite by a sum-of-squares decomposition that it computes itself. Nothing in this
script is trusted.

Usage: python3 lean/gen_sosdata.py [data/cert4d_h14.42.npy]   (about 5 minutes)
"""
import io
import contextlib
import os
import sys
from fractions import Fraction

import cvxpy as cp
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
certfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "data", "cert4d_h14.42.npy")
sys.argv = [sys.argv[0], certfile]
with contextlib.redirect_stdout(io.StringIO()):
    import cert_interval4d as ci

K = 24
SCALE = 4 * 10 ** 6
Z = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]  # exponents of (c1, s1, c2, s2)
BLOCKS = {"D": (0, 3), "O": (1, 2)}


def reduce_mono(m):
    """c_a^2 -> 1 - s_a^2, as a dict monomial -> coefficient."""
    for a in (0, 2):
        if m[a] >= 2:
            m1 = list(m); m1[a] -= 2
            m2 = list(m1); m2[a + 1] += 2
            out = dict(reduce_mono(tuple(m1)))
            for k, v in reduce_mono(tuple(m2)).items():
                out[k] = out.get(k, 0) - v
            return out
    return {m: 1}


PROD = {(a, b): reduce_mono(tuple(x + y for x, y in zip(Z[a], Z[b]))) for a in range(5) for b in range(5)}


def mpoly(j):
    """SCALE * M_p as {monomial: 12 x 12 integer matrix}, coordinates (free position, q)."""
    const, trig, _ = ci.plaquette_terms(j)
    terms = {(0, 0, 0, 0): [const], (1, 0, 0, 0): [trig[0]], (0, 1, 0, 0): [trig[1]], (0, 0, 1, 0): [trig[2]],
             (0, 0, 0, 1): [trig[3]], (1, 0, 1, 0): [trig[4]], (0, 1, 0, 1): [trig[4]], (0, 1, 1, 0): [trig[5]],
             (1, 0, 0, 1): [[[-v for v in r] for r in trig[5]]]}
    out = {}
    for m, ms in terms.items():
        A = [[sum(Fraction(M[a][b]) for M in ms) * SCALE for b in range(12)] for a in range(12)]
        assert all(v.denominator == 1 for r in A for v in r)
        A = [[int(v) for v in r] for r in A]
        if any(any(r) for r in A):
            out[m] = A
    return out


def block(Mp, qs):
    idx = [4 * i + q for i in range(3) for q in range(4) if q in qs]
    other = [k for k in range(12) if k not in idx]
    for A in Mp.values():
        assert all(A[a][b] == 0 for a in idx for b in other), "blocks are coupled"
    return {m: [[A[a][b] for b in idx] for a in idx] for m, A in Mp.items()}


def gram_poly(P, n):
    """z^T P z as {monomial: n x n matrix}; P indexed by a * n + i."""
    out = {}
    for a in range(5):
        for b in range(5):
            for m, c in PROD[(a, b)].items():
                A = out.setdefault(m, [[0] * n for _ in range(n)])
                for i in range(n):
                    for k in range(n):
                        A[i][k] += c * P[a * n + i][b * n + k]
    return {m: A for m, A in out.items() if any(any(r) for r in A)}


def solve(Mb, n=6):
    P = cp.Variable((5 * n, 5 * n), symmetric=True)
    t = cp.Variable()
    acc = {}
    for a in range(5):
        for b in range(5):
            for m, c in PROD[(a, b)].items():
                acc[m] = acc.get(m, 0) + c * P[n * a:n * a + n, n * b:n * b + n]
    cons = [P - t * np.eye(5 * n) >> 0]
    cons += [acc[m] == np.array(Mb.get(m, np.zeros((n, n))), dtype=float) / SCALE for m in acc]
    assert set(Mb) <= set(acc)
    cp.Problem(cp.Maximize(t), cons).solve(solver=cp.CLARABEL)
    return P.value, t.value


PIVOT = {(0, 0, 0, 0): (0, 0), (0, 2, 0, 0): (2, 2), (0, 0, 0, 2): (4, 4)}  # 1, s1^2, s2^2: the diagonal pair


def round_exact(Pf, Mb, n=6):
    """Round to integers (scale 2^K * SCALE) and repair the identity exactly by solving for one pivot entry per
    (monomial, entry): the pair (a, b) with z_a z_b = m (for 1, s1^2, s2^2 the pair (1,1), (s1,s1), (s2,s2))."""
    P = [[int(round(v * 2 ** K * SCALE)) for v in r] for r in Pf]
    P = [[P[min(a, b)][max(a, b)] for b in range(5 * n)] for a in range(5 * n)]
    pairs = {}
    for (a, b), d in PROD.items():
        for m in d:
            pairs.setdefault(m, []).append((a, b))
    for m, prs in pairs.items():
        if m in PIVOT:
            pa, pb = PIVOT[m]
        else:
            (pa, pb), = [(a, b) for (a, b) in prs if a <= b]
        for i in range(n):
            for k in range(i, n):
                target = 2 ** K * Mb.get(m, [[0] * n] * n)[i][k]
                rest, coef = 0, 0
                for (a, b) in prs:
                    c = PROD[(a, b)][m]
                    if {(a * n + i, b * n + k), (b * n + k, a * n + i)} & {(pa * n + i, pb * n + k),
                                                                            (pb * n + k, pa * n + i)}:
                        coef += c
                    else:
                        rest += c * P[a * n + i][b * n + k]
                assert coef in (1, 2) and (target - rest) % coef == 0
                v = (target - rest) // coef
                P[pa * n + i][pb * n + k] = P[pb * n + k][pa * n + i] = v
    G = gram_poly(P, n)
    for m in set(G) | set(Mb):
        assert G.get(m, [[0] * n] * n) == [[2 ** K * x for x in r] for r in Mb.get(m, [[0] * n] * n)], m
    return P


def ldl_pd(P):
    M = [[Fraction(v) for v in r] for r in P]
    n = len(M)
    for j in range(n):
        if M[j][j] <= 0:
            return False
        for i in range(j + 1, n):
            f = M[i][j] / M[j][j]
            for k in range(j + 1, n):
                M[i][k] -= f * M[j][k]
    return True


certs, index, margins = [], [], []
seen = {}
for j in range(len(ci.PL)):
    Mp = mpoly(j)
    key = repr(sorted(Mp.items()))
    if key not in seen:
        pair = []
        for name, qs in BLOCKS.items():
            Mb = block(Mp, qs)
            Pf, t = solve(Mb)
            P = round_exact(Pf, Mb)
            assert ldl_pd(P), f"plaquette {j} block {name}: rounded Gram matrix not positive definite"
            pair.append(P)
            margins.append(t)
        seen[key] = len(certs)
        certs.append(pair)
        print(f"plaquette {j}: new certificate {seen[key]}, margins {margins[-2]:.6f} {margins[-1]:.6f}",
              flush=True)
    index.append(seen[key])

lst = lambda xs: "[" + ", ".join(str(x) for x in xs) + "]"
tri = lambda P: lst(lst(P[i][:i + 1]) for i in range(len(P)))
out = [f"/- Generated by gen_sosdata.py from {os.path.basename(certfile)}. Do not edit. -/", "",
       "/-- the identity is z^T P z = sosScale * (4 * 10^6 * M_block). -/",
       f"def sosScale : Int := {2 ** K}",
       "/-- plaquette -> certificate (only a hint: every plaquette is checked against its own matrix). -/",
       "def sosIndex : List Nat := " + lst(index)]
for c, (PD, PO) in enumerate(certs):
    out.append(f"def sosD{c} : List (List Int) := {tri(PD)}")
    out.append(f"def sosO{c} : List (List Int) := {tri(PO)}")
out.append("/-- Gram matrices, lower triangles, indexed by (monomial, coordinate) = a * 6 + i. -/")
out.append("def sosCerts : List (List (List Int) × List (List Int)) := " +
           lst(f"(sosD{c}, sosO{c})" for c in range(len(certs))))
open(os.path.join(here, "SosData.lean"), "w").write("\n".join(out) + "\n")
print(f"wrote SosData.lean: {len(certs)} certificates for {len(index)} plaquettes, smallest SDP margin "
      f"{min(margins):.6f}")
