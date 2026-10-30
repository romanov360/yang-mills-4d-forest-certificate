#!/usr/bin/env python3
"""Exact machine checks of the identities in PROOF.md (d = 4).  No floating point anywhere.

A. Plaquette rewriting (PROOF.md section 1).  As identities of noncommutative polynomials in the letters Q_i, Q_i^*,
   X_i, with Q_i Q_i^* = Q_i^* Q_i = 1 and traces taken up to cyclic rotation:
     Tr Q_p(t) to second order in t, with Q_i -> e^{tX_i} Q_i, equals Tr(e^{tY1} e^{tY2} W e^{-tY3} e^{-tY4}) to
     second order, and 2 x its t^2 coefficient equals the trace polynomial whose real part is f_p (fp in the code).
   This holds for every N, since nothing about the matrix size is used.
B. Corner identities (section 2).  Tr(Y1 Y4), Tr(Y1 Y2), Tr(Y3 Y4), Tr(Y2 W Y3 W^*) equal the traces of the products
   of the stated ends, again as cyclic words.
C. Combinatorics (sections 0, 2, 5), by exhaustive enumeration on several boxes of Z^4:
   - every site is the head of exactly one fixed link, the fixed links contain no cycle, and every plaquette has
     exactly one fixed link;
   - the corner table of certify4d.PL is the geometric one: each corner names the two ends of the right links of
     its plaquette, and reduced mod 2 it is the table used by the certificate;
   - in a box, every pair of ends at a site in different directions, both links in the box, is a corner of exactly
     one plaquette of the box; parallel pairs are corners of none.
D. Reduction to U(2) (section 4).  For N = 3 and N = 4, with symbolic skew-Hermitian Y_1..Y_4, symbolic diagonal W
   and symbolic coefficients a_r, k: the form sum_r a_r|Y_r|^2 - f_p - c_p equals the sum of its U(2) pieces
   (diagonal pieces and off-diagonal pairs) as an exact polynomial identity.

Usage: python3 exact_check4d.py
"""
import itertools
import sys
from fractions import Fraction

import sympy as sp

import certify4d as cz
import forest4d as f4

d = 4


# ---------------------------------------------------------------- A, B: noncommutative words
def inv(l):
    return l[:-1] if l.endswith("*") else l + "*"


def is_q(l):
    return l.startswith("Q")


def red(w):
    out = []
    for l in w:
        if out and is_q(l) and out[-1] == inv(l):
            out.pop()
        else:
            out.append(l)
    return tuple(out)


def cyc(w):
    """Canonical form of Tr(w): reduce, cancel around the cycle, take the least rotation."""
    w = list(red(w))
    while len(w) >= 2 and is_q(w[0]) and w[-1] == inv(w[0]):
        w = list(red(w[1:-1]))
    if not w:
        return ()
    return min(tuple(w[i:] + w[:i]) for i in range(len(w)))


def P(*words, c=1):
    return {tuple(w): Fraction(c) for w in words}


def add(*ps):
    out = {}
    for p in ps:
        for w, c in p.items():
            out[w] = out.get(w, 0) + c
    return {w: c for w, c in out.items() if c}


def scal(c, p):
    return {w: c * v for w, v in p.items()}


def mul(*ps):
    out = {(): Fraction(1)}
    for p in ps:
        nxt = {}
        for w1, c1 in out.items():
            for w2, c2 in p.items():
                w = red(w1 + w2)
                nxt[w] = nxt.get(w, 0) + c1 * c2
        out = {w: c for w, c in nxt.items() if c}
    return out


def tr(p):
    out = {}
    for w, c in p.items():
        k = cyc(w)
        out[k] = out.get(k, 0) + c
    return {w: c for w, c in out.items() if c}


ONE = {(): Fraction(1)}


def series_mul(*ss, order=2):
    out = [ONE, {}, {}]
    for s in ss:
        nxt = [{}, {}, {}]
        for i in range(order + 1):
            for j in range(order + 1 - i):
                nxt[i + j] = add(nxt[i + j], mul(out[i], s[j]))
        out = nxt
    return out


def expo(Y, sign):
    return [ONE, scal(sign, Y), scal(Fraction(1, 2), mul(Y, Y))]


const = lambda p: [p, {}, {}]
Q = {i: P([f"Q{i}"]) for i in range(1, 5)}
Qs = {i: P([f"Q{i}*"]) for i in range(1, 5)}
X = {i: P([f"X{i}"]) for i in range(1, 5)}
W = mul(Q[1], Q[2], Qs[3], Qs[4])
Ws = mul(Q[4], Q[3], Qs[2], Qs[1])
Y = {1: X[1], 2: mul(Q[1], X[2], Qs[1]), 3: mul(Q[4], X[3], Qs[4]), 4: X[4]}

# Q_p(t) with Q_i -> e^{tX_i} Q_i, so Q_3^* -> Q_3^* e^{-tX_3}, Q_4^* -> Q_4^* e^{-tX_4}
lhs = series_mul(expo(X[1], 1), const(Q[1]), expo(X[2], 1), const(Q[2]), const(Qs[3]), expo(X[3], -1),
                 const(Qs[4]), expo(X[4], -1))
rhs = series_mul(expo(Y[1], 1), expo(Y[2], 1), const(W), expo(Y[3], -1), expo(Y[4], -1))
for k in range(3):
    assert tr(lhs[k]) == tr(rhs[k]), f"rewriting fails at order {k}"
fp_poly = add(*(mul(Y[i], Y[i], W) for i in range(1, 5)), scal(2, mul(Y[1], Y[2], W)), scal(2, mul(Y[3], Y[4], W)),
              *(scal(-2, mul(Y[i], W, Y[j])) for i in (1, 2) for j in (3, 4)))
assert tr(scal(2, lhs[2])) == tr(fp_poly), "second derivative is not fp"
print("A. Tr Q_p(t) = Tr(e^{tY1}e^{tY2}W e^{-tY3}e^{-tY4}) through order t^2, and d^2/dt^2 at 0 is the trace "
      f"polynomial of f_p ({len(tr(fp_poly))} cyclic words): exact, every N")

# ends of the four links of p, written in the same letters
o = {1: X[1], 2: X[2], 3: X[3], 4: X[4]}                      # o-end = X_e
i_ = {r: mul(Qs[r], X[r], Q[r]) for r in range(1, 5)}         # i-end = Ad(Q_e^*) X_e
checks = [("<Y1,Y4> = <o_mu,o_nu>(x)", mul(Y[1], Y[4]), mul(o[1], o[4])),
          ("<Y1,Y2> = <i_mu,o_nu>(x+mu)", mul(Y[1], Y[2]), mul(i_[1], o[2])),
          ("<Y3,Y4> = <o_mu,i_nu>(x+nu)", mul(Y[3], Y[4]), mul(o[3], i_[4])),
          ("<Y2,WY3W*> = <i_nu,i_mu>(x+mu+nu)", mul(Y[2], W, Y[3], Ws), mul(i_[2], i_[3]))]
for name, a, b in checks:
    assert tr(a) == tr(b), name
assert tr(mul(W, Ws)) == tr(ONE) and tr(mul(Ws, W)) == tr(ONE)
print("B. the four corner identities hold exactly: " + "; ".join(n for n, _, _ in checks))


# ---------------------------------------------------------------- C: combinatorics
C = f4.Cell()
fixed_dir = lambda x: C.mu[tuple(xi % 2 for xi in x)]     # site x is the head of (x - e_mu(x), mu(x))
e_ = lambda m: tuple(int(i == m) for i in range(d))
plus = lambda x, y, s=1: tuple(a + s * b for a, b in zip(x, y))
is_fixed = lambda x, mu: fixed_dir(plus(x, e_(mu))) == mu


def end_link(y, a):
    """The link whose end (y, a) is: a < 4 is o_a(y) on (y, a), a >= 4 is i_{a-4}(y) on (y - e, a - 4)."""
    return (y, a) if a < d else (plus(y, e_(a - d), -1), a - d)


def check_box(lo, hi):
    sites = list(itertools.product(*(range(l, h + 1) for l, h in zip(lo, hi))))
    S = set(sites)
    links = [(x, mu) for x in sites for mu in range(d) if plus(x, e_(mu)) in S]
    L = set(links)
    # forest: union-find on fixed links of the box
    par = {x: x for x in sites}

    def root(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for (x, mu) in links:
        if is_fixed(x, mu):
            a, b = root(x), root(plus(x, e_(mu)))
            assert a != b, f"cycle in F through {(x, mu)}"
            par[a] = b
    plaqs = []
    for x in sites:
        for mu in range(d):
            for nu in range(mu + 1, d):
                es = [(x, mu), (plus(x, e_(mu)), nu), (plus(x, e_(nu)), mu), (x, nu)]
                if all(e in L for e in es):
                    assert sum(is_fixed(*e) for e in es) == 1, f"plaquette {(x, mu, nu)}"
                    xm, xn = plus(x, e_(mu)), plus(x, e_(nu))
                    xmn = plus(xm, e_(nu))
                    corners = ((x, mu, nu), (xm, d + mu, nu), (xmn, d + nu, d + mu), (xn, mu, d + nu))
                    # the links each corner must pair: (e1,e4), (e1,e2), (e2,e3), (e3,e4), in this order of ends
                    want = ((0, 3), (0, 1), (1, 2), (2, 3))
                    for (s, r, q), (u, v) in zip(corners, want):
                        assert end_link(s, r) == es[u] and end_link(s, q) == es[v], "corner names wrong ends"
                    # mod 2 this is the certificate's table
                    cell = lambda z: tuple(zi % 2 for zi in z)
                    j = [k for k, (pp, cc, fp_) in enumerate(cz.PL)
                         if cc[0] == (cell(x), mu, nu)][0]
                    assert cz.PL[j][1] == tuple((cell(s), r, q) for s, r, q in corners), "cell table differs"
                    assert cz.PL[j][2] == [k for k in range(4) if not is_fixed(*es[k])], "free positions differ"
                    plaqs.append(corners)
    count = {}
    for corners in plaqs:
        for s, r, q in corners:
            key = (s, min(r, q), max(r, q))
            count[key] = count.get(key, 0) + 1
    for y in sites:
        for a, b in itertools.combinations(range(2 * d), 2):
            la, lb = end_link(y, a), end_link(y, b)
            if la not in L or lb not in L:
                assert (y, a, b) not in count
                continue
            n = count.get((y, a, b), 0)
            if a % d == b % d:
                assert n == 0, "parallel pair at a corner"
            else:
                assert n == 1, f"pair {(y, a, b)} occurs {n} times"
    for x in sites:  # x is the head of exactly one fixed link of Z^4
        assert sum(is_fixed(plus(x, e_(m), -1), m) for m in range(d)) == 1
    return len(sites), len(links), len(plaqs)


boxes = [((0,) * 4, (3,) * 4), ((0,) * 4, (4,) * 4), ((-1, 0, 2, 1), (2, 4, 3, 5)), ((1, 1, 1, 1), (2, 2, 2, 6)),
         ((0, 0, 0, 0), (1, 1, 1, 1))]
for lo, hi in boxes:
    ns, nl, npl = check_box(lo, hi)
    print(f"C. box {lo}..{hi}: {ns} sites, {nl} links, {npl} plaquettes: forest acyclic, one fixed link per "
          "plaquette, corner table geometric and equal to the certificate's mod 2, every perpendicular pair of "
          "ends at exactly one corner, no parallel pair")


# ---------------------------------------------------------------- D: reduction to U(2)
def form(Ys, w, a, k):
    """sum_r a_r |Y_r|^2 - Re Tr(fp) - c_p for W = diag(w), as an exact sympy expression."""
    n = len(w)
    Wm = sp.diag(*w)
    Wc = sp.diag(*[sp.conjugate(z) for z in w])
    T = lambda M: sp.expand(M.trace())
    Re = lambda z: sp.expand((z + sp.conjugate(z)) / 2)
    ipf = lambda A, B: -Re(T(A * B))
    Y1, Y2, Y3, Y4 = Ys
    f = (sum(T(y * y * Wm) for y in Ys) + 2 * T(Y1 * Y2 * Wm) + 2 * T(Y3 * Y4 * Wm)
         - 2 * sum(T(Yi * Wm * Yj) for Yi in (Y1, Y2) for Yj in (Y3, Y4)))
    c = k[0] * ipf(Y1, Y4) + k[1] * ipf(Y1, Y2) + k[2] * ipf(Y2, Wm * Y3 * Wc) + k[3] * ipf(Y3, Y4)
    return sp.expand(sum(a[r] * ipf(Ys[r], Ys[r]) for r in range(4)) - Re(f) - c)


a = sp.symbols("a0:4", real=True)
k = sp.symbols("k0:4", real=True)
for N in (3, 4):
    w = sp.symbols(f"w0:{N}")
    Ys, ent = [], {}
    for r in range(4):
        M = sp.zeros(N, N)
        for i in range(N):
            M[i, i] = sp.I * sp.Symbol(f"y{r}_{i}", real=True)
            for j in range(i + 1, N):
                z = sp.Symbol(f"u{r}_{i}{j}", real=True) + sp.I * sp.Symbol(f"v{r}_{i}{j}", real=True)
                M[i, j], M[j, i] = z, -sp.conjugate(z)
        Ys.append(M)
    full = form(Ys, w, a, k)
    pieces = 0
    for i in range(N):
        other = w[(i + 1) % N]
        pieces += form([sp.Matrix([[Yr[i, i], 0], [0, 0]]) for Yr in Ys], (w[i], other), a, k)
    for i, j in itertools.combinations(range(N), 2):
        pieces += form([sp.Matrix([[0, Yr[i, j]], [Yr[j, i], 0]]) for Yr in Ys], (w[i], w[j]), a, k)
    assert sp.expand(full - pieces) == 0, f"U({N}) form is not the sum of its U(2) pieces"
    print(f"D. U({N}): sum_r a_r|Y_r|^2 - f_p - c_p = sum of {N} diagonal and {N * (N - 1) // 2} off-diagonal U(2) "
          "pieces, exact polynomial identity in symbolic Y, W = diag(w), a, k")

print("ALL EXACT CHECKS PASSED")
