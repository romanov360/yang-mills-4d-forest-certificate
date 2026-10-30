#!/usr/bin/env python3
"""Sup of -Hess at the identity field for the d = 4 period-2 forest gauge, by Bloch theory (a lower bound for h_F).

At Q = I, -f_p = |Y1 + Y2 - Y3 - Y4|^2, so -Hess is 3 copies of sum_p (x_e1 + x_e2 - x_e3 - x_e4)^2 on free links,
periodic under 2Z^4; its sup is the max over Bloch momenta k of the top eigenvalue of the 48 x 48 Bloch matrix.
"""
import itertools

import numpy as np
from scipy.optimize import minimize

import forest4d as f4

T = f4.Cell()
fi = {e: i for i, e in enumerate(T.free)}
rows = []
for x in T.sites:
    for a in range(4):
        for b in range(a + 1, 4):
            terms = []
            for (y, c, sgn) in [(x, a, 1), (tuple(x[i] + (i == a) for i in range(4)), b, 1),
                                (tuple(x[i] + (i == b) for i in range(4)), a, -1), (x, b, -1)]:
                yc = tuple(v % 2 for v in y)
                n = tuple((v - w) // 2 for v, w in zip(y, yc))
                e = T.eidx[(yc, c)]
                if e in fi:
                    terms.append((fi[e], np.array(n), sgn))
            rows.append(terms)


def top(k):
    B = np.zeros((len(rows), len(fi)), complex)
    for r, terms in enumerate(rows):
        for j, n, s in terms:
            B[r, j] += s * np.exp(1j * np.dot(k, n))
    return np.linalg.eigvalsh(B.conj().T @ B)[-1]


best = (0, None)
g = np.linspace(0, 2 * np.pi, 8, endpoint=False)
for k in itertools.product(g, repeat=4):
    v = top(np.array(k))
    if v > best[0]:
        best = (v, k)
res = minimize(lambda k: -top(k), np.array(best[1]), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
print("grid best", best[0], "at", np.round(best[1], 4))
print("refined sup", -res.fun, "at k =", np.round(res.x % (2 * np.pi), 6), "; 9 + sqrt 17 =", 9 + 17 ** 0.5)
