#!/usr/bin/env python3
"""An independent re-implementation of the combinatorial and identity-field facts of this repository.

Written separately from forest4d.py and bloch4d.py (it shares no code with them) as a cross-check, by the authors of
the sympy_models fork of the OpenAI Yang-Mills release (github.com/brentharts/openai_math_yang_mills), whose
models/forest_gauge_4d.py checks the same facts inside a symbolic framework.  Needs only numpy and sympy; exact.

Checked, from the 16-letter pattern alone, on true Z^4 coordinates:
  * 16 of the 64 links of a period cell are fixed, every site is the head of one fixed link, the parent step lowers the
    coordinate sum by one (so the fixed links form a forest);
  * every plaquette contains exactly one fixed link;
  * each free link lies in 6 plaquettes and has 12 distinct free partners;
  * at the identity field and Bloch momentum 0 the 48 x 48 integer matrix B^T B has the factor x^2 - 18x + 64, so the
    top eigenvalue is 9 + sqrt 17.
Usage: python3 crosscheck4d.py
"""
import itertools
from collections import Counter

import sympy as sp

PAT = "0212331001332120"
cells = sorted(itertools.product(range(2), repeat=4))
mu = {c: int(PAT[i]) for i, c in enumerate(cells)}
e = lambda a: tuple(int(i == a) for i in range(4))
add = lambda x, y, s=1: tuple(p + s * q for p, q in zip(x, y))
head_dir = lambda x: mu[tuple(v % 2 for v in x)]
fixed = lambda x, a: head_dir(add(x, e(a))) == a


def plaq(x, a, b):
    return [(x, a), (add(x, e(a)), b), (add(x, e(b)), a), (x, b)]


links = [(x, a) for x in cells for a in range(4)]
assert sum(fixed(*l) for l in links) == 16 and len(links) == 64
assert set(Counter(tuple(v % 2 for v in add(x, e(a))) for x, a in links if fixed(x, a)).values()) == {1}
assert all(sum(add(x, e(head_dir(x)), -1)) == sum(x) - 1 for x in itertools.product(range(-3, 5), repeat=4))
window = itertools.product(range(-2, 4), repeat=4)
assert all(sum(fixed(*g) for g in plaq(x, a, b)) == 1 for x in window for a in range(4) for b in range(a + 1, 4))
print("forest: 16 of 64 fixed, one head per site, no cycle, exactly one fixed link in every plaquette")

counts = set()
for x, a in links:
    if fixed(x, a):
        continue
    part, nplaq = set(), 0
    for b in range(4):
        if b == a:
            continue
        for s in (0, -1):
            o = add(x, e(b), s)
            p = plaq(o, a, b) if a < b else plaq(o, b, a)
            nplaq += 1
            part |= {g for g in p if g != (x, a) and not fixed(*g)}
    counts.add((nplaq, len(part)))
assert counts == {(6, 12)}
print("each free link: 6 plaquettes, 12 distinct free partners")

free = [l for l in links if not fixed(*l)]
col = {l: i for i, l in enumerate(free)}
rows = []
for x in cells:
    for a in range(4):
        for b in range(a + 1, 4):
            r = [0] * 48
            for (y, c), sg in zip(plaq(x, a, b), (1, 1, -1, -1)):
                yc = tuple(v % 2 for v in y)
                if (yc, c) in col:
                    r[col[(yc, c)]] += sg
            rows.append(r)
B = sp.Matrix(rows)
x = sp.symbols("x")
fac = [f for f, _ in sp.factor_list((B.T * B).charpoly(x).as_expr())[1]]
assert any(sp.expand(f - (x**2 - 18 * x + 64)) == 0 for f in fac)
top = max(max(sp.Poly(f, x).nroots(n=30), key=lambda r: sp.re(r)) for f in fac)
assert abs(top - (9 + sp.sqrt(17))) < sp.Rational(1, 10**20)
print(f"identity field, k = 0: top eigenvalue {sp.N(top, 15)} = 9 + sqrt 17 (factor x^2 - 18x + 64)")
