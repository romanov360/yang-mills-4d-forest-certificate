#!/usr/bin/env python3
"""Period-2 forest gauge on Z^4 (pattern of the earlier note): period-cell bookkeeping for the certificate.

Site x is the head of the fixed link (x - e_mu(x), mu(x)); mu(x) depends on x mod 2.  Every plaquette has exactly one
fixed link (checked below), so every plaquette has three free links.  Links and plaquettes are indexed modulo the
period 2, which is all the certificate needs: its unknowns are periodic.
"""
import itertools

PAT4 = "0212331001332120"
D, P = 4, 2


class Cell:
    def __init__(self, d=D, pattern=PAT4, per=P):
        self.d = d
        self.sites = sorted(itertools.product(range(per), repeat=d))
        mu_of = {c: int(pattern[i]) for i, c in enumerate(self.sites)}
        self.mu = mu_of
        self.sh = lambda x, a, s=1: tuple((x[i] + (s if i == a else 0)) % per for i in range(d))
        self.edges = [(x, a) for x in self.sites for a in range(d)]
        self.eidx = {e: i for i, e in enumerate(self.edges)}
        self.fixed = {self.eidx[(y, a)] for (y, a) in self.edges if self.mu[self.sh(y, a)] == a}
        self.free = [e for e in range(len(self.edges)) if e not in self.fixed]


if __name__ == "__main__":
    C = Cell()
    cnt = []
    for x in C.sites:
        for a in range(4):
            for b in range(a + 1, 4):
                p = [C.eidx[(x, a)], C.eidx[(C.sh(x, a), b)], C.eidx[(C.sh(x, b), a)], C.eidx[(x, b)]]
                cnt.append(sum(e in C.fixed for e in p))
    print("fixed", len(C.fixed), "of", len(C.edges), "; fixed per plaquette", set(cnt), len(cnt), "plaquettes")
