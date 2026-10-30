"""Monte Carlo feasibility probe of the gauge Dobrushin-Shlosman condition GDS(V, eps)
(BBL, Definition def:gds / Theorem G) for SU(2) Wilson theory.

NOT A PROOF. kappa_V(y, b) is a supremum over boundary conditions; here it is replaced
by a maximum over a few sampled boundary conditions, and the probabilities are Monte
Carlo estimates. The output says whether a box certificate is plausible at a given
coupling, and how large the box must be.

Box.  Sites {0..L}^d. V is the set of links all of whose plaquettes lie in the box,
i.e. (x, mu) with x_mu in [0, L-1] and x_nu in [1, L-1] for nu != mu (for d = 2 these
are the 2L(L-1) "interior links" of BBL's Prop. prop:2d). dV is the set of box links
outside V that share a plaquette with V. Interior sites I(V) = {1..L-1}^d.

Coupling.  Interior gauge is fixed by an axial forest: for x in I(V) the link
(x - e_0, 0) is set to the identity (Lemma lem:lift lets the coupling be built on the
gauge quotient). The two box measures, for omega and omega' differing at one link y,
are coupled by running two heat-bath chains link by link; at each update the two
conditional laws are coupled maximally (identical staples give identical samples).
The stationary law of the pair chain is a coupling of the two box measures, so its
disagreement probabilities are admissible values of kappa_V(y, b) for that omega.

Criterion (symmetric box):  r = sum_{y in dV} sum_{b in V} kappa(y, b) / |V| <= 1 - eps.

The coupled-chain idea follows BBL's checks/cube/influence_mc.py (common random
numbers); the box, the maximal coupling and the criterion here are written for
Definition def:gds.

Run: python3 gds_mc.py --d 4 --L 2 --beta 0.3
"""
import argparse
import itertools
import math
import sys
import numpy as np
from scipy.special import ive


# ---------------------------------------------------------------- quaternions
def qmul(a, b):
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                     a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                     a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                     a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0], axis=-1)


CONJ = np.array([1.0, -1.0, -1.0, -1.0])


def haar(rng, shape):
    q = rng.standard_normal(shape + (4,))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def log_z(k):
    """log of the normaliser of exp(<u, v>) under normalised Haar, |v| = k: 2 I_1(k) / k."""
    k = np.maximum(k, 1e-12)
    return np.log(2 * ive(1, k) / k) + k


def sample_w0(rng, k):
    """w0 in [-1, 1] with density ~ sqrt(1 - w0^2) exp(k w0) (Kennedy-Pendleton for k >= 2)."""
    out = np.empty_like(k)
    todo = np.arange(k.size)
    while todo.size:
        kk = k[todo]
        small = kk < 2.0
        w = np.empty_like(kk)
        acc = np.zeros(kk.size, bool)
        if small.any():                       # uniform on S^3, accept with exp(k (w0 - 1))
            n = int(small.sum())
            w0 = haar(rng, (n,))[:, 0]
            a = rng.random(n) < np.exp(kk[small] * (w0 - 1))
            w[small], acc[small] = w0, a
        big = ~small
        if big.any():
            n = int(big.sum())
            r1, r2, r3, r4 = 1 - rng.random((4, n))
            lam2 = -(np.log(r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(r3)) / (2 * kk[big])
            a = r4 ** 2 <= 1 - lam2
            w[big], acc[big] = 1 - 2 * lam2, a
        out[todo[acc]] = w[acc]
        todo = todo[~acc]
    return out


def sample_link(rng, v):
    """u with density ~ exp(<u, v>) under Haar, for rows of v."""
    k = np.linalg.norm(v, axis=-1)
    w0 = sample_w0(rng, k)
    d = rng.standard_normal((len(k), 3))
    d *= (np.sqrt(np.maximum(1 - w0 ** 2, 0)) / np.linalg.norm(d, axis=-1))[:, None]
    w = np.concatenate([w0[:, None], d], axis=1)
    vhat = v / np.maximum(k, 1e-300)[:, None]
    return qmul(w, vhat)                      # <w vhat, vhat> = w0


def logdens(u, v):
    return np.einsum('ij,ij->i', u, v) - log_z(np.linalg.norm(v, axis=-1))


def maximal_coupling(rng, va, vb):
    """Maximal coupling of the two heat-bath laws, row by row."""
    x = sample_link(rng, va)
    y = x.copy()
    same = np.all(va == vb, axis=1)
    idx = np.nonzero(~same)[0]
    if idx.size:
        xa = x[idx]
        lr = logdens(xa, vb[idx]) - logdens(xa, va[idx])        # log q/p at X
        keep = np.log(rng.random(idx.size)) < np.minimum(lr, 0)
        rest = idx[~keep]
        while rest.size:                                           # residual (q - p)_+
            cand = sample_link(rng, vb[rest])
            lr = logdens(cand, va[rest]) - logdens(cand, vb[rest])   # log p/q at Y
            acc = rng.random(rest.size) < 1 - np.exp(np.minimum(lr, 0))
            y[rest[acc]] = cand[acc]
            rest = rest[~acc]
    return x, y


# ---------------------------------------------------------------------- box
class Box:
    def __init__(self, d, L):
        self.d, self.L = d, L
        self.index = {}
        self.links = []
        for x in itertools.product(range(L + 1), repeat=d):
            for mu in range(d):
                if x[mu] < L:
                    self.index[(x, mu)] = len(self.links)
                    self.links.append((x, mu))
        inV = lambda x, mu: all((0 <= x[n] <= L - 1) if n == mu else (1 <= x[n] <= L - 1)
                                for n in range(d))
        self.V = [i for i, (x, mu) in enumerate(self.links) if inV(x, mu)]
        Vset = set(self.V)
        plaqs = []
        for x in itertools.product(range(L + 1), repeat=d):
            for mu, nu in itertools.combinations(range(d), 2):
                if x[mu] < L and x[nu] < L:
                    xm = tuple(x[i] + (i == mu) for i in range(d))
                    xn = tuple(x[i] + (i == nu) for i in range(d))
                    p = [(self.index[(x, mu)], 1), (self.index[(xm, nu)], 1),
                         (self.index[(xn, mu)], -1), (self.index[(x, nu)], -1)]
                    if any(l in Vset for l, _ in p):
                        plaqs.append(p)
        self.dV = sorted({l for p in plaqs for l, _ in p} - Vset)
        # interior gauge fixing: (x - e0, 0) = 1 for x in I(V)
        fixed = set()
        for x in itertools.product(range(1, L), repeat=d):
            fixed.add(self.index[(tuple(x[i] - (i == 0) for i in range(d)), 0)])
        self.free = [l for l in self.V if l not in fixed]
        self.fixed = sorted(fixed)
        # staple recipe for each free link: list of (other three (link, sign) in cyclic order, own sign)
        self.recipe = {}
        for l in self.free:
            r = []
            for p in plaqs:
                for k, (m, s) in enumerate(p):
                    if m == l:
                        rest = [p[(k + j) % 4] for j in (1, 2, 3)]
                        r.append((rest, s))
            self.recipe[l] = r
        self.n_min = min(sum(1 for l in self.V if self.links[l][1] == mu) for mu in range(d))

    def staple(self, U, l):
        """sigma with weight exp(beta * Re(u sigma)) = exp(<u, beta * conj(sigma)>)."""
        tot = 0.0
        for rest, s in self.recipe[l]:
            S = None
            for m, t in rest:
                q = U[m] if t > 0 else U[m] * CONJ
                S = q if S is None else qmul(S, q)
            tot = tot + (S if s > 0 else S * CONJ)
        return tot

    def symmetry_classes(self):
        """Orbits of dV under the hypercubic group of the box."""
        d, L = self.d, self.L
        def key(l, perm, refl):
            x, mu = self.links[l]
            a = list(x)
            b = list(x); b[mu] += 1
            a2 = [L - a[perm[i]] if refl[i] else a[perm[i]] for i in range(d)]
            b2 = [L - b[perm[i]] if refl[i] else b[perm[i]] for i in range(d)]
            return tuple(sorted([tuple(a2), tuple(b2)]))
        group = [(p, r) for p in itertools.permutations(range(d))
                 for r in itertools.product((0, 1), repeat=d)]
        classes = {}
        for l in self.dV:
            c = min(key(l, p, r) for p, r in group)
            classes.setdefault(c, []).append(l)
        return list(classes.values())


def run(box, beta, y, omega_kind, R, therm, sweeps, rng):
    nl = len(box.links)
    U = np.zeros((nl, R, 4)); U[..., 0] = 1.0
    bd = box.dV
    if omega_kind == 'haar':
        U[bd] = haar(rng, (len(bd), R))
    for l in box.free:                       # hot start inside
        U[l] = haar(rng, (R,))
    for _ in range(therm):
        for l in box.free:
            U[l] = sample_link(rng, beta * box.staple(U, l) * CONJ)
    W = U.copy()
    if omega_kind == 'haar':
        W[y] = haar(rng, (R,))
    else:                                    # omega = 1, omega'_y = -1
        W[y] = -U[y]
    dis = np.zeros(nl)
    for t in range(2 * sweeps):
        for l in box.free:
            a, b = maximal_coupling(rng, beta * box.staple(U, l) * CONJ,
                                    beta * box.staple(W, l) * CONJ)
            U[l], W[l] = a, b
        if t >= sweeps:                      # first half is burn-in of the pair chain
            dis += np.any(U != W, axis=-1).mean(axis=1)
    return dis[box.V] / sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--beta', type=float, nargs='+', default=[0.3])
    ap.add_argument('--R', type=int, default=2000)
    ap.add_argument('--therm', type=int, default=100)
    ap.add_argument('--sweeps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    box = Box(a.d, a.L)
    cls = box.symmetry_classes()
    print(f'd={a.d} L={a.L} |V|={len(box.V)} free={len(box.free)} |dV|={len(box.dV)} '
          f'n_min={box.n_min} classes={[len(c) for c in cls]} R={a.R} sweeps={a.sweeps}')
    for beta in a.beta:
        rng = np.random.default_rng(a.seed)
        total, parts = 0.0, []
        for c in cls:
            k = np.maximum(*(run(box, beta, c[0], kind, a.R, a.therm, a.sweeps, rng)
                             for kind in ('haar', 'flip')))     # max over sampled omega, per b
            parts.append(k.sum())
            total += len(c) * k.sum()
        r = total / len(box.V)
        print(f'beta_W={beta:.3f}  r=K/n_min={r:.4f}  per-class sum_b kappa='
              + ' '.join(f'{p:.4f}' for p in parts), flush=True)


if __name__ == '__main__':
    main()
