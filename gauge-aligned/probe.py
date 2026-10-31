"""Gauge-aligned coupling probe for 4D SU(2) lattice Yang-Mills (Monte Carlo, NOT a proof).

Question: is the Dobrushin ceiling beta_W ~ 1/3 a property of the theory, or of measuring
disagreement on gauge-fixed links?

Setup: box {0..L}^4, Wilson weight exp(beta_W * sum_p Re Q_p) (quaternions, 1/2 Tr = Re).
Links lying in the boundary surface are frozen (boundary condition). Two copies share the
boundary except one boundary link y, which is flipped (U_y -> -U_y) in copy 2.
Both copies run heat-bath sweeps; each link update of copy 2 is maximally coupled to that of
copy 1 (copy 2 takes copy 1's new value with probability min(1, nu2/nu1), else a residual draw).

Two couplings:
  plain   : copy 2 updated as is.
  aligned : before every sweep, copy 2 is gauge-transformed at interior sites to minimise
            sum_e |U1_e - g_x U2_e g_{x+mu}^*|^2 (relative Landau gauge). The law of copy 2
            is invariant under these transformations (boundary sites untouched), so this is
            still a coupling of the two box measures.

Any coupling bounds W1 between the marginals from above. We report, at stationarity,
  link(r): E|U1_e - U2_e| for free links e at distance r from y (gauge-variant),
  plaq(r): E|Re Q1_p - Re Q2_p| for plaquettes p at distance r from y (gauge-invariant),
and their totals. A finite-box (Dobrushin-Shlosman type) criterion can only work if the
total gauge-invariant influence of one boundary link is small compared with |V|/|boundary|.
"""
import argparse
import sys
import time

import numpy as np
from scipy.special import ive


def qmul(a, b):
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0], axis=-1)


def qconj(a):
    return a * np.array([1.0, -1.0, -1.0, -1.0])


def fwd(A, ax):  # A[x + e_ax] (garbage at the far edge; always masked)
    return np.roll(A, -1, axis=ax)


def bwd(A, ax):  # A[x - e_ax]
    return np.roll(A, 1, axis=ax)


class Box:
    def __init__(self, L, R, beta, rng):
        self.L, self.R, self.beta, self.rng = L, R, beta, rng
        X = L + 1
        self.shape = (R, X, X, X, X)
        g = np.indices((X,) * 4)  # coords
        self.coord = g
        par = g.sum(axis=0) % 2
        self.par = par
        self.valid = [g[mu] < L for mu in range(4)]
        inner = [(g[nu] >= 1) & (g[nu] <= L - 1) for nu in range(4)]
        self.free = []
        for mu in range(4):
            m = self.valid[mu].copy()
            for nu in range(4):
                if nu != mu:
                    m &= inner[nu]
            self.free.append(m)
        self.isite = inner[0] & inner[1] & inner[2] & inner[3]
        self.table()

    # inverse CDF of z0 with density sqrt(1-z^2) exp(r z) on [-1,1]
    def table(self):
        self.rmax = 6.0 * self.beta * 1.0001 + 1e-9
        nr, nz, nu = 400, 6001, 2001
        self.rg = np.linspace(0, self.rmax, nr)
        z = np.linspace(-1, 1, nz)
        ug = np.linspace(0, 1, nu)
        T = np.empty((nr, nu))
        for i, r in enumerate(self.rg):
            w = np.sqrt(np.clip(1 - z * z, 0, None)) * np.exp(r * (z - 1))
            c = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]))])
            c /= c[-1]
            T[i] = np.interp(ug, c, z)
        self.T, self.nu = T, nu

    def z0(self, r, u):
        fr = np.clip(r / self.rmax, 0, 1) * (len(self.rg) - 1)
        fu = u * (self.nu - 1)
        i, j = np.minimum(fr.astype(int), len(self.rg) - 2), np.minimum(fu.astype(int), self.nu - 2)
        a, b = fr - i, fu - j
        T = self.T
        return ((1 - a) * (1 - b) * T[i, j] + a * (1 - b) * T[i + 1, j]
                + (1 - a) * b * T[i, j + 1] + a * b * T[i + 1, j + 1])

    def staple(self, U, mu):
        """Sum S over plaquettes containing (x,mu) with Re Q_p = Re(U_mu(x) S_p). Shape (R,X^4,4)."""
        L = self.L
        S = np.zeros(self.shape + (4,))
        a = 1 + mu
        for nu in range(4):
            if nu == mu:
                continue
            b = 1 + nu
            Umu, Unu = U[:, mu], U[:, nu]
            # forward: U_nu(x+mu) U_mu(x+nu)^* U_nu(x)^*
            Sf = qmul(qmul(fwd(Unu, a), qconj(fwd(Umu, b))), qconj(Unu))
            mf = self.coord[nu] < L
            # backward: U_nu(x-nu+mu)^* U_mu(x-nu)^* U_nu(x-nu)
            Unu_b = bwd(Unu, b)
            Sb = qmul(qmul(qconj(fwd(Unu_b, a)), qconj(bwd(Umu, b))), Unu_b)
            mb = self.coord[nu] >= 1
            S += Sf * mf[None, ..., None] + Sb * mb[None, ..., None]
        return S

    def plaqs(self, U):
        """Re Q_p for all (mu<nu); masked list."""
        out = []
        L = self.L
        for mu in range(4):
            for nu in range(mu + 1, 4):
                a, b = 1 + mu, 1 + nu
                P = qmul(qmul(U[:, mu], fwd(U[:, nu], a)), qmul(qconj(fwd(U[:, mu], b)), qconj(U[:, nu])))
                out.append(P[..., 0])
        return out

    def plaq_mask_and_centers(self):
        L = self.L
        masks, cents = [], []
        g = self.coord.astype(float)
        for mu in range(4):
            for nu in range(mu + 1, 4):
                m = (self.coord[mu] < L) & (self.coord[nu] < L)
                c = g.copy()
                c[mu] += 0.5
                c[nu] += 0.5
                masks.append(m)
                cents.append(c)
        return masks, cents

    def sample(self, c, u, n):
        r = np.linalg.norm(c, axis=-1)
        chat = c / np.maximum(r, 1e-300)[..., None]
        z0 = self.z0(r, u)
        s = np.sqrt(np.clip(1 - z0 * z0, 0, None))
        z = np.concatenate([z0[..., None], s[..., None] * n], axis=-1)
        return qmul(z, chat)

    def logdens(self, x, c):
        r = np.linalg.norm(c, axis=-1)
        # log(2 I1(r)/r) = log(2 ive(1,r)/r) + r ; r->0 limit 0
        rr = np.maximum(r, 1e-12)
        lz = np.log(2 * ive(1, rr) / rr) + rr
        return (x * c).sum(-1) - lz

    def rand_un(self):
        u = self.rng.random(self.shape)
        n = self.rng.normal(size=self.shape + (3,))
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return u, n

    def update(self, Us, mu, p, u, n):
        """Heat bath on links (x,mu), parity p. Copy 1 exact; copy 2 maximally coupled to copy 1."""
        sel = (self.free[mu] & (self.par == p))[None, ..., None]
        U1 = Us[0]
        c1 = self.beta * qconj(self.staple(U1, mu))
        x1 = self.sample(c1, u, n)
        if len(Us) > 1:
            U2 = Us[1]
            c2 = self.beta * qconj(self.staple(U2, mu))
            # accept x1 for copy 2 with prob min(1, nu2(x1)/nu1(x1))
            acc = np.log(self.rng.random(self.shape)) < (self.logdens(x1, c2) - self.logdens(x1, c1))
            x2 = x1.copy()
            todo = ~acc & sel[..., 0]
            for _ in range(200):
                if not todo.any():
                    break
                uu, nn = self.rand_un()
                y = self.sample(c2, uu, nn)
                # residual of nu2 - min(nu1,nu2): accept y with prob 1 - nu1(y)/nu2(y)
                ratio = np.exp(np.minimum(self.logdens(y, c1) - self.logdens(y, c2), 0.0))
                ok = todo & (self.rng.random(self.shape) < 1 - ratio)
                x2 = np.where(ok[..., None], y, x2)
                todo &= ~ok
            U2[:, mu] = np.where(sel, x2, U2[:, mu])
        U1[:, mu] = np.where(sel, x1, U1[:, mu])

    def sweep(self, Us):
        for mu in range(4):
            for p in (0, 1):
                u, n = self.rand_un()
                self.update(Us, mu, p, u, n)

    def align(self, U1, U2, sweeps=4):
        """Gauge-transform U2 at interior sites toward U1 (in place)."""
        for _ in range(sweeps):
            for p in (0, 1):
                M = np.zeros(self.shape + (4,))
                for mu in range(4):
                    a = 1 + mu
                    vo = self.valid[mu][None, ..., None]
                    M += qmul(U2[:, mu], qconj(U1[:, mu])) * vo
                    vi = (self.coord[mu] >= 1)[None, ..., None]
                    M += qmul(qconj(bwd(U2[:, mu], a)), bwd(U1[:, mu], a)) * vi
                g = qconj(M)
                g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-300)
                sel = (self.isite & (self.par == p))[None, ..., None]
                one = np.zeros_like(g)
                one[..., 0] = 1
                g = np.where(sel, g, one)
                for mu in range(4):
                    a = 1 + mu
                    # U_mu(x) -> g_x U_mu(x) g_{x+mu}^*
                    U2[:, mu] = qmul(qmul(g, U2[:, mu]), qconj(fwd(g, a)))


def run(L, beta, R, therm, meas, mode, bc, seed):
    rng = np.random.default_rng(seed)
    B = Box(L, R, beta, rng)
    U = np.zeros(B.shape[:1] + (4,) + B.shape[1:] + (4,))
    U[..., 0] = 1
    if bc == "haar":
        H = rng.normal(size=U.shape)
        U = H / np.linalg.norm(H, axis=-1, keepdims=True)
    elif bc == "therm":
        # thermalise the whole box (boundary included) with free boundary, then freeze
        allfree = B.free
        B.free = [B.valid[mu] for mu in range(4)]
        for _ in range(therm):
            B.sweep([U])
        B.free = allfree
    # boundary link y: on the face x0 = 0, direction 1, centred
    h = L // 2
    y = (0, h, h, h)
    ymu = 1
    U1 = U.copy()
    for _ in range(therm):
        B.sweep([U1])
    U2 = U1.copy()
    U2[:, ymu, y[0], y[1], y[2], y[3]] *= -1
    ymid = np.array(y, float)
    ymid[ymu] += 0.5
    # distances
    g = B.coord.astype(float)
    lmask, ldist = [], []
    for mu in range(4):
        c = g.copy()
        c[mu] += 0.5
        lmask.append(B.free[mu])
        ldist.append(np.sqrt(((c - ymid[:, None, None, None, None]) ** 2).sum(axis=0)))
    pm, pc = B.plaq_mask_and_centers()
    pdist = [np.sqrt(((c - ymid[:, None, None, None, None]) ** 2).sum(axis=0)) for c in pc]
    bins = np.arange(0, 2 * L + 1, 1.0)
    lsum = np.zeros(len(bins)); lcnt = np.zeros(len(bins))
    psum = np.zeros(len(bins)); pcnt = np.zeros(len(bins))
    ltot = 0.0; ptot = 0.0
    for mu in range(4):
        idx = np.digitize(ldist[mu][lmask[mu]], bins) - 1
        np.add.at(lcnt, idx, 1)
    for m, d in zip(pm, pdist):
        idx = np.digitize(d[m], bins) - 1
        np.add.at(pcnt, idx, 1)
    burn = therm
    for t in range(burn + meas):
        if mode == "aligned":
            B.align(U1, U2)
        B.sweep([U1, U2])
        if t < burn:
            continue
        for mu in range(4):
            dd = np.linalg.norm(U1[:, mu] - U2[:, mu], axis=-1).mean(axis=0)
            vals = dd[lmask[mu]]
            idx = np.digitize(ldist[mu][lmask[mu]], bins) - 1
            np.add.at(lsum, idx, vals)
            ltot += vals.sum()
        P1, P2 = B.plaqs(U1), B.plaqs(U2)
        for m, d, a, b in zip(pm, pdist, P1, P2):
            dd = np.abs(a - b).mean(axis=0)
            vals = dd[m]
            idx = np.digitize(d[m], bins) - 1
            np.add.at(psum, idx, vals)
            ptot += vals.sum()
    lsum /= meas; psum /= meas; ltot /= meas; ptot /= meas
    return bins, lsum / np.maximum(lcnt, 1), psum / np.maximum(pcnt, 1), ltot, ptot, lcnt, pcnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--therm", type=int, default=60)
    ap.add_argument("--meas", type=int, default=60)
    ap.add_argument("--bc", default="therm")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("betas", type=float, nargs="+")
    a = ap.parse_args()
    print("# L=%d R=%d therm=%d meas=%d bc=%s seed=%d" % (a.L, a.R, a.therm, a.meas, a.bc, a.seed))
    for beta in a.betas:
        for mode in ("plain", "aligned"):
            t0 = time.time()
            bins, lp, pp, lt, pt, lc, pc = run(a.L, beta, a.R, a.therm, a.meas, mode, a.bc, a.seed)
            print("beta_W=%.3f %-7s total_link=%.3f total_plaq=%.3f  (%.0fs)" % (beta, mode, lt, pt, time.time() - t0))
            print("   r    link(r)   plaq(r)")
            for i in range(len(bins)):
                if lc[i] or pc[i]:
                    print("  %2d  %.2e  %.2e" % (bins[i], lp[i], pp[i]))
            sys.stdout.flush()
