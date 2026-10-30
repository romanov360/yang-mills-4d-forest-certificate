"""SU(2) Wilson lattice gauge theory in d = 4: the one-step slice bound (S).

Measures, along the continuum tuning, the quantity of Proposition S in
continuum/SLICE.md,

    kappa_k = <u_k, (1 - T_k) u_k> / (a_k <u_k, u_k>) = (1 - D_k(a_k)/D_k(0)) / a_k ,

for the slice field u_k = zero-momentum spatial plaquette energy of one time
slice after the spatial gradient flow at a fixed physical flow time.  If
kappa_k stays bounded in physical units, Jensen's inequality gives the slice
ratio of BBL's forest paper (Proposition "sliceratio"), the open input of the
nontriviality criterion (Theorem "flowlimit"(iv)).

This is a Monte Carlo measurement, not a proof.

Links are unit quaternions q = (q0, q1, q2, q3) <-> q0 + i q.sigma.
Usage:  python3 su2_slice.py BETA L T NTHERM NMEAS SEP TAU_SIGMA SEED OUT
"""
import sys
import time
import numpy as np


# ---------------------------------------------------------------- quaternions
def qmul(p, q):
    """(p0 + i p.s)(q0 + i q.s) = p0 q0 - p.q + i (p0 q + q0 p - p x q).s"""
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + q0 * p1 - (p2 * q3 - p3 * q2),
        p0 * q2 + q0 * p2 - (p3 * q1 - p1 * q3),
        p0 * q3 + q0 * p3 - (p1 * q2 - p2 * q1),
    ], axis=-1)


def qdag(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qexp_vec(w):
    """exp(i w.sigma) for a real 3-vector field w (shape (..., 3))."""
    n = np.sqrt(np.sum(w * w, axis=-1))
    s = np.where(n > 1e-300, np.sin(n) / np.where(n > 1e-300, n, 1.0), 1.0)
    return np.concatenate([np.cos(n)[..., None], w * s[..., None]], axis=-1)


def renorm(q):
    return q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True))


# ------------------------------------------------------------------- lattice
# U has shape (4, T, L, L, L, 4); direction 0 is time.
def shift(a, mu, s):
    """a(x + s mu^) as an array over x (axis mu+0 of a site field)."""
    return np.roll(a, -s, axis=mu)


def staple_sum(U, mu, dirs):
    """Sum over nu in dirs, nu != mu, of the staples Sigma with plaquette = U_mu(x) Sigma.

    forward:  U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+
    backward: U_nu(x+mu-nu)^+ U_mu(x-nu)^+ U_nu(x-nu)
    """
    S = np.zeros_like(U[mu])
    for nu in dirs:
        if nu == mu:
            continue
        Unu_xmu = shift(U[nu], mu, 1)
        f = qmul(qmul(Unu_xmu, qdag(shift(U[mu], nu, 1))), qdag(U[nu]))
        b = qmul(qmul(qdag(shift(Unu_xmu, nu, -1)), qdag(shift(U[mu], nu, -1))),
                 shift(U[nu], nu, -1))
        S += f + b
    return S


def parity_mask(shape):
    idx = np.indices(shape)
    return (idx.sum(axis=0) % 2) == 0


def kp_sample(alpha, rng):
    """Kennedy-Pendleton: x0 with density ~ sqrt(1-x0^2) exp(alpha x0), alpha > 0 array."""
    x0 = np.empty_like(alpha)
    todo = np.ones(alpha.shape, dtype=bool)
    while todo.any():
        a = alpha[todo]
        r1, r2, r3, r4 = (rng.random(a.shape) for _ in range(4))
        lam2 = -(np.log(1 - r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(1 - r3)) / (2 * a)
        ok = r4 ** 2 <= 1 - lam2
        idx = np.flatnonzero(todo)
        x0[idx[ok]] = 1 - 2 * lam2[ok]
        todo[idx[ok]] = False
    return x0


def random_dir(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def heatbath(U, beta, rng, masks):
    for mu in range(4):
        for m in masks:
            S = staple_sum(U, mu, range(4))[m]           # (n, 4)
            k = np.sqrt(np.sum(S * S, axis=-1))
            V = S / k[:, None]                            # U V has density exp(beta k x0)
            x0 = kp_sample(beta * k, rng)
            xv = np.sqrt(np.maximum(0.0, 1 - x0 ** 2))[:, None] * random_dir(len(x0), rng)
            X = np.concatenate([x0[:, None], xv], axis=1)
            U[mu][m] = qmul(X, qdag(V))


def overrelax(U, masks):
    for mu in range(4):
        for m in masks:
            S = staple_sum(U, mu, range(4))[m]
            V = renorm(S)
            Vd = qdag(V)
            U[mu][m] = qmul(qmul(Vd, qdag(U[mu][m])), Vd)


# ------------------------------------------------------- spatial gradient flow
def flow_force(U):
    """Z_i = -P_ah(U_i Sigma_i) over the spatial plaquettes only, as 3-vectors.

    exp(eps Z) U increases Re Tr(U Sigma) to first order (verified: the energy decreases)."""
    Z = np.empty(U.shape[:-1] + (3,))
    Z[0] = 0.0
    for i in (1, 2, 3):
        W = qmul(U[i], staple_sum(U, i, (1, 2, 3)))
        Z[i] = -W[..., 1:]
    return Z


def spatial_flow(U, tau, eps):
    """Lüscher's RK3 integrator for the Wilson flow of the spatial links of every slice."""
    V = U.copy()
    nsteps = int(round(tau / eps))
    eps = tau / nsteps if nsteps else 0.0
    for _ in range(nsteps):
        Z0 = eps * flow_force(V)
        W1 = V.copy()
        W1[1:] = qmul(qexp_vec(0.25 * Z0[1:]), V[1:])
        Z1 = eps * flow_force(W1)
        W2 = W1.copy()
        W2[1:] = qmul(qexp_vec(8 / 9 * Z1[1:] - 17 / 36 * Z0[1:]), W1[1:])
        Z2 = eps * flow_force(W2)
        V[1:] = renorm(qmul(qexp_vec(3 / 4 * Z2[1:] - 8 / 9 * Z1[1:] + 17 / 36 * Z0[1:]), W2[1:]))
    return V


def slice_energy(V):
    """Phi_t = sum_x sum_{i<j} (1 - Re tr U_ij / 2), one number per time slice."""
    E = np.zeros(V.shape[1])
    for i in (1, 2, 3):
        for j in range(i + 1, 4):
            P = qmul(qmul(V[i], shift(V[j], i, 1)), qmul(qdag(shift(V[i], j, 1)), qdag(V[j])))
            E += (1 - P[..., 0]).sum(axis=(1, 2, 3))
    return E


def plaquette(U):
    tot = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            P = qmul(qmul(U[mu], shift(U[nu], mu, 1)), qmul(qdag(shift(U[mu], nu, 1)), qdag(U[nu])))
            tot += P[..., 0].mean()
    return tot / 6


def main():
    beta, L, T, ntherm, nmeas, sep, tau_sigma, seed, out = (
        float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8]), sys.argv[9])
    asig = A_SQRT_SIGMA[beta]
    tau = tau_sigma / asig ** 2          # flow time in lattice units, fixed tau*sigma
    eps = min(0.04, tau / 4) if tau > 0 else 0.0
    rng = np.random.default_rng(seed)
    shape = (T, L, L, L)
    m = parity_mask(shape)
    masks = (m, ~m)
    U = np.zeros((4,) + shape + (4,))
    U[..., 0] = 1.0
    t0 = time.time()
    for it in range(ntherm):
        heatbath(U, beta, rng, masks)
        for _ in range(3):
            overrelax(U, masks)
        U = renorm(U)
    print(f"beta={beta} L={L} T={T} a*sqrt(sigma)={asig} tau/a^2={tau:.4f} "
          f"therm {time.time() - t0:.0f}s plaq={plaquette(U):.6f}", flush=True)
    rows = []
    for it in range(nmeas):
        for _ in range(sep):
            heatbath(U, beta, rng, masks)
            for _ in range(3):
                overrelax(U, masks)
        U = renorm(U)
        V = spatial_flow(U, tau, eps) if tau > 0 else U
        rows.append(np.concatenate([[plaquette(U)], slice_energy(V), slice_energy(U)]))
        if it % 10 == 0:
            print(f"  meas {it} plaq={rows[-1][0]:.6f} t={time.time() - t0:.0f}s", flush=True)
        np.save(out, np.array(rows))


# a sqrt(sigma) for the SU(2) Wilson action (Fingberg-Heller-Karsch 1993;
# Lucini-Teper-Wenger 2004).  Only ratios along the tuning matter here.
A_SQRT_SIGMA = {2.3: 0.3690, 2.4: 0.2660, 2.5: 0.1881, 2.6: 0.1326, 2.7: 0.0911}

if __name__ == "__main__":
    main()
