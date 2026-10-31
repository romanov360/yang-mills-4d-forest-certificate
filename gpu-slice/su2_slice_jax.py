"""SU(2) Wilson lattice gauge theory on a GPU (JAX): the one-step slice bound along the continuum tuning.

A port of su2_slice.py (romanov360/big-bang-monotile, yang-mills-continuum/), with the same algorithm and
output format:

  * links are unit quaternions q = (q0, q1, q2, q3) <-> q0 + i q.sigma, array shape (d, T, L, ..., L, 4);
  * update: Kennedy-Pendleton heat bath, then NOR overrelaxation sweeps, both by direction and checkerboard
    parity (links of one direction and one parity share no plaquette);
  * measurement: Luscher's RK3 Wilson flow of the spatial links of every time slice (spatial plaquettes
    only), stopped at the requested flow times tau*sigma; at each, the zero-momentum slice field
    Phi_t = sum_x sum_{i<j} (1 - Re tr U_ij / 2) for every t.

Each row of the output .npy is [plaquette, Phi_t unflowed (T), Phi_t at tau_1 (T), Phi_t at tau_2 (T), ...].
A .json file next to it records the parameters.  analyse_jax.py computes E_k = -log(Gamma(1)/Gamma(0))/a,
the quantity of Proposition S in yang-mills-continuum/SLICE.md.

The run checkpoints the gauge field and the random key every --ckpt-every measurements and resumes from the
checkpoint if one exists, so it survives preemption.

Example (d = 4, beta = 2.7, 32^4, three flow times):
    python3 su2_slice_jax.py --beta 2.7 --L 32 --T 32 --ntherm 500 --nmeas 500 --sep 2 \
        --tau-sigma 0.045,0.1,0.2 --seed 1 --out out/b2.7_L32
"""
import argparse
import json
import os
import time

import numpy as np


def build(d, x64):
    """Return the jitted kernels for a d-dimensional lattice (imports jax lazily, so --x64 can be set)."""
    import jax
    jax.config.update("jax_enable_x64", x64)
    import jax.numpy as jnp
    from jax import lax

    fdt = jnp.float64 if x64 else jnp.float32
    SIGN = jnp.array([1.0, -1.0, -1.0, -1.0], dtype=fdt)

    def qmul(p, q):
        p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return jnp.stack([
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
            p0 * q1 + q0 * p1 - (p2 * q3 - p3 * q2),
            p0 * q2 + q0 * p2 - (p3 * q1 - p1 * q3),
            p0 * q3 + q0 * p3 - (p1 * q2 - p2 * q1),
        ], axis=-1)

    def qdag(q):
        return q * SIGN

    def renorm(q):
        return q / jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True))

    def qexp_vec(w):
        n = jnp.sqrt(jnp.sum(w * w, axis=-1))
        safe = jnp.where(n > 1e-30, n, 1.0)
        s = jnp.where(n > 1e-30, jnp.sin(n) / safe, 1.0)
        return jnp.concatenate([jnp.cos(n)[..., None], w * s[..., None]], axis=-1)

    def shift(a, mu, s):
        return jnp.roll(a, -s, axis=mu)

    def staple_sum(U, mu, dirs):
        S = jnp.zeros_like(U[mu])
        for nu in dirs:
            if nu == mu:
                continue
            Unu_xmu = shift(U[nu], mu, 1)
            f = qmul(qmul(Unu_xmu, qdag(shift(U[mu], nu, 1))), qdag(U[nu]))
            b = qmul(qmul(qdag(shift(Unu_xmu, nu, -1)), qdag(shift(U[mu], nu, -1))), shift(U[nu], nu, -1))
            S = S + f + b
        return S

    def kp_sample(key, alpha):
        """Kennedy-Pendleton: x0 ~ sqrt(1-x0^2) exp(alpha x0), all sites at once; retries the rejected ones."""
        def cond(state):
            return jnp.any(~state[2])

        def body(state):
            key, x0, done = state
            key, k1 = jax.random.split(key)
            r = jax.random.uniform(k1, (4,) + alpha.shape, dtype=fdt, minval=1e-12, maxval=1.0)
            lam2 = -(jnp.log(r[0]) + jnp.cos(2 * jnp.pi * r[1]) ** 2 * jnp.log(r[2])) / (2 * alpha)
            ok = r[3] ** 2 <= 1 - lam2
            new = ~done & ok
            return key, jnp.where(new, 1 - 2 * lam2, x0), done | ok

        _, x0, _ = lax.while_loop(cond, body, (key, jnp.zeros_like(alpha), jnp.zeros(alpha.shape, bool)))
        return x0

    def random_su2_with_x0(key, x0):
        v = jax.random.normal(key, x0.shape + (3,), dtype=fdt)
        v = v / jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True))
        return jnp.concatenate([x0[..., None], jnp.sqrt(jnp.maximum(0.0, 1 - x0 ** 2))[..., None] * v], axis=-1)

    def parity(shape):
        return (sum(jnp.indices(shape)) % 2) == 0

    def sweep(U, key, beta, masks, nor):
        alld = range(d)
        for mu in range(d):
            for m in masks:
                key, k1, k2 = jax.random.split(key, 3)
                S = staple_sum(U, mu, alld)
                k = jnp.sqrt(jnp.sum(S * S, axis=-1))
                V = S / k[..., None]
                X = random_su2_with_x0(k2, kp_sample(k1, beta * k))
                U = U.at[mu].set(jnp.where(m[..., None], qmul(X, qdag(V)), U[mu]))
        for _ in range(nor):
            for mu in range(d):
                for m in masks:
                    Vd = qdag(renorm(staple_sum(U, mu, alld)))
                    U = U.at[mu].set(jnp.where(m[..., None], qmul(qmul(Vd, qdag(U[mu])), Vd), U[mu]))
        return renorm(U), key

    sp = tuple(range(1, d))

    def flow_force(U):
        Z = jnp.zeros(U.shape[:-1] + (3,), dtype=U.dtype)
        for i in sp:
            Z = Z.at[i].set(-qmul(U[i], staple_sum(U, i, sp))[..., 1:])
        return Z

    def rk3_step(V, eps):
        """One step of Luscher's RK3 integrator; the temporal links are left unchanged."""
        def upd(Z, W):
            return W.at[1:].set(qmul(qexp_vec(Z[1:]), W[1:]))
        Z0 = eps * flow_force(V)
        W1 = upd(0.25 * Z0, V)
        Z1 = eps * flow_force(W1)
        W2 = upd(8 / 9 * Z1 - 17 / 36 * Z0, W1)
        Z2 = eps * flow_force(W2)
        V = upd(3 / 4 * Z2 - 8 / 9 * Z1 + 17 / 36 * Z0, W2)
        return V.at[1:].set(renorm(V[1:]))

    def flow(V, eps, n):
        return lax.fori_loop(0, n, lambda _, W: rk3_step(W, eps), V)

    def slice_energy(V):
        E = jnp.zeros(V.shape[1], dtype=V.dtype)
        axes = tuple(range(1, d))
        for i in range(1, d):
            for j in range(i + 1, d):
                P = qmul(qmul(V[i], shift(V[j], i, 1)), qmul(qdag(shift(V[i], j, 1)), qdag(V[j])))
                E = E + (1 - P[..., 0]).sum(axis=axes)
        return E

    def plaquette(U):
        tot = 0.0
        for mu in range(d):
            for nu in range(mu + 1, d):
                P = qmul(qmul(U[mu], shift(U[nu], mu, 1)), qmul(qdag(shift(U[mu], nu, 1)), qdag(U[nu])))
                tot = tot + P[..., 0].mean()
        return tot / (d * (d - 1) // 2)

    def measure(U, segments):
        """segments: tuple of (eps, nsteps) taking the flow from one checkpoint to the next."""
        out = [jnp.atleast_1d(plaquette(U)), slice_energy(U)]
        V = U
        for eps, n in segments:
            V = flow(V, eps, n) if n > 0 else V
            out.append(slice_energy(V))
        return jnp.concatenate(out)

    return dict(
        jax=jax, jnp=jnp, parity=parity,
        sweep=jax.jit(sweep, static_argnums=(4,)),
        measure=jax.jit(measure, static_argnums=(1,)),
        flow=jax.jit(flow, static_argnums=(2,)),
        slice_energy=jax.jit(slice_energy), plaquette=jax.jit(plaquette),
        qmul=qmul, qdag=qdag, renorm=renorm, kp_sample=kp_sample, staple_sum=staple_sum,
    )


# a sqrt(sigma) for the SU(2) Wilson action in d = 4 (Fingberg-Heller-Karsch 1993; Lucini-Teper-Wenger 2004).
# For other couplings pass --a-sqrt-sigma.  In d = 3, a g^2 = 4/beta and sqrt(sigma)/g^2 = 0.3353 (Teper 1998).
A_SQRT_SIGMA_4D = {2.3: 0.3690, 2.4: 0.2660, 2.5: 0.1881, 2.6: 0.1326, 2.7: 0.0911}
SQRT_SIGMA_OVER_G2_3D = 0.3353


def a_sqrt_sigma(beta, d):
    return A_SQRT_SIGMA_4D[beta] if d == 4 else 4.0 / beta * SQRT_SIGMA_OVER_G2_3D


def flow_segments(taus, eps_max):
    """Split the flow into segments ending exactly at each tau, with steps of at most eps_max."""
    segs, t = [], 0.0
    for tau in taus:
        n = int(np.ceil((tau - t) / eps_max - 1e-12)) if tau > t else 0
        segs.append(((tau - t) / n if n else 0.0, n))
        t = tau
    return tuple(segs)


def run(args, on_checkpoint=None):
    K = build(args.d, args.x64)
    jax, jnp = K["jax"], K["jnp"]
    asig = args.a_sqrt_sigma or a_sqrt_sigma(args.beta, args.d)
    tsig = [float(x) for x in args.tau_sigma.split(",")] if args.tau_sigma else []
    taus = [ts / asig ** 2 for ts in tsig]
    segs = flow_segments(taus, args.eps)
    shape = (args.T,) + (args.L,) * (args.d - 1)
    m = K["parity"](shape)
    masks = (m, ~m)
    meta = dict(beta=args.beta, d=args.d, L=args.L, T=args.T, sep=args.sep, nor=args.nor, seed=args.seed,
                tau_sigma=tsig, tau_lattice=taus, a_sqrt_sigma=asig, eps_max=args.eps, x64=args.x64,
                columns="plaquette, Phi_t unflowed (T), then Phi_t at each tau_sigma (T each)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    ckpt = args.out + ".ckpt.npz"
    rows = []
    t0 = time.time()
    if os.path.exists(ckpt):
        c = np.load(ckpt)
        U = jnp.asarray(c["U"])
        key = jax.random.wrap_key_data(jnp.asarray(c["key"]))
        rows = list(np.load(args.out + ".npy")) if os.path.exists(args.out + ".npy") else []
        therm_done = bool(c["therm_done"])
        print(f"resumed from {ckpt}: {len(rows)} measurements", flush=True)
    else:
        U = jnp.zeros((args.d,) + shape + (4,), dtype=jnp.float64 if args.x64 else jnp.float32).at[..., 0].set(1)
        key = jax.random.key(args.seed)
        therm_done = False

    def save(therm_flag):
        np.savez(ckpt + ".tmp.npz", U=np.asarray(U), key=np.asarray(jax.random.key_data(key)),
                 therm_done=therm_flag)
        os.replace(ckpt + ".tmp.npz", ckpt)
        if rows:
            np.save(args.out + ".tmp.npy", np.array(rows))
            os.replace(args.out + ".tmp.npy", args.out + ".npy")
        if on_checkpoint:
            on_checkpoint()

    if not therm_done:
        for it in range(args.ntherm):
            U, key = K["sweep"](U, key, args.beta, masks, args.nor)
        U.block_until_ready()
        print(f"d={args.d} beta={args.beta} L={args.L} T={args.T} a*sqrt(sigma)={asig:.4f} "
              f"tau/a^2={[round(t, 3) for t in taus]} therm {time.time() - t0:.0f}s "
              f"plaq={float(K['plaquette'](U)):.6f}", flush=True)
        save(True)
    while len(rows) < args.nmeas:
        for _ in range(args.sep):
            U, key = K["sweep"](U, key, args.beta, masks, args.nor)
        rows.append(np.asarray(K["measure"](U, segs), dtype=np.float64))
        n = len(rows)
        if n % args.ckpt_every == 0 or n == args.nmeas:
            save(True)
        if n % 10 == 0 or n == 1:
            print(f"  meas {n} plaq={rows[-1][0]:.6f} t={time.time() - t0:.0f}s", flush=True)
    return args.out + ".npy"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--L", type=int, required=True, help="spatial extent")
    p.add_argument("--T", type=int, required=True, help="temporal extent")
    p.add_argument("--d", type=int, default=4, choices=(3, 4))
    p.add_argument("--ntherm", type=int, default=500)
    p.add_argument("--nmeas", type=int, default=500)
    p.add_argument("--sep", type=int, default=2, help="update sweeps between measurements")
    p.add_argument("--nor", type=int, default=3, help="overrelaxation sweeps per heat-bath sweep")
    p.add_argument("--tau-sigma", default="0.045,0.1,0.2", help="flow times in units of 1/sigma")
    p.add_argument("--a-sqrt-sigma", type=float, default=None,
                   help="lattice spacing in units of 1/sqrt(sigma); required for couplings not in the table")
    p.add_argument("--eps", type=float, default=0.04, help="largest RK3 step")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--x64", action="store_true", help="double precision throughout")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--out", required=True, help="output prefix: writes OUT.npy, OUT.json, OUT.ckpt.npz")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
