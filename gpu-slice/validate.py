"""Checks of su2_slice_jax.py against exact results and the NumPy reference (reference/su2_slice.py).

Run before any long GPU job, on the machine that will run it:

    python3 validate.py            # quick checks (a minute on a CPU, seconds on a GPU)
    python3 validate.py --physics  # also a short beta = 2.3 run, compared with the CPU table
    python3 validate.py --bench 32 # time one sweep and one flow step on a 32^4 lattice (cost estimates)
    python3 validate.py --quick    # smaller samples, for a smoke test

Checks:
  1. Kennedy-Pendleton sampler: <x0> = I_2(alpha)/I_1(alpha) for the density sqrt(1-x0^2) exp(alpha x0).
  2. Flow: on a random gauge field, the flowed slice energies agree with the NumPy reference (same steps).
  3. Flow: gauge covariance (a random gauge transformation leaves the flowed slice energies unchanged).
  4. Heat bath: the plaquette at beta = 2.3 and 2.5 agrees with the literature (0.6024, 0.6517).
  5. (--physics) E_k/sqrt(sigma) at beta = 2.3, L = 8 agrees with the CPU runs (3.9-4.4 at tau*sigma = 0.045).
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.special import ive

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import su2_slice_jax as SJ

ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physics", action="store_true")
    p.add_argument("--bench", type=int, default=0)
    p.add_argument("--x64", action="store_true", help="run the statistical checks in double precision")
    p.add_argument("--quick", action="store_true", help="smaller samples (a smoke test, looser statistics)")
    a = p.parse_args()

    K = SJ.build(4, True)                      # the exact comparisons use double precision
    jax, jnp = K["jax"], K["jnp"]
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)

    # 1. Kennedy-Pendleton mean
    for alpha in (0.5, 4.0, 30.0):
        x = np.asarray(K["kp_sample"](jax.random.key(1), jnp.full((100000 if a.quick else 400000,), alpha)))
        exact = ive(2, alpha) / ive(1, alpha)
        err = x.std() / np.sqrt(len(x))
        report(f"KP sampler alpha={alpha}", abs(x.mean() - exact) < 5 * err,
               f"<x0>={x.mean():.5f}, exact {exact:.5f}, 5 sigma = {5 * err:.5f}")

    # 2. flow against the NumPy reference, on a random field
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference"))
    import su2_slice as REF
    rng = np.random.default_rng(3)
    shape = (4, 4, 4, 4)
    U0 = REF.renorm(rng.normal(size=(4,) + shape + (4,)))
    U0 = 0.6 * np.array([1.0, 0, 0, 0]) + 0.4 * U0          # not too rough, so the flow is smooth
    U0 = REF.renorm(U0)
    tau, eps = 0.3, 0.02
    ref = REF.slice_energy(REF.spatial_flow(U0, tau, eps))
    got = np.asarray(K["slice_energy"](K["flow"](jnp.asarray(U0), eps, int(round(tau / eps)))))
    rel = np.max(np.abs(got - ref) / np.abs(ref))
    report("flow vs NumPy reference", rel < 1e-10, f"max relative difference {rel:.1e}")

    # 3. gauge covariance
    g = REF.renorm(rng.normal(size=shape + (4,)))
    Ug = np.stack([REF.qmul(REF.qmul(g, U0[mu]), REF.qdag(np.roll(g, -1, axis=mu))) for mu in range(4)])
    e1 = np.asarray(K["slice_energy"](K["flow"](jnp.asarray(U0), eps, 15)))
    e2 = np.asarray(K["slice_energy"](K["flow"](jnp.asarray(Ug), eps, 15)))
    report("flow gauge covariance", np.max(np.abs(e1 - e2)) < 1e-10, f"max difference {np.max(np.abs(e1 - e2)):.1e}")

    # 4. plaquette
    Ks = SJ.build(4, a.x64)
    shape = (6, 6, 6, 6) if a.quick else (8, 8, 8, 8)
    nsw = 80 if a.quick else 150
    masks = Ks["parity_masks"](shape)
    for beta, lit in ((2.3, 0.6024), (2.5, 0.6517)):
        U = Ks["jnp"].zeros((4,) + shape + (4,), dtype=Ks["jnp"].float64 if a.x64 else Ks["jnp"].float32).at[..., 0].set(1)
        key = Ks["jax"].random.key(7)
        ps = []
        for it in range(nsw):
            U, key = Ks["sweep"](U, key, beta, masks, 3)
            if it >= nsw // 3:
                ps.append(float(Ks["plaquette"](U)))
        ps = np.array(ps)
        blk = np.array([b.mean() for b in np.array_split(ps, 10)])
        err = blk.std(ddof=1) / np.sqrt(len(blk))
        report(f"plaquette beta={beta}", abs(ps.mean() - lit) < max(5 * err, 3e-3 if a.quick else 1e-3),
               f"{ps.mean():.5f} +- {err:.5f}, literature {lit}")

    # 5. a short physics run
    if a.physics:
        import tempfile
        out = os.path.join(tempfile.mkdtemp(), "b2.3_L8")
        args = SJ.parser().parse_args(["--beta", "2.3", "--L", "8", "--T", "8", "--ntherm", "200", "--nmeas",
                                       "300", "--seed", "11", "--tau-sigma", "0.045", "--out", out]
                                      + (["--x64"] if a.x64 else []))
        SJ.run(args)
        import analyse_jax
        E, err = analyse_jax.effective_energy(out + ".npy", out + ".json")[0]
        report("E_k at beta=2.3, tau*sigma=0.045", 3.2 < E < 5.0,
               f"{E:.3f} +- {err:.3f} (CPU runs: 4.41(26) and 3.87(17))")

    # benchmark
    if a.bench:
        L = a.bench
        Kb = SJ.build(4, a.x64)
        jb = Kb["jnp"]
        shape = (L,) * 4
        masks = Kb["parity_masks"](shape)
        U = jb.zeros((4,) + shape + (4,), dtype=jb.float64 if a.x64 else jb.float32).at[..., 0].set(1)
        key = Kb["jax"].random.key(0)
        U, key = Kb["sweep"](U, key, 2.7, masks, 3)
        U.block_until_ready()
        t = time.time()
        for _ in range(3):
            U, key = Kb["sweep"](U, key, 2.7, masks, 3)
        U.block_until_ready()
        ts = (time.time() - t) / 3
        V = Kb["flow"](U, 0.04, 20)            # compile (the step count is a static argument)
        V.block_until_ready()
        t = time.time()
        V = Kb["flow"](U, 0.04, 20)
        V.block_until_ready()
        tf = (time.time() - t) / 20
        print(f"[BENCH] L={L}^4: {ts:.3f} s per sweep (1 heat bath + 3 overrelaxation), "
              f"{tf:.4f} s per RK3 flow step", flush=True)

    print("ALL PASSED" if ok_all else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
