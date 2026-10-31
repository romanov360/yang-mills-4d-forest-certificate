"""True boundary influence in a 4D SU(2) box (Monte Carlo, NOT a proof).

Delta(r) = | E^w[Re Q_p] - E^w'[Re Q_p] | averaged over plaquettes p at distance r from the
boundary link y, where w' is w with U_y -> -U_y. Independent chains, no coupling. This is a
lower bound for any coupling-based (W1 or TV) influence, so it shows what a finite-box
criterion could at best achieve, and separates "the coupling fails" from "the physics fails".
"""
import argparse, sys, time
import numpy as np
from probe import Box, qconj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--therm", type=int, default=100)
    ap.add_argument("--meas", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("betas", type=float, nargs="+")
    a = ap.parse_args()
    print("# L=%d R=%d therm=%d meas=%d seed=%d (cold boundary: all links 1, y flipped in copy 2)" % (a.L, a.R, a.therm, a.meas, a.seed))
    for beta in a.betas:
        t0 = time.time()
        rng = np.random.default_rng(a.seed)
        B = Box(a.L, a.R, beta, rng)
        U = np.zeros((a.R, 4) + B.shape[1:] + (4,)); U[..., 0] = 1
        h = a.L // 2; y = (0, h, h, h); ymu = 1
        U1 = U.copy(); U2 = U.copy(); U2[:, ymu, y[0], y[1], y[2], y[3]] *= -1
        for _ in range(a.therm):
            B.sweep([U1]); B.sweep([U2])
        pm, pc = B.plaq_mask_and_centers()
        ymid = np.array(y, float); ymid[ymu] += 0.5
        pd = [np.sqrt(((c - ymid[:, None, None, None, None]) ** 2).sum(0)) for c in pc]
        bins = np.arange(0, 2 * a.L + 1, 1.0)
        acc1 = [np.zeros((a.R,) + B.shape[1:]) for _ in pm]; acc2 = [np.zeros_like(x) for x in acc1]
        for _ in range(a.meas):
            B.sweep([U1]); B.sweep([U2])
            for k, (p1, p2) in enumerate(zip(B.plaqs(U1), B.plaqs(U2))):
                acc1[k] += p1; acc2[k] += p2
        # per-replica means -> mean and standard error of the difference, per plaquette
        print("beta_W=%.2f  (%.0fs)" % (beta, time.time() - t0))
        print("   r   |Delta(r)|   stderr   n_plaq")
        rows = {}
        for k, m in enumerate(pm):
            diff = (acc1[k] - acc2[k]) / a.meas  # (R, X^4)
            mu_ = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(a.R)
            idx = np.digitize(pd[k][m], bins) - 1
            for i, mm, ss in zip(idx, mu_[m], se[m]):
                rows.setdefault(i, []).append((mm, ss))
        for i in sorted(rows):
            v = np.array(rows[i])
            # average signed difference over the shell, error combined in quadrature
            mean = v[:, 0].mean(); se = np.sqrt((v[:, 1] ** 2).sum()) / len(v)
            print("  %2d   %.2e    %.1e   %d" % (bins[i], abs(mean), se, len(v)))
        sys.stdout.flush()

main()
