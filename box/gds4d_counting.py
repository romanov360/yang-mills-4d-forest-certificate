"""How large must a d = 4 box be for GDS(V, eps) with a "neighbourhood" coupling?  (GDS4D_SCOPING.md)

V = links with both endpoints in {0..L-1}^4.  The shell dV = links outside V sharing a plaquette with V.
A coupling that resamples a set D_y of m links next to each boundary link y independently has kappa(y,b) <= 1
on D_y; with kappa <= eta on the rest, condition (eq:gds) needs, for every direction nu,

    |dV_nu| (m + |V| eta) <= (1 - eps) n_min,     n_min = min_mu |V_mu|.

This script counts |dV_nu| and n_min exactly and prints the smallest L for given m (eta -> 0), and the
largest eta allowed at that L.
"""
import itertools


def counts(L, d=4):
    inside = lambda x: all(0 <= c < L for c in x)
    V = set()
    for x in itertools.product(range(L), repeat=d):
        for mu in range(d):
            y = list(x); y[mu] += 1
            if inside(y):
                V.add((x, mu))
    shell = set()
    for (x, mu) in V:
        for nu in range(d):
            if nu == mu:
                continue
            for s in (0, -1):                      # the two plaquettes in the (mu, nu) plane through (x, mu)
                base = list(x); base[nu] += s
                xm = list(base); xm[mu] += 1
                xn = list(base); xn[nu] += 1
                for link in ((tuple(base), mu), (tuple(xn), mu), (tuple(base), nu), (tuple(xm), nu)):
                    if link not in V:
                        shell.add(link)
    nmin = min(sum(1 for (_, m) in V if m == mu) for mu in range(d))
    dnu = max(sum(1 for (_, m) in shell if m == nu) for nu in range(d))
    return len(V), nmin, dnu


if __name__ == "__main__":
    print(f"{'L':>4} {'|V|':>9} {'n_min':>8} {'|dV_nu|':>8} {'n_min/|dV_nu|':>14}")
    for L in (3, 4, 6, 8, 12, 16):
        v, nmin, dnu = counts(L)
        print(f"{L:4d} {v:9d} {nmin:8d} {dnu:8d} {nmin / dnu:14.2f}")
    print("\nn_min/|dV_nu| grows like L/8, so a coupling that gives up m links per boundary link needs L >~ 8 m,")
    print("and eta <~ n_min / (|dV_nu| |V|) ~ 1/(32 L^4) on the remaining links.")
    for m in (1, 6, 18, 50):
        print(f"  m = {m:3d}:  L >~ {8 * m:4d},  |V| ~ {4 * (8 * m) ** 4:.1e} links,  eta <~ {1 / (32 * (8 * m) ** 4):.1e}")
