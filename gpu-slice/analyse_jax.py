"""Analyse su2_slice_jax.py output: E_k/sqrt(sigma) at each flow time, with jackknife errors.

E_k = -log(Gamma(1)/Gamma(0))/a, where Gamma(n) is the connected correlation of the slice field at time
separation n, averaged over t.  Proposition S (yang-mills-continuum/SLICE.md): if E_k stays bounded along the
tuning, the slice ratio holds.  Runs with the same parameters and different seeds are pooled.

    python3 analyse_jax.py out/*.npy
"""
import json
import sys
from collections import defaultdict

import numpy as np


def gamma(E, nmax):
    c = E - E.mean()
    return np.array([np.mean(c * np.roll(c, -n, axis=1)) for n in range(nmax + 1)])


def jackknife(E, f, nbin):
    bins = np.array_split(np.arange(len(E)), nbin)
    full = f(E)
    est = np.array([f(np.delete(E, b, axis=0)) for b in bins])
    return full, np.sqrt((nbin - 1) / nbin * np.sum((est - est.mean(axis=0)) ** 2, axis=0))


def columns(X, T, ntau):
    """Unflowed slice series, then one per flow time."""
    return [X[:, 1 + T * j:1 + T * (j + 1)] for j in range(ntau + 1)]


def effective_energy(npy, meta_path=None, X=None):
    """[(E/sqrt(sigma), error) for each flow time] + [(unflowed)]."""
    meta = json.load(open(meta_path or npy[:-4] + ".json"))
    X = np.load(npy) if X is None else X
    T, asig = meta["T"], meta["a_sqrt_sigma"]
    cols = columns(X, T, len(meta["tau_sigma"]))
    lg = lambda E: -np.log(gamma(E, 1)[1] / gamma(E, 1)[0])
    res = [jackknife(E, lg, min(20, len(X))) for E in cols[1:] + cols[:1]]
    return [(v / asig, e / asig) for v, e in res]


def main():
    groups = defaultdict(list)
    for path in sys.argv[1:]:
        meta = json.load(open(path[:-4] + ".json"))
        k = (meta["d"], meta["beta"], meta["L"], meta["T"], tuple(meta["tau_sigma"]))
        groups[k].append((path, meta))
    for (d, beta, L, T, tsig), runs in sorted(groups.items()):
        X = np.concatenate([np.load(p) for p, _ in runs])
        meta = runs[0][1]
        res = effective_energy(runs[0][0], X=X)
        print(f"d={d} beta={beta} L={L} T={T} a*sqrt(sigma)={meta['a_sqrt_sigma']:.4f} "
              f"ncfg={len(X)} chains={len(runs)} <P>={X[:, 0].mean():.5f}")
        for ts, (v, e) in zip(list(tsig) + ["unflowed"], res):
            print(f"    tau*sigma={ts}: E_k/sqrt(sigma) = {v:.3f} +- {e:.3f}")


if __name__ == "__main__":
    main()
