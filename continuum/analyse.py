"""Analyse the su2_slice.py runs: the one-step constant kappa_k of Proposition S, in units of sqrt(sigma).

For each coupling, Gamma(n) is the connected correlation of the zero-momentum flowed slice energy at time
separation n, averaged over t, with jackknife errors over bins of configurations.  Then

    kappa_k / sqrt(sigma) = (1 - Gamma(1)/Gamma(0)) / (a sqrt(sigma))        (linear form, <= 1/a always)
    E_k / sqrt(sigma)     = -log(Gamma(1)/Gamma(0)) / (a sqrt(sigma))        (log form, Proposition S)

and, for comparison, the unflowed slice field (which is not the field of BBL's theorem) is shown too.
Usage: python3 analyse.py data/*.npy
"""
import sys
import numpy as np
from su2_slice import A_SQRT_SIGMA


def gamma(E, nmax):
    """Connected correlator of each configuration's slice series, averaged over t (E: (ncfg, T))."""
    T = E.shape[1]
    c = E - E.mean()
    return np.array([np.mean(c * np.roll(c, -n, axis=1)) for n in range(nmax + 1)])


def jackknife(E, f, nbin):
    bins = np.array_split(np.arange(len(E)), nbin)
    full = f(E)
    est = np.array([f(np.delete(E, b, axis=0)) for b in bins])
    err = np.sqrt((nbin - 1) / nbin * np.sum((est - est.mean(axis=0)) ** 2, axis=0))
    return full, err


def main():
    print(f"{'beta':>5} {'L':>3} {'ncfg':>5} {'a*sqrt(s)':>9} {'<P>':>8} {'1/(a sqrt s)':>12} "
          f"{'kappa/sqrt(s)':>14} {'E/sqrt(s) flowed':>17} {'E/sqrt(s) unflowed':>19} {'D(r)/D(0), r sqrt(s)~0.74':>27}")
    runs = {}
    for path in sys.argv[1:]:                  # independent chains at one coupling are pooled
        beta = float(path.split("/b")[-1].split("_")[0])
        runs.setdefault(beta, []).append(np.load(path))
    for beta in sorted(runs):
        X = np.concatenate(runs[beta])
        T = (X.shape[1] - 1) // 2
        P, Ef, Eu = X[:, 0], X[:, 1:T + 1], X[:, T + 1:]
        asig = A_SQRT_SIGMA[beta]
        nbin = min(20, len(X))
        one = lambda E: 1 - gamma(E, 1)[1] / gamma(E, 1)[0]
        lg = lambda E: -np.log(gamma(E, 1)[1] / gamma(E, 1)[0])
        vk, ek = jackknife(Ef, one, nbin)
        vf, ef = jackknife(Ef, lg, nbin)
        vu, eu = jackknife(Eu, lg, nbin)
        n2 = int(round(0.74 / asig))
        rr = lambda E: gamma(E, n2)[n2] / gamma(E, n2)[0]
        vr, er = jackknife(Ef, rr, nbin)
        print(f"{beta:5.2f} {T:3d} {len(X):5d} {asig:9.4f} {P.mean():8.5f} {1 / asig:12.2f} "
              f"{vk / asig:8.3f}({ek / asig:.3f}) {vf / asig:10.3f}({ef / asig:.3f}) "
              f"{vu / asig:12.3f}({eu / asig:.3f}) {'n=%d: %.3f(%.3f)' % (n2, vr, er):>27}")




def multi(paths, tsig=(0.045, 0.1, 0.2)):
    """E_k/sqrt(sigma) at several flow times, for runs made with `su2_slice.py multi`."""
    print(f"{'beta':>5} {'ncfg':>5} {'a*sqrt(s)':>9} " + " ".join(f"{'tau*s=%g' % t:>14}" for t in tsig)
          + f" {'unflowed':>14}")
    for path in paths:
        X = np.load(path)
        beta = float(path.split("_b")[-1].split("_")[0])
        T = (X.shape[1] - 1) // (len(tsig) + 1)
        asig = A_SQRT_SIGMA[beta]
        lg = lambda E: -np.log(gamma(E, 1)[1] / gamma(E, 1)[0])
        cols = [X[:, 1 + T * (j + 1):1 + T * (j + 2)] for j in range(len(tsig))] + [X[:, 1:1 + T]]
        res = [jackknife(E, lg, min(20, len(X))) for E in cols]
        print(f"{beta:5.2f} {len(X):5d} {asig:9.4f} " + " ".join(f"{v / asig:7.3f}({e / asig:.3f})" for v, e in res))


if __name__ == "__main__":
    multi(sys.argv[2:]) if sys.argv[1] == "multi" else main()
