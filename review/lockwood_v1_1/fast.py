# Vectorized version of test.py: exact matrix of R_pm on u(N)^3 via bilinear terms; adversarial search over W in U(N).
import numpy as np, sys
from scipy.linalg import expm
from scipy.optimize import minimize
import test as slow   # reuse basis(), R(), lmin() for a cross-check (test.py's main block only runs with argv)
aa, ao, tau = 2.0547, 1.2471, 0.918
def G(B, A, C): return np.einsum('iab,bc,jcd,da->ij', B, A, B, C).real   # Re tr(B_i A B_j C)
def mat(W, k, sgn, B):
    N = W.shape[0]; I = np.eye(N); d = len(B); M = np.zeros((4 * d, 4 * d))
    def add(l, m, A, C, c):
        g = c * G(B, A, C)
        if l == m: M[l*d:(l+1)*d, l*d:(l+1)*d] += (g + g.T) / 2
        else: M[l*d:(l+1)*d, m*d:(m+1)*d] += g / 2; M[m*d:(m+1)*d, l*d:(l+1)*d] += g.T / 2
    f = [(l, l, I, W, 1) for l in range(4)] + [(0, 1, I, W, 2), (2, 3, I, W, 2)] + \
        [(i, j, W, I, -2) for i in (0, 1) for j in (2, 3)]
    c = [(0, 3, I, I, -2), (0, 1, I, I, 2), (1, 2, W, W.conj().T, -2), (2, 3, I, I, 2)]
    for l, m, A, C, x in f: add(l, m, A, C, -sgn * x)
    for l, m, A, C, x in c: add(l, m, A, C, -tau * x)
    for l in range(4):
        if l != k: M[l*d:(l+1)*d, l*d:(l+1)*d] += (ao if (l - k) % 4 == 2 else aa) * np.eye(d)
    act = [i for l in range(4) if l != k for i in range(l * d, (l + 1) * d)]
    return M[np.ix_(act, act)]
lmin = lambda W, k, sgn, B: np.linalg.eigvalsh(mat(W, k, sgn, B))[0]
def Wof(x, N):
    H = x[:N*N].reshape(N, N); K = x[N*N:].reshape(N, N)
    return expm(1j * ((H + H.T) / 2 + 1j * (K - K.T) / 2))
if __name__ == '__main__':
    N = int(sys.argv[1]); B = np.array(slow.basis(N)); rng = np.random.default_rng(7 + N)
    # cross-check against the slow polarisation code
    W = Wof(rng.normal(size=2*N*N), N)
    err = max(abs(lmin(W, k, s, B) - slow.lmin(W, k, s, list(B))) for k in range(4) for s in (1, -1))
    print(f"N={N}: fast vs slow lambda_min max diff {err:.1e}", flush=True)
    best = (np.inf,); rand = np.inf
    for _ in range(2000): rand = min(rand, min(lmin(Wof(rng.normal(size=2*N*N)*3, N), k, s, B) for k in range(4) for s in (1, -1)))
    for k in range(4):
        for s in (1, -1):
            for _ in range(10):
                r = minimize(lambda x: lmin(Wof(x, N), k, s, B), rng.normal(size=2*N*N)*3, method='Nelder-Mead',
                             options=dict(maxfev=6000, xatol=1e-9, fatol=1e-12))
                best = min(best, (r.fun, k, s))
    print(f"N={N}: 2000 Haar-ish W: min {rand:.6e}; 80 Nelder-Mead runs: min {best[0]:.6e} (k={best[1]}, sign={best[2]}); "
          f"Lemma 2 predicts 8.716949e-05", flush=True)
