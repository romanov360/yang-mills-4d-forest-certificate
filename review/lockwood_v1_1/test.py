# Independent test of Lockwood's Lemma 2: min eigenvalue of R_pm over u(N)^3 for W in U(N), no reduction used.
import numpy as np, sys
from scipy.linalg import expm
from scipy.optimize import minimize
from fractions import Fraction as Fr
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
aa, ao, tau = 2.0547, 1.2471, 0.918   # weights (B)
def basis(N):
    B = []
    for a in range(N):
        E = np.zeros((N, N), complex); E[a, a] = 1j; B.append(E)
        for b in range(a + 1, N):
            E = np.zeros((N, N), complex); E[a, b] = 1; E[b, a] = -1; B.append(E / np.sqrt(2))
            E = np.zeros((N, N), complex); E[a, b] = 1j; E[b, a] = 1j; B.append(E / np.sqrt(2))
    return B
ip = lambda A, B: -np.trace(A @ B).real
tr = lambda A: np.trace(A).real
def R(Y, W, k, sgn):
    f = sum(tr(Y[i] @ Y[i] @ W) for i in range(4)) + 2 * tr(Y[0] @ Y[1] @ W) + 2 * tr(Y[2] @ Y[3] @ W) \
        - 2 * sum(tr(Y[i] @ W @ Y[j]) for i in (0, 1) for j in (2, 3))
    c = 2 * ip(Y[0], Y[3]) - 2 * ip(Y[0], Y[1]) + 2 * ip(Y[1], W @ Y[2] @ W.conj().T) - 2 * ip(Y[2], Y[3])
    w = sum((ao if (l - k) % 4 == 2 else aa) * ip(Y[l], Y[l]) for l in range(4) if l != k)
    return w - sgn * f - tau * c
def lmin(W, k, sgn, B):
    N = W.shape[0]; act = [l for l in range(4) if l != k]; n = 3 * len(B)
    def Yof(v):
        Y = [np.zeros((N, N), complex) for _ in range(4)]
        for p, l in enumerate(act): Y[l] = sum(v[p * len(B) + q] * B[q] for q in range(len(B)))
        return Y
    E = np.eye(n); q = [R(Yof(E[i]), W, k, sgn) for i in range(n)]; M = np.diag(q)
    for i in range(n):
        for j in range(i + 1, n): M[i, j] = M[j, i] = (R(Yof(E[i] + E[j]), W, k, sgn) - q[i] - q[j]) / 2
    return np.linalg.eigvalsh(M)[0]
def Wof(x, N):
    H = x[:N * N].reshape(N, N); H = (H + H.T) / 2 + 1j * (x[N*N:].reshape(N, N) - x[N*N:].reshape(N, N).T) / 2
    return expm(1j * H)
if __name__ == "__main__":
    N = int(sys.argv[1]); B = basis(N); best = (np.inf,)
    for k in range(4):
        for sgn in (1, -1):
            for s in range(3 if N <= 3 else 2):
                x0 = rng.normal(size=2 * N * N) * 2
                r = minimize(lambda x: lmin(Wof(x, N), k, sgn, B), x0, method='Nelder-Mead',
                             options=dict(maxfev=1500 if N <= 3 else 600, xatol=1e-7, fatol=1e-10))
                best = min(best, (r.fun, k, sgn))
    print(f"N={N}: min over searched W of lambda_min(R_pm) = {best[0]:.6e} (k={best[1]}, sign={best[2]}); "
          f"Lemma-2 prediction (his 12x12 family) = 8.716949e-05")
