"""Exact checks of Proposition S (continuum/SLICE.md) on the two models where BBL verify the slice ratio.

1. The massless free lattice field of BBL Remark "freeslice": h = exp(-|x|^2), heat flow t_f = 0.05.
   We compute D_k(r) from their formula, reproduce their ratio D(1)/D(1/2) -> 0.6337, and compute the
   one-step constants kappa_k = (1 - D_k(a_k)/D_k(0))/a_k, E_k = -log(D_k(a_k)/D_k(0))/a_k and the mean
   energy <E>_k of the slice vector.  All converge as a_k -> 0, so Proposition S applies; its bound exp(-E_k) is printed
   next to the true ratio.
2. Two-dimensional SU(2) (BBL Remark "twodslice"), f = Re Tr V, g^2 = ell = 1: kappa_k -> E_F = 3/8.

Floating point (numpy quadrature), not a proof.
"""
import numpy as np
from scipy.special import ive


def free_field(a, tf=0.05, n=None):
    # Brillouin zone [-pi/a, pi/a]^3; |h_k^|^2 replaced by the continuum |h^|^2 = pi^3 exp(-p^2/2)
    # (aliasing corrections are below 1e-12 for a <= 1/2 because of the Gaussian factors).
    pmax = min(np.pi / a, 12.0)          # the integrand is < 1e-30 beyond |p| = 12
    n = n or 280
    p1 = (np.arange(n) + 0.5) / n * 2 * pmax - pmax
    px, py, pz = np.meshgrid(p1, p1, p1, indexing="ij")
    phat2 = sum((2 / a * np.sin(a * q / 2)) ** 2 for q in (px, py, pz))
    om = np.arccosh(1 + a * a * phat2 / 2) / a
    w = np.pi ** 3 * np.exp(-(px ** 2 + py ** 2 + pz ** 2) / 2) * np.exp(-2 * tf * phat2) \
        / (2 / a * np.sinh(a * om))
    D = lambda r: np.sum(w * np.exp(-r * om))
    D0 = D(0.0)
    kappa = (1 - D(a) / D0) / a
    Eeff = -np.log(D(a) / D0) / a
    meanE = np.sum(w * om) / D0
    return D, D0, kappa, Eeff, meanE


print("free massless field, h = exp(-|x|^2), t_f = 0.05")
print(f"{'a':>8} {'D(1)/D(1/2)':>12} {'kappa_k':>10} {'E_k':>10} {'<E>_k':>10} {'bound exp(-E_k)':>16} {'D(1)/D(0)':>10}")
for k in range(1, 7):
    a = 2.0 ** -k
    D, D0, kappa, Eeff, meanE = free_field(a)
    print(f"{a:8.5f} {D(1.0) / D(0.5):12.4f} {kappa:10.4f} {Eeff:10.4f} {meanE:10.4f} {np.exp(-Eeff):16.4f} "
          f"{D(1.0) / D0:10.4f}")

print("\ntwo-dimensional SU(2), f = Re Tr V, g^2 = ell = 1: kappa_k -> E_F = 3/8")
for k in range(1, 8):
    a = 2.0 ** -k
    beta = 4.0 / a ** 2                  # beta_k = 2N/(g^2 a^2), N = 2
    L = int(round(1 / a))
    c = ive(2, beta) / ive(1, beta)      # c_hat_{1/2}(beta) = I_2/I_1
    print(f"a={a:.5f}  E_k = E_F,k={-np.log(c) * L / a:.6f}  kappa_k={(1 - c ** L) / a:.6f}")
