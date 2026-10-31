"""Certified explicit boxes for BBL's gauge Dobrushin-Shlosman condition, two-dimensional SU(2).

BBL (4Y-M, strong-coupling/prd, Proposition "2d") prove that for every beta some L x L square satisfies
GDS(V, eps), with an existence argument (s with eta < 1/(80 s^2), then L = 8 s^2).  This script finds, for
given couplings, the smallest L that their construction certifies, in ball arithmetic (python-flint / Arb).

Construction (theirs).  For a boundary link y, integrate out the interior links of an s x s sub-square D
with y on its outer side (disk integration: the boundary holonomy of D carries the density f^{*s^2}).  Couple
the marginals of the remaining links maximally, and sample D independently.  Then kappa(y,b) <= 1 on the
2 s (s-1) links of D and kappa(y,b) <= eta_{s^2} elsewhere.  GDS holds if

    4 s (s-1) / (L-1) + 4 L eta_{s^2} < 1.

Bounds used.  With a_n = I_n(beta)/I_1(beta) (n = 2j+1, weight exp(beta_W cos theta), beta = beta_W):

    eps_m = sup_U |f^{*m}(U) - 1| = sum_{n >= 2} n^2 a_n^m,

and the total variation between the two marginals is at most eps_m.  This is BBL's Lemma "tanh"
consequence, sharper than the 2 eps/(1 - eps^2) of their proof; both are printed.  The tail n > N of the
series is bounded with I_{n+1}(x)/I_n(x) <= x/(2(n+1)) (Amos 1974), so the result is a rigorous enclosure.

    python3 gds2d_certified.py
"""
from flint import arb, ctx

ctx.prec = 200


def eps_m(beta, m, N=None):
    """Rigorous upper bound for eps_m = sum_{n>=2} n^2 a_n^m (returned as an arb ball; use .upper())."""
    b = arb(beta)
    I1 = b.bessel_i(1)
    N = N or max(40, int(3 * beta) + 20)
    s = arb(0)
    for n in range(2, N + 1):
        s += arb(n) ** 2 * (b.bessel_i(n) / I1) ** m
    # tail n > N: term ratio t_{n+1}/t_n <= ((n+1)/n)^2 (beta/(2(n+1)))^m =: q < 1 for n >= N
    aN = b.bessel_i(N) / I1
    q = (arb(N + 1) / N) ** 2 * (b / (2 * (N + 1))) ** m
    assert q < 1
    tail = arb(N) ** 2 * aN ** m * q / (1 - q)
    return s + tail


def smallest_box(beta, smax=60):
    """Smallest L certified by some s, with the sharper TV bound; also the L with BBL's 2e/(1-e^2)."""
    best, best_bbl = None, None
    for s in range(2, smax + 1):
        e = eps_m(beta, s * s).upper()
        if e >= 1:
            continue
        for eta, tag in ((e, "sharp"), (2 * e / (1 - e * e), "bbl")):
            # smallest integer L with 4s(s-1)/(L-1) + 4 L eta < 1  (L >= s)
            L = max(s, 2)
            while L < 10 ** 7:
                lhs = arb(4 * s * (s - 1)) / (L - 1) + 4 * L * arb(eta)
                if lhs < 1:
                    break
                if 4 * L * float(eta) > 1:
                    L = None
                    break
                L += 1 if L < 2000 else L // 200
            if L is None or L >= 10 ** 7:
                continue
            cand = (L, s, float(eta))
            if tag == "sharp" and (best is None or L < best[0]):
                best = cand
            if tag == "bbl" and (best_bbl is None or L < best_bbl[0]):
                best_bbl = cand
    return best, best_bbl


def main():
    print("2D SU(2) Wilson action, weight exp(beta_W cos theta).  Smallest L x L square certified by the")
    print("disk-integration coupling of BBL Prop. 2d (sub-square s x s), in ball arithmetic.")
    print(f"{'beta_W':>7} {'L (TV <= eps)':>14} {'s':>4} {'eta':>11}   {'L (BBL 2e/(1-e^2))':>19} {'s':>4}"
          f"   {'BBL existence L = 8 s^2':>24}")
    for beta in (0.5, 1, 2, 3, 4, 6, 8):
        (L, s, eta), bbl = smallest_box(beta)
        # BBL's own choice: smallest s with eta_{s^2} < 1/(80 s^2) (their eta), then L = 8 s^2
        s0 = next(s for s in range(1, 200)
                  if (lambda e: e < 1 and 2 * e / (1 - e * e) < 1 / (80 * s * s))(eps_m(beta, s * s).upper()))
        print(f"{beta:7.1f} {L:14d} {s:4d} {eta:11.3e}   {bbl[0]:19d} {bbl[1]:4d}   {8 * s0 * s0:24d}")


if __name__ == "__main__":
    main()
