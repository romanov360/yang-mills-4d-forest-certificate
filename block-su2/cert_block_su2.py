#!/usr/bin/env python3
"""
Rigorous (ball-arithmetic) evaluation of the SU(2) plaquette-block Dobrushin constant
alpha(beta_W) = 40 |beta| K, in two versions:

  --mode paper   the constants (3) of Bhattacharjee-Bhattacharya-Lockwood-Romanov, Sec. V
                 (reproduces Theorem 3: alpha(0.345) <= 0.99729);
  --mode sharp   the sharpened constants of SHARPENED.md (this directory):
                   s  = 1/4 + (e^w - 1) e^w sqrt(tau) / sqrt(3)            [Poincare, Lemma 154]
                   C  = 4|beta| s A20 + 20|beta| * 16 beta^2 M e^w / 3        [exponent interpolation]
                 where A20 >= max_{r <= 20|beta|} r * 2 phi'(r^2) = max |grad g| on the ball.

g(c) = log Z0(c) = phi(|c|^2),  Z0 = sum_j (t/4)^j / (j!(j+1)!),  t = |c|^2.
All analytic constants are enclosed with python-flint arb on a subdivision of the radius,
with a proved bound on the series tail.  Nothing is floating point.
"""
import argparse, sys, time
from flint import arb, ctx

ctx.prec = 128
J = 48  # number of series terms kept


def coeffs(m):
    """c_j for the m-th t-derivative of S(t) = sum_j t^j/(4^j j!(j+1)!): S^(m)(t) = sum_{j>=m} c_j t^(j-m)."""
    out = []
    for j in range(m, J):
        out.append(arb(1) / (arb(4) ** j * arb.fac_ui(j - m) * arb.fac_ui(j + 1)))
    return out


C = [coeffs(m) for m in range(4)]


def tail(m, T):
    """rigorous bound for sum_{j>=J} of the m-th derivative series at 0 <= t <= T (T an upper bound)."""
    j = J
    term = arb(T) ** (j - m) / (arb(4) ** j * arb.fac_ui(j - m) * arb.fac_ui(j + 1))
    ratio = arb(T) / (4 * (j - m + 1) * (j + 2))
    assert ratio.upper() < 0.5
    return 2 * term  # geometric tail with ratio < 1/2


def horner(cs, t):
    s = arb(0)
    for c in reversed(cs):
        s = s * t + c
    return s


def derivs(t, Tup):
    """enclosures of S, S', S'', S''' on the ball t (t >= 0), with tail."""
    out = []
    for m in range(4):
        e = tail(m, Tup)
        out.append(horner(C[m], t) + arb(0, e.upper()))
    return out


def phis(t, Tup):
    S0, S1, S2, S3 = derivs(t, Tup)
    p1 = S1 / S0
    p2 = S2 / S0 - p1 * p1
    p3 = S3 / S0 - 3 * S2 * S1 / (S0 * S0) + 2 * p1 ** 3
    return p1, p2, p3


def interval(lo, hi):
    lo = arb(lo); hi = arb(hi)
    return arb((lo + hi) / 2, ((hi - lo) / 2).upper())


def constants(Rball, R20, n=20000):
    """lam_lo <= every eigenvalue of Hess g <= tau_hi on |c| <= Rball;  M_hi >= |D^3 g| there;
    A20_hi >= max_{r <= R20} r * 2 phi'(r^2);  phi'' <= 0 checked on the whole ball."""
    lam_lo, tau_hi, M_hi, A20_hi = None, None, None, None
    Rb = arb(Rball)
    Tup = (Rb * Rb).upper() * 1.0001
    phi2_neg = True
    for k in range(n):
        r = interval(Rb * k / n, Rb * (k + 1) / n)
        t = interval(arb(r.lower() if k else 0) ** 2, arb(r.upper()) ** 2)
        p1, p2, p3 = phis(t, Tup)
        e_tan = 2 * p1
        e_rad = 2 * p1 + 4 * t * p2
        lo = min(e_tan.lower(), e_rad.lower())
        hi = max(e_tan.upper(), e_rad.upper())
        rabs = interval(r.lower() if k else 0, r.upper())
        A = 12 * rabs * p2
        G = 8 * rabs ** 3 * p3
        d3 = max(abs(A + G).upper(), (2 * abs(A) / 3).upper())
        if p2.upper() > 0:
            phi2_neg = False
        lam_lo = lo if lam_lo is None else min(lam_lo, lo)
        tau_hi = hi if tau_hi is None else max(tau_hi, hi)
        M_hi = d3 if M_hi is None else max(M_hi, d3)
        if rabs.lower() <= arb(R20).upper():
            a = (rabs * e_tan).upper()
            A20_hi = a if A20_hi is None else max(A20_hi, a)
    return lam_lo, tau_hi, M_hi, A20_hi, phi2_neg


def alpha(bW, consts, mode):
    lam, tau, M, A20 = consts
    b = arb(bW) / 4
    w = 8 * b * b * (tau - lam)
    ew = w.exp()
    if mode == "paper":
        s = arb(1) / 4 + (ew - 1) * ew * tau.sqrt()
        Cc = 20 * b * (4 * b * tau * s + 16 * b * b * M)
    else:
        s = arb(1) / 4 + (ew - 1) * ew * tau.sqrt() / arb(3).sqrt()
        Cc = 4 * b * s * A20 + 20 * b * 16 * b * b * M * ew / 3
    assert Cc.upper() < 1
    K = (1 + b) * s / (1 - Cc)
    a = 40 * b * K
    return dict(w=w.upper(), s=s.upper(), C=Cc.upper(), K=K.upper(), alpha=a.upper())


def f(x):
    """decimal string of an exact arb endpoint (printing only; comparisons are done in arb)."""
    return x.str(10, radius=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "sharp"], default="sharp")
    ap.add_argument("--ball", type=float, default=None, help="radius of the ball (default 24*beta_max)")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--lean", type=str, default=None, help="write the outward-rounded constants as a Lean data file")
    ap.add_argument("betaW", type=str, nargs="+", help="couplings beta_W; the LAST one fixes the ball")
    a = ap.parse_args()
    bmax = arb(a.betaW[-1]) / 4
    Rball = arb(a.ball) if a.ball else 24 * bmax
    R20 = 20 * bmax
    t0 = time.time()
    lam, tau, M, A20, neg = constants(Rball, R20, a.n)
    print(f"# mode={a.mode}  ball r <= {f(Rball.upper())}  subintervals={a.n}  series terms={J}  prec={ctx.prec}")
    print(f"# lambda_* >= {f(lam)}   tau <= {f(tau)}   M <= {f(M)}")
    print(f"# max_(r<=20beta) |grad g| = r*2phi'(r^2) <= {f(A20)}   (20beta = {f(R20.upper())})")
    print(f"# phi'' <= 0 on the whole ball: {neg}   [{time.time()-t0:.1f}s]")
    if a.lean:
        write_lean(a.lean, a.betaW, lam, tau, M, A20)
    for bw in a.betaW:
        r = alpha(bw, (lam, tau, M, A20), a.mode)
        ok = "alpha < 1" if r["alpha"] < 1 else "FAILS"
        print(f"beta_W={bw}: w<={f(r['w'])} s<={f(r['s'])} C<={f(r['C'])} K<={f(r['K'])} alpha<={f(r['alpha'])}  {ok}")


def rnd(x, up, D=10 ** 12):
    """outward rounding of an exact arb point to a fraction with denominator D (returns the integer numerator)."""
    y = x * D
    z = (y.ceil() if up else y.floor())
    return int(z.unique_fmpz())


def write_lean(path, betaWs, lam, tau, M, A20):
    D = 10 ** 12
    L, T, Mu, Au = rnd(lam, False), rnd(tau, True), rnd(M, True), rnd(A20, True)
    # recheck in arb with the rounded constants
    cons = (arb(L) / D, arb(T) / D, arb(Mu) / D, arb(Au) / D)
    for bw in betaWs:
        assert alpha(bw, cons, "sharp")["alpha"] < 1
    lines = ["/- Generated by cert_block_su2.py --lean. Outward-rounded Arb enclosures, denominator 10^12. -/",
             "", "def den : Nat := 1000000000000",
             f"/-- lower bound for every eigenvalue of Hess g on the ball |c| <= 24 beta -/\ndef lamLo : Nat := {L}",
             f"/-- upper bound for every eigenvalue of Hess g on that ball -/\ndef tauHi : Nat := {T}",
             f"/-- upper bound for the norm of D^3 g on that ball -/\ndef mHi : Nat := {Mu}",
             f"/-- upper bound for |grad g(c)| = |c| 2 phi'(|c|^2) on |c| <= 20 beta -/\ndef aHi : Nat := {Au}", ""]
    open(path, "w").write("\n".join(lines))
    print(f"# wrote {path}: lamLo={L} tauHi={T} mHi={Mu} aHi={Au} (/10^12)")


if __name__ == "__main__":
    main()
