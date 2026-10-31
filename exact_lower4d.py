#!/usr/bin/env python3
"""An exact rational lower bound for h_F in d = 4: h_F >= 13.97 with no floating point in the certified step.

check_field.py recomputes lambda_max = 13.974014 from the saved SU(2) field in floating point.  Here the same number
is turned into a theorem about rational arithmetic:

1. Every link variable of the saved field is moved to an exactly unitary rational matrix: a unit quaternion
   q = (1 - |t|^2, 2t) / (1 + |t|^2) with t rational (inverse stereographic projection), so |q| = 1 holds exactly.
2. A test vector v on the free links is taken from floating point (the top eigenvector) and rounded to rationals.
   Nothing about how v was found is trusted.
3. The Rayleigh quotient H(v, v) / |v|^2 is evaluated in exact rational arithmetic, straight from the definition used in
   lib/hessian.py: the curves Q_e(t) = exp(t X_e) Q_e are expanded to second order in t, the plaquette products
   Q_p(t) = Q_1(t) Q_2(t) Q_3(t)^* Q_4(t)^* are multiplied as truncated polynomials in t with exact entries, and
   H(v, v) = sum_p d^2/dt^2 Re Tr Q_p(t) at t = 0 is twice the t^2 coefficient.  No rewriting of the Hessian is used.
4. The vector is zero on the fixed links, so this is the gauge-fixed Hessian on the free links of the forest.

Since the field is periodic with period 2 and the vector lives on the L = 2 torus, a periodic field on Z^4 and cut-off
periodic vectors give h_F >= H(v, v) / |v|^2 (as in lib/torus3d.py).  The result is therefore a certified lower bound
for the constant that the proved upper bound 14.420001 controls.

Usage: python3 exact_lower4d.py [field.npy]     (about a minute)
"""
import os
import sys
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import torus3d as t  # noqa: E402
import forest4d as f4  # noqa: E402

DEN_T = 10 ** 9     # denominator bound for the stereographic coordinates of the field
DEN_V = 10 ** 9     # denominator bound for the test vector


# ---- exact arithmetic in M_2(C) with Fraction entries, as polynomials in t truncated at degree 2 -------------------
class C:
    """Gaussian rational a + b i."""
    __slots__ = ("a", "b")

    def __init__(self, a=Fr(0), b=Fr(0)):
        self.a, self.b = Fr(a), Fr(b)

    def __add__(s, o):
        return C(s.a + o.a, s.b + o.b)

    def __mul__(s, o):
        return C(s.a * o.a - s.b * o.b, s.a * o.b + s.b * o.a)

    def conj(s):
        return C(s.a, -s.b)


def mmul(A, B):
    return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] for j in range(2)] for i in range(2)]


def madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]


def mscale(A, s):
    s = Fr(s)
    return [[C(A[i][j].a * s, A[i][j].b * s) for j in range(2)] for i in range(2)]


def dag(A):
    return [[A[j][i].conj() for j in range(2)] for i in range(2)]


I2 = [[C(1), C(0)], [C(0), C(1)]]
SIGMA_I = [  # i sigma_a, a = x, y, z
    [[C(0), C(0, 1)], [C(0, 1), C(0)]],
    [[C(0), C(1)], [C(-1), C(0)]],
    [[C(0, 1), C(0)], [C(0), C(0, -1)]],
]


def pmul(P, R):
    """Product of matrix polynomials [c0, c1, c2] truncated at t^2."""
    out = [[[C(), C()], [C(), C()]] for _ in range(3)]
    for i in range(3):
        for j in range(3 - i):
            out[i + j] = madd(out[i + j], mmul(P[i], R[j]))
    return out


def exp_poly(X, sign):
    """exp(sign * t X) to second order: [1, sign X, X^2 / 2]."""
    return [I2, mscale(X, sign), mscale(mmul(X, X), Fr(1, 2))]


def const(Q):
    return [Q, mscale(Q, 0), mscale(Q, 0)]


def rat_unit_quaternion(u, den=DEN_T):
    """Exactly unit rational quaternion near the float unit quaternion u = (w, x, y, z).

    Inverse stereographic projection from the pole farther from u (t = v / (1 + w), or v / (1 - w) when w < -1/2), so
    that the rational coordinates t stay bounded and |q| = 1 holds exactly."""
    sgn = 1 if u[0] >= -0.5 else -1
    tv = [Fr(float(c / (1 + sgn * u[0]))).limit_denominator(den) for c in u[1:]]
    n2 = sum(c * c for c in tv)
    q = [sgn * (1 - n2) / (1 + n2)] + [2 * c / (1 + n2) for c in tv]
    assert sum(c * c for c in q) == 1
    return q


def quat_to_matrix(q):
    w, x, y, z = q
    return [[C(w, z), C(y, x)], [C(-y, x), C(w, -z)]]      # w + i(x sigma_x + y sigma_y + z sigma_z) up to convention


def matrix_to_quat_float(M):
    a, b = M[0][0], M[0][1]
    return np.array([a.real, b.imag, b.real, a.imag])


def tr_re(M):
    return M[0][0].a + M[1][1].a


def exact_hessian_form(T, Qr, X):
    """H(v, v) = sum_p d^2/dt^2 Re Tr Q_p(t) at t = 0, exactly, for Q_p = Q_1 Q_2 Q_3^* Q_4^*, Q_e(t) = exp(t X_e) Q_e."""
    tot = Fr(0)
    for p in T.plaqs:
        P = None
        for (e, fwd) in p:
            if fwd:
                M = pmul(exp_poly(X[e], 1), const(Qr[e]))
            else:                                                 # (exp(tX) Q)^* = Q^* exp(-tX)
                M = pmul(const(dag(Qr[e])), exp_poly(X[e], -1))
            P = M if P is None else pmul(P, M)
        tot += 2 * tr_re(P[2])
    return tot


def selftest(T, trials=3):
    """The exact evaluator against lib/fasthess.py on random fields and random vectors on all links (fixed ones too)."""
    import fasthess as fh
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(trials):
        qf = rng.normal(size=(T.nE, 4))
        qf /= np.linalg.norm(qf, axis=1)[:, None]
        Qr = [quat_to_matrix(rat_unit_quaternion(u)) for u in qf]
        Qc = np.array([[[complex(float(M[i][j].a), float(M[i][j].b)) for j in range(2)] for i in range(2)] for M in Qr])
        c = rng.normal(size=(T.nE, 3))
        u = [[Fr(float(c[e, a] / np.sqrt(2))).limit_denominator(10 ** 9) for a in range(3)] for e in range(T.nE)]
        X = [madd(madd(mscale(SIGMA_I[0], u[e][0]), mscale(SIGMA_I[1], u[e][1])), mscale(SIGMA_I[2], u[e][2]))
             for e in range(T.nE)]
        cu = np.array([[float(x) * np.sqrt(2) for x in row] for row in u])
        exact = float(exact_hessian_form(T, Qr, X) / (2 * sum(x * x for row in u for x in row)))
        worst = max(worst, abs(exact - fh.rayleigh(T, Qc, cu)))
    assert worst < 1e-9, f"exact evaluator disagrees with lib/fasthess.py: {worst}"
    print(f"self-test: exact evaluator agrees with lib/fasthess.py on random fields and vectors to {worst:.1e}")


def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bestQ_L2_s2_0.npy")
    T = t.Torus(2, d=4, pattern=f4.PAT4, P=2)
    selftest(T)
    Qf = np.load(fn)
    # float unit quaternions of the saved SU(2) matrices, in the convention quat_to_matrix inverts
    qf = np.array([matrix_to_quat_float(q) for q in Qf])
    Qr = []
    for u in qf:
        Qr.append(quat_to_matrix(rat_unit_quaternion(u)))
    err = max(abs(np.array([[complex(float(Qr[e][i][j].a), float(Qr[e][i][j].b)) for j in range(2)] for i in range(2)])
                  - Qf[e]).max() for e in range(T.nE))
    print(f"rational field: 64 exactly unitary matrices, max entry distance from the saved field {err:.2e}")

    # test vector from floating point, then rounded
    Qc = np.array([[[complex(float(Qr[e][i][j].a), float(Qr[e][i][j].b)) for j in range(2)] for i in range(2)]
                   for e in range(T.nE)])
    Hf = t.free_hessian(T, Qc)
    w, V = np.linalg.eigh(Hf)
    lam_float = float(w[-1])
    c = V[:, -1]                                                  # coordinates c_{e,a} in the orthonormal basis B_a
    u = {}                                                        # X_e = sum u_a (i sigma_a), u = c / sqrt 2, rounded
    for k, e in enumerate(T.free):
        u[e] = [Fr(float(c[3 * k + a] / np.sqrt(2))).limit_denominator(DEN_V) for a in range(3)]
    X = {e: ([[C(0), C(0)], [C(0), C(0)]] if e not in u else
             madd(madd(mscale(SIGMA_I[0], u[e][0]), mscale(SIGMA_I[1], u[e][1])), mscale(SIGMA_I[2], u[e][2])))
         for e in range(T.nE)}
    norm2 = 2 * sum(a * a for e in u for a in u[e])               # |v|^2 = sum_e Tr(X_e X_e^*) = 2 sum u_a^2

    tot = exact_hessian_form(T, Qr, X)
    rq = tot / norm2
    print(f"floating-point lambda_max of the rationalised field: {lam_float:.9f}")
    print(f"exact Rayleigh quotient H(v, v)/|v|^2 = {float(rq):.12f}")
    print(f"  numerator and denominator have {rq.numerator.bit_length()} and {rq.denominator.bit_length()} bits")
    lb = Fr(13970, 1000)
    assert rq > lb, "exact quotient does not exceed 13.97"
    assert abs(float(rq) - lam_float) < 1e-6, "exact and floating-point values disagree"
    floor6 = Fr((rq.numerator * 10 ** 6) // rq.denominator, 10 ** 6)       # exact rounding down to six decimals
    assert floor6 <= rq
    print(f"CERTIFIED (exact rational arithmetic): h_F >= {float(floor6):.6f}")


if __name__ == "__main__":
    main()
