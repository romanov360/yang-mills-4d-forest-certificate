#!/usr/bin/env python3
"""Independent floating-point cross-check (mpmath, Bessel closed forms) of the Arb constants of cert_block_su2.py.
Z0(c) = 2 I1(r)/r, r = |c|; tangential eigenvalue of Hess g = A(r)/r, radial = 1 - A^2 - 3A/r, A = I2/I1 = |grad g|.
Not part of the proof."""
import mpmath as mp
mp.mp.dps = 30
A = lambda r: mp.besseli(2, r) / mp.besseli(1, r)
def eig(r):
    if r == 0: return (mp.mpf(1)/4, mp.mpf(1)/4)
    a = A(r); return (a / r, 1 - a*a - 3*a/r)
phi = lambda t: mp.log(2*mp.besseli(1, mp.sqrt(t))/mp.sqrt(t))
def d3(r):
    t = mp.mpf(r)**2
    p2, p3 = mp.diff(phi, t, 2), mp.diff(phi, t, 3)
    a, g = 12*r*p2, 8*r**3*p3
    return max(abs(a+g), 2*abs(a)/3)
bW = mp.mpf('0.351'); b = bW/4; Rb, R20 = 24*b, 20*b
rs = [Rb*k/2000 for k in range(2001)]
lam = min(min(eig(r)) for r in rs); tau = max(max(eig(r)) for r in rs)
M = max(d3(r) for r in rs[1:])
print(f"lambda_* ~ {mp.nstr(lam,10)}  tau ~ {mp.nstr(tau,10)}  M ~ {mp.nstr(M,10)}  A(20beta) ~ {mp.nstr(A(R20),10)}")
for bw in ['0.3333333333333333','0.345','0.35','0.351']:
    b = mp.mpf(bw)/4; w = 8*b*b*(tau-lam); ew = mp.e**w
    s = mp.mpf(1)/4 + (ew-1)*ew*mp.sqrt(tau)/mp.sqrt(3)
    C = 4*b*s*A(R20) + 320*b**3*M*ew/3
    K = (1+b)*s/(1-C); print(bw, mp.nstr(40*b*K, 10))
