# Independent checks of the Hessian-bound package v1.1 (weights (B) unless stated).

1. Certificate check: check_A.log, check_B.log, check_PAPER.log are the outputs of your
   src/check_certificate.py on data/certificate_{A,B,PAPER}.json. All three: CERTIFICATE VALID.
   SHA256SUMS all match; src/exact.py regenerates data/exact_matrices.json byte for byte.

2. Lemma 2, numerical test without the reduction:
   test.py  - R_pm on u(N)^3 (three active links), matrix built by polarisation, straight from eqs. (1)-(2).
   fast.py  - same matrix built from bilinear terms (cross-checked against test.py to ~2e-15),
              then min eigenvalue over W in U(N): 2000 random W and 80 Nelder-Mead searches
              (4 fixed-link positions x 2 signs x 10 starts).
   Run: python3 fast.py N   (N = 2, 3, 4; needs numpy, scipy; keep test.py next to it)
   Results (out2/3/4.log): min over all searched W = 8.716949e-05 for N = 2, 3, 4,
   exactly the one-angle value of Lemma 2; nothing below it.

The package itself (NOTE v1.1, `hessian_bound_package_v1_1`) is J. M. Lockwood's and is not included here.
