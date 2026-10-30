# A proved bound h_F ≤ 14.420001 for the 4D forest-gauge Hessian

In forest gauge, Bhattacharjee, Bhattacharya and Lockwood (BBL) bound the gauge-fixed Wilson action Hessian by 29/2 for
every SU(N). By Bakry–Émery this gives a log-Sobolev inequality, decay of correlations and a mass gap for |β| < 1/29,
where β_W = N²β. This repository proves the sharper bound

    h_F = sup over fields of |Hess S_0(v, v)| / |v|²  ≤  14420001/10⁶ = 14.420001

on the d = 4 period-2 forest. So the range becomes |β| < 1/28.840002.

This is a strong-coupling result at a fixed lattice spacing. It says nothing about the continuum limit.

## What is new and what is not

| | value | source | status |
|---|---|---|---|
| identity field | 9 + √17 = 13.1231 | BBL; reproduced in `bloch4d.py` | exact |
| worst field found | 13.97401 | **BBL**; reproduced independently in `climb4d.py`, `check_field.py` | floating point |
| certificate of this form | ≈ 14.38 | **BBL** (14.3839, SDP); reproduced here (infeasible at 14.35, feasible at 14.42) | numerical |
| **proved upper bound** | **14.420001** | **this repository** | **proved, computer-assisted** |
| previous proved upper bound | 29/2 = 14.5 | BBL | proved |

The new part is the proved bound below 29/2. The lower bound, and the observation that the identity field is not the
worst case in d = 4, are BBL's. We found them independently and our numbers agree with theirs.

| range for β_W in d = 4 | SU(2) | SU(3) |
|---|---|---|
| BBL, 1/29 | 0.1379 | 0.3103 |
| this bound, 1/28.84 | 0.1387 | 0.3121 |
| ceiling of the method (h_F ≥ 13.974) | 0.1431 | 0.3220 |

Reference: D. Bhattacharjee, U. Bhattacharya and J. M. Lockwood, *Yang–Mills mass gap in 4D: lattice SU(N) and every
compact simple group*, Zenodo, 2026, doi:10.5281/zenodo.23729004. Code at github.com/creelie/4Y-M.

## The proof

This is the site-plus-plaquette certificate of the 3D paper (romanov360/yang-mills-3d-forest-gauge, Lemma 5.1),
applied to the d = 4 forest. The period cell has 16 sites, 64 links (16 fixed, 48 free) and 96 plaquettes, and
every plaquette contains exactly one fixed link.

1. **Finite part (exact rational arithmetic).** The certificate consists of 16 site matrices G_y (8×8) and
   nonnegative plaquette weights, all rational with denominator 10⁶. We check that every G_y is positive definite
   using exact LDLᵀ pivots, and that the maximum over the 48 free links of the per-link totals is exactly
   14420001/10⁶.
2. **All phases (exact sum of squares, Lean kernel).** For each of the 96 plaquettes, a 12×12 matrix M_p(φ) has
   to be positive definite for every pair of phases. In z = (1, cos φ1, sin φ1, cos φ2, sin φ2), M_p splits into two
   6×6 blocks, and each block satisfies
   z ⊗ x ↦ (z ⊗ x)ᵀ P (z ⊗ x) = 2²⁴·4·10⁶·xᵀM_block(φ)x on the torus, for an explicit integer 30×30 matrix P > 0.
   Lean recomputes M_p from its definition and the certificate data, checks this polynomial identity modulo
   cos² = 1 − sin², and checks that every P is positive definite by exact fraction-free elimination
   (`lean/Sos4.lean`, theorems `sos_gram` and `sos_pd`). The Gram matrices (`lean/SosData.lean`) come from an SDP
   and are rounded exactly by `lean/gen_sosdata.py`; nothing about how they were found is trusted.

   The earlier, independent proof of this step is still in the repository: an outward-rounded interval quadtree
   with Arb (python-flint), 68,968,000 squares, maximum depth 6 (`logs/interval_h14.42_*.log`).

Everything in the finite part, including all phases, is now a Lean 4 kernel theorem (`lean/`, PROOF.md section 6),
along with the forest and plaquette combinatorics on all of Z⁴ and the word identities of the plaquette rewriting and
the corners. Three steps remain written proofs, checked exactly in sympy but not in Lean: the lemma, the general-N
splitting to U(2), and the identification of the Hessian with Σ f_p. They need real matrices and calculus, so
Mathlib.

`PROOF.md` gives the written proof that these checks bound the Wilson Hessian: the plaquette rewriting, the site
identity with 8 ends per site, the lemma, and the reduction from U(N) to U(2) phases.

## Continuum side: the slice ratio from a one-step bound

`continuum/SLICE.md` is a separate note on BBL's continuum step. One of BBL's open inputs is the slice ratio
liminf D_k(r')/D_k(r) > 0, which gives nontriviality of the flowed continuum limit. The note shows that the slice
ratio follows from a bound on the first-step effective energy, −log(D_k(a_k)/D_k(0))/a_k ≤ E. This is a one
lattice-step, local-in-time condition. The proof is Jensen's inequality on the spectral measure, checked in Lean 4
+ Mathlib (`lean/mathlib/YMMathlib/SliceJensen.lean`). The note also has exact checks on the free field and on 2D
SU(2), and a Monte Carlo measurement in 4D SU(2), which is evidence and not a proof.

## Reproduce

Requirements: Python 3, numpy, scipy, sympy, cvxpy (with Clarabel), python-flint; Lean 4.19.0 for `lean/`.

    # finite part, exact (seconds)
    python3 cert_interval4d.py data/cert4d_h14.42.npy finite
    # all phases, interval arithmetic (independent of the Lean SOS check): about 3 h on one core, or split into ranges across cores (about 44 min on 4)
    python3 cert_interval4d.py data/cert4d_h14.42.npy centre 64 0 24     # ... 24 48, 48 72, 72 96
    # lower bounds
    python3 exact_check4d.py              # exact checks of the identities in PROOF.md (seconds)
    lean/check.sh                         # Lean 4.19.0 kernel checks, core only, no Mathlib (about 15 min on 4 cores)
    python3 validate4d.py                 # floating-point checks that the certificate is the Wilson Hessian
    python3 bloch4d.py                    # identity field, 9 + √17
    python3 check_field.py                # saved field, 13.974014
    python3 climb4d.py 2 2 <steps>        # the search that found it
    # regenerate a certificate (SDP, then exact rounding)
    python3 certify4d.py search 14.42

## Files

- `PROOF.md`: the analytic identification of the certificate with the Wilson Hessian in d = 4.
- `forest4d.py`: the d = 4 period-2 forest and its period cell.
- `certificate3d.py`: the u(2) quadratic forms, unchanged from the 3D repository (they do not depend on d).
- `certify4d.py`: the cutting-plane SDP and the exact rational rounding.
- `cert_interval4d.py`: the exact finite check and the interval proof.
- `bloch4d.py`, `climb4d.py`, `check_field.py`: the lower bounds.
- `exact_check4d.py`: exact (symbolic and combinatorial) checks of every identity in `PROOF.md`.
- `lean/`: Lean 4 kernel proofs. `Forest4.lean` covers the forest, one fixed link per plaquette and corners on Z⁴;
  `Words4.lean` the word identities; `Cert4.lean` the finite certificate checks, run on `CertData.lean`, which
  `gen_certdata.py` writes; `Sos4.lean` the all-phase sum-of-squares check, run on `SosData.lean`, which
  `gen_sosdata.py` writes, with one theorem per plaquette (`SosG0..7`) and per Gram matrix (`SosP0..7`),
  assembled in `SosAll.lean`.
  Run `check.sh`; output in `logs/lean_check.out`.
- `validate4d.py`: numerical checks, not part of the proof, of the identification used by Lemma lem:cert in d = 4. It checks the plaquette rewriting against fasthess (SU(2)) and finite differences (SU(3)), the site identity, the slack decomposition, the spectrum, and the U(3)/U(4) plaquette inequality. Output in `logs/validate4d.out`.
- `box/`: a Monte Carlo probe (not a proof) of BBL's finite-box criterion GDS for SU(2).
- `lib/`: `torus3d.py`, `fasthess.py` and `hessian.py` from the 3D repository. The only change is the import path.
- `data/cert4d_h14.42.npy`: the certificate, with exact rationals stored as strings.
- `data/bestQ_L2_s2_0.npy`: the extremal field.
- `logs/`: the SDP searches (`s_*.log`), the hill-climbs (`climb_s*.log`) and the interval proof
  (`interval_h14.42_*.log`).

## License

The code is under the MIT License (see `LICENSE`). The text is under CC BY 4.0.
