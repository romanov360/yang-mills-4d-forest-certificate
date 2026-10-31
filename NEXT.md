# What comes next: a map of BBL's open problems and what is in reach

The source is `strong-coupling/OPEN.md` in BBL's 4Y-M v3.19.0 release (Zenodo 10.5281/zenodo.23729004).

## Lockwood's proposed objective cannot work as stated

The proposal was to make "these positivity estimates uniform under lattice refinement as a → 0". The Hessian bound
does not depend on the spacing, and it only applies for |β| < 1/(2h_F), which is a fixed strong-coupling window. The
continuum limit requires β → ∞. BBL's own no-go theorem (`tex/main.tex`) also shows that a gap bounded below in
lattice units, uniformly in the spacing, forces every renormalised two-point function to vanish. So no estimate of
this kind can be the continuum step. The continuum step has to scale with a; this is their problems 3 and 4.

## Where strong coupling stands

| target | best proved (BBL) | what we could add | value |
|---|---|---|---|
| h_F, every SU(N) | ≤ 14.420001 (ours), ≥ 13.974 (numerical) | a certificate below 14.38 needs blocks of several plaquettes. Their holonomies do not commute, so the reduction to U(2) phases fails and the certificate becomes non-abelian | small: β improves by at most 3% |
| SU(2) | β_W ≤ 0.345 (plaquette blocks) | certify Remark 16 (keeping the middle link), which is about 0.357 in floating point | small: 3.5% |
| SU(3) | β_W ≤ 0.76 (blocks) | the same idea, which is about 0.78 in floating point | small |
| SU(2), 0.36 to 2.2 | no method | certify a finite-box (gauge Dobrushin–Shlosman, their Theorem G) at any coupling | large; this is the input that problem 2 needs |

The curvature route is essentially used up. The next real step at fixed spacing is a certified finite-box
criterion, and no box has been certified at any coupling.

## The bridge to weak coupling (their problem 2)

Bałaban's renormalisation group brings the effective action to an intermediate coupling. A finite-box criterion
that can be verified there would close the loop. The first milestone would be to certify GDS(V, ε) for SU(2) on
the smallest box, at a coupling where the result is already known (β_W ≈ 0.3), as a proof of concept, and then to
push the coupling up.

The difficulty is that κ_V(y, b) is a supremum over boundary holonomies of the disagreement probabilities of a
coupling. That is an integral over SU(2)^|V| inside a supremum, so a certified computation needs rigorous
quadrature and a cover of the boundary data.

## Recorded for later: a box coupling that integrates out the links near y

The Monte Carlo probe in `box/` shows that the naive coupling cannot certify a box. That coupling is link-by-link
heat bath with a maximal coupling at each update. It gives r < 1 only for β_W below about 0.2 (d = 4, L = 3), which
is under the 0.345 already proved, so it only reproduces single-site Dobrushin ranges. In d = 2 it fails at β = 2,
where BBL's Prop. prop:2d succeeds.

The plan, modelled on Prop. prop:2d:

1. Integrate out exactly the links of V that share a plaquette with the boundary link y, by character expansion in
   SU(2) (Bessel-function coefficients). The result is the marginal law of the remaining links, given the boundary.
2. Couple that marginal maximally between ω and ω' (they differ only at y). The disagreement probability is then a
   total-variation distance of explicit one- or few-link densities, rather than a product of per-update
   disagreements.
3. Bound the supremum over boundary holonomies with interval arithmetic on the class functions (they depend only on
   traces), so that the cover of the boundary data is low-dimensional.
4. First target: GDS(V, ε) for SU(2), d = 4, on the L = 2 or 3 box at β_W = 0.3, where the answer is known. Then
   push past 0.345.

## Update: scoping done (box/GDS4D_SCOPING.md)

Two results:

- **d = 2: certified explicit boxes.** `box/gds2d_certified.py` evaluates BBL's Prop. 2d coupling in ball
  arithmetic. The smallest certified squares are L = 10, 26, 36, 55, 85, 139, 209 at β_W = 0.5, 1, 2, 3, 4, 6, 8,
  against 32–512 from their existence choice.
- **d = 4: the plan above does not scale.** The exact counting in `box/gds4d_counting.py` shows that
  n_min/|∂V_ν| grows only like L/8. A neighbourhood coupling therefore needs boxes of 10⁴–10⁹ links, with a
  uniform η of 10⁻⁵–10⁻¹¹. That is not a finite computation; a new coupling idea is needed. See the scoping
  note.

