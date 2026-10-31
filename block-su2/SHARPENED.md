# SU(2) plaquette blocks: β_W ≤ 0.345 → β_W ≤ 0.351, certified

This note improves Theorem 3 of *Mass Gap of Four-Dimensional Lattice Yang–Mills Theory for Every Compact Simple Group*
(Bhattacharjee, Bhattacharya, Lockwood, Romanov; v. 30 Oct 2026; "the paper" below). All numbers come from ball
arithmetic (Arb via python-flint), and the final arithmetic is checked by the Lean kernel. Section, lemma and equation
numbers refer to the paper.

## Result

**Theorem 3′.** Let G = SU(2) and 0 ≤ β_W ≤ 0.351. The block number α′(β_W) of (S) below is increasing, with

| β_W | α (paper, Thm 3) | α′ (this note) | gap m ≥ −½ log α (paper) | gap m ≥ −½ log α′ (this note) |
|---|---|---|---|---|
| 1/3 | ≤ 0.9561 | ≤ 0.94280 | 0.0224 | **0.0294** |
| 0.345 | ≤ 0.99729 | ≤ 0.98046 | 0.00137 | **0.00987** (×7.2) |
| 0.350 | — (> 1) | ≤ 0.99669 | — | **0.00166** |
| 0.351 | — (> 1) | ≤ 0.99997 | — | **1.5·10⁻⁵** |

So the conclusions of Theorem 1 hold on the larger range: uniqueness, decay, and a transfer-matrix gap
m ≥ −½ log α′(β_W) > 0. For −0.351 ≤ β_W < 0 everything except reflection positivity and the gap holds.
The certificate fails at 0.3511, which is where this method ends.

The paper's Hessian certificate gave a 0.55% gain (29/2 → 14.420001). This note gives a 1.7% gain in the SU(2) range
and a 7× larger gap at the paper's old endpoint 0.345. Remark 18 of the paper estimates, in floating point and without
certification, that a sharper treatment of the middle link could reach about 0.357. A first-order computation (see the
last section) puts the ceiling of any plaquette-block Dobrushin bound near 0.366.

**Scope.** This is a fixed-lattice-spacing statement in the strong-coupling regime. Like every Dobrushin-type result, it
says nothing about the continuum limit; see `../FRONTIER.md`.

## The two changes

Everything else in Sec. V (Lemmas 12, 13 and 15, Proposition 17, the middle-link factor 1 + |β|, and the coupling
argument d₁ ≤ sε + Cd₃, d₃ ≤ Cd₁) is used unchanged. Only the constants (3) change, to

```
(S)   w  = 8β²(τ − λ*)                                   (unchanged)
      s′ = 1/4 + (e^w − 1) e^w √τ / √3                     (paper: no 1/√3)
      C′ = 4|β| s′ A₂₀ + 20|β| · 16β² M e^w / 3            (paper: 20|β|(4|β|τ s + 16β² M))
      K′ = (1 + |β|) s′ / (1 − C′),     α′ = 40|β| K′,
```

The constants are:

- λ*, τ and M are, as in the paper, bounds for the Hessian eigenvalues and for ‖D³g‖ on |c| ≤ 24|β|.
- A₂₀ ≥ max_{|c| ≤ 20|β|} |∇g(c)|. This equals max_{r ≤ 20|β|} r·2φ′(r²) = I₂(r)/I₁(r).

### Change 1: Poincaré in place of "sd ≤ 1" (gives s′)

In Lemma 14(i), Lemma 13 is applied with ν = ν_c, a von Mises–Fisher law, and the paper bounds sd_ν(f) ≤ 1 for
1-Lipschitz f. By Lemma 154 the vMF law satisfies Var_{ν_c}(f) ≤ ⅓∫|∇f|² dν_c for every c. A function that is
1-Lipschitz for the chordal metric is also 1-Lipschitz for the larger geodesic metric, so |∇f| ≤ 1 a.e. Hence
sd_ν(f) ≤ 1/√3. Lemma 13 then gives

 Cov_κ(f, ⟨x,h⟩) ≤ ¼|h| + (e^w − 1)e^w (1/√3) √τ |h| = s′|h|.  ∎

### Change 2: interpolate the exponents, not the field (gives C′)

The paper's Lemma 14(ii) moves c₀ along the chord from c₀ = b₂y₃ to c₀′ = b₂y₃′ and linearises at every point of the
chord. The chord passes inside the sphere |c| = |b₂|, where the Hessian is larger. This forces the factor
20|β|·τ with τ = ¼, the value at c = 0. We interpolate the measures instead.

**Lemma 14′(ii).** Let β ≥ 0, |a| ≤ 20β, and let c₀, c₀′ be points with |c₀| = |c₀′| = ρ ≤ 20β. Put
κ_c(dx) ∝ exp(⟨x,a⟩ + g(c + 4βx̄))dx. Then

 W₁(κ_{c₀}, κ_{c₀′}) ≤ (4β s′ · 2φ′(ρ²) + 16β² M e^w / 3) |c₀ − c₀′|.

*Proof.* Let κ_t ∝ exp(⟨x,a⟩ + (1−t)g(c₀+4βx̄) + t g(c₀′+4βx̄)), for t ∈ [0,1], and let f be 1-Lipschitz. Then
(d/dt)κ_t(f) = Cov_{κ_t}(f, D) with D(x) = g(c₀′+4βx̄) − g(c₀+4βx̄).

1. *κ_t is a tilted vMF law.* Taylor's formula at c ∈ {c₀, c₀′} gives
   g(c+4βx̄) = g(c) + 4β⟨∇g(c), x̄⟩ + R_c(x), with R_c(x) ∈ [8β²λ*, 8β²τ]. This holds because the intermediate point
   lies in |ξ| ≤ 24β. So κ_t = e^{R_t}ν/ν(e^{R_t}), where:
   - ν = ν_{c_t} with c_t = a + 4β·conj((1−t)∇g(c₀) + t∇g(c₀′));
   - |c_t| ≤ 24β, since |∇g| ≤ 1;
   - R_t = (1−t)R_{c₀} + tR_{c₀′} has oscillation ≤ w.

   Hence Change 1 applies to κ_t: Cov_{κ_t}(f, ⟨x,h⟩) ≤ s′|h|. The density of κ_t with respect to ν is at most e^w.
2. *Split D.* Up to a constant, D = 4β⟨x̄, ∇g(c₀′) − ∇g(c₀)⟩ + N(x), with N = R_{c₀′} − R_{c₀}.
3. *Linear part.* Since ∇g(c) = 2φ′(|c|²)c and |c₀| = |c₀′| = ρ, we have |∇g(c₀′) − ∇g(c₀)| = 2φ′(ρ²)|c₀′ − c₀|.
   Its covariance with f is at most 4βs′·2φ′(ρ²)|c₀′ − c₀|.
4. *Nonlinear part.* As a function of u = x̄ in the unit ball,
   ∇N(u) = 16β² ∫₀¹ [Hess g(c₀′+4βσu) − Hess g(c₀+4βσu)] u dσ.
   Both arguments lie in the convex ball |c| ≤ 24β, where ‖D³g‖ ≤ M. So |∇N| ≤ 16β²M|c₀′ − c₀|, and N is
   L-Lipschitz on S³ with L = 16β²M|c₀′ − c₀| (x ↦ x̄ is an isometry).
5. *Bound the nonlinear covariance.* For an L-Lipschitz function u,
   Var_{κ_t}(u) ≤ κ_t((u − νu)²) ≤ e^w Var_ν(u) ≤ e^w L²/3, by Lemma 154 again. So
   Cov_{κ_t}(f, N) ≤ sd(f)·sd(N) ≤ e^w L/3.
6. Integrate over t ∈ [0,1] and apply Kantorovich–Rubinstein duality.  ∎

*In Lemma 15.* For κ₁(·|y₃) we have c₀ = b₂y₃ and c₀′ = b₂y₃′. Both have norm ρ = |b₂| ≤ 20|β|, and
|c₀ − c₀′| = |b₂||y₃ − y₃′|. The constant is therefore

 ρ·2φ′(ρ²)·4|β|s′ + 16β²M e^w ρ/3 ≤ 4|β|s′A₂₀ + 20|β|·16β²M e^w/3 = C′.

The same holds for κ₃(·|y₁), with c₀ = y₁b₂. The rest of the proof of Lemma 15 is unchanged and gives
W₁(µ_b, µ_b′) ≤ K′|b_i − b_i′|.

Lemma 13 needs nothing new. Lemma 14′(i) is Change 1, and it also holds for every κ_t by step 1.

*Monotonicity.* Fix the constants λ*, τ, M on the ball |c| ≤ 24·(0.351/4) and A₂₀ on |c| ≤ 20·(0.351/4). For
smaller β these constants are still valid, because the balls shrink, and every quantity in (S) increases with |β|. One
evaluation at the endpoint therefore covers the whole range [0, 0.351].

## The constants (Arb, 200 000 subintervals of the radius, 48 series terms plus a proved tail bound, 128-bit)

```
ball r ≤ 2.106 (= 24·0.351/4):  λ* ≥ 0.1563819387   τ ≤ 0.2500000000   M ≤ 0.06151713657
r ≤ 1.755 (= 20·0.351/4):       A₂₀ = max r·2φ′(r²) ≤ 0.3914427967
φ″ ≤ 0 on the whole ball (so the tangential eigenvalue 2φ′ is the largest one)
β_W = 0.351: w ≤ 0.0057669  s′ ≤ 0.2516792  C′ ≤ 0.0390392  K′ ≤ 0.2848858  α′ ≤ 0.9999491
```

The paper's Lemma 16 ball (r ≤ 2.07) gives C′ ≤ 0.0377 at 0.345, against the paper's C ≤ 0.0502; the linear part of C
drops by about 10% and the cubic part by a factor of 3. The `--mode paper` option of the same program reproduces every
number the paper quotes in the proof of Theorem 3:

| quantity | paper | `--mode paper` |
|---|---|---|
| w | 0.0054459 | 0.0054444 |
| s | 0.252746 | 0.2527445 |
| C | 0.050243 | 0.0502342 |
| K | 0.289069 | 0.2890647 |
| α(0.345) | 0.99729 | 0.9972730 |

As an independent floating-point check, `validate_bessel.py` uses the closed forms Z₀ = 2I₁(r)/r and |∇g| = I₂/I₁,
and agrees with the Arb values to about 6 digits.

## Files

| file | role |
|---|---|
| `cert_block_su2.py` | Arb enclosures of λ*, τ, M, A₂₀ and of (S); `--mode paper` reproduces Theorem 3; `--lean` writes `lean/BlockData.lean` |
| `lean/BlockCheck.lean` | Lean 4 core (no Mathlib) kernel check of the chain (S) from the rounded constants: α′(0.351) ≤ 0.99997, α′(0.345) ≤ 0.98046, α′(1/3) ≤ 0.94280, and that the check fails at 0.352 (it is not vacuous) |
| `lean/check.sh` | `sh lean/check.sh` with Lean `v4.19.0` on the path |
| `validate_bessel.py` | independent floating-point cross-check; not part of the proof |
| `logs/` | outputs of the runs above |

Reproduce:

```
pip install python-flint mpmath
python3 cert_block_su2.py --mode paper --ball 2.07 --n 200000 0.333333333333333333 0.345
python3 cert_block_su2.py --mode sharp --n 200000 --lean lean/BlockData.lean 0.333333333333333333 0.345 0.35 0.351
sh lean/check.sh
```

The Lean check covers the finite arithmetic: the inputs are outward-rounded Arb bounds, and e^w ≤ 1/(1−w),
√τ ≤ q and 1/√3 ≤ r are checked as rational inequalities. The analytic lemmas above, and Lemma 154 of the paper, are
pen-and-paper proofs and are not formalised.

## Where the plaquette block ends

Take b = 0 and f = ⟨y₁,e⟩ + Re(y₂y₃e). This f is 1-Lipschitz for d, because it is 1-Lipschitz in each coordinate. A
first-order expansion in β gives Cov_{µ_b}(f, ⟨y₁,e⟩) = ¼ + β/4 + O(β²). So the true block sensitivity is
K ≥ (1+β)/4 + O(β²), and the factor 1 + |β| for the middle link cannot be removed. Then α ≥ 10|β|(1+|β|) + O(β³),
which reaches 1 near β_W ≈ 0.366.

This note uses about 40% of the room between the paper's 0.345 and Remark 18's floating-point 0.357. What remains is
mostly the two-site constant C′ ≈ 0.039. Remark 18's floating-point value, K ≈ 0.2783 at β_W = 0.34 against
(1+β)/4 = 0.2713 and our K′ ≤ 0.2836, suggests that the true y₁–y₃ feedback is about half of C′. Going past about 0.36 needs
larger blocks, for example the free links of a line-field cube, as Remark 18 already says.
