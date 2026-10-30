# The slice ratio from a one-step bound

This note concerns the continuum side of the problem. It does not touch the strong-coupling certificate in the
rest of this repository.

**Status.** One lemma is proved (Proposition S, checked in Lean 4 + Mathlib). Two exact model checks are
floating-point quadrature. One Monte Carlo study is evidence, not proof. Nothing here proves the slice ratio, the
existence of the continuum limit, or a mass gap.

## 1. Where the continuum problem stands (30 October 2026)

- **OpenAI (announced 26 October 2026)** claims a construction of 4D Yang–Mills/QCD. It is an existence claim with
  no mass gap. The proof and its Lean formalization are held by a working group chaired by D. Gross at KITP and
  have not been released. Nothing in it can be checked yet.
- **BBL + Romanov, 4Y-M v4.20.0** (Zenodo 10.5281/zenodo.23822000, 30 October 2026) proves lattice gaps at strong
  coupling for every compact simple group. Its continuum step is conditional. The forest paper (`strong-coupling/forest`)
  proves three implications (I1)–(I3) and leaves their inputs open:
  1. the cube condition D(n, α) along the tuning;
  2. the **slice ratio** `liminf_k D_k(r')/D_k(r) > 0` for one pair 0 < r < r' with r ≤ 2t₀
     (Proposition "sliceratio"). It gives the nontriviality criterion of Theorem "flowlimit"(iv);
  3. the tree bounds (T1)–(T3).
- Large-N lattice gap: Cao–Chatterjee et al., arXiv:2510.22788. This is a lattice result, not a continuum one.

This note works on input 2.

## 2. Proposition S

Notation as in BBL, Proposition "sliceratio". 𝕋_k is the transfer matrix (0 ≤ 𝕋_k ≤ 1, Lüscher). U_k is the
centred, spatially smeared slice field at time 0, after the spatial gradient flow at physical time τ.
u_k = Ψ_{U_k}, and D_k(r) = ⟨u_k, 𝕋_k^{r/a_k} u_k⟩.

**Proposition S.** Let E_k = −a_k⁻¹ log(D_k(a_k)/D_k(0)). This is the effective energy of the slice correlator at
zero separation. By log-convexity of D_k, it is the largest effective energy at any separation. Suppose that

    (S)   E_k ≤ E   for some E < ∞ and all large k.

Then for every such k and every r ≥ 0 in a_k ℤ:

1. D_k(r) ≥ e^{−Er} D_k(0). Hence D_k(r')/D_k(r) ≥ e^{−Er'} for all 0 ≤ r < r', so the slice ratio holds for
   every pair, including the pairs with r ≤ 2t₀.
2. For every packet g ⊗ h supported in times [t₀, t₁], C_k(σ)/C_k(0) ≥ e^{−E(2t₁+σ)}. By Theorem
   "flowlimit"(iv), every subsequential continuum limit of the renormalised flowed observables is nontrivial.

The linear form works the same way. If ⟨u_k, (1 − 𝕋_k)u_k⟩ ≤ κa_k⟨u_k, u_k⟩ with κa_k ≤ 1/2, the conclusions
hold with E = 2κ. As a_k → 0, both E_k and the linear constant κ_k = (1 − D_k(a_k)/D_k(0))/a_k tend to the mean
energy ∫E dμ of the slice vector, whenever that mean energy is finite.

*Proof.* Let ν be the spectral measure of 𝕋_k at u_k. It lives on [0,1] and has mass D_k(0). Let μ = ν/D_k(0).
Then D_k(na_k)/D_k(0) = ∫λⁿ dμ. Jensen's inequality for the convex function λⁿ gives
∫λⁿ dμ ≥ (∫λ dμ)ⁿ = e^{−nE_ka_k}. That is item 1, because D_k(r) ≤ D_k(0). For the linear form, also use
1 − x ≥ e^{−2x} on [0, 1/2]. For item 2, apply the left inequality of Proposition "sliceratio"(i):
C_k(σ)/C_k(0) ≥ D_k(2a_kn₁+σ)/D_k(2a_kn₀). Use D_k(2a_kn₀) ≤ D_k(0), 2a_kn₁ ≤ 2t₁, and item 1. ∎

The Lean file is `lean/mathlib/YMMathlib/SliceJensen.lean`, with theorems `jensen_pow`, `slice_step_log` (the log
form), `slice_step` and `slice_ratio` (the linear form). It is stated for a probability measure on ℝ carried by [0,1]. It depends only on propext,
Classical.choice and Quot.sound. Build it with `cd lean/mathlib && lake build YMMathlib.SliceJensen`
(Lean 4.19.0, Mathlib v4.19.0).

**What is new here.** BBL apply the same Jensen step to the time-smeared packet Ψ_k (Proposition "flowshift"(ii)).
They note that it would follow from the slice vector u_k "if u stays comparable to the packet", and that this
comparison is not proved. Applying Jensen to u_k directly and then using their Proposition "sliceratio"(i) removes
the need for the comparison. The step is short, but it changes the shape of the open input:

| | BBL open input | after Proposition S (sufficient, not necessary) |
|---|---|---|
| statement | D_k(r')/D_k(r) ≥ c at two fixed physical separations | −log(D_k(a_k)/D_k(0)) ≤ E a_k, one lattice step |
| range | correlations at physical distance, long-distance noise | adjacent slices only |
| meaning | the slice field keeps weight at finite energy | the slice field has bounded first-step effective energy, which tends to the mean energy |

(S) is strictly stronger. A spectral measure that puts weight √a_k at energy 1/a_k satisfies the ratio but not (S),
because its E_k grows like a_k^{−1/2}.
For the flowed field, however, finite mean energy is the expected behaviour (section 4).

**Lattice form.** For a real slice field at time 0, which the site reflection fixes,
⟨u_k, u_k⟩ = ω_k(U_k²) and ⟨u_k, 𝕋_k u_k⟩ = ω_k(U_k τ₁U_k). So

    E_k = −a_k⁻¹ log(ω_k(U_k τ₁U_k)/ω_k(U_k²)),       1 − e^{−a_kE_k} = ω_k((τ₁U_k − U_k)²)/(2ω_k(U_k²)).

So (S) says that the mean-square change of the flowed, smeared slice field over one lattice time step is O(a_k)
relative to its variance. Both sides are local in time, and the left side involves only adjacent slices.

## 3. Exact checks on the two models where BBL verify the slice ratio

`free_slice.py` (floating point quadrature):

- **Massless free field** (BBL Remark "freeslice", h = e^{−|x|²}, t_f = 0.05). E_k is 1.021, 1.088, 1.119,
  1.133, 1.139, 1.142 at a = 2^{−1}…2^{−6}. It converges to the mean energy 1.1447 of the slice vector (the
  continuum integral gives 1.1441; the difference is the error of the 280³-point grid). The script also reproduces
  BBL's D(1)/D(1/2) → 0.6337: the continuum quadrature gives 0.633655 and the grid gives 0.6333. Proposition S
  gives D(1)/D(0) ≥ e^{−E_k} ≈ 0.32. The true value is 0.373.
- **Two-dimensional SU(2)** (BBL Remark "twodslice", f = Re Tr V, g² = ℓ = 1). Here D_k is a single exponential,
  so E_k = E_{F,k} → 3/8 exactly and Proposition S is sharp.

Output: `logs/free_slice.out`.

## 4. Why (S) is plausible, and what proving it would involve (heuristic)

In the Hamiltonian (Kogut–Susskind) limit, U_k is a multiplication operator in the spatial links. It commutes with
the magnetic term, so the double commutator gives an f-sum rule:

    ⟨u, H u⟩ = ½⟨Ω, [U, [H, U]] Ω⟩ = (g₀²/2a) Σ_e ω(|∇_e U|²),

where ∇_e is the Lie derivative in the link e. Then (S) becomes

    g₀² Σ_e ω(|∇_e U_k|²) ≲ 2E a_k ω(U_k²)    (using 1 − 𝕋_k ≈ a_k H_k on low energies).

Both sides are expectations on a **single slice**. The left side is a regularity (Dirichlet-form) bound for the
flowed functional. Power counting with the flow kernel gives it the same order a_k⁹ g₀⁴ as the right side. The
right side needs a **lower** bound on the variance of the flowed smeared field, which is a non-degeneracy
statement. So (S) splits into:

- (S-a) an upper bound on the electric Dirichlet form of the flowed slice functional. This is a flow-regularity
  estimate, deterministic on small-field configurations. The large-field region is where it can fail, because
  the flow is only contracting near smooth fields.
- (S-b) a lower bound ω_k(U_k²) ≳ a_k⁸ g₀⁴ in physical normalisation. The flowed energy density must not
  become degenerate along the tuning.

On the lattice, the f-sum rule holds only to leading order. The exact statement is the lattice form above, with
the jump law of the Doob-transformed transfer kernel in place of the heat kernel. Making (S-a)/(S-b) rigorous is
open. (S-b) in particular is close to the nontriviality question itself. It is still a one-slice, one-point-type
estimate rather than a statement about long-distance correlations.

A caveat: finiteness of composite operators under the 4D gradient flow is Lüscher's theorem. For the
**spatial-only** flow that BBL use, I do not know a proof of the corresponding statement.

## 5. Monte Carlo: SU(2), d = 4, along the tuning (evidence, not proof)

`su2_slice.py` runs heat bath with 3 overrelaxation steps per sweep, using quaternion links. It applies Lüscher's
RK3 spatial Wilson flow on every slice. The slice field is the zero-momentum spatial plaquette energy (h = 1 on
the periodic spatial box). The flow time is τσ = 0.045 (√(8τ) = 0.6/√σ), and the physical box is L·a√σ ≈ 3.
`analyse.py` gives jackknife errors over 20 bins.

Checks: the plaquette agrees with the literature (0.6023 at β = 2.3, 0.6517 at β = 2.5). The flow is exactly
gauge covariant (difference 10⁻¹⁴ under a random gauge transformation). The energy decreases along the flow.
Changing the step from ε = 0.05 to 0.01 changes the result by 10⁻⁴.

Zero-momentum slice field, τσ = 0.045 (`python3 analyse.py data/b2*.npy`; β = 2.5 pools three independent chains):

| β | L = T | configs | a√σ | 1/(a√σ) | E_k/√σ, flowed | E_k/√σ, unflowed | κ_k/√σ (linear form) |
|---|---|---|---|---|---|---|---|
| 2.3 | 8 | 400 | 0.369 | 2.71 | 4.41(26) | 6.0(6) | 2.18(5) |
| 2.4 | 12 | 300 | 0.266 | 3.76 | 4.59(17) | 7.8(6) | 2.65(5) |
| 2.5 | 16 | 162 | 0.188 | 5.32 | 5.75(32) | 13.4(14) | 3.52(11) |

Several flow times in one run (`su2_slice.py multi`, `python3 analyse.py multi data/multi_*.npy`):

| β | configs | τσ = 0.045 | τσ = 0.1 | τσ = 0.2 | unflowed |
|---|---|---|---|---|---|
| 2.3 | 400 | 3.87(17) | 3.68(16) | 4.22(23) | 5.6(4) |

Reading, with caution:

- **Unflowed** E_k grows like 1/a: 6.0 → 7.8 → 13.4, against the 1/a ratios 1 : 1.39 : 1.96. This is the
  ultraviolet weight that the flow is there to remove. It confirms that the observable is sensitive to the
  failure (S) is meant to exclude.
- **Flowed** E_k stays near the scale of the lightest SU(2) glueball (m₀₊₊ ≈ 3.7–4√σ) at the two coarser
  couplings, and does not decrease as the flow time is raised fourfold at β = 2.3. That is the behaviour of a
  slice vector dominated by finite-energy states.
- At β = 2.5 it is higher, 5.75(32) against 4.59(17). That gap is about 3σ, but two independent runs at β = 2.3
  differ by 1.7σ (4.41 against 3.87), so the jackknife errors probably understate the autocorrelation. More
  importantly, τ/a² is only 0.33, 0.64 and 1.27 at τσ = 0.045. The coarse points are not in the regime
  √(8τ) ≫ a, where the flowed field is the continuum one. The data **do not decide** whether E_k converges.
  Deciding it needs a ≲ 0.1/√σ (β ≥ 2.7, L ≥ 32) at τσ ≈ 0.1–0.2. That is a GPU job (section 6).
- The physical-distance ratio D(r)/D(0) at r√σ ≈ 0.74 is 0.08(2), 0.05(2), 0.01(2). It is already at the noise
  level. This is why the one-step form is the measurable one.


## 6. Next steps

1. **Numerics at finer spacing** (β = 2.7, 2.85 at L a√σ ≈ 3, i.e. L = 32–48). This is where a GPU helps: a JAX
   or PyTorch port of `su2_slice.py` on one A100/H100 should run 50–100× faster than these 4 CPU cores. Modal is
   a good fit for that. Measure E_k for two flow times, and its dependence on the box.
2. **Measure both sides of the f-sum rule.** Use the adjoint (backpropagated) flow to compute Σ_e|∇_e U|². This
   separates (S-a) from (S-b) numerically.
3. **Prove (S-a) on small fields.** Use the flow linearisation bounds already in BBL ("torusscales",
   "smallscales"), plus the SU(2) plaquette bounds that are uniform in the coupling, to control large fields.
4. **Lean.** Restate Proposition S for the spectral measure of a positive contraction on a Hilbert space. This
   needs Mathlib's spectral theory for self-adjoint operators, which is partly available.
