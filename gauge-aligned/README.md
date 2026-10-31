# Where a finite-box criterion can and cannot come from (SU(2), d = 4)

**Status: Monte Carlo diagnostics and a heuristic analysis. Nothing in this directory is a proof.**

The question: beyond β_W ≈ 1/3, is the lack of a certified box criterion a limit of the physics, or a limit
of how we build couplings? The repo's own analysis (`NEXT.md`, `box/README.md`) says no box has been certified
at any coupling, and that link-by-link maximal-coupling heat bath fails at β_W ≈ 0.3. The two probes here
separate the two possibilities.

## 1. Why worst-case local methods stop near 1/3

These are my own estimates, made at linear order in β. They are heuristics, not theorems.

* **Single links.** With the sharp quarter, the Kantorovich/Dobrushin bound gives β_W < 1/3. Theorem 2 of the
  paper makes this a ceiling for single-link methods under any forest gauge.
* **Plaquette blocks (Sec. V).** The block law is ∝ exp(Σ⟨y_i, b_i⟩ + 4β Re(y₁y₂y₃)). At b = 0 the
  contraction constant is K ≥ (1+β)/4, so blocks stop near β_W ≈ 0.366. The proved value is 0.345, and the
  Remark 16 variant gives about 0.357 in floating point.
* **The counting reason.** A perturbation on link y enters the 6 plaquettes through y (each with 3 partner
  links, 2 of them free in the line-field gauge). A Dobrushin column sum pays for *every* free partner. That
  is a branching factor of about 2·6·(β_W/…) ≈ "10β" in the normalisation of Sec. V.
  The actual conditional response only has to restore the flux of each disturbed plaquette, and moving *one*
  link per plaquette is enough. That is a branching factor of about "5β".
  Methods that pay per partner link double-count by a factor of about 2. A partitioned block cannot avoid
  this on its surface links, so it only improves a constant. An averaged (Dobrushin–Shlosman, sum over the
  box) criterion can avoid it in the interior. At linear order this suggests a reachable window of roughly
  β_W ∈ (0.44, 0.8), depending on how much of the surface penalty a box of side L absorbs.
* **The DS criterion is not the limit.** The Dobrushin–Shlosman criterion holds whenever complete analyticity
  holds, so it is not mean-field limited. If DS fails at some β_W, the cause is that the coupling being certified
  is poor, not that the measure lacks the property.

## 2. Coupling probe: `probe.py`

The box is {0..L}⁴. Links in the boundary surface are frozen. Two copies differ only at the boundary link
y = (0, L/2, L/2, L/2), direction 1, where U_y → −U_y (a center flip). Both copies are updated by checkerboard heat bath,
with copy 2 maximally coupled to copy 1 at every update. The "aligned" mode also applies, at every sweep, a
relative Landau gauge transformation at interior sites, with g = conj(M)/|M| and
M = Σ_out U₂U₁† + Σ_in U₂†U₁. This removes the pure-gauge part of the disagreement.

The table gives the sums over the box of E|U₁ − U₂| (links) and E|ΔRe Q_p| (plaquettes), for L = 4,
R = 8 replicas, and 40 thermalisation plus 40 measurement sweeps (`logs/coupling_L4.log`).

| β_W | plain link | plain plaq | aligned link | aligned plaq |
|---|---|---|---|---|
| 0.3 | 23.9 | 52.2 | 32.7 | 70.9 |
| 0.4 | 89.5 | 170.6 | 95.8 | 182.3 |
| 0.5 | 144.4 | 252.3 | 144.8 | 252.3 |
| 0.7 | 245.3 | 352.5 | 229.3 | 340.6 |
| 1.0 | 355.6 | 408.0 | 317.7 | 391.4 |

At β_W = 1.0 the two copies are completely decorrelated: the per-plaquette difference is 0.37 → 0.18 at every
distance. Gauge alignment does not rescue the coupling. Per-update maximal coupling lets disagreement
percolate, and that disagreement is mostly not gauge.

A first attempt used the synchronous coupling (shared uniforms, the same rotation about ĉ). It decorrelated
completely even at small β, because ĉ = c/|c| is ill-conditioned when |c| is small. It was dropped.

## 3. True influence: `influence.py`

This probe uses two **independent** chains with a cold boundary (all boundary links = 1; y flipped in copy 2).
It reports Δ(r), the shell-averaged signed difference E[Re Q_p] − E'[Re Q_p] at distance r from y, with standard
errors. L = 4, R = 64, 100 thermalisation plus 400 measurement sweeps (`logs/influence_L4.log`). The r = 0 shell
contains the plaquettes through y itself, so a difference of about 0.38 there is forced and uninformative.

| β_W | Δ(1) | Δ(2) | Δ(3) | Δ(4) |
|---|---|---|---|---|
| 0.30 | 1.3e-04 ± 2e-04 | 2.2e-05 ± 1e-04 | 1.2e-04 ± 1e-04 | 6.3e-05 ± 5e-05 |
| 0.50 | 2.2e-04 ± 2e-04 | 4.1e-05 ± 1e-04 | 2.2e-06 ± 1e-04 | 5.3e-05 ± 5e-05 |
| 1.00 | 1.1e-03 ± 2e-04 | 4.5e-05 ± 9e-05 | 2.4e-04 ± 9e-05 | 2.7e-05 ± 4e-05 |
| 1.50 | 3.6e-03 ± 2e-04 | 2.0e-04 ± 8e-05 | 3.2e-06 ± 7e-05 | 4.5e-05 ± 3e-05 |
| 2.0, 2.5 | (running; will be added) | | | |

Read this with care. |ΔE[f]| ≤ E|f(U₁) − f(U₂)| holds for *every* coupling, so Δ(r) is a **lower bound** on
what any coupling can achieve for this one observable. It is not the full Wasserstein or total-variation influence
that DS needs. Still, the coupling sums at r ≥ 1 are tens to hundreds, while the true mean differences are
around 1e-3 at r = 1 and consistent with zero (at a noise level of about 1e-4) beyond that.
The coupling-based numbers exceed the observable lower bound by **two orders of magnitude or more**.
The physics leaves a great deal of room. The coupling technology wastes it.

## 4. What I think the next steps are, in order

1. **Flux-respecting couplings instead of link-by-link ones.** Couple plaquette variables, or whole hypertrees
   of links, jointly. When the boundary perturbation moves the flux of a plaquette, copy 2 should move a single
   link of that plaquette, chosen along a tree that carries the flux to the boundary. Per §1 this is the step
   that buys the factor of about 2 in branching. Check it first with the same Monte Carlo harness: replace
   `Box.update` and compare the totals in §2. That test is cheap.
2. **Exact integration near y.** Following `NEXT.md` and BBL's Prop. prop:2d, integrate out the links that share
   a plaquette with y by character expansion. In SU(2) the coefficients are I_n(β)/I_1(β) Bessel ratios, which
   `block-su2/` already certifies with intervals. Then couple the resulting few-link marginal maximally, so the
   disagreement probability becomes an explicit total-variation distance.
3. **Certify the smallest box at a known coupling.** Choose β_W ≈ 0.3, where uniqueness is already proved, as a
   proof of concept. Cover the boundary data with a quadtree over conjugacy-class angles, bound the class
   functions with interval arithmetic, and check the arithmetic in Lean, in the style of `lean/` and
   `block-su2/lean/`. No box has ever been certified, so this first certificate is the milestone, even below
   1/3.
4. **Then push the coupling upward** into the 0.36–0.8 window that §1 suggests, and compare it against the
   observable lower bounds from §3 to see how much room is left.

## Reproduce

```
python3 probe.py --L 4 --R 8 --therm 40 --meas 40 0.3 0.4 0.5 0.7 1.0
python3 influence.py --L 4 --R 64 --therm 100 --meas 400 0.3 0.5 1.0 1.5 2.0 2.5
```
