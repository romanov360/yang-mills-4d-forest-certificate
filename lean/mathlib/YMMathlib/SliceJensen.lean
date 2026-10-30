/-
The one-step reduction of the slice ratio (continuum/SLICE.md, Proposition S).

Let ν be the spectral measure of the transfer matrix 𝕋_k at the slice vector u_k, normalised to a
probability measure μ on [0,1], so that D_k(n a_k)/D_k(0) = ∫ λ^n dμ.  Then

* `jensen_pow`:   (∫ λ dμ)^n ≤ ∫ λ^n dμ;
* `slice_step_log`: if ∫ λ dμ ≥ exp(-c), then ∫ λ^n dμ ≥ exp(-c n)  (no condition on c);
* `slice_step`:   if ∫ λ dμ ≥ 1 - ε with 0 ≤ ε ≤ 1/2, then ∫ λ^n dμ ≥ exp(-2 ε n);
* `slice_ratio`:  under the same hypothesis, for every m ≤ n, ∫ λ^n dμ ≥ exp(-2 ε n) ∫ λ^m dμ.

With ε = κ a_k and n a_k = r' this is D_k(r') ≥ e^{-2κr'} D_k(0) ≥ e^{-2κr'} D_k(r); with
c = a_k E_k, E_k = -a_k⁻¹ log(D_k(a_k)/D_k(0)), it is D_k(r') ≥ e^{-E_k r'} D_k(0).
-/
import Mathlib

open MeasureTheory Set Real

namespace SliceJensen

variable {μ : Measure ℝ} [IsProbabilityMeasure μ]

lemma integrable_pow_of_mem (h01 : ∀ᵐ x ∂μ, x ∈ Icc (0 : ℝ) 1) (n : ℕ) :
    Integrable (fun x : ℝ => x ^ n) μ := by
  refine Integrable.mono' (integrable_const (1 : ℝ)) (by fun_prop) ?_
  filter_upwards [h01] with x hx
  rw [Real.norm_eq_abs, abs_pow]
  exact pow_le_one₀ (abs_nonneg x) (abs_le.mpr ⟨by linarith [hx.1], hx.2⟩)

/-- Jensen's inequality for `λ ↦ λ^n` on a probability measure carried by `[0,1]`. -/
theorem jensen_pow (h01 : ∀ᵐ x ∂μ, x ∈ Icc (0 : ℝ) 1) (n : ℕ) :
    (∫ x, x ∂μ) ^ n ≤ ∫ x, x ^ n ∂μ := by
  have hs : ∀ᵐ x ∂μ, x ∈ Ici (0 : ℝ) := by
    filter_upwards [h01] with x hx using hx.1
  have h1 := integrable_pow_of_mem h01 1
  simp only [pow_one] at h1
  exact (convexOn_pow n).map_integral_le (continuous_pow n).continuousOn isClosed_Ici hs h1
    (integrable_pow_of_mem h01 n)

/-- The log form: a bound `E` on the first-step effective energy bounds the decay at every step. -/
theorem slice_step_log (h01 : ∀ᵐ x ∂μ, x ∈ Icc (0 : ℝ) 1) {c : ℝ} (hstep : exp (-c) ≤ ∫ x, x ∂μ)
    (n : ℕ) : exp (-c * n) ≤ ∫ x, x ^ n ∂μ := by
  calc exp (-c * n) = exp (-c) ^ n := by rw [← exp_nat_mul]; ring_nf
    _ ≤ (∫ x, x ∂μ) ^ n := pow_le_pow_left₀ (exp_pos _).le hstep n
    _ ≤ _ := jensen_pow h01 n

/-- `1 - ε ≥ exp(-2ε)` on `[0, 1/2]`. -/
lemma exp_neg_two_mul_le (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1 / 2) : exp (-2 * ε) ≤ 1 - ε := by
  have key : 1 ≤ exp (2 * ε) * (1 - ε) := by
    have := add_one_le_exp (2 * ε)
    nlinarith
  have hpos : 0 < exp (2 * ε) := exp_pos _
  rw [show -2 * ε = -(2 * ε) by ring, exp_neg]
  rw [inv_eq_one_div, div_le_iff₀ hpos]
  linarith [mul_comm (exp (2 * ε)) (1 - ε)]

/-- The one-step bound gives exponential decay at most `2ε` per step. -/
theorem slice_step (h01 : ∀ᵐ x ∂μ, x ∈ Icc (0 : ℝ) 1) {ε : ℝ} (h0 : 0 ≤ ε) (h1 : ε ≤ 1 / 2)
    (hstep : 1 - ε ≤ ∫ x, x ∂μ) (n : ℕ) : exp (-2 * ε * n) ≤ ∫ x, x ^ n ∂μ := by
  have hb : exp (-2 * ε) ^ n ≤ (1 - ε) ^ n :=
    pow_le_pow_left₀ (exp_pos _).le (exp_neg_two_mul_le ε h0 h1) n
  have hc : (1 - ε) ^ n ≤ (∫ x, x ∂μ) ^ n :=
    pow_le_pow_left₀ (by linarith) hstep n
  calc exp (-2 * ε * n) = exp (-2 * ε) ^ n := by rw [← exp_nat_mul]; ring_nf
    _ ≤ _ := hb.trans (hc.trans (jensen_pow h01 n))

/-- The slice ratio: `D(n)/D(m) ≥ exp(-2 ε n)` for `m ≤ n` (here `D(0) = 1`). -/
theorem slice_ratio (h01 : ∀ᵐ x ∂μ, x ∈ Icc (0 : ℝ) 1) {ε : ℝ} (h0 : 0 ≤ ε) (h1 : ε ≤ 1 / 2)
    (hstep : 1 - ε ≤ ∫ x, x ∂μ) (m n : ℕ) :
    exp (-2 * ε * n) * ∫ x, x ^ m ∂μ ≤ ∫ x, x ^ n ∂μ := by
  have hm : ∫ x, x ^ m ∂μ ≤ 1 := by
    calc ∫ x, x ^ m ∂μ ≤ ∫ _x, (1 : ℝ) ∂μ := by
          refine integral_mono_ae (integrable_pow_of_mem h01 m) (integrable_const _) ?_
          filter_upwards [h01] with x hx using pow_le_one₀ hx.1 hx.2
      _ = 1 := by simp
  calc exp (-2 * ε * n) * ∫ x, x ^ m ∂μ ≤ exp (-2 * ε * n) * 1 :=
        mul_le_mul_of_nonneg_left hm (exp_pos _).le
    _ = exp (-2 * ε * n) := mul_one _
    _ ≤ _ := slice_step h01 h0 h1 hstep n

end SliceJensen
