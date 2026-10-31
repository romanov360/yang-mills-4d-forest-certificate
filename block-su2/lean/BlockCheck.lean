/-
Kernel-checked arithmetic of the sharpened plaquette-block constant for SU(2) (Lean core only, no Mathlib).

Inputs (BlockData.lean, written by cert_block_su2.py from Arb enclosures, rounded outward to 10^-12):
  lamLo <= every eigenvalue of Hess g,  tauHi >= every eigenvalue,  mHi >= |D^3 g|   on |c| <= 24 beta,
  aHi >= |grad g(c)|                                                                on |c| <= 20 beta,
for g = log Z0 and beta = beta_W / 4 at the endpoint beta_W = 0.351.

Three elementary facts are used as rational inequalities, each checked below:
  e^w <= 1/(1-w) for 0 <= w < 1          (encoded as E := 1/(1-W)),
  sqrt(tau) <= q   because q^2 >= tau     (sq_ok),
  1/sqrt 3  <= r   because 3 r^2 >= 1     (r_ok).

Checked (SHARPENED.md, eq. (S)):
  W  = 8 beta^2 (tau - lambda_*)
  S  = 1/4 + (E - 1) E q r
  Cc = 4 beta S aHi + 20 beta * 16 beta^2 mHi E / 3          < 1
  alpha = 40 beta (1 + beta) S / (1 - Cc)                   < 1      (and <= 0.99997)
Every quantity is increasing in each input bound, so the checked numbers bound the true ones.
-/
import BlockData

/-- nonnegative rationals as numerator / denominator (denominators are positive throughout). -/
structure Q where
  n : Nat
  d : Nat

def Q.add (a b : Q) : Q := ⟨a.n * b.d + b.n * a.d, a.d * b.d⟩
def Q.mul (a b : Q) : Q := ⟨a.n * b.n, a.d * b.d⟩
/-- truncated difference; only used where `b ≤ a` is checked. -/
def Q.sub (a b : Q) : Q := ⟨a.n * b.d - b.n * a.d, a.d * b.d⟩
def Q.div (a b : Q) : Q := ⟨a.n * b.d, a.d * b.n⟩
def Q.le (a b : Q) : Bool := a.n * b.d ≤ b.n * a.d
def Q.lt (a b : Q) : Bool := a.n * b.d < b.n * a.d
def Q.nat (k : Nat) : Q := ⟨k, 1⟩

def lam : Q := ⟨lamLo, den⟩
def tau : Q := ⟨tauHi, den⟩
def mM : Q := ⟨mHi, den⟩
def aA : Q := ⟨aHi, den⟩
def q : Q := ⟨500000000008, den⟩
def r : Q := ⟨577350269190, den⟩
def one : Q := Q.nat 1

def sq_ok : Bool := tau.le (q.mul q)
def r_ok : Bool := one.le ((Q.nat 3).mul (r.mul r))
def tau_lam_ok : Bool := lam.le tau

/-- the chain of (S) at beta = bW / 4: returns (W, S, Cc, alpha). -/
def chain (bW : Q) : Q × Q × Q × Q :=
  let b := bW.div (Q.nat 4)
  let W := (Q.nat 8).mul (b.mul (b.mul (tau.sub lam)))
  let E := one.div (one.sub W)
  let S := (⟨1, 4⟩ : Q).add ((E.sub one).mul (E.mul (q.mul r)))
  let Cc := ((Q.nat 4).mul (b.mul (S.mul aA))).add
            ((⟨320, 3⟩ : Q).mul (b.mul (b.mul (b.mul (mM.mul E)))))
  let alpha := ((Q.nat 40).mul (b.mul ((one.add b).mul S))).div (one.sub Cc)
  (W, S, Cc, alpha)

/-- all side conditions (W < 1, Cc < 1, E >= 1) and alpha <= bound. -/
def ok (bW bound : Q) : Bool :=
  let (W, _, Cc, alpha) := chain bW
  W.lt one && Cc.lt one && alpha.le bound

theorem side_ok : (sq_ok && r_ok && tau_lam_ok) = true := by decide +kernel

/-- the endpoint: alpha(0.351) <= 0.99997 < 1. -/
theorem alpha_0351 : ok ⟨351, 1000⟩ ⟨99997, 100000⟩ = true := by decide +kernel

/-- with the same constants (valid on the smaller balls): alpha(0.345) <= 0.98046, alpha(1/3) <= 0.94280. -/
theorem alpha_0345 : ok ⟨345, 1000⟩ ⟨98046, 100000⟩ = true := by decide +kernel
theorem alpha_third : ok ⟨1, 3⟩ ⟨94280, 100000⟩ = true := by decide +kernel

/-- sanity: the check is not vacuous; alpha(0.352) with these constants exceeds 1. -/
theorem alpha_0352_fails : ok ⟨352, 1000⟩ ⟨1, 1⟩ = false := by decide +kernel
