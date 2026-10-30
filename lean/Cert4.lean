/-
Kernel-checked finite part of the certificate (Lean core only). Everything is an integer: the certificate's
rationals all have denominator dividing 10^6, so G, a and h are stored multiplied by 10^6 (CertData.lean, written by
gen_certdata.py; nothing in that script is trusted, only the identities checked here).

Checked (PROOF.md sections 3 and 5):
  * `sites_ok`: every G_y is symmetric, 8 x 8, has G_y[o_mu, i_mu] = 0, and satisfies
        S_y * G_y = sum_k c_k u_k u_k^T  with S_y > 0, c_k > 0,
    so x^T G_y x = (1/S_y) sum_k c_k (u_k . x)^2 >= 0: G_y is positive semidefinite (positive definite, since the
    u_k form a triangular basis);
  * `weights_ok`: every a_{p,r} >= 0, and a_{p,r} = 0 at the fixed position;
  * `totals_ok`, `totals_max`: for every free link e of Z^4,
        sum_{p ∋ e} a_{p,e} + G_tail(e)[o_mu, o_mu] + G_head(e)[i_mu, i_mu]  <=  h,
    with equality at some link. The plaquettes containing e are found by enumerating all plaquettes with base in
    tail(e) + {-1, 0}^4 (every link of the plaquette with base x has its tail in x + {0, 1}^4), so this is the
    geometric definition, not a table.
-/
import Forest4
import CertData

def entry (g : List (List Int)) (i j : Nat) : Int := (g.getD i []).getD j 0

def Gat (y : Site) : List (List Int) := certG.getD y.cellIdx []

def idx8 : List Nat := List.range 8

def dot (u v : List Int) : Int := (u.zip v).foldl (fun s (a, b) => s + a * b) 0

def siteOK (n : Nat) : Bool :=
  let g := certG.getD n []
  let S := certS.getD n 0
  let cs := certC.getD n []
  let us := certU.getD n []
  g.length == 8 && g.all (·.length == 8) && cs.length == 8 && us.length == 8 && us.all (·.length == 8) &&
  S > 0 && cs.all (· > 0) &&
  idx8.all (fun i => idx8.all fun j => entry g i j == entry g j i) &&
  (List.range 4).all (fun mu => entry g mu (mu + 4) == 0) &&
  idx8.all (fun i => idx8.all fun j =>
    S * entry g i j == dot cs (us.map fun u => u.getD i 0 * u.getD j 0))

theorem sites_ok : (List.range 16).all siteOK = true := by decide +kernel

def pairIdx (mu nu : Fin 4) : Nat := (pairs.findIdx? (· == (mu, nu))).getD 0

/-- the weight a_{p,r} of the plaquette (x; mu, nu) at position r; periodic, through x mod 2. -/
def weight (x : Site) (mu nu : Fin 4) (r : Nat) : Int :=
  (certA.getD (x.cellIdx * 6 + pairIdx mu nu) []).getD r 0

def weightsOK : Bool :=
  certA.length == 96 && certA.all (·.length == 4) && certA.all (·.all (· ≥ 0)) &&
  cellSites.all fun x => pairs.all fun (mu, nu) =>
    ((List.range 4).zip (plaqLinks x mu nu)).all fun (r, l) => !fixed l || weight x mu nu r == 0

theorem weights_ok : weightsOK = true := by decide +kernel

def offsets : List Site :=
  [0, -1].flatMap fun a => [0, -1].flatMap fun b => [0, -1].flatMap fun c => [0, -1].map fun d => ⟨a, b, c, d⟩

def addS (y z : Site) : Site := ⟨y.x0 + z.x0, y.x1 + z.x1, y.x2 + z.x2, y.x3 + z.x3⟩

/-- all (plaquette, position) incidences of the link e, and their number. -/
def incid (e : Link) : List (Site × Fin 4 × Fin 4 × Nat) :=
  offsets.flatMap fun o => pairs.flatMap fun (mu, nu) =>
    ((List.range 4).zip (plaqLinks (addS e.1 o) mu nu)).filterMap fun (r, l) =>
      if l == e then some (addS e.1 o, mu, nu, r) else none

def total (e : Link) : Int :=
  (incid e).foldl (fun s (x, mu, nu, r) => s + weight x mu nu r) 0 +
    entry (Gat e.1) e.2.val e.2.val + entry (Gat (head e)) (e.2.val + 4) (e.2.val + 4)

def cellLinks : List Link := cellSites.flatMap fun y => (List.finRange 4).map fun a => (y, a)

def freeCell : List Link := cellLinks.filter fun l => !fixed l

/-- every free link of the cell lies in exactly 6 plaquettes, and its total is at most h. -/
def totalsOK : Bool := freeCell.length == 48 && freeCell.all fun e => (incid e).length == 6 && total e ≤ certH

theorem totals_ok : totalsOK = true := by decide +kernel

theorem totals_max : (freeCell.map total).foldl max 0 = certH := by decide +kernel

theorem h_value : certH = 14420001 := by decide

#print axioms sites_ok
#print axioms weights_ok
#print axioms totals_ok
#print axioms totals_max
