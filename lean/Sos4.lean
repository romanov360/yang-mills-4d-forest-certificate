/-
Kernel-checked all-phase part of the certificate (Lean core only). It replaces the interval cover of
cert_interval4d.py (PROOF.md section 5), which stays as an independent check.

For a plaquette p and phases phi_1, phi_2 put c_a = cos phi_a, s_a = sin phi_a and W = diag(c1 + i s1, c2 + i s2).
With the weights a_r and the corner coefficients k = 2 G[s][r][q] of the certificate, the plaquette form is

    Phi_p(Y) = sum_r a_r |Y_r|^2 - f_p(Y, W) - k14 <Y1,Y4> - k12 <Y1,Y2> - k34 <Y3,Y4> - k23 <Y2, W Y3 W*>,
    f_p(Y, W) = sum_r ReTr(Y_r Y_r W) + 2 ReTr(Y1 Y2 W) + 2 ReTr(Y3 Y4 W) - 2 sum_{i in 1,2; j in 3,4} ReTr(Y_i W Y_j),

on the free positions, <X,Y> = -ReTr(XY). Here we recompute its matrix M_p in the orthonormal basis B_q / sqrt 2
of u(2) (B = iI, i sigma_1, i sigma_2, i sigma_3) as a polynomial matrix in z = (1, c1, s1, c2, s2), from the
definition above and the certificate data, and check:

  * `sos_gram`: M_p splits into the blocks q in {0, 3} and q in {1, 2} (the cross entries vanish), and for each block
        sosScale * 4 * 10^6 * M_block  =  sum_{a,b} z_a z_b P[a*6 + ., b*6 + .]
    as polynomials modulo c_a^2 = 1 - s_a^2, with P the symmetric 30 x 30 integer matrix of SosData.lean;
  * `sos_pd`: every such P is positive definite. `pdCheck` is fraction-free elimination: with p = P_00 > 0,
    first row (p, r) and trailing block P', the identity  p x^T P x = (p x_0 + r.x')^2 + x'^T (p P' - r r^T) x'
    reduces positivity of P to that of p P' - r r^T, which is checked to be prev times an integer matrix (prev > 0
    the previous pivot) and treated recursively.

On the torus c_a^2 + s_a^2 = 1, so for x in R^6 and w = z (x) x (w != 0 since z_0 = 1),
x^T (sosScale * 4 * 10^6 * M_block(phi)) x = w^T P w > 0: every M_p(phi) is positive definite for every phi.
SosData.lean is written by gen_sosdata.py (an SDP and an exact rounding); nothing in it is trusted.
-/
import Cert4
import SosData

/-! ### Gaussian integers and 2 x 2 matrices -/

abbrev GI := Int × Int

def gadd (a b : GI) : GI := (a.1 + b.1, a.2 + b.2)
def gmul (a b : GI) : GI := (a.1 * b.1 - a.2 * b.2, a.1 * b.2 + a.2 * b.1)

/-- the matrix [[a, b], [c, d]]. -/
structure M2 where
  a : GI
  b : GI
  c : GI
  d : GI

/-- sqrt 2 times the orthonormal basis of u(2): i I, i sigma_1, i sigma_2, i sigma_3. -/
def basis : Nat → M2
  | 0 => ⟨(0, 1), (0, 0), (0, 0), (0, 1)⟩
  | 1 => ⟨(0, 0), (0, 1), (0, 1), (0, 0)⟩
  | 2 => ⟨(0, 0), (1, 0), (-1, 0), (0, 0)⟩
  | _ => ⟨(0, 1), (0, 0), (0, 0), (0, -1)⟩

/-! ### Diagonal matrices of degree <= 1 in z = (1, c1, s1, c2, s2), as (a, d1, d2): z_a diag(d1, d2) -/

def lI : List (Nat × GI × GI) := [(0, (1, 0), (1, 0))]
/-- W = diag(c1 + i s1, c2 + i s2) and W*. -/
def lW : List (Nat × GI × GI) := [(1, (1, 0), (0, 0)), (2, (0, 1), (0, 0)), (3, (0, 0), (1, 0)), (4, (0, 0), (0, 1))]
def lWs : List (Nat × GI × GI) :=
  [(1, (1, 0), (0, 0)), (2, (0, -1), (0, 0)), (3, (0, 0), (1, 0)), (4, (0, 0), (0, -1))]

/-- quadratic polynomials as lists (a, b, w): w z_a z_b. -/
abbrev Quad := List (Nat × Nat × Int)

/-- ReTr(X D Y E) for constant X, Y and diagonal linear D, E: Tr = sum_{p,q} X_pq D_qq Y_qp E_pp. -/
def quadTr (X : M2) (D : List (Nat × GI × GI)) (Y : M2) (E : List (Nat × GI × GI)) : Quad :=
  D.flatMap fun (a, d1, d2) => E.map fun (b, e1, e2) =>
    (a, b, (gadd (gadd (gmul (gmul (gmul X.a d1) Y.a) e1) (gmul (gmul (gmul X.b d2) Y.c) e1))
      (gadd (gmul (gmul (gmul X.c d1) Y.b) e2) (gmul (gmul (gmul X.d d2) Y.d) e2))).1)

/-- normal form modulo c_a^2 = 1 - s_a^2 (z_1 = c1, z_2 = s1, z_3 = c2, z_4 = s2): z_a z_b is a combination of the
monomials z_a' z_b' with a' <= b', (a', b') != (1, 1), (3, 3), keyed by 5 a' + b'. -/
def nkey (a b : Nat) : List (Nat × Int) :=
  let m := if a ≤ b then 5 * a + b else 5 * b + a
  if m == 6 then [(0, 1), (12, -1)] else if m == 18 then [(0, 1), (24, -1)] else [(m, 1)]

/-- the keys of the normal monomials. -/
def nmon : List Nat := [0, 1, 2, 3, 4, 7, 8, 9, 12, 13, 14, 19, 24]

def normal (q : Quad) : List (Nat × Int) := q.flatMap fun (a, b, w) => (nkey a b).map fun (m, e) => (m, e * w)

def coeff (q : List (Nat × Int)) (m : Nat) : Int := q.foldl (fun s (m', w) => if m' == m then s + w else s) 0

/-- the normal form as the list of coefficients of `nmon`. -/
def dense (q : Quad) : List Int := let n := normal q; nmon.map (coeff n)

/-! ### The plaquette form -/

/-- a term (c, i, A, j, B) is c * ReTr(Y_i A Y_j B), positions 0..3. -/
abbrev Term := Int × Nat × List (Nat × GI × GI) × Nat × List (Nat × GI × GI)

def E6 : Int := 1000000

def terms (x : Site) (mu nu : Fin 4) : List Term :=
  let k := (corners x mu nu).map fun (s, r, q) => 2 * entry (Gat s) r.val q.val
  (List.range 4).map (fun r => (-(weight x mu nu r), r, lI, r, lI)) ++
  (List.range 4).map (fun r => (-E6, r, lI, r, lW)) ++
  [(-2 * E6, 0, lI, 1, lW), (-2 * E6, 2, lI, 3, lW)] ++
  ([0, 1].flatMap fun i => [2, 3].map fun j => (2 * E6, i, lW, j, lI)) ++
  [(k.getD 0 0, 0, lI, 3, lI), (k.getD 1 0, 0, lI, 1, lI), (k.getD 3 0, 2, lI, 3, lI), (k.getD 2 0, 1, lW, 2, lWs)]

/-- 4 * 10^6 * M_p at the coordinates (position, q), (position', q'), by polarisation. -/
def entryQ (ts : List Term) (u v : Nat × Nat) : Quad :=
  ts.flatMap fun (c, i, A, j, B) =>
    let one (u v : Nat × Nat) : Quad :=
      if u.1 == i && v.1 == j then (quadTr (basis u.2) A (basis v.2) B).map fun (a, b, w) => (a, b, c * w) else []
    one u v ++ one v u

/-! ### The Gram side -/

/-- the transpose of a list of rows (structural; rows of equal length). -/
def transp : List (List Int) → List (List Int)
  | [] => []
  | r :: rs => match rs with
    | [] => r.map fun x => [x]
    | _ => (r.zip (transp rs)).map fun (x, c) => x :: c

def madd (A B : List (List Int)) : List (List Int) := (A.zip B).map fun (r, s) => (r.zip s).map fun (x, y) => x + y
def msub (A B : List (List Int)) : List (List Int) := (A.zip B).map fun (r, s) => (r.zip s).map fun (x, y) => x - y

/-- the symmetric 30 x 30 matrix with lower triangle T: L + (strict L)^T, rows padded with zeros. -/
def full (T : List (List Int)) : List (List Int) :=
  let pad (l : List Int) := l ++ List.replicate (30 - l.length) 0
  madd (T.map pad) (transp (T.map fun r => pad r.dropLast))

/-- consecutive groups of 6. -/
def chunks6 : Nat → List Int → List (List Int)
  | 0, _ => []
  | n + 1, l => l.take 6 :: chunks6 n (l.drop 6)

def chunkRows : Nat → List (List Int) → List (List (List Int))
  | 0, _ => []
  | n + 1, l => l.take 6 :: chunkRows n (l.drop 6)

/-- the 6 x 6 blocks B[a][b] of a 30 x 30 matrix, b-th column group of the a-th row group. -/
def blocks (F : List (List Int)) : List (List (List (List Int))) :=
  (chunkRows 5 F).map fun G =>
    let cs := G.map (chunks6 5)
    (List.range 5).map fun b => cs.map fun r => r.getD b []

/-- the normal form of sum_{a,b} z_a z_b P[a*6 + i, b*6 + k] for all i, k, as 6 x 6 lists of coefficients on `nmon`
(z_1^2 = 1 - z_2^2, z_3^2 = 1 - z_4^2). -/
def gramN (F : List (List Int)) : List (List (List Int)) :=
  let B := blocks F
  let g (a b : Nat) := (B.getD a []).getD b []
  let s (a b : Nat) := madd (g a b) (g b a)
  let comb := [madd (madd (g 0 0) (g 1 1)) (g 3 3), s 0 1, s 0 2, s 0 3, s 0 4, s 1 2, s 1 3, s 1 4,
    msub (g 2 2) (g 1 1), s 2 3, s 2 4, s 3 4, msub (g 4 4) (g 3 3)]
  (List.range 6).map fun i => transp (comb.map fun M => M.getD i [])

def triOK (T : List (List Int)) : Bool := T.length == 30 && (List.range 30).all fun i => (T.getD i []).length == i + 1

def blockOK (ts : List Term) (cs : List (Nat × Nat)) (T : List (List Int)) : Bool :=
  triOK T && (((List.range 6).zip (gramN (full T))).all fun (i, row) => ((List.range 6).zip row).all fun (k, g) => k < i ||
    (dense (entryQ ts (cs.getD i (0, 0)) (cs.getD k (0, 0)))).map (sosScale * ·) == g)

def plaqs : List (Site × Fin 4 × Fin 4) := cellSites.flatMap fun x => pairs.map fun (mu, nu) => (x, mu, nu)

def plaqOK (j : Nat) : Bool :=
  let (x, mu, nu) := plaqs.getD j (⟨0, 0, 0, 0⟩, 0, 0)
  let ts := terms x mu nu
  let free := (List.range 4).filter fun r => !fixed ((plaqLinks x mu nu).getD r (x, 0))
  let cD := free.flatMap fun r => [(r, 0), (r, 3)]
  let cO := free.flatMap fun r => [(r, 1), (r, 2)]
  let (PD, PO) := sosCerts.getD (sosIndex.getD j 0) ([], [])
  free.length == 3 &&
  cD.all (fun u => cO.all fun v => (dense (entryQ ts u v)).all (· == 0)) &&
  blockOK ts cD PD && blockOK ts cO PO

/-! ### Positive definiteness of the Gram matrices -/

def pdAux : Nat → Int → List (List Int) → Bool
  | 0, _, rows => rows.isEmpty
  | _, _, [] => true
  | fuel + 1, prev, r0 :: rows =>
    match r0 with
    | [] => false
    | p :: rt =>
      let S := rows.map fun r => match r with
        | [] => []
        | h :: t => (t.zip rt).map fun (x, y) => p * x - h * y
      p > 0 && rows.all (·.length == r0.length) && S.all (·.all (· % prev == 0)) &&
        pdAux fuel p (S.map (·.map (· / prev)))

def pdCheck (T : List (List Int)) : Bool :=
  triOK T && pdAux 30 1 (full T)

theorem plaqs_length : plaqs.length = 96 := by decide +kernel
