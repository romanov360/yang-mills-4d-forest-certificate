/-
Kernel-checked word identities (Lean core only): parts A and B of exact_check4d.py, PROOF.md sections 1 and 2.

Noncommutative polynomials with integer coefficients in the letters Q_i, Q_i^* and X_i (i = 1..4), with
Q_i Q_i^* = Q_i^* Q_i = 1, and traces taken as cyclic words. Nothing about the matrix size is used, so the identities
hold for every N. Letters are integers: Q_i = i, Q_i^* = -i, X_i = 10 + i.

Power series in t are truncated after t^2 and stored in divided-power form [c_0, c_1, 2! c_2], so every coefficient
is an integer: (ab)_n = sum_i binom(n, i) a_i b_{n-i}, and e^{sY} = [1, sY, Y^2].

A. `rewriting`: Tr Q_p(t), with Q_i -> e^{tX_i} Q_i, equals Tr(e^{tY1} e^{tY2} W e^{-tY3} e^{-tY4}) through order
   t^2, and its second derivative at 0 is the trace polynomial of f_p.
B. `corners`: Tr(Y1 Y4) = Tr(o1 o4), Tr(Y1 Y2) = Tr(i1 o2), Tr(Y3 Y4) = Tr(o3 i4), Tr(Y2 W Y3 W^*) = Tr(i2 i3),
   with o_r = X_r and i_r = Q_r^* X_r Q_r.
-/

abbrev Word := List Int
abbrev Poly := List (Word × Int)

def isQ (l : Int) : Bool := l != 0 && l.natAbs ≤ 4

/-- free reduction of Q Q^* and Q^* Q, with a stack. -/
def red (w : Word) : Word :=
  (w.foldl (fun st l => match st with
    | a :: rest => if isQ l && a == -l then rest else l :: st
    | [] => [l]) []).reverse

/-- cancel Q ... Q^* around the cycle. -/
def strip : Nat → Word → Word
  | 0, w => w
  | n + 1, w => match w with
    | a :: rest => if isQ a && rest.getLast? == some (-a) then strip n (red rest.dropLast) else w
    | [] => []

def lexLt : Word → Word → Bool
  | [], _ :: _ => true
  | a :: u, b :: v => a < b || (a == b && lexLt u v)
  | _, _ => false

/-- canonical form of Tr(w): reduce, cancel around the cycle, least rotation. -/
def cyc (w : Word) : Word :=
  let w := strip w.length (red w)
  ((List.range w.length).map fun i => w.drop i ++ w.take i).foldl (fun m r => if lexLt r m then r else m) w

def coeff (w : Word) (p : Poly) : Int := p.foldl (fun s (v, c) => if v == w then s + c else s) 0

/-- merge equal words and drop zero coefficients. -/
def compact (p : Poly) : Poly :=
  (p.foldl (fun acc (w, c) => if acc.any (·.1 == w) then acc.map (fun (v, d) => if v == w then (v, d + c) else (v, d))
    else acc ++ [(w, c)]) []).filter (·.2 != 0)

def add (p q : Poly) : Poly := compact (p ++ q)
def scal (c : Int) (p : Poly) : Poly := compact (p.map fun (w, d) => (w, c * d))
def mul (p q : Poly) : Poly := compact (p.flatMap fun (w1, c1) => q.map fun (w2, c2) => (red (w1 ++ w2), c1 * c2))
def mulL (ps : List Poly) : Poly := ps.foldl mul [([], 1)]
def tr (p : Poly) : Poly := compact (p.map fun (w, c) => (cyc w, c))

/-- equality of polynomials as functions word -> coefficient. -/
def peq (p q : Poly) : Bool := (p ++ q).all fun (w, _) => coeff w p == coeff w q
def treq (p q : Poly) : Bool := peq (tr p) (tr q)

abbrev Ser := Poly × Poly × Poly

def smul (a b : Ser) : Ser :=
  (mul a.1 b.1, add (mul a.1 b.2.1) (mul a.2.1 b.1),
    add (add (mul a.1 b.2.2) (scal 2 (mul a.2.1 b.2.1))) (mul a.2.2 b.1))
def smulL (ss : List Ser) : Ser := ss.foldl smul ([([], 1)], [], [])
def expo (y : Poly) (s : Int) : Ser := ([([], 1)], scal s y, mul y y)
def const (p : Poly) : Ser := (p, [], [])

def Q (i : Int) : Poly := [([i], 1)]
def Qs (i : Int) : Poly := [([-i], 1)]
def X (i : Int) : Poly := [([10 + i], 1)]
def W : Poly := mulL [Q 1, Q 2, Qs 3, Qs 4]
def Ws : Poly := mulL [Q 4, Q 3, Qs 2, Qs 1]
def Y : Int → Poly
  | 1 => X 1
  | 2 => mulL [Q 1, X 2, Qs 1]
  | 3 => mulL [Q 4, X 3, Qs 4]
  | _ => X 4

def lhs : Ser := smulL [expo (X 1) 1, const (Q 1), expo (X 2) 1, const (Q 2), const (Qs 3), expo (X 3) (-1),
  const (Qs 4), expo (X 4) (-1)]
def rhs : Ser := smulL [expo (Y 1) 1, expo (Y 2) 1, const W, expo (Y 3) (-1), expo (Y 4) (-1)]

def fpPoly : Poly :=
  ([1, 2, 3, 4].map fun i => mulL [Y i, Y i, W]) ++ [scal 2 (mulL [Y 1, Y 2, W]), scal 2 (mulL [Y 3, Y 4, W])] ++
    ([1, 2].flatMap fun i => [3, 4].map fun j => scal (-2) (mulL [Y i, W, Y j])) |>.foldl add []

/-- A: the plaquette rewriting through order t^2, and the second derivative is f_p. -/
theorem rewriting :
    (treq lhs.1 rhs.1 && treq lhs.2.1 rhs.2.1 && treq lhs.2.2 rhs.2.2 && treq lhs.2.2 fpPoly &&
      (tr fpPoly).length == 10) = true := by decide +kernel

def iE (r : Int) : Poly := mulL [Qs r, X r, Q r]

/-- B: the four corner identities, and W W^* = W^* W = 1. -/
theorem corners :
    (treq (mul (Y 1) (Y 4)) (mul (X 1) (X 4)) && treq (mul (Y 1) (Y 2)) (mul (iE 1) (X 2)) &&
      treq (mul (Y 3) (Y 4)) (mul (X 3) (iE 4)) && treq (mulL [Y 2, W, Y 3, Ws]) (mul (iE 2) (iE 3)) &&
      peq (mul W Ws) [([], 1)] && peq (mul Ws W) [([], 1)]) = true := by decide +kernel

#print axioms rewriting
#print axioms corners
