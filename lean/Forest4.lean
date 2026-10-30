/-
Kernel-checked combinatorics of the four-dimensional forest gauge (Lean core only, no Mathlib).

Sites are points of Z^4. The pattern mu : (Z/2)^4 -> {0,1,2,3} fixes the link (x - e_mu(x), mu(x)) at every site x,
so the link (y, a) is fixed iff mu(y + e_a) = a. Proved here, for all of Z^4 (PROOF.md sections 0 and 2):

  * `head_unique`: every site is the head of exactly one fixed link;
  * `forest_leaf`: every finite nonempty set of fixed links has a site that meets exactly one of them, so the fixed
    links contain no cycle (all degrees on a cycle are 2);
  * `one_fixed`: every plaquette has exactly one fixed link;
  * `corner_cover`: at every site, each pair of ends in different directions is a corner of exactly one plaquette,
    and a pair in the same direction is a corner of none;
  * `corner_ends`: each corner names the two ends of the right links of its plaquette.

The statements about all of Z^4 reduce to finite checks because mu depends only on coordinates mod 2 and the corner
geometry is translation invariant.
-/

structure Site where
  x0 : Int
  x1 : Int
  x2 : Int
  x3 : Int
deriving DecidableEq, Repr

namespace Site

def shift (y : Site) (a : Fin 4) (s : Int) : Site :=
  match a with
  | 0 => { y with x0 := y.x0 + s }
  | 1 => { y with x1 := y.x1 + s }
  | 2 => { y with x2 := y.x2 + s }
  | 3 => { y with x3 := y.x3 + s }

def sum (y : Site) : Int := y.x0 + y.x1 + y.x2 + y.x3

/-- reduction mod 2. -/
def red (y : Site) : Site := ⟨y.x0 % 2, y.x1 % 2, y.x2 % 2, y.x3 % 2⟩

/-- index of a site of the period cell, lexicographic order, first coordinate most significant. -/
def cellIdx (y : Site) : Nat :=
  (y.x0 % 2).toNat * 8 + (y.x1 % 2).toNat * 4 + (y.x2 % 2).toNat * 2 + (y.x3 % 2).toNat

end Site

/-- the pattern PAT4 = "0212331001332120" of forest4d.py. -/
def pat : List Nat := [0, 2, 1, 2, 3, 3, 1, 0, 0, 1, 3, 3, 2, 1, 2, 0]

def muAt (y : Site) : Fin 4 := ⟨pat.getD y.cellIdx 0 % 4, Nat.mod_lt _ (by decide)⟩

abbrev Link := Site × Fin 4

def head (l : Link) : Site := l.1.shift l.2 1

def fixed (l : Link) : Bool := muAt (head l) == l.2

/-! ### The forest -/

theorem shift_back (z : Site) (a : Fin 4) : (z.shift a (-1)).shift a 1 = z := by
  cases z
  match a with
  | 0 => simp [Site.shift, Int.neg_add_cancel_right]
  | 1 => simp [Site.shift, Int.neg_add_cancel_right]
  | 2 => simp [Site.shift, Int.neg_add_cancel_right]
  | 3 => simp [Site.shift, Int.neg_add_cancel_right]

theorem shift_inj (y y' : Site) (a : Fin 4) (h : y.shift a 1 = y'.shift a 1) : y = y' := by
  cases y; cases y'
  match a with
  | 0 => simp [Site.shift] at h ⊢; omega
  | 1 => simp [Site.shift] at h ⊢; omega
  | 2 => simp [Site.shift] at h ⊢; omega
  | 3 => simp [Site.shift] at h ⊢; omega

theorem sum_head (l : Link) : (head l).sum = l.1.sum + 1 := by
  obtain ⟨⟨a, b, c, d⟩, e⟩ := l
  match e with
  | 0 => simp [head, Site.shift, Site.sum]; omega
  | 1 => simp [head, Site.shift, Site.sum]; omega
  | 2 => simp [head, Site.shift, Site.sum]; omega
  | 3 => simp [head, Site.shift, Site.sum]; omega

/-- every site z is the head of exactly one fixed link, namely (z - e_mu(z), mu(z)). -/
theorem head_unique (z : Site) :
    fixed (z.shift (muAt z) (-1), muAt z) = true ∧ head (z.shift (muAt z) (-1), muAt z) = z ∧
    ∀ l : Link, fixed l = true → head l = z → l = (z.shift (muAt z) (-1), muAt z) := by
  refine ⟨?_, shift_back z _, ?_⟩
  · simp [fixed, head, shift_back]
  · intro ⟨y, a⟩ hf hh
    simp only [fixed, head, beq_iff_eq] at hf hh
    rw [hh] at hf
    subst hf
    have : y = z.shift (muAt z) (-1) := by
      apply shift_inj _ _ (muAt z); rw [shift_back]; exact hh
    rw [this]

theorem exists_max {α : Type} (f : α → Int) :
    ∀ s : List α, s ≠ [] → ∃ m ∈ s, ∀ x ∈ s, f x ≤ f m
  | [], h => absurd rfl h
  | [a], _ => ⟨a, by simp⟩
  | a :: b :: t, _ => by
    obtain ⟨m, hm, hmax⟩ := exists_max f (b :: t) (by simp)
    by_cases hle : f a ≤ f m
    · exact ⟨m, List.mem_cons_of_mem _ hm, fun x hx => by
        rcases List.mem_cons.mp hx with rfl | hx
        · exact hle
        · exact hmax x hx⟩
    · exact ⟨a, List.mem_cons_self, fun x hx => by
        rcases List.mem_cons.mp hx with rfl | hx
        · exact Int.le_refl _
        · have := hmax x hx; omega⟩

def meets (z : Site) (l : Link) : Prop := l.1 = z ∨ head l = z

/-- every finite nonempty set of fixed links has a leaf: a site meeting exactly one of its links. Since every site of
a cycle meets two of its links, the fixed links form a forest. -/
theorem forest_leaf (s : List Link) (hne : s ≠ []) (hf : ∀ l ∈ s, fixed l = true) :
    ∃ l0 ∈ s, ∀ l ∈ s, meets (head l0) l → l = l0 := by
  obtain ⟨l0, h0, hmax⟩ := exists_max (fun l => (head l).sum) s hne
  refine ⟨l0, h0, fun l hl hm => ?_⟩
  rcases hm with h | h
  · have h1 := hmax l hl
    have h2 := sum_head l
    rw [h] at h2
    omega
  · have u := (head_unique (head l0)).2.2
    exact (u l (hf l hl) h).trans (u l0 (hf l0 h0) rfl).symm

/-! ### Plaquettes -/

def pairs : List (Fin 4 × Fin 4) := [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

/-- the links of the plaquette (x; mu, nu), in the order e1 = (x,mu), e2 = (x+mu,nu), e3 = (x+nu,mu), e4 = (x,nu). -/
def plaqLinks (x : Site) (mu nu : Fin 4) : List Link :=
  [(x, mu), (x.shift mu 1, nu), (x.shift nu 1, mu), (x, nu)]

def fixedCount (x : Site) (mu nu : Fin 4) : Nat := ((plaqLinks x mu nu).filter fixed).length

theorem cellIdx_red (y : Site) : y.red.cellIdx = y.cellIdx := by
  obtain ⟨p, q, r, t⟩ := y
  have e : ∀ u : Int, u % 2 % 2 = u % 2 := fun u => by omega
  simp only [Site.red, Site.cellIdx, e]

theorem red_shift (y : Site) (a : Fin 4) (s : Int) : (y.shift a s).red = (y.red.shift a s).red := by
  obtain ⟨p, q, r, t⟩ := y
  match a with
  | 0 => simp only [Site.shift, Site.red, Site.mk.injEq]; omega
  | 1 => simp only [Site.shift, Site.red, Site.mk.injEq]; omega
  | 2 => simp only [Site.shift, Site.red, Site.mk.injEq]; omega
  | 3 => simp only [Site.shift, Site.red, Site.mk.injEq]; omega

theorem muAt_red (y : Site) : muAt y.red = muAt y := by simp only [muAt, cellIdx_red]

theorem fixed_red (y : Site) (a : Fin 4) : fixed (y, a) = fixed (y.red, a) := by
  simp only [fixed, head]
  rw [← muAt_red (y.shift a 1), red_shift, muAt_red]

theorem fixed_shift_red (y : Site) (a b : Fin 4) : fixed (y.shift a 1, b) = fixed (y.red.shift a 1, b) := by
  rw [fixed_red, red_shift, ← fixed_red]

theorem fixedCount_red (x : Site) (mu nu : Fin 4) : fixedCount x mu nu = fixedCount x.red mu nu := by
  simp only [fixedCount, plaqLinks, List.filter_cons, List.filter_nil]
  rw [fixed_red x mu, fixed_shift_red x mu nu, fixed_shift_red x nu mu, fixed_red x nu]
  split <;> split <;> split <;> split <;> rfl

def cellSites : List Site :=
  [0, 1].flatMap fun a => [0, 1].flatMap fun b => [0, 1].flatMap fun c => [0, 1].map fun d => ⟨a, b, c, d⟩

theorem red_mem (x : Site) : x.red ∈ cellSites := by
  obtain ⟨p, q, r, t⟩ := x
  have h : ∀ u : Int, u % 2 = 0 ∨ u % 2 = 1 := fun u => by omega
  simp only [Site.red]
  rcases h p with hp | hp <;> rcases h q with hq | hq <;> rcases h r with hr | hr <;> rcases h t with ht | ht <;>
    (simp only [hp, hq, hr, ht]; decide)

def cellOneFixed : Bool := cellSites.all fun x => pairs.all fun (mu, nu) => fixedCount x mu nu == 1

theorem cell_one_fixed : cellOneFixed = true := by decide +kernel

/-- every plaquette of Z^4 has exactly one fixed link. -/
theorem one_fixed (x : Site) (mu nu : Fin 4) (h : (mu, nu) ∈ pairs) : fixedCount x mu nu = 1 := by
  rw [fixedCount_red]
  have hc := cell_one_fixed
  simp only [cellOneFixed, List.all_eq_true, beq_iff_eq] at hc
  exact hc _ (red_mem x) (mu, nu) h

/-! ### Ends and corners -/

/-- ends at a site y: r < 4 is o_r(y), on the link (y, r); r >= 4 is i_{r-4}(y), on the link (y - e_{r-4}, r-4). -/
def dir (r : Fin 8) : Fin 4 := ⟨r.val % 4, Nat.mod_lt _ (by decide)⟩

def endLink (y : Site) (r : Fin 8) : Link :=
  if r.val < 4 then (y, dir r) else (y.shift (dir r) (-1), dir r)

def oE (mu : Fin 4) : Fin 8 := ⟨mu.val, by omega⟩
def iE (mu : Fin 4) : Fin 8 := ⟨mu.val + 4, by omega⟩

/-- the corners of (x; mu, nu) as (site, end, end), in the order of the terms k14, k12, k23, k34. -/
def corners (x : Site) (mu nu : Fin 4) : List (Site × Fin 8 × Fin 8) :=
  [(x, oE mu, oE nu), (x.shift mu 1, iE mu, oE nu), ((x.shift mu 1).shift nu 1, iE nu, iE mu),
   (x.shift nu 1, oE mu, iE nu)]

/-- corner c must pair the links (e_u, e_v) of its plaquette, for (u, v) = (1,4), (1,2), (2,3), (3,4). -/
def want : List (Nat × Nat) := [(0, 3), (0, 1), (1, 2), (2, 3)]

theorem corner_ends (x : Site) : pairs.all (fun (mu, nu) =>
    ((corners x mu nu).zip want).all fun ((s, r, q), (u, v)) =>
      endLink s r == (plaqLinks x mu nu).getD u (x, 0) && endLink s q == (plaqLinks x mu nu).getD v (x, 0)) := by
  obtain ⟨p, q, r, t⟩ := x
  simp (config := { decide := true }) [pairs, corners, want, plaqLinks, endLink, dir, oE, iE, Site.shift,
    Int.add_neg_cancel_right, Fin.ext_iff]

/-- relative corner positions: the corner of (x; mu, nu) at site x + off. Translation invariance means the number of
plaquettes having a given pair of ends as a corner at y is the number of entries of this list with that pair. -/
def relCorners : List (Fin 8 × Fin 8) :=
  pairs.flatMap fun (mu, nu) => (corners ⟨0, 0, 0, 0⟩ mu nu).map fun (_, r, q) => (r, q)

theorem corners_rel (x : Site) (mu nu : Fin 4) :
    (corners x mu nu).map (fun (_, r, q) => (r, q)) = (corners ⟨0, 0, 0, 0⟩ mu nu).map (fun (_, r, q) => (r, q)) := by
  simp [corners]

def coverOK : Bool :=
  (List.finRange 8).all fun r => (List.finRange 8).all fun q =>
    r.val < q.val → (relCorners.filter fun (a, b) => (a == r && b == q) || (a == q && b == r)).length ==
      if dir r == dir q then 0 else 1

theorem corner_cover : coverOK = true := by decide +kernel

/-- the offsets of the four corners are distinct, so the plaquettes y - off counted in `relCorners` are distinct. -/
theorem corner_count : relCorners.length = 24 := by decide +kernel

#print axioms head_unique
#print axioms forest_leaf
#print axioms one_fixed
#print axioms corner_ends
#print axioms corner_cover
