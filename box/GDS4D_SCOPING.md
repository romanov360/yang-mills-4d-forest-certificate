# Can BBL's finite-box criterion be certified by computer in d = 4?

This is the scoping that `NEXT.md` ("the bridge to weak coupling") asked for. **Short answer:** in two dimensions,
yes. It is done here, with explicit certified boxes. In four dimensions, not with couplings of the kind BBL
use. The obstruction is the size of the box, not compute.

## The condition

BBL's Theorem G (4Y-M `strong-coupling/prd`, Definition "gds") asks for couplings of the box measure under two
boundary conditions that differ at one link y. It needs bounds κ(y,b) on the disagreement probability at each
b ∈ V, uniform over all boundary data, such that for every direction ν

    K_V(ν) = Σ_{y ∈ ∂V, dir y = ν} Σ_{b ∈ V} κ(y,b)  ≤  (1 − ε) n_min.

Read differently: **the total influence of one boundary link on the whole box, Σ_b κ(y,b), must be below
n_min/|∂V_ν|**. That ratio is a pure counting quantity.

## d = 2: done (`gds2d_certified.py`)

BBL's Proposition "2d" is an existence proof (L = 8s² with s chosen so that η_{s²} < 1/(80s²)). Their coupling
integrates out the interior of an s × s square next to y, which is exact by disk integration. Evaluated in ball
arithmetic (python-flint), with their sharper total-variation bound, it gives these smallest certified squares:

| β_W | 0.5 | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| certified L | 10 | 26 | 36 | 55 | 85 | 139 | 209 |
| BBL's existence choice 8s² | 32 | 72 | 128 | 200 | 200 | 392 | 512 |

The tail of the character series is bounded with I_{n+1}/I_n ≤ x/(2(n+1)), so these are rigorous. Two
dimensions has no ultraviolet problem and the theory is gapped at every β, so the table is a pipeline check, not
new physics.

## d = 4: the counting (`gds4d_counting.py`)

Take V = the links of {0..L−1}⁴, with ∂V its plaquette shell. Exact counts:

| L | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| n_min / \|∂V_ν\| | 0.33 | 0.46 | 0.71 | 0.97 | 1.47 | 1.97 |

The ratio grows like L/8. So, for this V:

- **Single-link influence already exceeds what small boxes allow.** Any coupling has κ(y,b) of order the
  single-link influence for the ~18 links b sharing a plaquette with y. At β_W ≈ 0.35–0.5 that sums to an O(1)
  total influence, above the budget 0.33–0.97 of boxes with L ≤ 8. Small boxes (L = 2, 3) cannot satisfy the
  condition for V of this shape. Another shape of V changes the constants, not the scaling.
- **Neighbourhood couplings need huge boxes.** BBL's 2D coupling, and the plan in `NEXT.md`, integrate out a
  neighbourhood D_y of m links where the coupling gives up (κ ≤ 1), and bound κ ≤ η elsewhere. They need
  L ≳ 8m and η ≲ 1/(32 L⁴):

  | m | 1 | 6 | 18 | 50 |
  |---|---|---|---|---|
  | L ≳ | 8 | 48 | 144 | 400 |
  | links in V | 1.6·10⁴ | 2·10⁷ | 2·10⁹ | 10¹¹ |
  | η ≲ | 8·10⁻⁶ | 6·10⁻⁹ | 7·10⁻¹¹ | 10⁻¹² |

  η must hold uniformly over all boundary data, for the marginal of all the other links of V.
- **In 2D the same construction is cheap because two facts hold there.** Disk integration is exact, and η decays
  like a^{s²}, the plaquette character coefficient to the power of the area. In 4D no exact integration exists.
  A rigorous η for a 10⁴–10⁹-link marginal would have to come from a cluster or polymer expansion. Those converge
  only at strong coupling, where single-link and plaquette-block methods already give the gap (β_W ≤ 0.345 for
  SU(2)).

## Consequence for the research plan

Certifying GDS in 4D at β_W between 0.35 and the crossover (~2.2) is not a finite computation with couplings of
this kind. It needs a new idea. That would be either a coupling whose near-boundary disagreements are well below
the single-link influence (a genuinely collective move), or an analytic bound for η beyond the cluster-expansion
regime. Neither is in sight. The strong-coupling computations that remain in reach are the small ones listed in
`NEXT.md`:

- certifying BBL's Remark 16 (SU(2) from 0.345 to about 0.357);
- the SU(3) block step (0.76 to about 0.78);
- a certificate for h_F below 14.38.

Each is worth a few percent in β.

This corrects the suggestion in the session summary that a certified finite box might be the best next
direction: in 4D it is not within reach.
