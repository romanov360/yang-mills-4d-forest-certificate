# Box criterion probe (Monte Carlo, not a proof)

`gds_mc.py` estimates r = K_V(ν)/n_min for BBL's gauge Dobrushin–Shlosman condition (Definition def:gds,
Theorem G) for SU(2). The coupling of the two box measures is the stationary law of two heat-bath chains whose link
updates are coupled maximally, with the interior gauge fixed by an axial forest. The supremum over boundary
conditions is replaced by a maximum over two kinds of boundary data (Haar random, and identity with the link y
flipped to −1). GDS needs r < 1.

| d | box side L | β_W | r | |
|---|---|---|---|---|
| 4 | 2 (8-link star) | 0.30 | 1.39 | fails |
| 4 | 3 (96 links) | 0.15 | 0.32 | passes |
| 4 | 3 | 0.30 | 1.48 | fails |
| 4 | 3 | 0.45 | 10.1 | fails |
| 2 | 4 | 1 | 0.45 | passes |
| 2 | 4 | 2 | 2.13 | fails |
| 2 | 4 | 4 | 6.35 | fails |

R = 500 replicas in 4D, R = 1000 in 2D, with 100 measured sweeps (`logs/`).

## Reading

The crossing in 4D is between 0.15 and 0.30, below BBL's proved 0.345. Going from L = 2 to L = 3 did not help at
0.30. A coupling built from link-by-link maximal couplings only propagates single-site influences, so it reproduces
Dobrushin ranges and cannot exploit the box. Their 2D proof (Prop. prop:2d) works because it couples the
*marginal* of the far part of the box maximally, after integrating out a sub-square exactly.

So the next step is not more compute. It is a better coupling: integrate out the links near y exactly (or by
character expansion) and couple the remaining marginal maximally, as in the 2D proof, then measure r again.
