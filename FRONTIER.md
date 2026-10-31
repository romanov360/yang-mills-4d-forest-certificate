# Yang–Mills: frontier, status of the four open items, next steps

Written 2026-10-31 against *Mass Gap of Four-Dimensional Lattice Yang–Mills Theory for Every Compact Simple Group*
(Bhattacharjee, Bhattacharya, Lockwood, Romanov, 30 Oct 2026; "the paper"), this repository and
`romanov360/yang-mills-3d-forest-gauge`, and a web search done the same day.
Items marked *(unverified)* come from secondary sources.

## 1. What was done here

**[block-su2/SHARPENED.md](block-su2/SHARPENED.md): the SU(2) mass gap with plaquette blocks extends from
β_W ≤ 0.345 to β_W ≤ 0.351.** At the old endpoint 0.345, the gap bound rises from 0.00137 to 0.00987 (×7.2). At
β_W = 1/3 it rises from 0.0224 to 0.0294.

This follows the same pattern as the Hessian certificate (29/2 → 14.420001):

1. Two pen-and-paper sharpenings of the paper's Lemma 14:
   - a Poincaré inequality in place of sd ≤ 1;
   - interpolating exponents rather than fields, which removes the Hessian at the centre of the ball.
2. Arb ball arithmetic on 200 000 radius cells for the analytic constants.
3. A Lean-kernel check of the final rational arithmetic.

The same program reproduces every constant of the paper's Theorem 3.

## 2. The four items: honest status

| item | status | what would move it |
|---|---|---|
| **1. Fibres with no calm field** | As stated, "every fibre has a calm field" is false: a frustrated fibre is a counterexample, so no small-field bound can hold on every fibre. The usable form is "bad fibres are rare and harmless". | Disagreement percolation: combine the cluster bounds (Thm 131, Cor 132) with the fibre-wise decay (Thm 122, Cor 128). This is a real proof project at finite β, and probably the most tractable of the four. |
| **2. Cube estimate (D)** | Open. Theorem 69: (D) fails at a fixed cube side. The Gaussian benchmarks (Props 62–63) show the forest gauge alone fails. Block averages (Sec. XII) give D^c(n, α) only for n ≥ 2·10¹⁵/α (Thm 90), or n ≥ 10⁸²⁷/α at second order (Thm 100). Floating point says n = 64 suffices. | Sharpen the constants of Thm 90 and Cor 98 from 10¹⁴–10¹⁵ toward the floating-point 24–61. This is the same kind of work as §1, but the Combes–Thomas/Agmon routes I tried stall at n ~ 10⁵–10⁹. Composition (Remark 101) is the real gap. |
| **3. Continuum existence (E)** | Only the easy half is proved (Thm 10.6). | The Bałaban regime is needed. Nothing at fixed lattice spacing (this note included) touches it. |
| **4. Physical gap** | It reduces to (D) + (E), via Thms 45–46. | — |

Items 2–4 are the Clay problem. The "pessimistic" earlier assessment is, on the mathematics, accurate. What it
undersold is that the finite-β, computer-assisted constants can still move, and each move is checkable. §1 is one such
move.

## 3. Frontier (AI + formal methods, 2025–2026)

- **Douglas–Rajasekaran, Lean 4 `lgt` library.** It proves a U(n) lattice gauge theory strong-coupling mass gap,
  β < 1/(32n(d−1)), via the Dobrushin condition (Chatterjee's lecture notes, Ch. 16), using only the standard axioms.
  For SU(2) in d = 4 that is β_W tiny compared with this line of work (1/3 → 0.345 → 0.351).
  [Reservoir: mrdouglasny/LGT](https://reservoir.lean-lang.org/@mrdouglasny/LGT)
- **AI-assisted formalisation of QFT / Osterwalder–Schrader axioms in Lean.**
  [arXiv:2603.15770](https://arxiv.org/pdf/2603.15770)
- **Douglas, Seiberg–Witten theory in Lean** (formalised with explicit axioms).
  [arXiv:2607.06379](https://arxiv.org/abs/2607.06379)
- **lean-dojo `problems`.** A formal statement of the Millennium problem.
  [Reservoir: lean-dojo/problems](https://reservoir.lean-lang.org/@lean-dojo/problems)
- **OpenAI's October 2026 claims** (Kakeya, ETH, Yang–Mills) *(unverified)*. Third-party write-ups do not show a
  continuum Yang–Mills result.
  [metirai](https://www.metirai.com/blog/openai-kakeya-eth-yang-mills-claims-verification-2026),
  [syntaxandsignal](https://syntaxandsignal.tech/blog/openai-lean-formal-proofs-kakeya-conjecture-disproof-eth-yang-mills-2026.md)
- **Shen–Zhu–Zhu, Bakry–Émery curvature for lattice Yang–Mills.** The baseline improved by the forest gauge and the
  14.42 certificate. [arXiv:2204.12737](https://arxiv.org/pdf/2204.12737)
- **Related recent lattice work.** [arXiv:2509.03513](https://arxiv.org/html/2509.03513v2)

## 4. Ranked next steps (incremental, certifiable)

1. **SU(3), Theorem 7 (0.76 → ~0.77).** Lemma 36 has the same structure as Lemma 14, so both changes of §1 carry over
   with Lemma 38's constants:
   - the linear part of C uses max ‖∇g‖ on 5NβK instead of 5Nβ·τ;
   - the cubic part gains e^w/γ-type Poincaré factors.

   The paper's floating-point ceiling is about 0.78. This reuses the 2341-box machinery that already exists.
2. **SU(2), close the rest of Remark 18 (0.351 → ~0.355).** Replace the two-site Dobrushin step (d₁ ≤ sε + Cd₃) by a
   bound on the actual y₁–y₃ correlation, for example with a Brascamp–Lieb or H⁻¹ bound on (S³)², which is
   certifiable with the same Arb machinery.
3. **SU(2), larger blocks (past ~0.366).** The plaquette block cannot pass about 0.366: its sensitivity is
   (1+β)/4 + O(β²) at b = 0, as shown in SHARPENED.md. The next block is the set of free links of a line-field cube.
   The sensitivity is no longer two-site, so it needs a certified finite-dimensional covariance bound. That is the
   finite-box GDS item of [NEXT.md](NEXT.md).
4. **Formal bridge.** Port Lemma 13, Lemma 14′ and Proposition 17 into Lean/Mathlib on top of `lgt`'s Dobrushin
   framework, with the numeric constants imported from Arb as here. That would give a fully formal SU(2) gap for
   β_W ≤ 1/3 (Theorem 1), and then for 0.351. That is about 60× the coupling range formalised so far, and it is the
   kind of verified-AI result the frontier currently rewards.
5. **Item 1 reformulated** (rare bad fibres), by disagreement percolation. This is pen-and-paper, and the biggest step
   toward the physics.
6. **(D) constants.** Make the strip width of Theorem 96 explicit (10⁻³⁶ now) by a certified local eigenvalue
   computation on the few geometry types it actually needs. This is the most direct certificate-style attack on the
   n ≥ 10⁸²⁷ figure.
