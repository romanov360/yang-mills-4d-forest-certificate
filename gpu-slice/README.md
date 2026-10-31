# GPU runs: the one-step slice bound in 4D SU(2) lattice gauge theory

This folder is a self-contained JAX port of a Monte Carlo study, with a launcher for Modal GPUs. The
launcher has **not been run on Modal yet**. Every check below was run on a CPU.

## What it is for

In the continuum programme of Bhattacharjee, Bhattacharya and Lockwood (BBL, 4Y-M, Zenodo
10.5281/zenodo.23822000), one open input is the **slice ratio**. It requires that correlations of the
gradient-flowed slice field at two fixed physical time separations keep a positive ratio as the lattice spacing
a → 0. If it holds, the continuum limit they construct is nontrivial.

The note `yang-mills-continuum/SLICE.md` in romanov360/big-bang-monotile proves (Proposition S, checked in Lean 4
+ Mathlib) that the slice ratio follows from a one-step condition:

    E_k = −log(Γ(1)/Γ(0)) / a   stays bounded in physical units as a → 0,

where Γ(n) is the connected correlator of the flowed slice field at a separation of n lattice time steps.

The CPU runs found the following.

- **d = 3:** E_k is flat over a factor 2.7 in a, while the unflowed one grows like 1/a.
- **d = 4:** only β = 2.3–2.5 was reachable, with a√σ = 0.37–0.19, and there the flow radius is barely a lattice
  spacing. The result is inconclusive: E_k/√σ = 4.4(3), 4.6(2), 5.9(3).

These runs decide the d = 4 case at a√σ ≈ 0.13, 0.09 and 0.06.

## Algorithm

- **Field.** SU(2) Wilson action on a periodic d = 4 lattice. Links are stored as unit quaternions.
- **Update.** One Kennedy–Pendleton heat-bath sweep, then 3 overrelaxation sweeps. Each sweep updates one link
  direction and one checkerboard parity at a time. Links of the same direction and parity share no plaquette, so
  each half-lattice updates at once.
- **Measurement**, every 2 sweeps:
  1. Run Lüscher's RK3 Wilson flow on the spatial links only (spatial plaquettes, all time slices at once).
  2. Stop at τσ = 0.045, 0.1 and 0.2 in physical units, so τ/a² grows like 1/a² along the tuning.
  3. At each stop, record the zero-momentum slice field Φ_t = Σ_x Σ_{i<j} (1 − ½ Re tr U_ij), one number per t.
- **Analysis.** The correlator over t gives Γ(0) and Γ(1), hence E_k, with jackknife errors over
  configurations. Independent chains are pooled.

Almost all GPU time goes to the flow, which takes 275–1222 RK3 steps per configuration at β = 2.85.

## Software

Python, JAX (`jax[cuda12]==0.11.2`), NumPy, SciPy and, for the launcher, `modal==1.6.3`.

- Every sweep and every flow segment is compiled with `jax.jit`.
- Neighbour shifts are `jnp.roll`.
- Random numbers come from JAX's counter-based keys, so a seed reproduces a run.
- The Kennedy–Pendleton rejection loop is a `lax.while_loop` that redraws only the rejected links. This makes it
  exact, not a fixed-trial approximation.
- The field runs in float32 by default. The slice sums are accumulated in the field's precision and written
  as float64. Use `--x64` for double precision throughout.

| file | what |
|---|---|
| `su2_slice_jax.py` | the simulation (CLI; `--help`), with checkpoint/resume |
| `analyse_jax.py` | E_k/√σ per flow time, jackknife errors, pools chains |
| `validate.py` | correctness checks and a benchmark; run it first on the target machine |
| `modal_app.py` | Modal launcher: `validate` and `production` entrypoints, results on the volume `su2-slice` |
| `reference/su2_slice.py` | the NumPy original, used by `validate.py` |
| `requirements.txt` | pinned versions |

## Validation (CPU, this repository; rerun on the GPU)

    python3 validate.py --physics --bench 16

| check | result |
|---|---|
| Kennedy–Pendleton: ⟨x₀⟩ = I₂(α)/I₁(α), α = 0.5, 4, 30 | agree within 5σ (400k samples each) |
| flow vs NumPy reference, random field, 15 RK3 steps, double precision | max relative difference 8·10⁻¹⁵ |
| flow gauge covariance | 1.4·10⁻¹⁴ |
| plaquette, 8⁴, float32: β = 2.3 / 2.5 | 0.60159(65) / 0.65250(45); literature 0.6024 / 0.6517 |
| E_k/√σ at β = 2.3, 8⁴, τσ = 0.045, 300 configurations | 4.03(26); CPU NumPy runs gave 4.41(26) and 3.87(17) |
| checkpoint and resume | a 12-measurement run resumed to 20 measurements |
| CPU timing, 16⁴ | 0.25 s per sweep, 0.15 s per RK3 flow step (NumPy: about 1.5 s per sweep) |

## Running

Locally (CPU or one GPU):

    pip install -r requirements.txt          # on a GPU machine: pip install "jax[cuda12]==0.11.2" numpy scipy
    python3 validate.py --physics --bench 32
    python3 su2_slice_jax.py --beta 2.7 --L 32 --T 32 --nmeas 300 --seed 1 --out out/b2.7_L32_s1
    python3 analyse_jax.py out/*.npy

On Modal:

    pip install modal==1.6.3 && modal setup
    modal run modal_app.py::validate                    # L4, a few minutes: checks + 32^4 benchmark
    modal run modal_app.py::validate --gpu H100         # benchmark on the production GPU
    modal run --detach modal_app.py::production         # the plan below, in parallel
    modal volume get su2-slice / ./out && python3 analyse_jax.py out/*.npy

## Plan and cost

Every run uses a physical box L·a√σ ≈ 2.9–3.2, 500 thermalisation sweeps, 2 sweeps between measurements, and
three flow times.

| run | lattice | a√σ | flow steps per config | configurations |
|---|---|---|---|---|
| β = 2.5 (overlap with CPU data) | 16⁴ | 0.188 | 142 | 500 |
| β = 2.6 | 24⁴ | 0.133 | 286 | 500 |
| β = 2.7 | 32⁴ | 0.091 | 604 | 2 × 300 |
| β = 2.85 | 48⁴ | 0.064* | 1222 | 2 × 200 |

\* At β = 2.85, a√σ = 0.064 is a two-loop extrapolation from β = 2.7. Measure it (string tension, or the flow
scale t₀) before quoting physical units.

**Cost.** The GPU speed is not measured yet; the validate entrypoint measures it. If an H100 runs this about
100× faster than the CPU timing above, the plan takes roughly:

- β = 2.5 and 2.6: about 1 h;
- β = 2.7: about 2 × 1.5 h;
- β = 2.85: about 2 × 8 h.

That is 20–25 GPU-hours, about $80–100 at Modal's ~$3.95/h for an H100. Two ways to cut it:

- drop τσ = 0.2 at β = 2.85 (the flow steps halve), or raise `--eps` after checking the step-size dependence;
- run β = 2.85 last, and only if β = 2.6–2.7 leave the question open.

## Reading the result

Proposition S needs E_k/√σ (flowed) to level off as a → 0, as it does in d = 3. The unflowed column should keep
growing like 1/a, which shows the observable can detect a divergence. If the flowed E_k keeps growing at
β = 2.7–2.85 for every flow time, the one-step route to the slice ratio fails in d = 4 for this observable.
The slice ratio itself could still hold, because Proposition S is only sufficient.
