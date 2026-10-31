"""Run su2_slice_jax.py on Modal GPUs.  Nothing here has been run on Modal yet.

Setup (once):  pip install modal==1.6.3 && modal setup

    modal run modal_app.py::validate                 # checks + benchmark on a cheap GPU (L4), a few minutes
    modal run modal_app.py::validate --gpu H100      # the same on the production GPU
    modal run --detach modal_app.py::production      # all production runs in parallel, one GPU each
    modal run --detach modal_app.py::production --only b2.6   # a subset (substring of the run name)
    modal volume get su2-slice / ./out               # download; then: python3 analyse_jax.py out/*.npy

Runs checkpoint every 10 measurements to the volume and resume after preemption or a retry.
"""
import os

import modal

GPU = os.environ.get("SU2_GPU", "H100")
HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("su2-slice")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda12]==0.11.2", "numpy==2.5.3", "scipy==1.18.1")
    .add_local_python_source("su2_slice_jax", "analyse_jax", "validate")
    .add_local_dir(os.path.join(HERE, "reference"), "/root/reference")
)
vol = modal.Volume.from_name("su2-slice", create_if_missing=True)

# The production plan, d = 4, physical box L a sqrt(sigma) ~ 3, flow times tau*sigma = 0.045, 0.1, 0.2.
# beta = 2.5 overlaps the CPU data (E_k = 5.87(33) at tau*sigma = 0.045) as a cross-check.
# beta = 2.85: a sqrt(sigma) = 0.064 is a two-loop extrapolation from beta = 2.7, to be replaced by a
# measured value (string tension, or r0 / t0 from the same runs) before quoting physical units.
TAUS = "0.045,0.1,0.2"
PLAN = {
    "b2.5_L16_s1": ["--beta", "2.5", "--L", "16", "--T", "16", "--nmeas", "500", "--seed", "1"],
    "b2.6_L24_s1": ["--beta", "2.6", "--L", "24", "--T", "24", "--nmeas", "500", "--seed", "1"],
    "b2.7_L32_s1": ["--beta", "2.7", "--L", "32", "--T", "32", "--nmeas", "300", "--seed", "1"],
    "b2.7_L32_s2": ["--beta", "2.7", "--L", "32", "--T", "32", "--nmeas", "300", "--seed", "2"],
    "b2.85_L48_s1": ["--beta", "2.85", "--L", "48", "--T", "48", "--nmeas", "200", "--seed", "1",
                     "--a-sqrt-sigma", "0.064"],
    "b2.85_L48_s2": ["--beta", "2.85", "--L", "48", "--T", "48", "--nmeas", "200", "--seed", "2",
                     "--a-sqrt-sigma", "0.064"],
}


@app.function(image=image, gpu=GPU, timeout=24 * 3600, volumes={"/out": vol},
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def run_one(name: str, argv: list):
    import su2_slice_jax as SJ
    args = SJ.parser().parse_args(argv + ["--tau-sigma", TAUS, "--ntherm", "500", "--out", f"/out/{name}"])
    SJ.run(args, on_checkpoint=vol.commit)
    vol.commit()
    return name


@app.function(image=image, gpu="L4", timeout=3600)
def validate_on(bench_L: int):
    import subprocess
    r = subprocess.run(["python3", "/root/validate.py", "--physics", "--bench", str(bench_L)],
                       capture_output=True, text=True)
    return r.stdout + r.stderr


@app.local_entrypoint()
def validate(gpu: str = "L4", bench: int = 32):
    print(validate_on.with_options(gpu=gpu).remote(bench))


@app.local_entrypoint()
def production(only: str = ""):
    runs = [(k, v) for k, v in PLAN.items() if only in k]
    print("launching", [k for k, _ in runs])
    for res in run_one.starmap(runs, return_exceptions=True):
        print("finished:", res)
