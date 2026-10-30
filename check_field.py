#!/usr/bin/env python3
"""Recompute the lower bound h_F >= 13.974 from the saved SU(2) field on the L = 2 torus (floating point)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import torus3d as t  # noqa: E402
import forest4d as f4  # noqa: E402

T = t.Torus(2, d=4, pattern=f4.PAT4, P=2)
Q = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/bestQ_L2_s2_0.npy")
print("unitarity error", max(np.abs(q @ q.conj().T - np.eye(2)).max() for q in Q))
print("lambda_max of the gauge-fixed Hessian", t.lam(T, Q))
