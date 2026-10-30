import Sos4

/-! Positive definiteness of the Gram matrices of the certificates 0 .. 5 (see SosAll.lean). -/

theorem pd_D0 : pdCheck sosD0 = true := by decide +kernel
theorem pd_O0 : pdCheck sosO0 = true := by decide +kernel
theorem pd_D1 : pdCheck sosD1 = true := by decide +kernel
theorem pd_O1 : pdCheck sosO1 = true := by decide +kernel
theorem pd_D2 : pdCheck sosD2 = true := by decide +kernel
theorem pd_O2 : pdCheck sosO2 = true := by decide +kernel
theorem pd_D3 : pdCheck sosD3 = true := by decide +kernel
theorem pd_O3 : pdCheck sosO3 = true := by decide +kernel
theorem pd_D4 : pdCheck sosD4 = true := by decide +kernel
theorem pd_O4 : pdCheck sosO4 = true := by decide +kernel
theorem pd_D5 : pdCheck sosD5 = true := by decide +kernel
theorem pd_O5 : pdCheck sosO5 = true := by decide +kernel
