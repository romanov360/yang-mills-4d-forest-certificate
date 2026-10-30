import Sos4

/-- positive definiteness for the certificates 0 .. 11 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_pd_0 : (List.range' 0 12).all pdPair = true := by decide +kernel
