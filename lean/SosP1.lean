import Sos4

/-- positive definiteness for the certificates 12 .. 23 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_pd_1 : (List.range' 12 12).all pdPair = true := by decide +kernel
