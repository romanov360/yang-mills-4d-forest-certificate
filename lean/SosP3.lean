import Sos4

/-- positive definiteness for the certificates 36 .. 47 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_pd_3 : (List.range' 36 12).all pdPair = true := by decide +kernel
