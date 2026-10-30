import Sos4

/-- positive definiteness for the certificates 24 .. 35 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_pd_2 : (List.range' 24 12).all pdPair = true := by decide +kernel
