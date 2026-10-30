import Sos4

/-- the Gram identity for the plaquettes 72 .. 95 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_gram_3 : (List.range' 72 24).all plaqOK = true := by decide +kernel
