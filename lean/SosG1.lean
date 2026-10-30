import Sos4

/-- the Gram identity for the plaquettes 24 .. 47 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_gram_1 : (List.range' 24 24).all plaqOK = true := by decide +kernel
