import Sos4

/-- the Gram identity for the plaquettes 48 .. 71 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_gram_2 : (List.range' 48 24).all plaqOK = true := by decide +kernel
