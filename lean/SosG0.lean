import Sos4

/-- the Gram identity for the plaquettes 0 .. 23 (one of four parallel chunks, see SosAll.lean). -/
theorem sos_gram_0 : (List.range' 0 24).all plaqOK = true := by decide +kernel
