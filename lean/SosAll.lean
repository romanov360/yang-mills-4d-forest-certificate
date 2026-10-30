/-
The all-phase theorems, assembled from the chunks SosG0..3 and SosP0..3 (checked as separate files, in parallel).
-/
import SosG0
import SosG1
import SosG2
import SosG3
import SosP0
import SosP1
import SosP2
import SosP3

theorem sos_gram : plaqs.length = 96 ∧ (List.range 96).all plaqOK = true := by
  have h : List.range 96 = List.range' 0 24 ++ List.range' 24 24 ++ List.range' 48 24 ++ List.range' 72 24 := by
    decide
  refine ⟨plaqs_length, ?_⟩
  rw [h]
  simp only [List.all_append, sos_gram_0, sos_gram_1, sos_gram_2, sos_gram_3, Bool.and_self]

theorem sos_pd : sosCerts.length = 48 ∧ (List.range 48).all pdPair = true := by
  have h : List.range 48 = List.range' 0 12 ++ List.range' 12 12 ++ List.range' 24 12 ++ List.range' 36 12 := by
    decide
  refine ⟨certs_length, ?_⟩
  rw [h]
  simp only [List.all_append, sos_pd_0, sos_pd_1, sos_pd_2, sos_pd_3, Bool.and_self]

#print axioms sos_gram
#print axioms sos_pd
