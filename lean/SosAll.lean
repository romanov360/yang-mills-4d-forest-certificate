/-
The all-phase theorems, assembled from one kernel theorem per plaquette (SosG0..7) and per Gram matrix (SosP0..7).
The files are checked separately, in parallel, since a single `decide` over everything runs out of memory.
-/
import SosG0
import SosG1
import SosG2
import SosG3
import SosG4
import SosG5
import SosG6
import SosG7
import SosP0
import SosP1
import SosP2
import SosP3
import SosP4
import SosP5
import SosP6
import SosP7

theorem sos_gram : plaqs.length = 96 ∧ (List.range 96).all plaqOK = true := by
  refine ⟨plaqs_length, ?_⟩
  rw [show List.range 96 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95] from rfl]
  simp only [List.all_cons, List.all_nil, Bool.and_true, plaq_0, plaq_1, plaq_2, plaq_3, plaq_4, plaq_5, plaq_6, plaq_7, plaq_8, plaq_9, plaq_10, plaq_11, plaq_12, plaq_13, plaq_14, plaq_15, plaq_16, plaq_17, plaq_18, plaq_19, plaq_20, plaq_21, plaq_22, plaq_23, plaq_24, plaq_25, plaq_26, plaq_27, plaq_28, plaq_29, plaq_30, plaq_31, plaq_32, plaq_33, plaq_34, plaq_35, plaq_36, plaq_37, plaq_38, plaq_39, plaq_40, plaq_41, plaq_42, plaq_43, plaq_44, plaq_45, plaq_46, plaq_47, plaq_48, plaq_49, plaq_50, plaq_51, plaq_52, plaq_53, plaq_54, plaq_55, plaq_56, plaq_57, plaq_58, plaq_59, plaq_60, plaq_61, plaq_62, plaq_63, plaq_64, plaq_65, plaq_66, plaq_67, plaq_68, plaq_69, plaq_70, plaq_71, plaq_72, plaq_73, plaq_74, plaq_75, plaq_76, plaq_77, plaq_78, plaq_79, plaq_80, plaq_81, plaq_82, plaq_83, plaq_84, plaq_85, plaq_86, plaq_87, plaq_88, plaq_89, plaq_90, plaq_91, plaq_92, plaq_93, plaq_94, plaq_95]

theorem sos_pd : sosCerts.all (fun (PD, PO) => pdCheck PD && pdCheck PO) = true := by
  simp only [sosCerts, List.all_cons, List.all_nil, Bool.and_true,
    pd_D0, pd_O0, pd_D1, pd_O1, pd_D2, pd_O2, pd_D3, pd_O3, pd_D4, pd_O4, pd_D5, pd_O5, pd_D6, pd_O6, pd_D7, pd_O7, pd_D8, pd_O8, pd_D9, pd_O9, pd_D10, pd_O10, pd_D11, pd_O11, pd_D12, pd_O12, pd_D13, pd_O13, pd_D14, pd_O14, pd_D15, pd_O15, pd_D16, pd_O16, pd_D17, pd_O17, pd_D18, pd_O18, pd_D19, pd_O19, pd_D20, pd_O20, pd_D21, pd_O21, pd_D22, pd_O22, pd_D23, pd_O23, pd_D24, pd_O24, pd_D25, pd_O25, pd_D26, pd_O26, pd_D27, pd_O27, pd_D28, pd_O28, pd_D29, pd_O29, pd_D30, pd_O30, pd_D31, pd_O31, pd_D32, pd_O32, pd_D33, pd_O33, pd_D34, pd_O34, pd_D35, pd_O35, pd_D36, pd_O36, pd_D37, pd_O37, pd_D38, pd_O38, pd_D39, pd_O39, pd_D40, pd_O40, pd_D41, pd_O41, pd_D42, pd_O42, pd_D43, pd_O43, pd_D44, pd_O44, pd_D45, pd_O45, pd_D46, pd_O46, pd_D47, pd_O47]

#print axioms sos_gram
#print axioms sos_pd
