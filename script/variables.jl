module Vars

jets = mapreduce(vcat, 0:5) do i
    ["Jet_Px[$i]", "Jet_Py[$i]", "Jet_Pz[$i]", "Jet_E[$i]"]
end
tight_lepton = "TightLepton_" .* ["Px", "Py", "Pz", "E"] .* "[0]"
jet_csv = string.("Jet_CSV[", 0:5, "]")

scalar_features = [
    "Evt_CSV_avg_tagged",
    "Evt_CSV_avg",
    "Evt_M_minDrLepTag",
    "Evt_Pt_JetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Deta_JetsAverage",
    #"N_Jets",
    "Reco_JABDT_ttbar_Jet_CSV_whaddau2",
    "Reco_ttbar_toplep_m",
    "Reco_tHq_bestJABDToutput",
    "Reco_JABDT_tHq_abs_ljet_eta",
    "Reco_JABDT_tHq_Jet_CSV_hdau1",
    "Reco_JABDT_tHW_Jet_CSV_btop",
    #"memDBp",
]

all_features = [jets; tight_lepton; jet_csv; scalar_features]

weights = [
    "Weight_XS",
    #"Weight_btagSF",
    "Weight_CSV",
    "Weight_GEN_nom",
    "lumiWeight",
]

classes = [:ttH, :ttbb, :ttcc, :ttlf]


names_lola_out = [
    mapreduce(vcat, 0:5) do i
        ["Jet_m²[$i]"; "Jet_p_T[$i]"; "Jet_weighted_E[$i]"; string.("Jet_weighted_d", 1:4, "[$i]")]
    end
    ["Lepton_m²"; "Lepton_p_T"; "Lepton_weighted_E"; string.("Lepton_weighted_d", 1:4)]
]

tier1 = [
    "Evt_CSV_avg_tagged",
    "Evt_CSV_avg",
    "Evt_M_minDrLepTag",
    "Evt_Pt_JetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Deta_TaggedJetsAverage",
]
tier2 = [
    "Evt_Deta_JetsAverage",
]
tier3 = [
    "Reco_JABDT_ttbar_Jet_CSV_whaddau2",
    "Reco_ttbar_toplep_m",
    "Reco_tHq_bestJABDToutput",
    "Reco_JABDT_tHq_abs_ljet_eta",
    "Reco_JABDT_tHq_Jet_CSV_hdau1",
    "Reco_JABDT_tHW_Jet_CSV_btop",
]

end
