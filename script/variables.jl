module Vars

jets = mapreduce(vcat, 0:5) do i
    ["Jet_X[$i]", "Jet_Y[$i]", "Jet_Z[$i]", "Jet_T[$i]"]
end
tight_lepton = "TightLepton_" .* ["X", "Y", "Z", "T"] .* "[0]"
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

all_features = [jets; tight_lepton; jet_csv]

weights = [
    "Weight_XS",
    #"Weight_btagSF",
    "Weight_CSV",
    "Weight_GEN_nom",
    "lumiWeight",
]

classes = [:ttH, :ttbb, :ttcc, :ttlf]

end
