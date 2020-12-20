module Vars

variables = Dict{String,Vector{String}}()
variables["ge4j_ge3t"] = [

    # X boson stuff
    "RecoX_jet1_Eta",
    "RecoX_jet2_Eta",
    "RecoX_jet1_Phi",
    "RecoX_jet2_Phi",
    "RecoX_jet1_btagValue",
    "RecoX_jet2_btagValue",
    "RecoX_jet1_M",
    "RecoX_jet2_M",
    "RecoX_jet1_E",
    "RecoX_jet2_E",
    "RecoX_jet1_Pt",
    "RecoX_jet2_Pt",
    "RecoX_jet1_idx",
    "RecoX_jet2_idx",
    "N_Jets",
    "N_BTagsM",
    "RecoX_X_Pt",
    "RecoX_X_Eta",
    "RecoX_X_M",
    "RecoX_X_E",
    "RecoX_X_openingAngle",
    "RecoX_X_dPhi",
    "RecoX_X_dEta",
    "RecoX_X_dPt",
    "RecoX_X_dR",

    #"RecoX_X_dKin",
    "RecoX_X_btagAverage",

    # lepton variables
    "TightLepton_E_0",
    "TightLepton_Eta_0",
    "TightLepton_M_0",
    "TightLepton_Phi_0",
    "TightLepton_Pt_0",

    # boson-lepton variables
    "RecoX_jet1_dEta_lept",
    "RecoX_jet2_dEta_lept",
    "RecoX_jet1_dPhi_lept",
    "RecoX_jet2_dPhi_lept",
    "RecoX_jet1_dR_lept",
    "RecoX_jet2_dR_lept",
    "RecoX_X_dEta_lept",
    "RecoX_X_dPhi_lept",
    "RecoX_X_dR_lept",

    "RecoX_jet1_CvsL_deepJet",
    "RecoX_jet1_CvsB_deepJet",
    "RecoX_jet2_CvsL_deepJet",
    "RecoX_jet2_CvsB_deepJet",

]
variables["lorentzlayer"] = [
    #"N_Jets",
    #"N_BTagsM",

    "Jet_X[0]",
    "Jet_Y[0]",
    "Jet_Z[0]",
    "Jet_T[0]",
    "Jet_X[1]",
    "Jet_Y[1]",
    "Jet_Z[1]",
    "Jet_T[1]",
    "Jet_X[2]",
    "Jet_Y[2]",
    "Jet_Z[2]",
    "Jet_T[2]",
    "Jet_X[3]",
    "Jet_Y[3]",
    "Jet_Z[3]",
    "Jet_T[3]",
    "Jet_X[4]",
    "Jet_Y[4]",
    "Jet_Z[4]",
    "Jet_T[4]",
    "Jet_X[5]",
    "Jet_Y[5]",
    "Jet_Z[5]",
    "Jet_T[5]",
 ]

weights = [
    "Weight_XS",
    "Weight_btagSF",
    "Weight_GEN_nom",
    "lumiWeight",
]

end
