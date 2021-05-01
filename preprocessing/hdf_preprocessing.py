import pandas as pd
import ROOT

def add_four_vectors(df, inputs, outputs):
    def four_vector(row):
        x = ROOT.Math.PtEtaPhiMVector(*row[inputs])
        return pd.Series([x.Px(), x.Py(), x.Pz(), x.E()], index=outputs)

    return pd.concat([df, df.apply(four_vector, axis=1)], axis=1)


for file in ["ttbb_lola.h5", "ttcc_lola.h5", "ttH_lola.h5", "ttlf_lola.h5"]:
    print("processing " + file)
    df = pd.read_hdf("/work/sschaub/h5files/ttH_sebastian_all_n_jets/" + file)
    df = df.query("N_Jets == 6")

    for i in range(6):
        inputs = ['Jet_Pt[{}]'.format(i), 'Jet_Eta[{}]'.format(i), 'Jet_Phi[{}]'.format(i), 'Jet_M[{}]'.format(i)]
        outputs = ['Jet_Px[{}]'.format(i), 'Jet_Py[{}]'.format(i), 'Jet_Pz[{}]'.format(i), 'Jet_E_[{}]'.format(i)]

        df = add_four_vectors(df, inputs, outputs)

    ####
    inputs = ['TightLepton_Pt[0]', 'TightLepton_Eta[0]', 'TightLepton_Phi[0]', 'TightLepton_M[0]']
    outputs = ['TightLepton_Px[0]', 'TightLepton_Py[0]', 'TightLepton_Pz[0]', 'TightLepton_E_[0]']

    df = add_four_vectors(df, inputs, outputs)

    df.to_hdf("/local/scratch/ssd/sschaub/h5files/ttH_sebastian-processed/" + file, key='data', mode='w')
