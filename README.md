# LorentzLayerNetworks.jl

Julia implementation of Lorentz layer networks inspired by https://arxiv.org/abs/1707.08966

## Files

#### `script`: all scripts used for variable extraction, model creation, training, plotting, etc.

Start Julia with `julia --project=script` to automatically get correct versions of all dependencies required.

- `run.jl`: main entry point for network training. Change `basedir` to location of the HDF5 files and `output_dir` inside the call to `save_model` to specify output location.
- `variables.jl`: all relevant variables for trining and plotting
- `plots.jl`: plot measures of the network. (change variable `basedir` to location of model output)
- `likelihood.jl`: perform likelihood analyses and plot results.
- `plot_input_dists.jl`/`plot_recorded_measures.jl`: plot input distributions/recorded measures

#### `src`: implementation of the LLN and utils

Package should be loaded automatically if specifying `script` as the project. Can be imported with `using JuliaForHEP`.

#### `preprocessing`

- `hdf_preprocessing.py`: quick and dirty Python script using Pandas + PyROOT, which extracts Cartesian four momenta from detector coordinates and filters out events with != 6 jets.
