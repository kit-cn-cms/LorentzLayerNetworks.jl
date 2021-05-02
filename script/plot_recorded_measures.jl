using Arrow, DataFrames
using PyPlot

include("plot_styles.jl")

basedir = "/work/sschaub/JuliaForHEP/final_plotting/"
output_dir = joinpath(basedir, "lola+all_1")
recorded_measures = DataFrame(Arrow.Table(joinpath(output_dir, "recorded_measures.arrow")))

fig, ax = subplots(figsize=(12, 9))
ax.plot([recorded_measures.train_loss recorded_measures.test_loss recorded_measures.validation_loss])
ax.set_title("Loss Values")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(["Training Loss", "Testing Loss", "Validation Loss"])

fig.savefig(joinpath(output_dir, "losses.pdf"))
