using Arrow, DataFrames
using PyPlot

include("variables.jl")
include("plot_styles.jl")
using .Vars: classes

basedir = "/work/sschaub/JuliaForHEP/final_plotting/"
output_dir = joinpath(basedir, "lola+all_1")
all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))

fig, axs = subplots(4, 4, figsize=(30, 30))

for (ax, feature) in zip(permutedims(reshape(axs, 4, 4)), Vars.scalar_features)
    #ax.hist(all_features[!, feature], bins=30)
    df = groupby(all_features, :output_expected; sort=true)
    ax.hist([i[!, feature] for i in df],
        weights=[i.weights for i in df],
        bins=30, stacked=true, label=first.(keys(df)))
    ax.set_title("\\verb|$feature|\n")
    ax.legend()
    annotate_cms(ax)
end

fig.suptitle("Distributions of Scalar Features")
fig.tight_layout()
display(fig)
fig.savefig(joinpath(output_dir, "scalar_features.pdf"))

#=
fig, axs = subplots(7, 15, figsize=(30, 30))
for (ax, feature) in zip(permutedims(reshape(axs, 4, 4)), Vars.scalar_features)
    #ax.hist(all_features[!, feature], bins=30)
    df = groupby(all_features, :output_expected; sort=true)
    ax.hist([i[!, feature] for i in df],
        weights=[i.weights for i in df],
        bins=30, stacked=true, label=first.(keys(df)))
    ax.set_title("\\verb|$feature|")
    ax.legend()
end
fig.suptitle("FOo")
fig.tight_layout()
display(fig)
=#
