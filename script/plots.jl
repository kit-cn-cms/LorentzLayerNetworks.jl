using Arrow, DataFrames#, StatsPlots
#pyplot()
using PyPlot

include("variables.jl")
output_dir = "/work/sschaub/JuliaForHEP/run1"

layer_params = DataFrame(Arrow.Table(joinpath(output_dir, "layer_params.arrow")))
all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))

for ((kind,), df) in pairs(groupby(all_features, :kind))
    fig, axes = subplots(ncols=2, nrows=1, figsize=(10, 5))

    foreach(pairs(groupby(df, :output_expected)), axes) do ((class,), df), ax
        @show class
        ax.hist(
            [df.output_predicted_Hbb, df.output_predicted_Zbb],
            label=["p(Hbb)", "p(Zbb)"], stacked=true,
            density=true, bins=20, rwidth=.8
        )
        ax.set_title("$class events")
        ax.legend()
    end
    fig.suptitle(kind)
    display(fig)
    fig.savefig(joinpath(output_dir, "output_features_hist_$kind.pdf"))
end
