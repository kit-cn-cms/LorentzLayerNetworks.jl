using Arrow, DataFrames#, StatsPlots
#pyplot()
using PyPlot
using Printf

include("variables.jl")
output_dir = "/work/sschaub/JuliaForHEP/run1"

layer_params = DataFrame(Arrow.Table(joinpath(output_dir, "layer_params.arrow")))
all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))

fig1, ax1 = subplots()
linestyles = (train = "-", test = "--", validation="-.")
for ((kind,), df) in pairs(groupby(all_features, :kind))
    foreach(pairs(groupby(df, :output_expected, sort=true)), [:C0, :C1]) do ((class,), df), color
        ax1.hist(
            df.output_predicted_Hbb;
            label="true $class ($kind)", density=true,
            bins=30, histtype=:step, color, ls=linestyles[kind],
        )
    end

    confusion_mat = zeros(2, 2)
    for r in eachrow(df)
        i = r.output_expected === :Hbb ? 1 : 2
        j = r.output_predicted_Hbb > .5 ? 1 : 2
        confusion_mat[i, j] += r.weights
    end
    confusion_mat ./= sum(confusion_mat, dims=2)
    fig, ax = subplots()
    img = ax.imshow(confusion_mat)
    for i in 1:2, j in 1:2
        ax.text(j-1, i-1, @sprintf("%.3g", confusion_mat[i, j]), ha=:center, va=:center)
    end
    ax.set_xlabel("predicted output")
    xticks(0:1, [:Hbb,:Zbb])
    ax.set_ylabel("true output")
    yticks(0:1, [:Hbb,:Zbb])
    fig.colorbar(img)
    fig.suptitle(kind)
    fig.savefig(joinpath(output_dir, "confusion_matrix_$kind.pdf"))

    fig, axes = subplots(ncols=6, nrows=8, figsize=(25, 25))
    foreach(Vars.variables["ge4j_ge3t"], axes) do i, ax
        ax.hist(
            df[:, i],
            bins=20, rwidth=.8, density=true,
        )
        ax.set_title("feature $i")
    end
    fig.suptitle(kind)
    fig.savefig(joinpath(output_dir, "input_features_hist_$kind.pdf"))


    fig, axes = subplots(ncols=6, nrows=8, figsize=(25, 25))
    foreach(Vars.variables["ge4j_ge3t"], axes) do i, ax
        ax.hist(
            df[:, i * "_norm"],
            bins=20, rwidth=.8, density=true,
        )
        ax.set_title("feature $i")
    end
    fig.suptitle(kind)
    fig.savefig(joinpath(output_dir, "input_features_norm_hist_$kind.pdf"))
end

ax1.set_xlabel("p(Hbb)")
ax1.legend()
fig1.suptitle("Predicted Hbb")
fig1.savefig(joinpath(output_dir, "output_features_hist.pdf"))
