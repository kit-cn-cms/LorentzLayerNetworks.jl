using Arrow, DataFrames#, StatsPlots
#pyplot()
using PyPlot
using Printf

include("variables.jl")
output_dir = "/work/sschaub/JuliaForHEP/run1"

layer_params = DataFrame(Arrow.Table(joinpath(output_dir, "layer_params.arrow")))
all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))

for ((kind,), df) in pairs(groupby(all_features, :kind))
    fig, axes = subplots(ncols=2, nrows=1, figsize=(10, 5))
    _df = select(df, :, :output_predicted_Hbb => (x -> ifelse.(x .> .5, :Hbb, :Zbb)) => :output_predicted)
    foreach(pairs(groupby(_df, :output_predicted, sort=true)), axes) do ((class,), df), ax
        idx_Hbb = df.output_expected .=== :Hbb#class
        ax.hist(
            [df[idx_Hbb, Symbol(:output_predicted_, class)], df[.!idx_Hbb, Symbol(:output_predicted_, class)]],
            weights = [df[idx_Hbb, :weights], df[.!idx_Hbb, :weights]] ./ sum(_df.weights),
            label=["true Hbb", "true Zbb"], stacked=true, #density=true,
            bins=20, rwidth=.8
        )
        ax.set_title("classified as $class")
        ax.set_xlabel("p($class)")
        ax.legend()
    end
    fig.suptitle(kind)
    fig.savefig(joinpath(output_dir, "output_features_hist_$kind.pdf"))

    confusion_mat = zeros(2, 2)
    for r in eachrow(df)
        i = r.output_expected === :Hbb ? 1 : 2
        j = r.output_predicted_Hbb > .5 ? 1 : 2
        confusion_mat[i, j] += r.weights_norm
    end
    fig, ax = subplots()
    img = ax.imshow(confusion_mat)
    for i in 1:2, j in 1:2
        ax.text(j-1, i-1, @sprintf("%.3g", confusion_mat[i, j]), ha=:center, va=:center)
    end
    ax.set_xlabel("predicted output")
    ax.set_xticks(0:1, [:Hbb,:Zbb])
    ax.set_ylabel("true output")
    ax.set_yticks(0:1, [:Hbb,:Zbb])
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
