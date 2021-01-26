using Arrow, DataFrames
using Statistics, Distributions, StatsBase
using Roots
using PyPlot, Printf

include("variables.jl")
using .Vars: classes

all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))
transform!(all_features,
    AsTable(r"output_predicted_.+") => ByRow(x -> Symbol(split(string(argmax(x)), '_')[end])) => :prediction,
)
validation_split = .75 * .2

for ((kind,), df) in pairs(groupby(all_features, :kind))
    kind === :validation || continue

    hists = DataFrame()
    for ((node, true_class), df) in pairs(groupby(df, [:prediction, :output_expected], sort=true))
        data = df[!, Symbol(:output_predicted_, node)]
        weights = Weights(df.weights ./ validation_split)
        bins = range(.25, 1; length=11)
        values = fit(Histogram, data, weights, bins).weights
        append!(hists, DataFrame(; bins=bins[1:end-1], values, node, true_class))
    end
    hists = groupby(hists, [:bins, :node])
    hists = combine(hists) do x
        sig = x.true_class .=== :ttH
        (signal=round(Int, sum(x[sig, :values])), bg=round(Int, sum(x[Not(sig), :values])))
    end
    #filter!([:signal, :bg] => (s, b) -> s + b > 0, hists)
    k = hists.signal .+ hists.bg
    λ(μ) = μ .* hists.signal .+ hists.bg
    NLL(μ) = -sum(logpdf.(Poisson.(λ(μ)), k))
    z1, z2 = find_zeros(μ -> 2(NLL(μ) - NLL(1)) - 1, 0, 2)

    μ = range(.5, 1.5; length=100)
    t = 2 .* (NLL.(μ) .- NLL(1))
    fig, ax = subplots()
    ax.plot(μ, t)
    ax.axhline(1; color=:grey, ls="--")
    ax.axvline.([z1, z2]; color=:grey, ls="--")
    ax.set_title("ttH")
    ax.set_xlabel(L"\mu")
    ax.set_ylabel(L"-2\log(L(\mu) / L(1))")
    ax.legend(["\$\\mu = 1^{+$(@sprintf "%.3f" z2-1)}_{-$(@sprintf "%.3f" 1-z1)}\$"])
    display(fig)
    fig.savefig(joinpath(output_dir, "likelihood_ttH.pdf"))
end
