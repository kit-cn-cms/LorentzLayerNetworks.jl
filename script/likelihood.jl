using Arrow, DataFrames
using Statistics, Distributions, StatsBase
using PyPlot

include("variables.jl")
using .Vars: classes

all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))
transform!(all_features,
    AsTable(r"output_predicted_.+") => ByRow(x -> Symbol(split(string(argmax(x)), '_')[end])) => :prediction,
)
validation_split = .75 * .2

for ((kind,), df) in pairs(groupby(all_features, :kind))
    kind === :validation || continue

    for ((true_class,), df) in pairs(groupby(df, :output_expected, sort=true))
        hists = DataFrame()
        for ((class_predicted,), df) in pairs(groupby(df, :prediction, sort=true))

            data = df[!, Symbol(:output_predicted_, class_predicted)]
            weights = Weights(df.weights ./ validation_split)
            bins = range(.25, 1; length=11)
            hists[:, class_predicted] = fit(Histogram, data, weights, bins).weights
        end
        k = map(x -> round(Int, sum(x)), eachrow(hists))
        @show true_class k
        λ(μ) = μ .* hists[!, true_class] .+ sum.(eachrow(hists[!, Not(true_class)]))
        NLL(μ) = -sum(logpdf.(Poisson.(λ(μ)), k))

        μ = range(.75, 1.25; length=100)
        t = NLL.(μ) .- NLL(1)
        fig, ax = subplots()
        ax.plot(μ, t)
        ax.set_title(true_class)
        ax.set_xlabel(L"\mu")
        ax.set_ylabel(L"-\log(L(\mu) / L(1))")
        display(fig)
        fig.savefig(joinpath(output_dir, "likelihood_$true_class.pdf"))
    end
end
