using Arrow, DataFrames
using Statistics, Distributions, StatsBase
using Roots
using PyPlot, Printf, PyCall


include("variables.jl")
include("plot_styles.jl")
using .Vars: classes

measures = DataFrame()
basedir = "/work/sschaub/JuliaForHEP/feature_evaluation_0125/"

for feature in ["lola+" .* ["none"; Vars.scalar_features; "scalars11"]; "scalars2"]
    @show feature
    output_dir = joinpath(basedir, "$feature/")
    feature = replace!([feature], "lola+scalars11" => "lola+all scalars", "scalars2" => "only scalars")[]

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
            weights = Weights(df.weights ./ validation_split)# .* (300 / 41.5))
            bins = range(.25, 1; length=11)
            values = fit(Histogram, data, weights, bins).weights
            append!(hists, DataFrame(; bins=bins[1:end-1], values, node, true_class))
        end
        @show combine(groupby(hists, [:true_class, :node]), :values => sum)
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
        push!(measures, (; feature, σ_μ = (z2 - z1) / 2))

        μ = range(.5, 1.5; length=100)
        t = 2 .* (NLL.(μ) .- NLL(1))

        fig, ax = subplots()
        ax.plot(μ, t)
        ax.axhline(1; color=:grey, ls="--", lw=1)
        ax.axvline.([z1, z2]; color=:grey, ls="--", lw=1)
        fig.suptitle("Likelihood Fit ttH")
        ax.set_title("\\verb|$feature|"; fontsize=16)
        ax.set_xlabel(L"\mu")
        ax.set_xlim(.5, 1.5)
        ax.set_ylabel(L"-2\log(\mathcal L(\mu) / \mathcal L(1))")
        ax.legend(["\$\\mu = 1^{+$(@sprintf "%.3f" z2-1)}_{-$(@sprintf "%.3f" 1-z1)}\$"]; loc="upper center")
        annotate_cms(ax)
        display(fig)
        fig.savefig(joinpath(output_dir, "likelihood_ttH.pdf"))

        fig1, axs1 = subplots(ncols=2, nrows=2, figsize=(15, 12))
        foreach(
            pairs(groupby(hists, :node, sort=true)),
            axs1,
        ) do ((node,), df), ax1
            ax1.set_title("$node node")
            ax1.step(df.bins, df.bg; label="background")
            ax1.step(df.bins, df.bg + df.signal; label="total")
            ax1.legend()
            ax1.set_xlabel("p($node)")
            ax1.set_yscale(:log)
            ax1.set_ylim(minimum(filter(!=(0), df.bg)) * .8, maximum(df.bg + df.signal) * 2)
            ax1.set_ylabel("Events")
            annotate_cms(ax1)
        end
        fig1.suptitle("Binning \\verb|$feature|")
        fig1.tight_layout()
        display(fig1)
        fig1.savefig(joinpath(output_dir, "binning_ttH.pdf"))
    end
end

#replace!(measures.feature, "lola+scalars11" => "lola+all scalars", "scalars2" => "only scalars")

fig, ax = subplots(figsize=(11, 11))
x = axes(measures, 1)
ax.scatter(x, measures.σ_μ, color=[:red; fill(:blue, length(x)-3); :green; :orange])

ax.set_xticks(x)
ax.set_xticklabels(string.("\\verb|", measures.feature, "|"); rotation=90)
ax.set_ylabel(L"\sigma_\mu")
fig.suptitle("Standard Deviations of \$\\mu\$ for training with LoLa + X")
annotate_cms(ax)
fig.tight_layout()
fig.savefig(joinpath(basedir, "lola+x.pdf"))
display(fig)
