using Arrow, DataFrames
using PyPlot
using JuliaForHEP

include("variables.jl")
include("plot_styles.jl")
using .Vars: classes

basedir = "/work/sschaub/JuliaForHEP/final_plotting/"
output_dir = joinpath(basedir, "lola+all_1")
all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))

fig1, axs1 = subplots(4, 2, figsize=(18, 30))
fig2, axs2 = subplots(3, 2, figsize=(18, 22.5))
df = groupby(all_features, :output_expected; sort=true)
for (ax, feature) in zip(permutedims([axs1; axs2]), Vars.scalar_features)
    ax.hist([i[!, feature] for i in df],
        weights=[i.weights for i in df],
        bins=30, stacked=true, label=first.(keys(df)))
    unit = any(occursin(feature), ["Evt_M", "Evt_Pt", "toplep_m"]) ? " in GeV" : ""
    ax.set_xlabel("\\verb|$feature|$unit")
    ax.set_ylabel("Events/Bin")
    ax.set_yscale(:log)

    leg = ax.legend()
    if feature in Vars.scalar_features[[end-3; end-1:end]]
        annotate_cms(ax; sel_pos=(.25, .95))
    else
        annotate_cms(ax; sel_pos=(.05, .95))
    end
end
axs2[end].axis(:off)
for (i, fig) in enumerate([fig1, fig2])
    fig.suptitle("Distributions of Scalar Features ($i)")
    fig.tight_layout()
    display(fig)
    fig.savefig(joinpath(output_dir, "scalar_features_$i.pdf"))
end

_all_features = let
    basedir = "/local/scratch/ssd/sschaub/h5files/ttH_sebastian-processed/"
    h5files = joinpath.(basedir, ["ttH", "ttbb", "ttcc", "ttlf"] .* "_lola.h5")
    names = [
        mapreduce(vcat, 0:5) do i
            ["Jet_Pt[$i]", "Jet_Eta[$i]", "Jet_Phi[$i]", "Jet_M[$i]"]
        end
        "TightLepton_" .* ["Pt", "Eta", "Phi", "M"] .* "[0]"
        "Weight_XS"
    ]
    df = DataFrame(extract_cols(h5files, names)', names)
    [df all_features]
end

fig, axs = subplots(2, 2, figsize=(18, 16))
for (ax, feature, unit) in zip(axs, [r"_Pt\[", r"_Eta\[", r"_Phi\[", r"_M\["],
                               [" in GeV", "", "", " in GeV"])
    _df = groupby(stack(_all_features, feature), :output_expected; sort=true)
    #_df = [stack(i, feature) for i in _df]
    ax.hist([i.value for i in _df],
        weights=[i.weights for i in _df],
        bins=30, stacked=true, label=first.(keys(_df)))
    ax.set_xlabel("\\verb|$(feature.pattern[2:end-2])|$unit")
    ax.set_ylabel("Events/Bin")
    ax.set_yscale(:log)

    ax.legend()
    annotate_cms(ax; sel_pos=(.05, .95))
end
fig.suptitle("Distributions of Four Momenta")
fig.tight_layout()
display(fig)
fig.savefig(joinpath(output_dir, "four_momenta.pdf"))

for i in 0:6
fig, axs = subplots(4, 2, figsize=(18, 30))
axs[2, 2].axis(:off)
for (ax, feature, (lo, hi), unit) in zip([axs[1:2, :][1:3]; axs[3:4, :][:]], Vars.names_lola_out[7i+1:end],
                                [(0, 10000); (0, 1000); (-200, 1200); fill((-5e5, 5e5), 2); fill((-5e5, 0), 2)],
                                ["GeV\$^2\$"; "GeV"; "GeV"; fill("GeV\$^2\$", 4)])
    ax.hist([i[!, feature] for i in df],
        weights=[i.weights for i in df],
        bins=range(lo, hi; length=31), stacked=true, label=first.(keys(df)))
    feature = replace(feature, r"Jet_(.*)\[([0-5])\]" => s"\\verb|\1| of Jet \2")
    feature = replace(feature, r"Lepton_(.*)" => s"\\verb|\1| of the Lepton")
    ax.set_xlabel("$feature in $unit")
    ax.set_ylabel("Events/Bin")
    ax.set_yscale(:log)

    ax.legend()
    annotate_cms(ax; sel_pos=(.05, .95))
end
fig.suptitle("LoLa outputs")
fig.tight_layout()
display(fig)
fig.savefig(joinpath(output_dir, "lola_outputs_$i.pdf"))
end
