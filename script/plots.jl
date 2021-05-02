using Arrow, DataFrames, Flux
using PyPlot
using Printf
using Trapz
using Statistics

include("variables.jl")
include("plot_styles.jl")
using .Vars: classes

measures = DataFrame()
basedir = "/work/sschaub/JuliaForHEP/final_plotting/"
tex = true

for i in 1:10, feature in filter(x -> endswith(x, "_$i") && isdir(joinpath(basedir, x)), readdir(basedir))
    output_dir = joinpath(basedir, "$feature/")
    feature = replace(feature, Regex("_$i\$") => "")
    @show feature, i

    layer_params = DataFrame(Arrow.Table(joinpath(output_dir, "layer_params.arrow")))
    all_features = DataFrame(Arrow.Table(joinpath(output_dir, "all_features.arrow")))

    fig1, axs1 = subplots(ncols=2, nrows=2, figsize=(16, 12))
    linestyles = (train = "-", test = "--", validation="-.")
    for ((kind,), df) in pairs(groupby(all_features, :kind))
        kind !== :test && foreach(
            pairs(groupby(df, :output_expected, sort=true)),
            [:C0, :C1, :C2, :C3],
        ) do ((class,), df), color
            foreach(axs1, classes) do ax1, class_predicted
                ax1.hist(
                    df[!, Symbol(:output_predicted_, class_predicted)];
                    label="true $class ($kind)", density=true,
                    bins=30, histtype=:step, color, ls=linestyles[kind],
                )
            end
        end

        ŷ = mapreduce(vcat, classes) do class
            df[!, Symbol(:output_predicted_, class)]'
        end
        outputs_onehot = Flux.onehotbatch(df.output_expected, classes)

        confusion_mat = sum(
            reshape(Flux.onehotbatch(Flux.onecold(ŷ, classes), classes), 1, 4, :) .===
                reshape(outputs_onehot, 4, 1, :);
            dims=3,
        )
        confusion_mat = confusion_mat ./ sum(confusion_mat; dims=2)

        fig, ax = subplots()

        img = ax.imshow(confusion_mat)
        for i in axes(confusion_mat, 1), j in axes(confusion_mat, 2)
            ax.text(j-1, i-1, @sprintf("%.3g", confusion_mat[i, j]), ha=:center, va=:center)
        end
        ax.set_xlabel("predicted output")
        ax.set_xticks(eachindex(classes).-1)
        ax.set_xticklabels(classes)
        ax.set_ylabel("true output")
        ax.set_yticks(eachindex(classes).-1)
        ax.set_yticklabels(classes)
        annotate_cms(ax)
        fig.colorbar(img)
        fig.suptitle(kind)
        fig.savefig(joinpath(output_dir, "confusion_matrix_$kind.pdf"))

        fig_eff, axs_eff = subplots(ncols=2, nrows=2, figsize=(16, 12))
        fig_roc, axs_roc = subplots(ncols=2, nrows=2, figsize=(16, 12))
        for (ax_roc, ax_eff, class) in zip(axs_roc, axs_eff, classes)
            predictions_positives = df[df.output_expected .=== class, Symbol(:output_predicted_, class)]
            predictions_background = df[df.output_expected .!== class, Symbol(:output_predicted_, class)]
            eff_s, = ax_eff.hist(
                predictions_positives;
                cumulative=true, bins=range(0, 1; length=30), density=true, histtype=:step,
                label="signal efficiency"
            )
            eff_bg, = ax_eff.hist(
                predictions_background;
                cumulative=true, bins=range(0, 1; length=30), density=true, histtype=:step,
                label="background efficiency",
            )
            ax_eff.legend(; loc="lower right")
            ax_eff.set_ylabel("efficiency")
            ax_eff.set_xlabel("1 - specificity")
            ax_eff.set_title("$class")
            annotate_cms(ax_eff)

            x, y = 1 .- eff_s, eff_bg
            ax_roc.plot(x, y)
            idx = sortperm(x)
            auc = trapz(x[idx], y[idx])
            ax_roc.text(.2, .2, @sprintf("AUC = %.3f", auc); fontsize=24)
            ax_roc.set_xlim(0, 1)
            ax_roc.set_ylim(0, 1)
            ax_roc.set_ylabel("background rejection")
            ax_roc.set_xlabel("signal efficiency")
            ax_roc.set_title("$class")
            annotate_cms(ax_roc)

            push!(measures, (; feature, kind, ROC_AUC=auc, node=class, i))
        end
        fig_eff.suptitle("Efficiencies - $kind")
        fig_eff.tight_layout()
        fig_eff.savefig(joinpath(output_dir, "efficiencies_$kind.pdf"))

        fig_roc.suptitle("ROC curves - $kind")
        fig_roc.tight_layout()
        fig_roc.savefig(joinpath(output_dir, "roc_curves_$kind.pdf"))

        _features = [Vars.all_features; string.("output_predicted_", classes)]
        _features = [Vars.names_lola_out; _features[4*7+1:end]]
        _features = filter(in(names(df)), _features)
        n = length(_features)
        vars = df[!, _features]
        tex && (_features = string.("\\verb|", _features, "|"))

        correlations = [cor(u, v) for u in eachcol(vars), v in eachcol(vars)]
        fig, ax = subplots(figsize=(14, 12.5))
        img = ax.imshow(correlations; vmin=-1, vmax=1)
        fig.colorbar(img)
        ax.set_xticks(axes(vars, 2).-1)
        ax.set_xticklabels(_features; rotation=90, fontsize=10)
        ax.set_yticks(axes(vars, 2).-1)
        ax.set_yticklabels(_features, fontsize=10)
        annotate_cms(ax)
        # dividers
        for (i, opts) in zip([7*7 + 6, n - 4], [(; linestyle=:dashed, linewidth=1), (; linewidth=3)])
            feature == "only scalars" && i == 7*7 + 6 && continue
            ax.vlines(i - .5, -.5, n - .5; color=:red, opts...)
            ax.hlines(i - .5, -.5, n - .5; color=:red, opts...)
        end
        fig.suptitle("Correlations - $kind")
        fig.tight_layout()
        fig.savefig(joinpath(output_dir, "correlations_$kind.pdf"))
    end

    foreach(axs1, classes) do ax1, class_predicted
        ax1.set_title(String(class_predicted))
        ax1.set_xlabel("p($class_predicted)")
        ax1.legend(; fontsize=16)
        annotate_cms(ax1)
    end
    #fig1.suptitle("Predicted Hbb")
    fig1.tight_layout()
    fig1.savefig(joinpath(output_dir, "output_features_hist.pdf"))
end

Arrow.write(
    joinpath(basedir, "roc_scores.arrow"),
    measures,
)

begin
idx = sortperm(measures.feature; by=x -> findfirst(==(x), ["lola+" .* ["none", "tier3", "tier2+3", "all"]; "jets_as_scalars+all"; "none+all"]))
measures = measures[idx, :]
fig, ax = subplots(figsize=(11, 11))
marker = (train=:v, test=:P, validation=:o)
foreach(pairs(groupby(measures, [:kind, :node]))) do ((kind, node), df)
    node === :ttH || return
    kind === :validation || return
    df = combine(groupby(df, :feature), (:ROC_AUC .=> (mean, std))...)
    x = axes(df, 1)
    ax.errorbar(
        x, df.ROC_AUC_mean; yerr=df.ROC_AUC_std,
        color=:blue#=[:red; fill(:blue, length(x)-3); :green; :orange]=#, marker=marker[kind],
        lw=0, elinewidth=1, capsize=8, markersize=8,
        label=kind,
    )
end

x = unique(measures.feature)
ax.set_xticks(eachindex(x))
ax.set_xticklabels(string.("\\verb|", x, "|"); rotation=90)
ax.set_ylabel("AUC")
ax.legend(; #=loc="upper left", =#fontsize=16, frameon=true)
fig.suptitle("ttH ROC integrals for training with LoLa + X")
annotate_cms(ax)
fig.tight_layout()
fig.savefig(joinpath(basedir, "roc_scores.pdf"))
display(fig)
end
