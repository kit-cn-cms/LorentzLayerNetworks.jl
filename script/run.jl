using JuliaForHEP
using JuliaForHEP: step!
using Flux
using Flux: Data.DataLoader
using DataFrames
using Random
using LinearAlgebra
using Compat
using PyPlot

Random.seed!(123)

include("variables.jl")
include("model.jl")
include("training.jl")
include("save_model.jl")

#basedir = "/work/sschaub/h5files/ttH-processed/"
basedir = "/local/scratch/ssd/sschaub/h5files/ttH_sebastian-processed/"
h5files = joinpath.(basedir, ["ttH", "ttbb", "ttcc", "ttlf"] .* "_lola.h5")

for i in 1:10

for (scalar_features, include_jets) in zip([
    [String[]]; #=Base.vect.(Vars.scalar_features); =#
    [Vars.tier3, [Vars.tier2; Vars.tier3], Vars.scalar_features];
    [Vars.scalar_features, Vars.scalar_features]
],
    [fill(:lola, 4); :jets_as_scalars; :none]
)
    @show scalar_features, include_jets
    feature_names = if include_jets === :none
        scalar_features
    else
        [Vars.jets; Vars.tight_lepton; Vars.jet_csv; scalar_features]
    end
    inputs = extract_cols(h5files, feature_names) .|> Float32

    classes = Vars.classes
    outputs = Symbol.(extract_cols(h5files, ["class_label"]))
    outputs_onehot = Flux.onehotbatch(vec(outputs), classes)

    weights = prod(Float32, extract_cols(h5files, Vars.weights); dims=1)
    total_weights = scale_weights(weights, outputs)

    ds_train, ds_validation, ds_test = split_datasets(
        gpu,
        (inputs, outputs_onehot, total_weights, axes(inputs, 2)),
        [.75 * .8, .75 * .2, .25],
    )
    idx_train, idx_validation, idx_test = last.((ds_train, ds_validation, ds_test))
    ds_train = DataLoader(ds_train; batchsize=1024, shuffle=true)
    ds_validation = DataLoader(ds_validation; batchsize=1024, shuffle=false)
    ds_test = DataLoader(ds_test; batchsize=1024, shuffle=false)

    m = include_jets === :lola ? 15 : 0
    n_jets = include_jets === :lola ? length([Vars.jets; Vars.tight_lepton]) ÷ 4 : 0
    n_inputs = size(inputs, 1)
    n_outputs = length(classes)
    model = build_model(;
        m, n_jets, n_inputs, n_outputs,
        w_d_reducers = ((+) => 2, min => 2),
        neurons = [1024, 2048, 512, 512],
        activation = _leakyrelu,
        dropout = Dropout(0.5),
    ) |> gpu

    optimizer = ADAM(5e-5)

    penalty = l2_penalty(model)
    loss(ŷ, y; weights) = Flux.logitcrossentropy(ŷ, y; agg = x -> weighted_mean(x, weights))# + 1f-5 * penalty()

    measures = (; loss=st->st.loss, accuracy=st->accuracy(softmax(st.ŷ), st.y))

    recorded_measures = DataFrame()

    if include_jets === :lola
        # pretrain Lo+CoLa
        n_outputs_lola = (n_jets + m) * (3 + 4) + length(Vars.jet_csv)
        _model = Chain(
            model[1:end-1]...,
            #x -> view(x, 1:n_outputs_lola, :),
            x -> x[1:n_outputs_lola, :],
            build_dnn(;
                n_inputs = n_outputs_lola,
                neurons = [1024, 2048, 512, 512],
                n_outputs,
                activation = _leakyrelu,
                dropout = Dropout(0.5),
            ) |> gpu,
        )
        train!(;
            model=_model, ds_train, ds_test, ds_validation,
            loss, optimizer, measures, recorded_measures,
            max_epochs=20, min_epochs=50, early_stopping_n=10, early_stopping_percentage=2,
        )
    else
        model = model[2:end]
    end

    # main training
    train!(;
        model, ds_train, ds_test, ds_validation,
        loss, optimizer, measures, recorded_measures,
        max_epochs=120, min_epochs=50, early_stopping_n=7, early_stopping_percentage=2,
    )

    name = string(include_jets) * "+" * if isempty(scalar_features)
        "none"
    elseif length(scalar_features) == 1
        scalar_features[]
    elseif scalar_features == Vars.tier3
        "tier3"
    elseif scalar_features == [Vars.tier2; Vars.tier3]
        "tier2+3"
    elseif scalar_features == Vars.scalar_features
        "all"
    end

    save_model(;
        output_dir = "/work/sschaub/JuliaForHEP/feature_evaluation_0208/$(name)_$i/",
        fig = nothing,
        recorded_measures,
        model,
        inputs,
        feature_names,
        n_jets,
        classes,
        outputs,
        weights,
        total_weights,
        idx_train, idx_test, idx_validation,
    )
end

end
