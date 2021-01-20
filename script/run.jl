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

#basedir = "/work/sschaub/h5files/ttH-processed/"
basedir = "/local/scratch/ssd/sschaub/h5files/ttH_sebastian-processed/"
h5files = joinpath.(basedir, ["ttH", "ttbb", "ttcc", "ttlf"] .* "_lola.h5")

feature_names = Vars.all_features
inputs = extract_cols(h5files, feature_names)
inputs = Float32.(inputs)

using .Vars: classes
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

m = 15
n_jets = length(Vars.jets) ÷ 4
n_inputs = size(inputs, 1)
n_outputs = length(classes)
model = build_model(;
    m, n_jets, n_inputs, n_outputs,
    w_d_reducers = ((+) => 2, min => 2),
    #w_d_reducers = (min => 4,),
    neurons = [1024, 2048, 512, 512],
    activation = _leakyrelu,
    dropout = Dropout(0.5),
) |> gpu

optimizer = ADAM(5e-5)

penalty = l2_penalty(model)
loss(ŷ, y; weights) = Flux.logitcrossentropy(ŷ, y; agg = x -> weighted_mean(x, weights))# + 1f-5 * penalty()

measures = (; loss=st->st.loss, accuracy=st->accuracy(softmax(st.ŷ), st.y))

recorded_measures = DataFrame()

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
    max_epochs=50, min_epochs=50, early_stopping_n=10, early_stopping_percentage=2,
)

# main training
train!(;
    model, ds_train, ds_test, ds_validation,
    loss, optimizer, measures, recorded_measures,
    max_epochs=120, min_epochs=50, early_stopping_n=10, early_stopping_percentage=2,
)

using Arrow

output_dir = "/work/sschaub/JuliaForHEP/lola+scalars8/"
isdir(output_dir) || mkdir(output_dir)

fig.savefig(joinpath(output_dir, "losses.pdf"))

Arrow.write(
    joinpath(output_dir, "recorded_measures.arrow"),
    recorded_measures,
)

let ps = Flux.params(cpu(model)) |> collect
    df = DataFrame()
    if n_jets > 0
        cola = popfirst!(ps)
        @assert cola isa CoLa
        lola = popfirst!(ps)
        @assert lola isa LoLa
        df = DataFrame(
            :cola => [cola.C],
            :lola_w_E => [lola.w_E],
            (Symbol(:lola_w_d_, i) => [lola.w_ds[i]] for i in eachindex(lola.w_ds))...,
        )
    end
    df = hcat(
        df,
        DataFrame(
            (Symbol(i % Bool ? :W_ : :b_, fld1(i, 2)) => [p] for (i, p) in enumerate(ps))...,
        )
    )
    Arrow.write(
        joinpath(output_dir, "layer_params.arrow"),
        df,
    )
end

all_features = DataFrame();
ŷ = softmax(model(gpu(inputs))) |> cpu
for (label, idx) in [
    :train => idx_train,
    :test => idx_test,
    :validation => idx_validation,
]
    idx = cpu(idx)
    input_features = Symbol.(feature_names) .=> eachrow(view(inputs, :, idx))
    #input_features_norm = Symbol.(Vars.variables["ge4j_ge3t"], :_norm) .=> eachrow(view(inputs, :, idx))
    output_expected = :output_expected => outputs[idx]
    output_predicted = [Symbol(:output_predicted_, l) => ŷ[i, idx] for (i, l) in enumerate(classes)]
    _weights = [:weights => weights[idx], :weights_norm => total_weights[idx]]
    kind = :kind => fill(label, length(idx))
    append!(all_features, DataFrame(input_features..., #=input_features_norm..., =#output_expected, output_predicted..., _weights..., kind))
end

Arrow.write(
    joinpath(output_dir, "all_features.arrow"),
    all_features,
)
