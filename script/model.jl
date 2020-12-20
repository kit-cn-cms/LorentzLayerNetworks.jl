using JuliaForHEP
using Flux
using Flux: Data.DataLoader
using DataFrames
using Random
using LinearAlgebra
using Compat
using StaticArrays
Random.seed!(123)

include("variables.jl")

#basedir = "/work/sschaub/h5files/ttH-processed/"
basedir = "/local/scratch/ssd/sschaub/h5files/ttH-processed/"
h5files = joinpath.(basedir, ["ttH_ttH.h5", "ttbb_ttH.h5", "ttcc_ttH.h5", "ttlf_ttH.h5"])

raw_inputs = extract_cols(h5files, mapreduce(vcat, 0:5) do i
    ["Jet_X[$i]", "Jet_Y[$i]", "Jet_Z[$i]", "Jet_T[$i]"]
end)#, 1:1000)
raw_inputs = Float32.(reshape(raw_inputs, 4, 6, :))
raw_inputs = reinterpret(reshape, SVector{4,Float32}, raw_inputs)
#inputs = Flux.normalise(raw_inputs; dims=2)
inputs = raw_inputs

outputs = Symbol.(extract_cols(h5files, ["class_label"]))
outputs_onehot = Flux.onehotbatch(vec(outputs), [:ttH, :ttbb, :ttcc, :ttlf])

weights = prod(Float32, extract_cols(h5files, Vars.weights); dims=1)
total_weights = scale_weights(weights, outputs)

ds_train, ds_validation, ds_test = split_datasets(
    gpu,
    (inputs, outputs_onehot, total_weights, axes(inputs, 2)),
    [.75 * .8, .75 * .2, .25],
)
idx_train, idx_validation, idx_test = last.((ds_train, ds_validation, ds_test))
ds_train = DataLoader(ds_train; batchsize=512, shuffle=true)
ds_validation = DataLoader(ds_validation; batchsize=128, shuffle=false)
ds_test = DataLoader(ds_test; batchsize=512, shuffle=false)

n_input, n_output = size(inputs, 1), 4
n_hidden = [200, 200] # [200, 200, 200]

_leakyrelu(x) = max(x * 0.3f0, x)

dbg(x) = (@show summary(x); x)
model = Chain(
    Linear(CoLa(rand(Float32, 15, n_input))),
    LoLa(rand(Float32, n_input+15, n_input+15), ntuple(_ -> rand(Float32, n_input+15, n_input+15), 4), (+, +, min, min)),
    Flux.flatten,
    foldl(n_hidden; init=((), (n_input+15)*7)) do (c, n_in), n_out
        (c..., Dense(n_in, n_out, _leakyrelu), Dropout(0.1)), n_out
    end[1]...,
    Dense(n_hidden[end], n_output),
) |> gpu

optimizer = ADAGrad(0.001)
optimizer = ADAM(5e-5)

let ps = Flux.params(model)
    global penalty() = sum(x -> norm(x .+ 1e-8), ps)
end
loss(ŷ, y; weights) = Flux.logitcrossentropy(ŷ, y; agg = x -> weighted_mean(x, weights)) #+ 1e-5 * penalty()

measures = (; loss=st->st.loss, accuracy=st->accuracy(softmax(st.ŷ), st.y))

using StatsPlots

recorded_measures = DataFrame()
for i in 1:100
    @info "Epoch $i"
    train, test, validation = step!(
        model, ds_train, ds_test, ds_validation;
        loss_function=loss, optimizer, measures,
    )
    @show train
    @show test
    @show validation
    push!(
        recorded_measures,
        merge(prefix_labels.((train, test, validation), (:train_, :test_, :validation_))...),
    )

    plt = @df recorded_measures Plots.plot(
        [:train_loss, :test_loss, :validation_loss],
        labels=["train loss" "test loss" "validation loss"],
    )
    display(plt)

    if i >= 20
        if argmin(recorded_measures[!, :test_loss]) <= i - 5
            @info "Loss has not decreased in the last 5 epochs, stopping training"
            break
        elseif test.loss / train.loss > 1.1
            @info "Test loss more than 10% greater than training loss, stopping training"
            break
        end
    end
end

using Arrow

output_dir = "/work/sschaub/JuliaForHEP/foo"
mkdir(output_dir)

Plots.savefig(joinpath(output_dir, "losses.pdf"))

Arrow.write(
    joinpath(output_dir, "recorded_measures.arrow"),
    recorded_measures,
)

#Arrow.write(
#    joinpath(output_dir, "layer_params.arrow"),
#    DataFrame((Symbol(i % Bool ? :W_ : :b_, fld1(i, 2)) => [p] for (i, p) in enumerate(Flux.params(model)))...)
#)
#
#all_features = DataFrame();
#ŷ = softmax(model(gpu(inputs))) |> cpu
#for (label, idx) in [
#    :train => idx_train,
#    :test => idx_test,
#    :validation => idx_validation,
#]
#    idx = cpu(idx)
#    input_features = Symbol.(Vars.variables["ge4j_ge3t"]) .=> eachrow(view(raw_inputs, :, idx))
#    input_features_norm = Symbol.(Vars.variables["ge4j_ge3t"], :_norm) .=> eachrow(view(inputs, :, idx))
#    output_expected = :output_expected => outputs[idx]
#    output_predicted = [Symbol(:output_predicted_, l) => ŷ[i, idx] for (i, l) in enumerate([:Hbb, :Zbb])]
#    _weights = [:weights => weights[idx], :weights_norm => total_weights[idx]]
#    kind = :kind => fill(label, length(idx))
#    append!(all_features, DataFrame(input_features..., input_features_norm..., output_expected, output_predicted..., _weights..., kind))
#end
#
#Arrow.write(
#    joinpath(output_dir, "all_features.arrow"),
#    all_features,
#)

