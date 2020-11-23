using JuliaForHEP
using Flux
using Flux: Data.DataLoader
using Random
using DataFrames

include("variables.jl")

basedir = "/ceph/larmbruster/multiclassJAN/bkg_merging/cTag_infos/v1/"
h5files = joinpath.(basedir, ["Hbb_dnn.h5", "Zbb_dnn.h5"])

raw_inputs = extract_cols(h5files, Vars.variables["ge4j_ge3t"])
inputs = Flux.normalise(raw_inputs; dims=2)

outputs = Symbol.(extract_cols(h5files, ["class_label"]))

weights = prod(Float32, extract_cols(h5files, Vars.weights); dims=1)
total_weights = scale_weights(weights, outputs)

test_split = 0.25
n_train = floor(Int, length(outputs) * (1 - test_split))

idx = shuffle(eachindex(outputs))
train_idx, test_idx = idx[1:n_train], idx[n_train+1:end]

ds_train = (
    inputs[:, train_idx],
    Flux.onehotbatch(outputs[train_idx], [:Hbb, :Zbb]),
    total_weights[:, train_idx],
) .|> gpu
ds_train = DataLoader(ds_train, batchsize=128, shuffle=true)

ds_test = (
    inputs[:, test_idx],
    Flux.onehotbatch(outputs[test_idx], [:Hbb, :Zbb]),
    total_weights[:, test_idx],
) .|> gpu
ds_test = DataLoader(ds_test, batchsize=128, shuffle=false)

n_input, n_output = size(inputs, 1), 2
n_hidden = [200, 200, 200]

_leakyrelu(x) = max(x * 0.3f0, x)

model = Chain(
    foldl(n_hidden; init=((), n_input)) do (c, n_in), n_out
        (c..., Dense(n_in, n_out, _leakyrelu), Dropout(0.1)), n_out
    end[1]...,
    Dense(n_hidden[end], n_output),
) |> gpu

opt = ADAGrad(0.001)

loss(ŷ, y; weights) = Flux.logitcrossentropy(ŷ, y; agg = x -> weighted_mean(x, weights))

measures = (; loss=st->st.loss, accuracy=st->accuracy(softmax(st.ŷ), st.y))

measures_train, measures_test = DataFrame(), DataFrame()

for i in 1:20
    @info "Epoch $i"
    train, test = step!(model, ds_train, ds_test, loss, opt, measures)
    @show train
    @show test
    push!(measures_train, train)
    push!(measures_test, test)
end
