using Flux, Zygote
using Flux: crossentropy, Data.DataLoader
using HDF5, DataFrames
using WeightedOnlineStats
using Random
using CUDA

_a = CuArray{Float32,0}(undef)

function accuracy(_a, ŷ, y::Flux.OneHotMatrix)
    _a[] = count(axes(ŷ, 2)) do i
        @inbounds y[argmax(view(ŷ, :, i)), i]
    end / Float32(size(ŷ, 2))
    nothing
end

function step!(model, ds_train, ds_test, loss_function, opt)
    ps = Flux.params(model)

    trainmode!(model)

    train_loss, train_accuracy = WeightedMean(), WeightedMean()
    for (x, y, weights) in ds_train
        local ŷ
        loss, pb = Zygote.pullback(ps) do
            ŷ = model(x)
            return loss_function(ŷ, y; weights)
        end
        gs = pb(one(loss))
        Flux.update!(opt, ps, gs)
        fit!(train_loss, loss, size(y, 2))
        @cuda threads=size(ŷ, 2) accuracy(_a, softmax(ŷ), y)
        #global _ŷ, _y = ŷ, y
        fit!(train_accuracy, _a[], size(y, 2))
    end

    testmode!(model)

    test_loss, test_accuracy = WeightedMean(), WeightedMean()
    for (x, y, weights) in ds_test
        ŷ = model(x)
        loss = loss_function(ŷ, y; weights)
        fit!(test_loss, loss, size(y, 2))
        @cuda threads=size(ŷ, 2) accuracy(_a, softmax(ŷ), y)
        fit!(test_accuracy, _a[], size(y, 2))
    end

    @show value.([train_loss, train_accuracy])
    @show value.([test_loss, test_accuracy])
    nothing
end

basedir = "/ceph/larmbruster/multiclassJAN/bkg_merging/cTag_infos/v1/"
h5files = joinpath.(basedir, ["Hbb_dnn.h5", "Zbb_dnn.h5"])

using Pandas

function extract_cols(names)
    return mapreduce(hcat, h5files) do f
        df = read_hdf(f)
        PermutedDimsArray(Array(df[names]), (2, 1))
    end
end

include("variables.jl")

raw_inputs = extract_cols(Vars.variables["ge4j_ge3t"])
inputs = Flux.normalise(raw_inputs)
outputs = Symbol.(extract_cols(["class_label"]))
weights = prod(Float32, extract_cols(Vars.weights); dims=1)

function scale_weights(weights, outputs)
    sums_weights = Dict{Symbol,Float32}()
    for (i, class) in pairs(outputs)
        sums_weights[class] = get(sums_weights, class, 0f0) + weights[i]
    end
    return map(CartesianIndices(weights)) do i
        weights[i] / sums_weights[outputs[i]]
    end
end

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

wmean(x, weights) = sum(x .* weights ./ sum(weights))

loss(ŷ, y; weights) = Flux.logitcrossentropy(ŷ, y; agg = x -> wmean(x, weights))

@time Flux.@epochs 100 step!(model, ds_train, ds_test, loss, opt)
