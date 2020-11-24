using JuliaForHEP
using Flux
using Flux: Data.DataLoader
using DataFrames

include("variables.jl")

basedir = "/ceph/larmbruster/multiclassJAN/bkg_merging/cTag_infos/v1/"
h5files = joinpath.(basedir, ["Hbb_dnn.h5", "Zbb_dnn.h5"])

raw_inputs = extract_cols(h5files, Vars.variables["ge4j_ge3t"])
inputs = Flux.normalise(raw_inputs; dims=2)

outputs = Symbol.(extract_cols(h5files, ["class_label"]))
outputs_onehot = Flux.onehotbatch(vec(outputs), [:Hbb, :Zbb])

weights = prod(Float32, extract_cols(h5files, Vars.weights); dims=1)
total_weights = scale_weights(weights, outputs)

ds_train, ds_validation, ds_test = split_datasets(
    gpu,
    (inputs, outputs_onehot, total_weights),
    [.75 * .8, .75 * .2, .25],
)
ds_train = DataLoader(ds_train; batchsize=128, shuffle=true)
ds_validation = DataLoader(ds_validation; batchsize=128, shuffle=false)
ds_test = DataLoader(ds_test; batchsize=128, shuffle=false)

n_input, n_output = size(inputs, 1), 2
n_hidden = [200, 200, 200]

_leakyrelu(x) = max(x * 0.3f0, x)

model = Chain(
    foldl(n_hidden; init=((), n_input)) do (c, n_in), n_out
        (c..., Dense(n_in, n_out, _leakyrelu), Dropout(0.1)), n_out
    end[1]...,
    Dense(n_hidden[end], n_output),
) |> gpu

optimizer = ADAGrad(0.001)
optimizer = ADAM(5e-5)

loss(ŷ, y; weights) = Flux.logitcrossentropy(ŷ, y; agg = x -> weighted_mean(x, weights))

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

    plt = @df recorded_measures plot(
        [:train_loss, :test_loss, :validation_loss],
        labels=["train loss" "test loss" "validation loss"],
    )
    display(plt)

    if i >= 5
        if argmin(recorded_measures[!, :test_loss]) <= i - 3
            @info "Loss has not decreased in the last 3 epochs, stopping training"
            break
        elseif test.loss / train.loss > 1.1
            @info "Test loss more than 10% greater than training loss, stopping training"
            break
        end
    end
end
