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

#basedir = "/work/sschaub/h5files/ttH-processed/"
basedir = "/local/scratch/ssd/sschaub/h5files/ttH_sebastian-processed/"
h5files = joinpath.(basedir, ["ttH", "ttbb", "ttcc", "ttlf"] .* "_lola.h5")

feature_names = [
    mapreduce(vcat, 0:5) do i
        ["Jet_X[$i]", "Jet_Y[$i]", "Jet_Z[$i]", "Jet_T[$i]"]
    end;
    "TightLepton_" .* ["X", "Y", "Z", "T"] .* "[0]";
    string.("Jet_CSV[", 0:5, "]");
]
append!(feature_names, [
    "Evt_CSV_avg_tagged",
    "Evt_CSV_avg",
    "Evt_M_minDrLepTag",
    "Evt_Pt_JetsAverage",
    "Evt_Pt_minDrTaggedJets",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_Deta_JetsAverage",
    "N_Jets",
    "Reco_JABDT_ttbar_Jet_CSV_whaddau2",
    "Reco_ttbar_toplep_m",
    "Reco_tHq_bestJABDToutput",
    "Reco_JABDT_tHq_abs_ljet_eta",
    "Reco_JABDT_tHq_Jet_CSV_hdau1",
    "Reco_JABDT_tHW_Jet_CSV_btop",
    #"memDBp",
])
raw_inputs = extract_cols(h5files, feature_names)#, 1:1000)
raw_inputs = Float32.(raw_inputs)
#raw_inputs = Float32.(reshape(raw_inputs, 4, 6, :))
#raw_inputs = reinterpret(reshape, SVector{4,Float32}, raw_inputs)
#inputs = Flux.normalise(raw_inputs; dims=2)
inputs = raw_inputs

classes = [:ttH, :ttbb, :ttcc, :ttlf]
outputs = Symbol.(extract_cols(h5files, ["class_label"]))
outputs_onehot = Flux.onehotbatch(vec(outputs), classes)

weights = prod(Float32, extract_cols(h5files, Vars.weights); dims=1)
idx = view(inputs, findfirst(==("N_Jets"), feature_names), :) .<= 6
inputs, outputs_onehot, weights = inputs[:, idx], outputs_onehot[:, idx], weights[:, idx]
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

n_input, n_output = size(inputs, 1), 4
n_jets = 7
n_scalar_input = n_input - 4n_jets
n_hidden = [1024,2048,512,512]

_leakyrelu(x) = max(x * 0.3f0, x)

m = 15
n_wd_sum = 2
n_wd_min = 2

model = Chain(
    LorentzSidechain(
        n_jets, m,
        (ntuple(_ -> +, n_wd_sum)..., ntuple(_ -> min, n_wd_min)...),
    ),
    x -> Flux.normalise(x; dims=2),
    foldl(n_hidden; init=((), (n_jets + m) * (3 + n_wd_sum + n_wd_min) + n_scalar_input)) do (c, n_in), n_out
        (c..., Dense(n_in, n_out, _leakyrelu), Dropout(0.5)), n_out
    end[1]...,
    Dense(n_hidden[end], n_output),
) |> gpu

optimizer = ADAM(5e-5)

penalty = let ps = Flux.params(model)
    nrm(x::AbstractArray) = norm(abs.(x) .+ eps(0f0))^2
    nrm(x::CoLa) = nrm(x.C)
    nrm(x::LoLa) = nrm(x.w_E) + sum(nrm, x.w_ds)
    penalty() = sum(nrm, ps)
end
loss(ŷ, y; weights) = Flux.logitcrossentropy(ŷ, y; agg = x -> weighted_mean(x, weights)) + 1f-5 * penalty()

measures = (; loss=st->st.loss, )#accuracy=st->accuracy(softmax(st.ŷ), st.y))

recorded_measures = DataFrame()
for i in 1:200
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

    let
        global fig, ax = subplots()
        losses = ["train_loss", "test_loss", "validation_loss"]
        plts = foreach(1:3, losses) do i, loss
            ax.plot(
                getproperty(recorded_measures, loss),
                label=loss
            )
        end
        ax.legend()
        display(fig)
    end

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

output_dir = "/work/sschaub/JuliaForHEP/foo10"
isdir(output_dir) || mkdir(output_dir)

fig.savefig(joinpath(output_dir, "losses.pdf"))

Arrow.write(
    joinpath(output_dir, "recorded_measures.arrow"),
    recorded_measures,
)

let ps = Flux.params(cpu(model)) |> collect
    cola = popfirst!(ps)
    @assert cola isa CoLa
    lola = popfirst!(ps)
    @assert lola isa LoLa
    Arrow.write(
        joinpath(output_dir, "layer_params.arrow"),
        DataFrame(
            :cola => [cola.C],
            :lola_w_E => [lola.w_E],
            (Symbol(:lola_w_d_, i) => [lola.w_ds[i]] for i in eachindex(lola.w_ds))...,
            (Symbol(i % Bool ? :W_ : :b_, fld1(i, 2)) => [p] for (i, p) in enumerate(ps))...,
        )
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
    input_features = Symbol.(feature_names) .=> eachrow(view(raw_inputs, :, idx))
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
