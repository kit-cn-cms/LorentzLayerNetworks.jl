using Arrow

function save_model(;
    output_dir,
    fig,
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
    isdir(output_dir) || mkdir(output_dir)

    if fig !== nothing
        fig.savefig(joinpath(output_dir, "losses.pdf"))
    end

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
    if n_jets > 0
        lola_out = model[1](gpu(inputs))[1:length(Vars.names_lola_out), :] |> cpu
    end
    for (label, idx) in [
        :train => idx_train,
        :test => idx_test,
        :validation => idx_validation,
    ]
        idx = cpu(idx)
        input_features = Symbol.(feature_names) .=> eachrow(view(inputs, :, idx))
        output_expected = :output_expected => outputs[idx]
        ŷ = softmax(model(gpu(inputs[:, idx]))) |> cpu
        output_predicted = [Symbol(:output_predicted_, l) => ŷ[i, :] for (i, l) in enumerate(classes)]
        _weights = [:weights => weights[idx], :weights_norm => total_weights[idx]]
        _lola_out = if n_jets > 0
            Symbol.(Vars.names_lola_out) .=> eachrow(view(lola_out, :, idx))
        else
            ()
        end
        kind = :kind => fill(label, length(idx))
        append!(all_features, DataFrame(
            input_features...,
            output_expected,
            output_predicted...,
            _weights...,
            _lola_out...,
            kind,
        ))
    end

    Arrow.write(
        joinpath(output_dir, "all_features.arrow"),
        all_features,
    )
end
