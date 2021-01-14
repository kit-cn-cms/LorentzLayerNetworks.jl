_leakyrelu(x) = max(x * 0.3f0, x)

function build_dnn(; n_inputs, neurons, n_outputs)
    return foldl(n_hidden; init=((), n_inputs)) do (c, n_in), n_out
        (c..., Dense(n_in, n_out, _leakyrelu), Dropout(0.5)), n_out
    end[1]
end

function build_model(; m, n_jets, n_inputs, n_outputs, w_d_reducers, neurons, dropout=identity, activation=_leakyrelu)
    n_scalar_input = n_input - 4n_jets
    n_wd = sum(last, w_d_reducers)
    reducers = foldl(w_d_reducers; init=()) do t, (f, n)
        (t..., ntuple(_ -> f, n)...)
    end
    return Chain(
        LorentzSidechain(
            n_jets, m,
            (ntuple(_ -> +, n_wd_sum)..., ntuple(_ -> min, n_wd_min)...),
        ),
        x -> Flux.normalise(x; dims=2),
        build_dnn(;
            n_inputs = (n_jets + m) * (3 + n_wd) + n_scalar_input,
            neurons,
            n_outputs,
        )...,
        Dense(n_hidden[end], n_output),
    )
end
