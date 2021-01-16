_leakyrelu(x) = max(x * 0.3f0, x)

function build_dnn(; n_inputs, neurons, n_outputs, activation, dropout)
    return Chain(
        foldl(neurons; init=((), n_inputs)) do (c, n_in), n_out
            (c..., Dense(n_in, n_out, _leakyrelu), dropout), n_out
        end[1]...,
        Dense(neurons[end], n_outputs),
    )
end

function build_model(; m, n_jets, n_inputs, n_outputs, w_d_reducers, neurons, activation=_leakyrelu, dropout=identity)
    n_scalar_input = n_inputs - 4n_jets
    n_wd = sum(last, w_d_reducers)
    reducers = foldl(w_d_reducers; init=()) do t, (f, n)
        (t..., ntuple(_ -> f, n)...)
    end
    return Chain(
        LorentzSidechain(n_jets, m, reducers),
        x -> Flux.normalise(x; dims=2),
        build_dnn(;
            n_inputs = (n_jets + m) * (3 + n_wd) + n_scalar_input,
            neurons,
            n_outputs,
            activation,
            dropout,
        )...,
    )
end

function l2_penalty(model)
    ps = Flux.params(model)
    return () -> sum(_nrm2, ps)
end
_nrm2(x::AbstractArray) = norm(abs.(x) .+ eps(0f0))^2
_nrm2(x::CoLa) = _nrm2(x.C)
_nrm2(x::LoLa) = _nrm2(x.w_E) + sum(_nrm2, x.w_ds)
