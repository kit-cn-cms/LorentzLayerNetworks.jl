struct LorentzSidechain{L}
    layers::L
    n_jets::Int
end

Flux.@functor LorentzSidechain

function LorentzSidechain(n_jets::Int, n_lincomb::Int)
    layers = Chain(
        x -> reinterpret(reshape, SVector{4,Float32}, reshape(x, 4, n_jets, :)),
        cola(n_jets, n_lincomb),
        LoLa(n_jets + n_lincomb, (+, +, min, min)),
        Flux.flatten,
    )
    return LorentzSidechain(layers, n_jets)
end

function (s::LorentzSidechain)(x)
    #return [s.layers(view(x, 1:4*s.n_jets, :)); view(x, 4*s.n_jets+1:size(x, 1), :)]
    return [s.layers(x[1:4*s.n_jets, :]); x[4*s.n_jets+1:size(x, 1), :]]
end
