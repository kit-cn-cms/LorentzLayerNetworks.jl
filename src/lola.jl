using LinearAlgebra
using Tullio
using StaticArrays

struct LoLa{A<:AbstractMatrix,T<:Tuple,Fs<:Tuple}
    w_E::A
    w_ds::T
    w_d_reducers::Fs
end

function Flux.update!(opt, l::LoLa, dl)
    if haskey(dl, :w_E)
        Flux.update!(opt, l.w_E, dl[:w_E])
    end
    if haskey(dl, :w_ds)
        for i in eachindex(dl[:w_ds])
            Flux.update!(opt, l.w_ds[i], dl[:w_ds][i])
        end
    end
    return l
end

Flux.functor(l::LoLa) = (l.w_E, l.w_ds), x -> LoLa(x[1], x[2], l.w_d_reducers)

Flux.params!(p::Zygote.Params, l::LoLa, seen=IdSet()) = push!(p, l)
Flux.params!(p::Zygote.Params, k::AbstractArray{<:SArray}, seen=IdSet()) = push!(p, k)

m²(k) = sum(abs2, k)
p_T(k) = hypot(k[1], k[2])
E(k) = k[4]

using Compat

include("lola_kernel.jl")

E!(_E::AbstractVector, w, k::AbstractVector) = @tullio _E[i] = w[i, j] * E(k[j])
E!(_E::AbstractMatrix, w, k::AbstractMatrix) = @tullio _E[i, l] = w[i, j] * E(k[j, l])

slice(res, i) = view(res, i, Base.tail(axes(res))...)

function _lola3(l, k)
    T = eltype(eltype(k))
    res = similar(k, T, 3 + length(l.w_ds), axes(k)...)
    map!(m², slice(res, 1), k)
    map!(p_T, slice(res, 2), k)
    E!(slice(res, 3), l.w_E, k)
    _k = reinterpret(reshape, T, k)
    return res, _k
end

function (l::LoLa)(k)
    res, _k = _lola3(l, k)
    for i in 1:length(l.w_ds)
        wd!(slice(res, 3 + i), l.w_ds[i], _k, l.w_d_reducers[i])
    end
    return res
end

function ChainRulesCore.rrule(l::LoLa, k)
    Ω, _k = _lola3(l, k)
    pullbacks_wd = ntuple(length(l.w_ds)) do i
        _, pb_w, pb_k = wd_adjoint!(slice(Ω, 3 + i), l.w_ds[i], _k, l.w_d_reducers[i])
        return pb_w, pb_k
    end

    function lola_pullback(Δ)
        T = eltype(Δ)
        dk = @thunk begin
            if ndims(k) == 1
                @tullio dE[i] := $(l.w_E)[j, i] * Δ[3, j]
            else
                @tullio dE[i, l] := $(l.w_E)[j, i] * Δ[3, j, l]
            end
            dk = similar(k)
            map!(dk, k, slice(Δ, 1), slice(Δ, 2), slice(Ω, 2), dE) do k_i, Δ1, Δ2, Ω2, dE
                2Δ1 * k_i + SA{eltype(Δ2)}[Δ2 / Ω2 * k_i[1], Δ2 / Ω2 * k_i[2], 0, dE]
            end
            _dk = reinterpret(reshape, T, dk)
            for i in 1:length(l.w_ds)
                pullbacks_wd[i][2](_dk, slice(Δ, 3 + i))
            end
            return dk
        end
        dw_E = @thunk if ndims(k) == 1
            @tullio dw_E[i, j] := Δ[3, i] * E(k[j])
        else
            @tullio dw_E[i, l] := Δ[3, i, j] * E(k[l, j])
        end
        dw_ds = ntuple(length(l.w_ds)) do i
            @thunk begin
                dw_d = reinterpret(reshape, T, zero(l.w_ds[i]))
                pullbacks_wd[i][1](dw_d, slice(Δ, 3 + i))
                return dw_d
            end
        end
        return Composite{typeof(l)}(; w_E=dw_E, w_ds=dw_ds), dk
    end
    return Ω, lola_pullback
end

# FiniteDifferences.to_vec(l::LoLa) = [vec(l.w_E); vec.(l.w_ds)...], v -> LoLa(reshape(v[1:length(l.w_E)], axes(l.w_E)), ntuple(i -> reshape(v[length(l.w_E) + mapreduce(i->length(l.w_ds[i]), +, 1:i-1; init=0) .+ (1:length(l.w_ds[i]))], axes(l.w_ds[i])), length(l.w_ds)), l.w_d_reducers)
