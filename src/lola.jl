using LinearAlgebra
#using Tullio

struct LoLa{A<:AbstractMatrix,T<:Tuple,Fs<:Tuple}
    w_E::A
    w_ds::T
    w_d_reducers::Fs
end

Flux.@functor LoLa #(w_E, w_ds)

m²(k) = sum(abs2, k)
p_T(k) = hypot(k[1], k[2])

E(k) = k[4]

g(x, y) = x[4] * y[4] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3] #dot(x[StaticArrays.SOneTo(3)], y[StaticArrays.SOneTo(3)])
minkowski(x) = g(x, x)
function ChainRulesCore.rrule(::typeof(minkowski), x)
    function minkowski_pullback(Δ)
        return NO_FIELDS, -2Δ * typeof(x)(-x[1], -x[2], -x[3], x[4])
    end
    return minkowski(x), minkowski_pullback
end
# Zygote._zero(xs::AbstractArray{<:StaticArray}, T) = map(zero, xs)
function w_d²(w_d, k::AbstractVector, reducer!)
    res = similar(k, eltype(eltype(k)))
    reducer!(res, CartesianIndices((axes(k)..., axes(k, 1)))) do _I
        I = Tuple(I_)
        j, l, m = I[1], Base.front(Base.tail(I)), I[end]
        @inbounds w_d[j, m] * minkowski(k[j, l...] - k[m, l...])
    end
    return res
end
function ChainRulesCore.rrule(::typeof(w_d²), w_d, k::AbstractMatrix, reducer)
    function w_d²_pullback(Δ)
        dw_d = zero(w_d)
        dk = mapreduce(hcat, eachcol(k), eachcol(Δ)) do k, Δ
            pb = Zygote.pullback(w_d², w_d, k, reducer)[2](Δ)
            dw_d .+= pb[1]
            return pb[2]
        end
        return NO_FIELDS, dw_d, dk, DoesNotExist()
    end
    return w_d²(w_d, k, reducer), w_d²_pullback
end
#w_d²(w_d, k::AbstractVector, reducer) = @tullio reducer k̃[j] := w_d[j, m] * minkowski(k[j] - k[m])
#w_d²(w_d, k::AbstractMatrix, reducer) = @tullio reducer k̃[j, l] := w_d[j, m] * minkowski(k[j, l] - k[m, l])

_map(f, x) = map(f, x)
function ChainRulesCore.rrule(::typeof(_map), f, x)
    function _map_pullback(Δ)
        NO_FIELDS, NO_FIELDS, map((x, dx) -> Zygote.pullback(f, x)[2](dx), x, Δ)
    end
    return map(f, x), _map_pullback
end

(l::LoLa)(k) = vstack(
    m².(k),
    p_T.(k),
    (l.w_E * E.(k)),
    ntuple(length(l.w_ds)) do i
        w_d²(l.w_ds[i], k, l.w_d_reducers[i])
    end...,
)

using StaticArrays
using StaticArrays: SOneTo

function ChainRulesCore.rrule(l::LoLa, k)
    Ω = l(k)

    function lola_pullback(Δ)
        T = eltype(Δ)
        dE = l.w_E' * Δ
        dmink = similar(k, axes(k, 1), axes(k, 1))
        broadcast(k, reshape(k, 1, :), axes(k, 1), axes(k, 2)) do k_j, k_m, j, m
            (k_j .- k[m])
        end
            sum(eachindex(k)) do m
                SA{T}[k_j .- k[m] ]
        end
        dw_d = map(Δ) do Δ_i

        end
        map(pairs(k)) do (i, k_i)
            @. Δ * (2 * k_i + k_i / 2Ω[2, i] + SA{T}[0, 0, 0, dE[i]])
        end

    end
    return map(f, x), _map_pullback
end

vstack(xs...) = cat((reshape(x, 1, size(x)...) for x in xs)...; dims=1)
function ChainRulesCore.rrule(::typeof(vstack), x...)
    function vstack_pullback(Δ)
        return NO_FIELDS, ntuple(i -> @thunk(Δ[i, ntuple(_ -> :, ndims(Δ) - 1)...]), length(x))...
    end
    vstack(x...), vstack_pullback
end
