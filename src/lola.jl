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

#(l::LoLa)(k) = vstack(
#    m².(k),
#    p_T.(k),
#    (l.w_E * E.(k)),
#    ntuple(length(l.w_ds)) do i
#        w_d²(l.w_ds[i], k, l.w_d_reducers[i])
#    end...,
#)

using Compat
include("lola_kernel.jl")

# FIXME
#_mul!(x...) = mul!(x...)
#function _mul!(Y::CUDA.StridedCuVector, A::StridedCuMatrix, X::StridedCuVector)
#    @assert size(A, 1) == size(Y, 1) && size(A, 2) == size(X, 1) && size(Y, 2) == size(X, 2)
#    CUDA.CUBLAS.cublasSgemv_v2(CUDA.CUBLAS.handle(), 'N', size(A)..., true, A, stride(A, 2), X, stride(X, 1), false, Y, stride(Y, 1))
#    return Y
#end
E!(_E::AbstractVector, w, k::AbstractVector) = @tullio _E[i] = w[i, j] * E(k[j])
E!(_E::AbstractMatrix, w, k::AbstractMatrix) = @tullio _E[i, l] = w[i, j] * E(k[j, l])

function _lola3(l, k)
    T = eltype(eltype(k))
    res = similar(k, T, 3 + length(l.w_ds), axes(k)...)
    slice(i) = view(res, i, axes(k)...)
    map!(m², slice(1), k)
    map!(p_T, slice(2), k)
    E!(slice(3), l.w_E, k)
    _k = reinterpret(reshape, T, k)
    return res, _k
end

function (l::LoLa)(k)
    res, _k = _lola3(l, k)
    for i in 1:length(l.w_ds)
        wd!(slice(3 + i), l.w_ds[i], _k, l.w_d_reducers[i])
    end
    return res
end

using StaticArrays
using StaticArrays: SOneTo

function ChainRulesCore.rrule(l::LoLa, k)
    Ω, _k = _lola3(l, k)
    pullbacks_wd = ntuple(length(l.w_ds)) do i
        _, pb_w, pb_k = wd_adjoint!(slice(3 + i), l.w_ds[i], _k, l.w_d_reducers[i])
        return pb_w, pb_k
    end

    function lola_pullback(Δ)
        T = eltype(Δ)
        dk = @thunk begin
            dE = l.w_E' * Δ
            dk = similar(k)
            map!(dk, CartesianIndices(k), k) do (i, k_i)
                @. 2Δ[1, i] * k_i + Δ[2, i] / 2Ω[2, i] * k_i + SA{T}[0, 0, 0, Δ[3, i], dE[i]])
            end
            for i in 1:length(l.w_ds)
                pullbacks_wd[1][2](dk, Δ)
            end
            return dk
        end
        dw_E = @thunk if ndims(k) == 1
            @tullio dw_E[1, i] := w[j, i] * E(k[j])
        else
            @tullio dw_E[i, l] := w[i, j] * E(k[l, j])
        end
        dw_ds = ntuple(length(l.w_ds)) do i
            @thunk begin
                dw_d = reinterpret(reshape, T, zero(k))
                pullbacks_wd[i][1](dw_d, Δ)
                return dw_d
            end
        end
        return Composite{typeof(l)}(; w_E=dw_E, w_ds=dw_ds), dk
    end
    return Ω, lola_pullback
end

vstack(xs...) = cat((reshape(x, 1, size(x)...) for x in xs)...; dims=1)
function ChainRulesCore.rrule(::typeof(vstack), x...)
    function vstack_pullback(Δ)
        return NO_FIELDS, ntuple(i -> @thunk(Δ[i, ntuple(_ -> :, ndims(Δ) - 1)...]), length(x))...
    end
    vstack(x...), vstack_pullback
end
