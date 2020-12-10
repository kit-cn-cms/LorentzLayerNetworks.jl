using LinearAlgebra, ChainRulesCore
using Compat

struct CoLa{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    C::A
end

Base.size(c::CoLa) = (size(c.C, 1) + size(c.C, 2), size(c.C, 2))

@inline function Base.getindex(c::CoLa{T}, i::Int, j::Int) where T
    Base.@boundscheck Base.checkbounds(c, i, j)
    N = size(c.C, 2)
    return if i <= N
        T(i == j)
    else
        @inbounds c.C[i-N, j]
    end
end

function _boundschecks(k_, c, k)
    Base.require_one_based_indexing(k_, k)
    @assert size(k, 1) == size(c, 2)
    @assert size(k_, 1) == size(c, 1)
    @assert size(k, 2) == size(k_, 2)
end

function _kmul!(k_, c, k, α, β; N=0)
    r_k_ = reinterpret(reshape, eltype(eltype(k_)), k_)
    r_k = reinterpret(reshape, eltype(eltype(k)), k)
    @show summary.((k_, c, k))
    @show summary.((r_k_, c, r_k))
    for μ in 1:4
        mul!(@view(r_k_[μ, N+1:end, :]), c, view(r_k, μ, :, :), α, β)
    end
    return k_
end
function _mul!(k_, c, k, α, β, _add)
    N = size(k, 1)
    @inbounds k_[1:N, :] .= _add.(k, view(k_, 1:N, :))
    _kmul!(k_, c.C, k, α, β; N)
    return k_
end

function LinearAlgebra.mul!(k_::AbstractMatrix, c::CoLa, k::AbstractMatrix, α::Number, β::Number)
    _boundschecks(k_, c, k)
    return _mul!(k_, c, k, α, β, LinearAlgebra.MulAddMul(α, β))
end

function LinearAlgebra.mul!(k_::AbstractVector, c::CoLa, k::AbstractVector, α::Number, β::Number)
    _boundschecks(k_, c, k)
    return _mul!(k_, c, k, α, β, LinearAlgebra.MulAddMul(α, β))
end

using ChainRulesCore

function _kmul(c, k)
    k_ = similar(k, size(c, 1), size(k, 2))
    @show summary.((k_, c, k))
    _kmul!(k_, c, k, true, false)
    return k_
end

function ChainRulesCore.rrule(
    ::typeof(*),
    c::CoLa{<:Union{Real,Complex}},
    k::AbstractVecOrMat,
)
    function mul_pullback(Δ)
        t(k) = k isa AbstractVector ? reshape(k, 1, :) : PermutedDimsArray(k, (2, 1))
        dc = @thunk let
            N = size(k, 1)
            #dC = adjoint.(@view(Δ[N+1:end, :])) * t(k)
            dC = _kmul(adjoint.(@view(Δ[N+1:end, :])), t(k))
            Composite{typeof(dC)}(; C = dC)
        end
        return NO_FIELDS, dc, @thunk(c' * Δ)
    end
    return c * k, mul_pullback
end

function Flux.update!(opt, c::CoLa, dc)
    if haskey(dc, :C)
        c.C .-= Flux.update!(opt, c.C, dc[:C])
    end
    return c
end

struct Linear{A<:AbstractMatrix}
    m::A
end

Flux.@functor Linear

(l::Linear)(x) = l.m * x
