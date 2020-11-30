using LinearAlgebra, ChainRulesCore

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

function _mul!(k_, c, k, α, β, _add)
    N = size(k, 1)
    @inbounds k_[1:N, :] .= _add.(k, view(k_, 1:N, :))
    mul!(@view(k_[N+1:end, :]), c.C, k, α, β)
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

function ChainRulesCore.rrule(
    ::typeof(*),
    c::CoLa{<:Union{Real,Complex}},
    k::AbstractMatrix{<:Union{Real,Complex}},
)
    function mul_pullback(Δ)
        dc = @thunk let
            N = size(k, 1)
            dC = @view(Δ[N+1:end, :]) * k'
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
