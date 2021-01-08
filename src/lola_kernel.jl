struct ArgMinArray{I,M,N,AI<:AbstractArray{I,N},AM<:AbstractArray{M,N}} <: AbstractArray{Tuple{I,M},N}
    idx::AI
    min::AM
end

Base.size(a::ArgMinArray) = size(a.min)
Base.@propagate_inbounds function Base.setindex!(a::ArgMinArray, x, i::Int...)
    setindex!(a.idx, x[1], i...)
    setindex!(a.min, x[2], i...)
    return x
end
Adapt.adapt_structure(T, a::ArgMinArray) = ArgMinArray(adapt(T, a.idx), adapt(T, a.min))
Tullio.storage_type(a::ArgMinArray) = Tullio.storage_type(a.min)

argmin_inner(x, y) = ifelse(isless(x[2], y[2]), x, y);

d_kk(j, m) = :(
    (k[4, $(j...)] - k[4, $(m...)])^2 - (k[1, $(j...)] - k[1, $(m...)])^2 -
        (k[2, $(j...)] - k[2, $(m...)])^2 - (k[3, $(j...)] - k[3, $(m...)])^2
)

kernel_wd = :(
    f[m, l] = w[j, m] * $(d_kk([:j, :l], [:m, :l]))
)
kernel_wd_argmin = :(
    f[m, l] = (j, w[j, m] * $(d_kk([:j, :l], [:m, :l])))
)

kernel_wd_dk_add = :(
    dk[μ, i, l] += 2g(μ) * (Δ[j, l] * w[i, j] + Δ[i, l] * w[j, i]) * (k[μ, i, l] - k[μ, j, l])
)
kernel_wd_dw_add = :(
    dw[i, h] += Δ[h, l] * $(d_kk([:i, :l], [:h, :l]))
)

g(μ) = ifelse(μ == 4, 1, -1)
kernel_wd_dk_min = :(
    dk[μ, i, l] += 2g(μ) * Δ[m, l] * w[j_min[m, l], m] * ((i == j_min[m, l]) - (i == m)) *
        (k[μ, j_min[m, l], l] - k[μ, m, l])
)
kernel_wd_dw_min = :(
    dw[i, h] += (i == j_min[h, l]) ? Δ[h, l] * $(d_kk([:(j_min[h, l]), :l], [:h, :l])) : zero(T)
)


for red in [:+, :min]
    @eval function wd!(f::AbstractMatrix, w, k::AbstractArray{<:Any,3}, ::typeof($red))
        return @tullio ($red) $kernel_wd
    end
end

@eval function wd_adjoint!(
    _f::AbstractMatrix{T}, w, k::AbstractArray{<:Any,3}, ::typeof(+),
) where {T}
    f = wd!(_f, w, k, +)
    function wd_pullback_w!(dw, Δ)
        @tullio $kernel_wd_dw_add
    end
    function wd_pullback_k!(dk, Δ)
        @tullio $kernel_wd_dk_add
    end
    return f, wd_pullback_w!, wd_pullback_k!
end

@eval function wd_adjoint!(
    _f::AbstractMatrix{T}, w, k::AbstractArray{<:Any,3}, ::typeof(min),
) where {T}
    f = ArgMinArray(similar(_f, Int), _f)
    @tullio (argmin_inner) $kernel_wd_argmin init=(1, typemax(T))
    j_min = f.idx
    function wd_pullback_w!(dw, Δ)
        @tullio $kernel_wd_dw_min
    end
    function wd_pullback_k!(dk, Δ)
        @tullio $kernel_wd_dk_min
    end
    return f.min, wd_pullback_w!, wd_pullback_k!
end
