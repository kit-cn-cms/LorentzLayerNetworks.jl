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
    (k[4, $(j...)] - k[4, $(m...)])^2 - (k[1, $(j...)] - k[1, $(m...)])^2 - (k[2, $(j...)] - k[2, $(m...)])^2 - (k[3, $(j...)] - k[3, $(m...)])^2
)

kernel_wd_vec = :(
    f[m] = w[j, m] * $(d_kk([:j], [:m]))
)
kernel_wd_mat = :(
    f[m, l] = w[j, m] * $(d_kk([:j, :l], [:m, :l]))
)

kernel_wd_argmin_vec = :(
    f[m] = (j, w[j, m] * $(d_kk([:j], [:m])))
)
kernel_wd_argmin_mat = :(
    f[m, l] = (j, w[j, m] * $(d_kk([:j, :l], [:m, :l])))
)

kernel_wd_dk_add_vec = :(
    dk[μ, i] += 2g(μ) * (Δ[j] * w[i, j] + Δ[i] * w[j, i]) * (k[μ, i] - k[μ, j])
)
kernel_wd_dk_add_mat = :(
    dk[μ, i, l] += 2g(μ) * (Δ[j, l] * w[i, j] + Δ[i, l] * w[j, i]) * (k[μ, i, l] - k[μ, j, l])
)

kernel_wd_dw_add_vec = :(
    dw[i, h] += Δ[h] * $(d_kk([:i], [:h]))
)
kernel_wd_dw_add_mat = :(
    dw[i, h] += Δ[h, l] * $(d_kk([:i, :l], [:h, :l]))
)

g(μ) = ifelse(μ == 4, 1, -1)
kernel_wd_dk_min_vec = :(
    dk[μ, i] += 2g(μ) * Δ[m] * w[j_min[m], m] * ((i == j_min[m]) - (i == m)) * (k[μ, j_min[m]] - k[μ, m])
)
kernel_wd_dk_min_mat = :(
    dk[μ, i, l] += 2g(μ) * Δ[m, l] * w[j_min[m, l], m] * ((i == j_min[m, l]) - (i == m)) * (k[μ, j_min[m, l], l] - k[μ, m, l])
)

kernel_wd_dw_min_vec = :(
    dw[i, h] += (i == j_min[h]) ? Δ[h] * $(d_kk([:(j_min[h])], [:h])) : zero(T)
)
kernel_wd_dw_min_mat = :(
    dw[i, h] += (i == j_min[h, l]) ? Δ[h, l] * $(d_kk([:(j_min[h, l]), :l], [:h, :l])) : zero(T)
)


for (N, kernel, kernel_argmin, kernel_dk_add, kernel_dw_add, kernel_dk_min, kernel_dw_min) in [
    (1, kernel_wd_vec, kernel_wd_argmin_vec, kernel_wd_dk_add_vec, kernel_wd_dw_add_vec, kernel_wd_dk_min_vec, kernel_wd_dw_min_vec),
    (2, kernel_wd_mat, kernel_wd_argmin_mat, kernel_wd_dk_add_mat, kernel_wd_dw_add_mat, kernel_wd_dk_min_mat, kernel_wd_dw_min_mat),
]
    for red in [:+, :min]
        @eval wd!(f::AbstractArray{<:Any,$N}, w, k::AbstractArray{<:Any,$(N+1)}, ::typeof($red)) = @tullio ($red) $kernel
    end

    @eval function wd_adjoint!(_f::AbstractArray{T,$N}, w, k::AbstractArray{<:Any,$(N+1)}, ::typeof(+)) where {T}
        f = wd!(_f, w, k, +)
        function wd_pullback_w!(dw, Δ)
            @tullio $kernel_dw_add
        end
        function wd_pullback_k!(dk, Δ)
            @tullio $kernel_dk_add
        end
        return f, wd_pullback_w!, wd_pullback_k!
    end

    @eval function wd_adjoint!(_f::AbstractArray{T,$N}, w, k::AbstractArray{<:Any,$(N+1)}, ::typeof(min)) where {T}
        f = ArgMinArray(similar(_f, Int), _f)
        @tullio (argmin_inner) $kernel_argmin init=(1, typemax(T))
        j_min = f.idx
        function wd_pullback_w!(dw, Δ)
            @tullio $kernel_dw_min
        end
        function wd_pullback_k!(dk, Δ)
            @tullio $kernel_dk_min
        end
        return f.min, wd_pullback_w!, wd_pullback_k!
    end
end
