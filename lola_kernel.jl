using Tullio, CUDA, KernelAbstractions, Zygote, FiniteDifferences, Adapt

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
Tullio.storage_type(a::ArgMinArray) = typeof(a.min)

argmin_inner(x, y) = ifelse(isless(x[2], y[2]), x, y);

kernel_wd_vec = :(
    f[m] = w[j, m] *
        ((k[j, 4] - k[m, 4])^2 - (k[j, 1] - k[m, 1])^2 - (k[j, 2] - k[m, 2])^2 - (k[j, 3] - k[m, 3])^2)
)
kernel_wd_mat = :(
    f[m, l] = w[j, m] *
        ((k[j, l, 4] - k[m, l, 4])^2 - (k[j, l, 1] - k[m, l, 1])^2 - (k[j, l, 2] - k[m, l, 2])^2 - (k[j, l, 3] - k[m, l, 3])^2)
)

for red in [:+, :min], (N, kernel) in [1 => kernel_wd_vec, 2 => kernel_wd_mat]
    @eval wd!(f::AbstractArray{<:Any,$N}, w, k::AbstractArray{<:Any,$(N+1)}, ::typeof($red)) = @tullio ($red) $kernel
end

@eval function wd_pullback!(f)
end
