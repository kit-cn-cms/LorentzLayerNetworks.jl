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

g(x, y) = x[4] * y[4] - dot(x[1:3], y[1:3])
minkowski(x) = g(x, x)
w_d²(w_d, k::AbstractVector, reducer) = map(axes(w_d, 1)) do j
    reducer(axes(w_d, 2)) do m
        @inbounds w_d[j, m] * minkowski(k[j] - k[m])
    end
end
w_d²(w_d, k::AbstractMatrix, reducer) = map(Iterators.product(axes(k)...)) do (j, l)
    reducer(axes(w_d, 2)) do m
        @inbounds w_d[j, m] * minkowski(k[j, l] - k[m, l])
    end
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

(l::LoLa)(k) = vstack(
    m².(k),
    p_T.(k),
    (l.w_E * E.(k)),
    ntuple(length(l.w_ds)) do i
        w_d²(l.w_ds[i], k, l.w_d_reducers[i])
    end...,
)

vstack(xs...) = cat((reshape(x, 1, size(x)...) for x in xs)...; dims=1)
function ChainRulesCore.rrule(::typeof(vstack), x...)
    function vstack_pullback(Δ)
        return NO_FIELDS, ntuple(i -> @thunk(Δ[i, ntuple(_ -> :, ndims(Δ) - 1)...]), length(x))...
    end
    vstack(x...), vstack_pullback
end
