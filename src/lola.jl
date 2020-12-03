using LinearAlgebra

struct LoLa{A<:AbstractMatrix,T<:Tuple,Fs<:Tuple}
    w_E::A
    w_ds::T
    w_d_reducers::Fs
end

Flux.@functor LoLa (w_E, w_ds)

m²(k) = sum(abs2, k)
p_T(k) = hypot(k[1], k[2])

E(k) = k[4]

g(x, y) = x[4] * y[4] - dot(x[1:3], y[1:3])
minkowski(x) = g(x, x)
w_d²(w_d, k, reducer) = map(axes(w_d, 1)) do j
    reducer(axes(w_d, 2)) do m
        @inbounds w_d[j, m] * minkowski(k[j] - k[m])
    end
end

(l::LoLa)(k) = vstack(
    m².(k),
    p_T.(k),
    (l.w_E * E.(k)),
    ntuple(length(l.w_ds)) do i
        w_d²(l.w_ds[i], k, l.w_d_reducers[i])
    end...,
)

vstack(x::AbstractVector...) = vcat(adjoint.(x)...)
function ChainRulesCore.rrule(::typeof(vstack), x...)
    function vstack_pullback(Δ)
        return NO_FIELDS, ntuple(i -> @thunk(Δ[i, :]), length(x))...
    end
    vstack(x...), vstack_pullback
end
