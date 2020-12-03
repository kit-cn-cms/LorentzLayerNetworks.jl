using LinearAlgebra

struct LoLa{A<:AbstractMatrix}
    w_E::A
    w_d::A
end

Flux.@functor LoLa

m²(k) = sum(abs2, k)
p_T(k) = hypot(k[1], k[2])

E(k) = k[4]

g(x, y) = x[4] * y[4] - dot(x[1:3], y[1:3])
minkowski(x) = g(x, x)
w_d²(w_d, k) = w_d .* minkowski.(reshape(k, 1, :) .- k)

(l::LoLa)(k) = [
    m².(k)'
    p_T.(k)'
    (l.w_E * E.(k))'
    w_d²(l.w_d, k)
]

#Zygote.@adjoint function (l::LoLa)(k)
#    m, dm = Zygote._pullback(__context__, map, m², k)
#    p, dp = Zygote._pullback(__context__, map, p_T, k)
#    e, de = Zygote._pullback(__context__, (w_E, k) -> w_E * map(E, k), l.w_E, k)
#    d, dd = Zygote._pullback(__context__, w_d², l.w_d, k)
#
#    function lola_pullback(Δ)
#        _, dw_E, dk_E = de(view(Δ, 3, :))
#        _, dw_d, dk_d = dd(@view Δ[4:end, :])
#        return (
#            Composite{typeof(l)}(w_E=dw_E, w_d=dw_d),
#            dm(view(Δ, 1, :))[3] + dp(view(Δ, 2, :))[3] + dk_E + dk_d,
#        )
#    end
#    return [m'; p'; e'; d'], lola_pullback
#end
function ChainRulesCore.rrule(l::LoLa, k)
    m, dm = Zygote.pullback(map, m², k)
    p, dp = Zygote.pullback(map, p_T, k)
    e, de = Zygote.pullback((w_E, k) -> w_E * map(E, k), l.w_E, k)
    d, dd = Zygote.pullback(w_d², l.w_d, k)

    function lola_pullback(Δ)
        dw_E, dk_E = de(view(Δ, 3, :))
        dw_d, dk_d = dd(@view Δ[4:end, :])
        return (
            Composite{typeof(l)}(w_E=dw_E, w_d=dw_d),
            dm(view(Δ, 1, :))[2] + dp(view(Δ, 2, :))[2] + dk_E + dk_d,
        )
    end
    return [m'; p'; e'; d], lola_pullback
end
