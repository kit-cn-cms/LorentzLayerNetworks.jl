function Base.reinterpret(::typeof(reshape), ::Type{T}, a::A) where {T,S,A<:CuArray{S}}
    isbitstype(T) || throwbits(S, T, T)
    isbitstype(S) || throwbits(S, T, S)
    if sizeof(S) == sizeof(T)
        N = ndims(a)
    elseif sizeof(S) > sizeof(T)
        rem(sizeof(S), sizeof(T)) == 0 || throwintmult(S, T)
        N = ndims(a) + 1
    else
        rem(sizeof(T), sizeof(S)) == 0 || throwintmult(S, T)
        N = ndims(a) - 1
        N > -1 || throwsize0(S, T, "larger")
        axes(a, 1) == Base.OneTo(sizeof(T) ÷ sizeof(S)) || throwsize1(a, T)
    end
    paxs = axes(a)
    new_axes = if sizeof(S) > sizeof(T)
        (Base.OneTo(div(sizeof(S), sizeof(T))), paxs...)
    elseif sizeof(S) < sizeof(T)
        Base.tail(paxs)
    else
        paxs
    end
    reshape(reinterpret(T, vec(a)), new_axes)
end

@noinline function throwintmult(S::Type, T::Type)
    throw(ArgumentError("`reinterpret(reshape, T, a)` requires that one of `sizeof(T)` (got $(sizeof(T))) and `sizeof(eltype(a))` (got $(sizeof(S))) be an integer multiple of the other"))
end
@noinline function throwsize1(a::AbstractArray, T::Type)
    throw(ArgumentError("`reinterpret(reshape, $T, a)` where `eltype(a)` is $(eltype(a)) requires that `axes(a, 1)` (got $(axes(a, 1))) be equal to 1:$(sizeof(T) ÷ sizeof(eltype(a))) (from the ratio of element sizes)"))
end
@noinline function throwbits(S::Type, T::Type, U::Type)
    throw(ArgumentError("cannot reinterpret `$(S)` as `$(T)`, type `$(U)` is not a bits type"))
end
@noinline function throwsize0(S::Type, T::Type, msg)
    throw(ArgumentError("cannot reinterpret a zero-dimensional `$(S)` array to `$(T)` which is of a $msg size"))
end
