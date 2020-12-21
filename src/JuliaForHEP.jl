module JuliaForHEP

const __revise_mode__ = :sigs

using Flux, Zygote, CUDA, WeightedOnlineStats, ProgressMeter
using Pandas: read_hdf
import Random

export accuracy, extract_cols, scale_weights, split_indices, split_datasets, weighted_mean,
    step!, prefix_labels

"""
    accuracy(ŷ::AbstractMatrix, y::AbstractMatrix{Bool})

Calculate accuracy of predictions `ŷ` against ground truth `y`.
"""
function accuracy(ŷ::AbstractMatrix{T}, y::AbstractMatrix{Bool}) where {T}
    _a = Ref{T}()
    _accuracy_kernel!(_a, ŷ, y)
    return _a[]
end

function accuracy(ŷ::CuMatrix{T}, y::AbstractMatrix{Bool}) where {T}
    _a = CuArray{T}(undef)
    @cuda threads=size(ŷ, 2) _accuracy_kernel!(_a, ŷ, y)
    return CUDA.@allowscalar _a[]
end

function _accuracy_kernel!(_a, ŷ, y)
    _a[] = count(axes(ŷ, 2)) do i
        @inbounds y[argmax(view(ŷ, :, i)), i]
    end / convert(eltype(_a), size(ŷ, 2))
    nothing
end

"""
    extract_cols(h5files, column_names)

Extract features given by `column_names` from all files in `h5files` and concatenate them.
"""
function extract_cols(h5files, column_names, idx=:)
    return mapreduce(hcat, h5files) do f
        df = read_hdf(f)
        PermutedDimsArray(Array(df[column_names][idx]), (2, 1))
    end
end

"""
    scale_weights(weights, outputs)

Scale `weights`, so that the sum over all weights for each output feature is always one.
"""
function scale_weights(weights, outputs)
    sums_weights = Dict{Symbol,Float32}()
    for (i, class) in pairs(outputs)
        sums_weights[class] = get(sums_weights, class, 0f0) + weights[i]
    end
    return map(CartesianIndices(weights)) do i
        weights[i] / sums_weights[outputs[i]]
    end
end

function split_indices(indices, proportions; shuffle=true, rng=Random.GLOBAL_RNG)
    if shuffle
        indices = Random.shuffle(rng, indices)
    end
    split = [0; round.(Int, cumsum(proportions) .* length(indices))]

    if split[end] > length(indices)
        throw(ArgumentError("Invalid proportions $proportions, sum is > 1"))
    elseif split[end] < length(indices)
        @warn "$(length(indices) - slit[end]) indices not assigned to any dataset"
    end

    return ntuple(length(split) - 1) do i
        indices[split[i]+1:split[i+1]]
    end
end

function split_datasets(datasets, proportions; kwargs...)
    return split_datasets(identity, datasets, proportions; kwargs...)
end
function split_datasets(f, datasets, proportions; kwargs...)
    if !all(ds -> axes(ds)[end] == axes(datasets[1])[end], datasets[2:end])
        throw(ArgumentError("Last axes of datasets don't match"))
    end
    return map(split_indices(axes(datasets[1])[end], proportions; kwargs...)) do idx
        map(datasets) do ds
            f(ds[ntuple(_ -> (:), ndims(ds)-1)..., idx])
        end
    end
end

"""
    weighted_mean(x, weights)

AD-friendly weighted mean. `weights` is always normalized.
"""
weighted_mean(x, weights) = sum(x .* weights ./ sum(weights))


function step!(model, ds_train, ds_tests...; loss_function, optimizer, measures=(;))
    ps = Flux.params(model)

    trainmode!(model)

    measures_train = map(_ -> WeightedMean(), measures)
    p = Progress(div(ds_train.nobs, ds_train.batchsize, RoundUp))
    for (x, y, weights) in ds_train
        local ŷ
        loss, pb = Zygote.pullback(ps) do
            ŷ = model(x)
            return loss_function(ŷ, y; weights)
        end
        gs = pb(one(loss))
        Flux.update!(optimizer, ps, gs)

        state = (; x, y, weights, loss, ŷ, mode=:train)
        for (k, f) in pairs(measures)
            fit!(measures_train[k], f(state), size(y, 2))
        end
        next!(p)
    end

    testmode!(model)

    measures_tests = map(enumerate(ds_tests)) do (i, ds_test)
        measures_test = map(_ -> WeightedMean(), measures)
        for (x, y, weights) in ds_test
            ŷ = model(x)
            loss = loss_function(ŷ, y; weights)

            state = (; x, y, weights, loss, ŷ, mode=Symbol(:test_, i))
            for (k, f) in pairs(measures)
                fit!(measures_test[k], f(state), size(y, 2))
            end
        end
        return measures_test
    end

    return map.(value, (measures_train, measures_tests...))
end

function prefix_labels(nt::NamedTuple{keys}, prefix) where {keys}
    return NamedTuple{Symbol.(prefix, keys)}(Tuple(nt))
end

export CoLa, Linear, LoLa, LorentzSidechain

using Tullio, CUDA, KernelAbstractions, Adapt, Compat, LinearAlgebra
using StaticArrays
using ChainRulesCore

# 🏴‍☠️
Flux.params!(p::Zygote.Params, k::AbstractArray{<:SArray}, seen=IdSet()) = push!(p, k)
Flux.fmap(f, a::AbstractArray{<:SArray}) = f(a)
using ChainRulesCore: backing
Zygote.accum(x::Union{NamedTuple,Composite}, y::Union{NamedTuple,Composite}) = Zygote.accum(backing(x), backing(y))

function ChainRulesCore.rrule(::typeof(reinterpret), ::typeof(reshape), ::Type{T}, x::AbstractArray) where {T}
    function reinterpret_reshape_pullback(Δ)
        return NO_FIELDS, NO_FIELDS, NO_FIELDS, reinterpret(reshape, eltype(x), Δ)
    end
    return reinterpret(reshape, T, x), reinterpret_reshape_pullback
end

include("cola.jl")
include("lola.jl")
include("lorentz_sidechain.jl")

end
