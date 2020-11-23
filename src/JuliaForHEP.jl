module JuliaForHEP

using Flux, Zygote, CUDA, WeightedOnlineStats, ProgressMeter
using Pandas: read_hdf

export accuracy, extract_cols, scale_weights, weighted_mean, step!

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
    return _a[]
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
function extract_cols(h5files, column_names)
    return mapreduce(hcat, h5files) do f
        df = read_hdf(f)
        PermutedDimsArray(Array(df[column_names]), (2, 1))
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

"""
    weighted_mean(x, weights)

AD-friendly weighted mean. `weights` is always normalized.
"""
weighted_mean(x, weights) = sum(x .* weights ./ sum(weights))


function step!(model, ds_train, ds_test, loss_function, opt, measures)
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
        Flux.update!(opt, ps, gs)

        state = (; x, y, weights, loss, ŷ, mode=:train)
        for (k, f) in pairs(measures)
            fit!(measures_train[k], f(state), size(y, 2))
        end
        next!(p)
    end

    testmode!(model)

    measures_test = map(_ -> WeightedMean(), measures)
    for (x, y, weights) in ds_test
        ŷ = model(x)
        loss = loss_function(ŷ, y; weights)

        state = (; x, y, weights, loss, ŷ, mode=:test)
        for (k, f) in pairs(measures)
            fit!(measures_test[k], f(state), size(y, 2))
        end
    end

    return map.(value, (measures_train, measures_test))
end

end
