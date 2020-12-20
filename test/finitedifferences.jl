function FiniteDifferences.to_vec(l::LoLa)
    n = length(l.w_ds[i])
    return (
        [vec(l.w_E); vec.(l.w_ds)...],
        function(v)
            LoLa(
                reshape(v[1:length(l.w_E)], axes(l.w_E)),
                ntuple(length(l.w_ds)) do i
                    reshape(
                        v[length(l.w_E) + mapreduce(i->length(l.w_ds[i]), +, 1:i-1; init=0) .+ (1:n)],
                        axes(l.w_ds[i]),
                    )
                end,
                l.w_d_reducers,
            )
        end,
    )
end
