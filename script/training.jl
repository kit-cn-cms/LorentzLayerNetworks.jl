function train!(;
    model, ds_train, ds_test, ds_validation,
    loss, optimizer, measures, recorded_measures,
    max_epochs, min_epochs, early_stopping_n, early_stopping_percentage,
)
    for i in 1:max_epochs
        @info "Epoch $i"
        train, test, validation = step!(
            model, ds_train, ds_test, ds_validation;
            loss_function=loss, optimizer, measures,
        )
        @show train
        @show test
        @show validation
        push!(
            recorded_measures,
            merge(prefix_labels.((train, test, validation), (:train_, :test_, :validation_))...),
        )

        #let
        #    global fig, ax = subplots()
        #    losses = ["train_loss", "test_loss", "validation_loss"]
        #    plts = foreach(1:3, losses) do i, loss
        #        ax.plot(
        #            getproperty(recorded_measures, loss),
        #            label=replace(loss, '_' => ' '),
        #        )
        #    end
        #    ax.legend()
        #    display(fig)
        #end

        if i >= min_epochs
            if argmin(recorded_measures[!, :test_loss]) <= i - early_stopping_n
                @info "Loss has not decreased in the last 10 epochs, stopping training"
                break
            elseif test.loss / train.loss > 1 + early_stopping_percentage / 100
                @info "Test loss more than $early_stopping_percentage% greater than training loss, stopping training"
                break
            end
        end
    end
end
