function [P, Q, true_states] = updateEnvironmentStates(P, Q, true_states, trial, t, num_states, chosen_action, D, B, bb, food_sources)
    for factor = 1:2
        Q_prev = Q{t - 1, factor}';
        
        % Update Q based on the factor
        if factor == 1
            B_current = B{1}(:, :, chosen_action(t - 1));
            Q{t, factor} = (B_current * Q_prev)';
            true_states{trial}(factor, t) = find(cumsum(B{1}(:, true_states{trial}(factor, t - 1), chosen_action(t - 1))) >= rand, 1);
        else
            B_current = bb{2}(:, :, chosen_action(t - 1));
            Q{t, factor} = (B_current * Q_prev)';
            true_states{trial}(factor, t) = find(cumsum(B{2}(:, true_states{trial}(factor, t - 1), 1)) >= rand, 1);
        end
    end
end
