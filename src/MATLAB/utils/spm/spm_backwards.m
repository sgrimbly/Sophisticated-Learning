function [L] = spm_backwards(O, Q, A, B, u, t, T)
    % Backwards smoothing to evaluate posterior over initial states
    %--------------------------------------------------------------------------
    L = Q{t, 2};
    p = 1;

    for timestep = (t + 1):T

        % belief propagation over hidden states
        %------------------------------------------------------------------

        p = B{2}(:, :, 1) * p;

        for state = 1:numel(L)
            % and accumulate likelihood
            %------------------------------------------------------------------
            for g = 3:3
                % possible_states = O{g,timestep}*A{g}(:,:);
                obs_dist = O{g, timestep};
                obs_dist = obs_dist(:);
                temp = sum(A{g} .* reshape(obs_dist, [], 1, 1), 1);
                temp = permute(temp, [3, 2, 1]);
                temp = temp * Q{timestep, 1}';
                aaa = temp' * p(:, state);
                L(state) = L(state) .* aaa;

            end

        end

    end

    % marginal distribution over states
    %--------------------------------------------------------------------------
    L = spm_norm(L(:));
end
