%% =======================================================================
%% calculate_posterior (GPU-optimised)
%% =======================================================================
function P = calculate_posterior(P, A, O, t)
% GPU-based version of calculate_posterior
%
%   - We assume that P, A, and O refer to cell arrays containing gpuArrays.
%   - The random calls (rand) also occur on the GPU courtesy of the global RNG.

    % Ensure P{t, 2} is a column vector
    if size(P{t, 2}, 2) > 1
        P{t, 2} = P{t, 2}';
    end

    for fact = 2:2
        L   = gpuArray(1);
        num = numel(A);

        for modal = 2:num
            obs = find(cumsum(O{modal, t}) >= rand, 1);  % GPU-based cumsum, rand
            temp = A{modal}(obs, :, :);
            temp = permute(temp, [3, 2, 1]);
            L = L .* temp;
        end

        % multiply across the other factor
        for f = 1:2
            if f ~= fact
                if f == 2
                    LL = P{t, f} * L;
                else
                    LL = L * P{t, f}';
                end
            end
        end

        y = LL .* P{t, fact};
        P{t, fact} = normalise(y)';  % ensure GPU-based normalisation
    end
end
% -------------------------------------------------------------------------
