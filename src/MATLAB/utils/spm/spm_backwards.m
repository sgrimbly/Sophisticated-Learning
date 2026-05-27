function [L] = spm_backwards(O, Q, A, B, u, t, T)
    % Backwards smoothing to evaluate posterior over initial states
    %--------------------------------------------------------------------------
    % Optimisation 2026-05-03: the inner loop body's `temp` computation
    % depends only on (timestep, g) — not on `state` — but was being recomputed
    % numel(L)x per timestep (4 redundant repeats). Hoisted out and the
    % per-state update vectorised: was `for state=1:numel(L); aaa = temp' *
    % p(:,state); L(state) = L(state) * aaa; end`, now `L = L .* (temp' * p)'`.
    % spm_backwards was 39% of SL wall time per profile; biggest single
    % SL-side target. FP-reorder vs original (matrix-multiply reduction order
    % differs from per-element accumulation), verified statistically.

    L = Q{t, 2};
    L = L(:);  % column form so the vectorised update preserves the original
               % spm_norm(L(:)) return shape.
    p = 1;

    for timestep = (t + 1):T

        % belief propagation over hidden states
        %------------------------------------------------------------------
        p = B{2}(:, :, 1) * p;

        % outcome marginal over contexts — only g=3 in current call sites,
        % loop preserved for forward compatibility (negligible cost).
        for g = 3:3
            obs_dist = O{g, timestep};
            obs_dist = obs_dist(:);
            temp = sum(A{g} .* reshape(obs_dist, [], 1, 1), 1);
            temp = permute(temp, [3, 2, 1]);
            temp = temp * Q{timestep, 1}';
            % Per-state likelihood update vectorised:
            %   for state, aaa = temp' * p(:,state); L(state) = L(state)*aaa
            % is equivalent to L = L .* (temp' * p)' for all states at once.
            L = L .* (temp' * p)';
        end

    end

    % marginal distribution over states
    %--------------------------------------------------------------------------
    L = spm_norm(L);
end
