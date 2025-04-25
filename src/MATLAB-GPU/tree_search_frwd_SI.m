function [G, P, short_term_memory, best_actions, memory_accessed] = tree_search_frwd_SI(short_term_memory, O, P, a, A, y, B, b, t, T, N, ...
    t_food, t_water, t_sleep, true_t, chosen_action, true_t_food, true_t_water, true_t_sleep, ...
    best_actions, learning_weight, novelty_weight, epistemic_weight, preference_weight, memory_accessed)
% TREE_SEARCH_FRWD_SI - GPU-accelerated version of the original tree_search_frwd_SI
%
% Inputs (gpuArray-based wherever large arrays):
%   short_term_memory - 4D memory array on the GPU
%   O, P, a, A, y, B, b - cell arrays (some 3D) used throughout the search
%   t, T, N, t_food, etc. - scalars (can be GPU or CPU; they are small)
%   chosen_action, best_actions, memory_accessed - progress trackers
%   learning_weight, novelty_weight, epistemic_weight, preference_weight - numeric hyperparameters
%
% Outputs:
%   G - the accumulated quantity (EFE, etc.)
%   P - the updated posterior cell array
%   short_term_memory - updated 4D memory
%   best_actions - updated array of best-chosen actions
%   memory_accessed - count of memory lookups

%% 1) Basic definitions
G = gpuArray(0.02);           % baseline G, stored on GPU
P_prior = P;                  % store prior copy of P
P = calculate_posterior(P, y, O, t);  % compute posterior on GPU
bb{2} = normalise_matrix(b{2});       % normalise b{2} in GPU context

t_food_approx   = round(t_food + 1);
t_water_approx  = round(t_water + 1);
t_sleep_approx  = round(t_sleep + 1);
num_factors     = 2;

%% 2) If t > true_t, incorporate novelty, epistemics, extrinsic
if t > true_t
    novelty = gpuArray(0);

    % Single iteration from t:t, but keep the loop structure to match original
    for timey = t:t
        if timey ~= t
            L = spm_backwards(O, P, A, bb, chosen_action, timey, t);
        else
            L = P{t, 2};
        end
        LL{2} = L;
        LL{1} = P{timey, 1};
        a_prior = a{2};

        for modality = 2:2
            a_learning = O{modality, timey}';

            % Outer product across factors
            for factor = 1:num_factors
                a_learning = spm_cross(a_learning, LL{factor});
            end

            a_learning = a_learning .* (a{modality} > 0);
            a_learning_weighted = a_learning;
            a_learning_weighted(2:end, :) = learning_weight * a_learning(2:end, :);
            a_learning_weighted(1, :)     = a_learning(1, :);

            a_temp = a_prior + a_learning_weighted;
        end

        w = kldir(normalise(a_temp(:)), normalise(a_prior(:)));
        novelty = novelty + w;
    end

    % Add epistemic term
    epi = G_epistemic_value(y, P_prior(t, :)');

    % Accumulate in G
    G = G + novelty_weight  * novelty;
    G = G + epistemic_weight * epi;

    % Compute extrinsic with preference
    for modality = 2:2
        if modality == 2
            C = determineObservationPreference(t_food, t_water, t_sleep);
            C{modality} = C{modality} / preference_weight;  % reduce preference precision
        end

        if modality == 2
            extrinsic = O{2, t} * C{2}';
            G = G + extrinsic;
        end
    end

    % Update times (food, water, sleep) from observation
    t_food  = round(t_food  * (1 - O{2, t}(2))) + 1;
    t_water = round(t_water * (1 - O{2, t}(3))) + 1;
    t_sleep = round(t_sleep * (1 - O{2, t}(4))) + 1;
    t_food_approx   = t_food;
    t_water_approx  = t_water;
    t_sleep_approx  = t_sleep;
end

%% 3) If t < N, explore future actions
if t < N
    actions = randperm(5, 'gpuArray');  % random permutation on GPU
    efe = gpuArray.zeros(1,5);          % expected free energy array on GPU

    % We'll store partial results in a GPU vector K:
    % short_term_memory dimension 4 is 'state', so we match that size
    nStates = size(short_term_memory,4);
    K = gpuArray.zeros(nStates,1);

    for action = gather(actions) % gather => standard for-loop iteration over [1..5] in random order
        % Q next states given chosen action
        Qtmp{1} = (B{1}(:, :, action) * P{t, 1}')';
        Qtmp{2} = (bb{2}(:, :, 1)      * P{t, 2}')';

        qs = spm_cross(Qtmp);  % outer product across factors
        qs = qs(:);

        likely_states = find(qs > 1/8);
        if isempty(likely_states)
            threshold = 1 / numel(qs) * 1 / numel(qs);
            likely_states = find(qs > (1 / numel(qs) - threshold));
        end

        for state = reshape(likely_states,1,[])
            % Check short_term_memory for stored EFE
            if short_term_memory(t_food_approx, t_water_approx, t_sleep_approx, state) ~= 0
                sh      = short_term_memory(t_food_approx, t_water_approx, t_sleep_approx, state);
                K(state) = sh;
                memory_accessed = memory_accessed + 1;
            else
                % Next observation for each modality
                for modal = 1:numel(A)
                    O{modal, t + 1} = normalise(y{modal}(:, state)');
                end

                % Prior for next states
                P{t + 1, 1}     = Qtmp{1};
                P{t + 1, 2}     = Qtmp{2};
                chosen_action(t) = action;

                % Recurse deeper
                [expected_free_energy, P, short_term_memory, best_actions, memory_accessed] = tree_search_frwd_SI( ...
                    short_term_memory, O, P, a, A, y, B, b, ...
                    t + 1, T, N, t_food_approx, t_water_approx, t_sleep_approx, ...
                    true_t, chosen_action, true_t_food, true_t_water, true_t_sleep, ...
                    best_actions, learning_weight, novelty_weight, epistemic_weight, preference_weight, memory_accessed);

                S = max(expected_free_energy);
                K(state) = S;

                % Store in short_term_memory
                short_term_memory(t_food_approx, t_water_approx, t_sleep_approx, state) = S;
            end
        end

        % Weighted sum by qs
        partial_fe = K(likely_states)' .* qs(likely_states);
        efe(action) = efe(action) + 0.7 * sum(partial_fe);
    end

    [maxi, chosen_act] = max(efe);
    G = G + maxi;
    best_actions = [chosen_act, best_actions];
end

end % MAIN FUNCTION: tree_search_frwd_SI
% -------------------------------------------------------------------------

