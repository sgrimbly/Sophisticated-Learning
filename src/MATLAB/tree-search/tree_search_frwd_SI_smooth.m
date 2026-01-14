function [G, P, short_term_memory, best_actions, memory_accessed, efe_components] = tree_search_frwd_SI_smooth(short_term_memory, O, P, a, A, y, B, b, t, T, N, t_food, t_water, t_sleep, true_t, chosen_action, true_t_food, true_t_water, true_t_sleep, best_actions, learning_weight, novelty_weight, epistemic_weight, preference_inverse_precision, memory_accessed, varargin)

    G = 0.02;
    collect_efe_components = false;
    if ~isempty(varargin)
        collect_efe_components = logical(varargin{1});
    end
    efe_components = [];

    efe_base_term = 0.02;
    efe_novelty_raw = 0;
    efe_novelty_term = 0;
    efe_epistemic_raw = 0;
    efe_epistemic_term = 0;
    efe_extrinsic_term = 0;
    efe_future_term = 0;
    P_prior = P;
    P = calculate_posterior(P, y, O, t);
    bb{2} = normalise_matrix(b{2});
    t_food_idx = min(max(round(t_food) + 1, 1), 35);
    t_water_idx = min(max(round(t_water) + 1, 1), 35);
    t_sleep_idx = min(max(round(t_sleep) + 1, 1), 35);
    num_factors = 2;
    %fprintf('At time step %d, non-zero memmory length: %d\n', t, length(short_term_memory(find(short_term_memory > 0))));

    if t > true_t
        novelty = 0;
        if novelty_weight ~= 0
            start = t - 6;

            if start <= 0
                start = 1;
            end

            for timey = start:t

                if timey ~= t
                    L = spm_backwards(O, P, A, bb, chosen_action, timey, t);
                else
                    L = P{t, 2};
                end

                LL{2} = L;
                LL{1} = P{timey, 1};
                a_prior = a{2};

                for modality = 2:2
                    a_learning = O(modality, timey)';

                    for factor = 1:num_factors
                        a_learning = spm_cross(a_learning, LL{factor});
                    end

                    a_learning = a_learning .* (a{modality} > 0);
                    a_learning_weighted = a_learning;
                    a_learning_weighted(2:end, :) = learning_weight * a_learning(2:end, :);
                    a_learning_weighted(1, :) = a_learning(1, :);
                    a_temp = a_prior + a_learning_weighted;
                end

                w = kldir(normalise(a_temp(:)), normalise(a_prior(:)));
                novelty = novelty + w;
            end

            G = G + novelty_weight * novelty;
            efe_novelty_raw = novelty;
            efe_novelty_term = novelty_weight * novelty;
        end

        if epistemic_weight ~= 0
            epi = G_epistemic_value(y, P_prior(t, :)');
            G = G + epistemic_weight * epi;
            efe_epistemic_raw = epi;
            efe_epistemic_term = epistemic_weight * epi;
        end

        for modality = 2:2

            if modality == 2
                C = determineObservationPreference(t_food, t_water, t_sleep);
                %reduce preference precision
                C{modality} = C{modality} / preference_inverse_precision;
            end

            if modality == 2
                % add extrinsic term (see EFE equation)
                extrinsic = O{2, t} * C{2}';
                G = G + extrinsic;
                efe_extrinsic_term = extrinsic;
                % extrinsic
            end

        end

        % epi
        % extrinsic
        % novelty

        t_food = round((t_food + 1) * (1 - O{2, t}(2)));
        t_water = round((t_water + 1) * (1 - O{2, t}(3)));
        t_sleep = round((t_sleep + 1) * (1 - O{2, t}(4)));

        t_food_idx = min(max(t_food + 1, 1), 35);
        t_water_idx = min(max(t_water + 1, 1), 35);
        t_sleep_idx = min(max(t_sleep + 1, 1), 35);

    end

    if t < N

        actions = 1:5;
        % actions = 1:5;
        efe = [0, 0, 0, 0, 0];

        for action = actions

            Q{1, action} = (B{1}(:, :, action) * P{t, 1}')';
            Q{2, action} = (bb{2}(:, :, 1) * P{t, 2}');
            s = Q(:, action);
            qs = spm_cross(s);
            qs = qs(:);
            % only consider relatively likely states
            likely_states = find(qs > 1/8);

            if isempty(likely_states)
                threshold = 1 / numel(qs) * 1 / numel(qs);
                likely_states = find(qs > (1 / numel(qs) - threshold));
            end

            % for each of those likely states
            for state = likely_states(:)'

                if short_term_memory(t_food_idx, t_water_idx, t_sleep_idx, state) ~= 0
                    sh = short_term_memory(t_food_idx, t_water_idx, t_sleep_idx, state);
                    K(state) = sh;
                    memory_accessed = memory_accessed + 1;
                else

                    for modal = 1:numel(A)
                        O{modal, t + 1} = normalise(y{modal}(:, state)');
                    end

                    % prior over next states given transition function
                    % (calculated earlier)
                    P{t + 1, 1} = Q{1, action};
                    P{t + 1, 2} = Q{2, action};
                    chosen_action(t) = action;

                    % recursively move to the next node (likely state) of
                    % the tree
                    [expected_free_energy, P, short_term_memory, best_actions, memory_accessed] = tree_search_frwd_SI_smooth(short_term_memory, O, P, a, A, y, B, b, t + 1, T, N, t_food, t_water, t_sleep, true_t, chosen_action, true_t_food, true_t_water, true_t_sleep, best_actions, learning_weight, novelty_weight, epistemic_weight, preference_inverse_precision, memory_accessed);
                    S = max(expected_free_energy);
                    K(state) = S;
                    short_term_memory(t_food_idx, t_water_idx, t_sleep_idx, state) = S;
                end

            end

            action_fe = K(likely_states) * qs(likely_states);
            efe(action) = efe(action) + 0.7 * action_fe;
        end

        [maxi, chosen_action] = max(efe);
        G = G + maxi;
        efe_future_term = maxi;
        best_actions = [chosen_action best_actions];
    end

    if collect_efe_components
        efe_components = struct(...
            'base_term', efe_base_term, ...
            'novelty_raw', efe_novelty_raw, ...
            'novelty_term', efe_novelty_term, ...
            'epistemic_raw', efe_epistemic_raw, ...
            'epistemic_term', efe_epistemic_term, ...
            'extrinsic_term', efe_extrinsic_term, ...
            'future_term', efe_future_term, ...
            'G_total', G ...
        );
    end
