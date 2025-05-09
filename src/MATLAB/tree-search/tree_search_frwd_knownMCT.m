function [G, P, D, short_term_memory, long_term_memory, optimal_traj, best_actions, Tree] = tree_search_frwd_knownMCT(long_term_memory, short_term_memory, O, P, a, A, y, D, B, b, t, T, N, t_food, t_water, t_sleep, resource_locations, current_state, true_t, chosen_action, novelty, surety, simulated_time, true_t_food, true_t_water, true_t_sleep, hill_visited, optimal_traj, best_actions, history, k_factor, mct, num_rollouts)

    Tree = {};
    global tree_history;
    global searches;
    global auto_rest;

    G = 0.02;
    P_prior = P;
    P = calculate_posterior(P, y, O, t);

    if t > true_t
        epi = G_epistemic_value(y, P_prior(t, :)');
        G = G + epi;

        for modality = 2:2

            if modality == 2
                C = determineObservationPreference(t_food, t_water, t_sleep);
                C{modality} = C{modality} / 10;
            end

            if modality == 2
                extrinsic = O{2, t} * C{2}';
                G = G + extrinsic;
            end

        end

    end

    t_food = round(t_food * (1 - O{2, t}(2))) + 1;
    t_water = round(t_water * (1 - O{2, t}(3))) + 1;
    t_sleep = round(t_sleep * (1 - O{2, t}(4))) + 1;
    t_food_approx = t_food;
    t_water_approx = t_water;
    t_sleep_approx = t_sleep;

    if t >= N
        G_mct = zeros(1, num_rollouts);
        P_back = P;
        y_back = y;
        O_back = O;
        t_back = t;
        t_food_back = t_food;
        t_water_back = t_water;
        t_sleep_back = t_sleep;

        for roll = 1:num_rollouts
            mct_dep = 1;
            P = P_back;
            y = y_back;
            O = O_back;
            t = t_back;
            t_food = t_food_back;
            t_water = t_water_back;
            t_sleep = t_sleep_back;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            while mct_dep <= mct
                G_roll = 0.02;

                if mct_dep > 1
                    P_prior = P;
                    P = calculate_posterior(P, y, O, t);
                end

                epi = G_epistemic_value(y, P_prior(t, :)');
                G_roll = G_roll + epi;

                for modality = 2:2

                    if modality == 2
                        C = determineObservationPreference(t_food, t_water, t_sleep);
                        C{modality} = C{modality} / 10;
                    end

                    if modality == 2
                        extrinsic = O{2, t} * C{2}';
                        G_roll = G_roll + extrinsic;
                    end

                end

                t_food = round(t_food * (1 - O{2, t}(2))) + 1;
                t_water = round(t_water * (1 - O{2, t}(3))) + 1;
                t_sleep = round(t_sleep * (1 - O{2, t}(4))) + 1;
                t_food_approx = t_food;
                t_water_approx = t_water;
                t_sleep_approx = t_sleep;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                act = randi([1, 5]); %selecting only one act
                Q{1, act} = (B{1}(:, :, act) * P{t, 1}')';
                Q{2, act} = (B{2}(:, :, 1) * P{t, 2}');
                s = Q(:, act);
                qs = spm_cross(s);
                qs = qs(:);
                likely_states = randsample(1:numel(qs), 1, true, qs); %just probabilistaclly select a state.

                for modal = 1:numel(A)
                    O{modal, t + 1} = normalise(y{modal}(:, likely_states)');
                end

                P{t + 1, 1} = Q{1, act};
                P{t + 1, 2} = Q{2, act};
                chosen_action(t) = act;
                G_roll = G_roll * qs(likely_states); %try with qs multiple
                G_mct(1, roll) = G_mct(1, roll) + (k_factor ^ mct_dep) * G_roll;
                mct_dep = mct_dep + 1;
                t = t + 1;
                % P_prior = P;
            end

            %%%%%%%%%%%%%%%%%%%%%%
            %mct_dep = 1;
            P = P_back;
            y = y_back;
            O = O_back;
            t = t_back;
            t_food = t_food_back;
            t_water = t_water_back;
            t_sleep = t_sleep_back;
        end

        G = G + mean(G_mct);
    end

    if t < N
        actions = randperm(5);
        efe = [0, 0, 0, 0, 0];
        validFieldName = ['h_', history];

        for action = actions
            Q{1, action} = (B{1}(:, :, action) * P{t, 1}')';
            Q{2, action} = (B{2}(:, :, 1) * P{t, 2}');
            s = Q(:, action);
            qs = spm_cross(s);
            qs = qs(:);
            likely_states = find(qs > 1/8);

            if isempty(likely_states)
                threshold = 1 / numel(qs) * 1 / numel(qs);
                likely_states = find(qs > (1 / numel(qs) - threshold));
            end

            for state = likely_states(:)'

                if ~auto_rest && short_term_memory(t_food_approx, t_water_approx, t_sleep_approx, state) ~= 0
                    sh = short_term_memory(t_food_approx, t_water_approx, t_sleep_approx, state);
                    searches = searches + 1;
                    K(state) = sh;
                else

                    for modal = 1:numel(A)
                        O{modal, t + 1} = normalise(y{modal}(:, state)');
                    end

                    P{t + 1, 1} = Q{1, action};
                    P{t + 1, 2} = Q{2, action};
                    chosen_action(t) = action;
                    Tree.history = history;
                    [expected_free_energy, d, D, short_term_memory, long_term_memory, optimal_traj, best_actions, Tree] = tree_search_frwd_knownMCT(long_term_memory, short_term_memory, O, P, a, A, y, D, B, b, t + 1, T, N, t_food_approx, t_water_approx, t_sleep_approx, resource_locations, state, true_t, chosen_action, novelty, surety, 0, true_t_food, true_t_water, true_t_sleep, hill_visited, optimal_traj, best_actions, [history, num2str(action)], k_factor, mct, num_rollouts);
                    S = max(expected_free_energy);
                    K(state) = S;
                    short_term_memory(t_food_approx, t_water_approx, t_sleep_approx, state) = S;
                end

            end

            action_fe = K(likely_states) * qs(likely_states);
            efe(action) = efe(action) + k_factor * action_fe;
        end

        tree_history.(validFieldName) = efe;
        [maxi, chosen_action] = max(efe);
        G = G + maxi;
        best_actions = [chosen_action best_actions];
    end

end
