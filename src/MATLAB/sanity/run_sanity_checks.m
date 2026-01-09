function run_sanity_checks()

    currentDir = fileparts(mfilename('fullpath'));
    srcPath = fullfile(currentDir, '..');
    addpath(genpath(srcPath));

    fprintf('Running Sophisticated-Learning sanity checks...\n');

    sanity_preference_scaling();
    sanity_rng_not_advanced_by_calculate_posterior();
    sanity_rng_not_advanced_by_spm_backwards();
    sanity_ucb_bonus_monotonicity();
    sanity_ucb_Nt_updates_only_at_root();
    sanity_config_hash_changes_with_config();
    sanity_state_roundtrip();
    sanity_time_since_update_convention();

    fprintf('All sanity checks passed.\n');

end

function sanity_preference_scaling()

    t_food = 10;
    t_water = 5;
    t_sleep = 3;

    C = determineObservationPreference(t_food, t_water, t_sleep);
    o_food = [0, 1, 0, 0];

    inv_precision = [1, 2, 5, 10];
    extrinsic = zeros(size(inv_precision));

    for i = 1:numel(inv_precision)
        C_scaled = C{2} / inv_precision(i);
        extrinsic(i) = o_food * C_scaled(:);
    end

    assert(all(diff(abs(extrinsic)) < 0), 'Preference scaling sanity check failed: |extrinsic| did not decrease with inverse precision.');

end

function sanity_rng_not_advanced_by_calculate_posterior()

    rng(0, 'twister');
    expected_first = rand;

    rng(0, 'twister');

    % Minimal shapes for calculate_posterior(P, A, O, t)
    P = cell(1, 2);
    P{1, 1} = [0.7, 0.3];
    P{1, 2} = [0.5, 0.5];

    A = cell(1, 2);
    A{1} = [];
    A{2} = reshape(1:8, [2, 2, 2]);

    O = cell(2, 1);
    O{2, 1} = [1, 0];

    calculate_posterior(P, A, O, 1);

    after = rand;
    assert(after == expected_first, 'calculate_posterior() advanced RNG; expected it to be deterministic for fixed O.');

end

function sanity_config_hash_changes_with_config()

    cfg = struct(...
        'algorithm', 'SI', ...
        'seed', 1, ...
        'grid_size', 10, ...
        'start_position', 51, ...
        'hill_pos', 55, ...
        'food_sources', [71, 43, 57, 78], ...
        'water_sources', [73, 33, 48, 67], ...
        'sleep_sources', [64, 44, 49, 59], ...
        'weights', [10, 40, 1, 10], ...
        'num_states', 100, ...
        'num_trials', 200, ...
        'grid_id', '' ...
    );

    id1 = config_hash(cfg);

    cfg.weights = [10, 40, 1, 20];
    id2 = config_hash(cfg);

    assert(~strcmp(id1, id2), 'config_hash() did not change when weights changed.');

end

function sanity_time_since_update_convention()

    % Convention: time-since variables are raw (0 means just consumed).
    % If food is certain (O_food = 1), next time-since should remain 0.

    t_food = 0;
    O_food_prob = 1;
    next_food = round((t_food + 1) * (1 - O_food_prob));
    assert(next_food == 0, 'Time-since update failed: expected reset to 0 when food is observed with prob=1.');

    t_food = 0;
    O_food_prob = 0;
    next_food = round((t_food + 1) * (1 - O_food_prob));
    assert(next_food == 1, 'Time-since update failed: expected increment to 1 when food is absent with prob=0.');

end

function sanity_ucb_bonus_monotonicity()

    ucb_scale = 5;

    counts = [1, 2, 5, 10];
    base_visits = 100;
    total_visits = base_visits + counts;
    bonus = ucb_scale * sqrt(log(total_visits + 1) ./ counts);

    assert(all(diff(bonus) < 0), 'UCB bonus did not decrease with increasing visit count.');

end

function sanity_state_roundtrip()

    tmp = [tempname, '.mat'];

    rng(42, 'twister');
    rand(1, 3);
    saved_rng = rng;
    expected = rand(1, 3);

    cfg = struct('algorithm', 'unit', 'seed', 1);
    id = config_hash(cfg);
    meta = struct('config_id', id, 'run_config', cfg);

    Nt = reshape(1:4, [2, 2]);
    state = {saved_rng, 1, meta, Nt};
    save_state(tmp, state);

    [loaded, isNew] = load_state(tmp);
    assert(~isNew, 'Expected existing state file to load.');
    assert(iscell(loaded) && numel(loaded) == 4, 'Unexpected loaded state shape.');
    assert(isstruct(loaded{3}) && isfield(loaded{3}, 'config_id'), 'Missing run_meta in loaded state.');
    assert(strcmp(loaded{3}.config_id, id), 'run_meta.config_id did not round-trip.');
    assert(isequal(loaded{4}, Nt), 'Nt did not round-trip.');

    rng(loaded{1});
    resumed = rand(1, 3);
    assert(isequal(expected, resumed), 'RNG resume did not reproduce expected sequence.');

    delete(tmp);

end


function sanity_rng_not_advanced_by_spm_backwards()

    rng(0, 'twister');
    expected_first = rand;

    rng(0, 'twister');

    num_states = 2;
    num_contexts = 2;

    O = cell(3, 2);
    O{3, 2} = [1, 0];

    Q = cell(2, 2);
    Q{1, 2} = [0.5, 0.5];
    Q{2, 1} = [0.6, 0.4];

    A = cell(1, 3);
    A{3} = ones(2, num_states, num_contexts);

    B = cell(1, 2);
    B{2} = zeros(num_contexts, num_contexts, 1);
    B{2}(:, :, 1) = eye(num_contexts);

    spm_backwards(O, Q, A, B, zeros(1, 1), 1, 2);

    after = rand;
    assert(after == expected_first, 'spm_backwards() advanced RNG; expected it to be deterministic for fixed O.');

end

function sanity_ucb_Nt_updates_only_at_root()

    num_states = 2;
    num_contexts = 2;
    num_joint_states = num_states * num_contexts;

    Nt0 = ones(num_states, num_contexts);

    P = cell(2, 2);
    P{1, 1} = [1, 0];
    P{1, 2} = [0, 1];

    O = cell(2, 2);
    O{2, 1} = [0, 1, 0, 0];

    y = cell(1, 2);
    y{1} = ones(1, num_joint_states);
    y{2} = ones(4, num_states, num_contexts);

    A = cell(1, 2);
    A{1} = [];
    A{2} = [];

    b = cell(1, 2);
    b{2} = ones(num_contexts, num_contexts, 1);

    B = cell(1, 1);
    B{1} = zeros(num_states, num_states, 5);
    for a = 1:5
        B{1}(:, :, a) = eye(num_states);
    end

    chosen_action = zeros(1, 2);
    short_term_memory = zeros(35, 35, 35, num_joint_states, 5);
    long_term_memory = 0;

    current_joint_state = sub2ind([num_states, num_contexts], 1, 2);

    t = 1;
    true_t = 1;
    T = 2;
    N = 2;

    [~, ~, ~, ~, ~, ~, ~, Nt_out, ~] = tree_search_frwd_UCB( ...
        long_term_memory, short_term_memory, O, P, [], A, y, [], B, b, t, T, N, ...
        0, 0, 0, 0, 0, 0, current_joint_state, true_t, chosen_action, 0, 0, 0, 0, 0, 0, 0, [], [], Nt0, 0, 1, 5);

    assert(sum(Nt_out(:)) == sum(Nt0(:)) + 1, 'Nt should increment exactly once per root call (not per recursion).');
    assert(Nt_out(1, 2) == Nt0(1, 2) + 1, 'Nt incremented at wrong (state, context) index.');

end
