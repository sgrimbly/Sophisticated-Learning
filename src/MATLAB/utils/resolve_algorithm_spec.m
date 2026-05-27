function spec = resolve_algorithm_spec(label)
%RESOLVE_ALGORITHM_SPEC Map each public algorithm label to explicit flags.
%
% This centralizes algorithm semantics so entrypoints do not infer behavior
% from substrings such as "novelty" or "noSmooth".

    arguments
        label char
    end

    switch char(label)
        case 'SI'
            spec = make_spec(label, 'SI', 'SI', true, false, false, 'SI-runs', true);
        case 'SI_noNovelty'
            spec = make_spec(label, 'SI', 'SI', false, false, false, 'SI-runs', true);
        case 'SI_novelty'
            spec = make_spec(label, 'SI', 'SI', true, false, false, 'SI-runs', true);
        case 'SI_smooth_noNovelty'
            spec = make_spec(label, 'SI', 'SI_smooth', false, true, false, 'SI-runs', true);
        case 'SI_smooth'
            spec = make_spec(label, 'SI', 'SI_smooth', false, true, false, 'SI-runs', true);
        case 'SI_novelty_smooth'
            spec = make_spec(label, 'SI', 'SI_smooth', true, true, false, 'SI-runs', true);
        case 'SL'
            spec = make_spec(label, 'SL', 'SL', true, true, false, 'SL-runs', true);
        case 'SL_noNovelty'
            spec = make_spec(label, 'SL', 'SL', false, true, false, 'SL-runs', true);
        case 'SL_adaptivePlan'
            spec = make_spec(label, 'SL', 'SL', true, true, true, 'SL-runs', true);
        case 'SL_noNovelty_adaptivePlan'
            spec = make_spec(label, 'SL', 'SL', false, true, true, 'SL-runs', true);
        case 'SL_noAdaptivePlan'
            spec = make_spec(label, 'SL', 'SL', true, true, false, 'SL-runs', true);
        case 'SL_noSmooth'
            spec = make_spec(label, 'SL', 'SL_noSmooth', true, false, false, 'SL-runs', true);
        case 'SL_noNovelty_noSmooth'
            spec = make_spec(label, 'SL', 'SL_noSmooth', false, false, false, 'SL-runs', true);
        case 'SL_noSmooth_adaptivePlan'
            spec = make_spec(label, 'SL', 'SL_noSmooth', true, false, true, 'SL-runs', true);
        case 'SL_noNovelty_noSmooth_adaptivePlan'
            spec = make_spec(label, 'SL', 'SL_noSmooth', false, false, true, 'SL-runs', true);
        case 'SL_noSmooth_noAdaptivePlan'
            spec = make_spec(label, 'SL', 'SL_noSmooth', true, false, false, 'SL-runs', true);
        case 'BA'
            spec = make_spec(label, 'BA', 'BA', false, false, false, 'BA-runs', true);
        case 'BAUCB'
            spec = make_spec(label, 'BAUCB', 'BAUCB', false, false, false, 'BAUCB-runs', true);
        case 'known_large_MCT'
            spec = make_spec(label, 'known_large_MCT', 'known_large_MCT', false, false, false, 'MCT-runs', false);
        case 'model_free_RL'
            spec = make_spec(label, 'model_free_RL', 'model_free_RL', false, false, false, 'RL-runs', false);
        case 'model_mixed_RL'
            spec = make_spec(label, 'model_mixed_RL', 'model_mixed_RL', false, false, false, 'RL-runs', false);
        otherwise
            error('resolve_algorithm_spec:UnknownLabel', 'Unknown algorithm label "%s".', label);
    end

    spec.file_prefix = ['results_' spec.label];
end

function spec = make_spec(label, family, implementation, novelty_on, smoothing_on, adaptive_plan_on, run_folder, is_unknown_model)
    spec = struct( ...
        'label', label, ...
        'family', family, ...
        'implementation', implementation, ...
        'novelty_on', logical(novelty_on), ...
        'smoothing_on', logical(smoothing_on), ...
        'adaptive_plan_on', logical(adaptive_plan_on), ...
        'run_folder', run_folder, ...
        'is_unknown_model', logical(is_unknown_model) ...
    );
end
