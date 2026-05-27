function G = G_epistemic_value(A, s)

    % auxiliary function for Bayesian suprise or mutual information
    % FORMAT [G] = spm_MDP_G(A,s)
    %
    % A   - likelihood array (probability of outcomes given causes)
    % s   - probability density of causes

    % Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

    % Karl Friston
    % $Id: spm_MDP_G.m 7306 2018-05-07 13:42:02Z karl $

    % probability distribution over the hidden causes: i.e., Q(s)

    qx = spm_cross(s); % this is the outer product of the posterior over states
    % calculated with respect to itself

    % accumulate expectation of entropy: i.e., E[lnP(o|s)]
    G = 0;
    qo = 0;

    % Hoist constants out of the per-likely-state loop: numel(A) and the cell
    % derefs of A{g} are loop-invariant. Round 2 (2026-05-03) — biggest
    % remaining SI hotspot per profile.
    nA = numel(A);
    A1 = A{1};
    A2 = A{2};
    A3 = A{3};
    likely = find(qx > exp(-16))';

    % Round 2.4 (2026-05-03): replace the 3-spm_cross chain with a single
    % kron-of-kron call, and inline nat_log to remove ~209k function dispatches
    % per SI run (G_epistemic_value's nat_log calls were 4.1s in the previous
    % profile). kron(a3, kron(a2, a1)) computes the same flat outer product
    % as `po(:)` from the spm_cross chain bit-shape-wise; FP order differs.
    NAT_LOG_FLOOR = exp(-500);  % constant, hoisted out of per-iteration loop

    for i = likely
        % flat outer product over modalities directly, skipping intermediate
        % N-D shapes that we'd just flatten with po(:) anyway.
        if nA == 3
            po = kron(A3(:, i), kron(A2(:, i), A1(:, i)));
        elseif nA == 2
            po = kron(A2(:, i), A1(:, i));
        elseif nA == 1
            po = A1(:, i);
        else
            % fallback: general case, build via spm_cross then flatten
            po = 1;
            for g = 1:nA, po = spm_cross(po, A{g}(:, i)); end
            po = po(:);
        end

        qx_i = qx(i);
        qo = qo + qx_i * po;
        % inlined nat_log(po) = log(po + exp(-500))
        G = G + qx_i * po' * log(po + NAT_LOG_FLOOR);
    end

    % subtract entropy of expectations: i.e., E[lnQ(o)]
    % inlined nat_log
    G = G - qo' * log(qo + NAT_LOG_FLOOR);

end
