function metrics = compare_fhmm_dhmm_metrics(S, M, P, Aidx, Bidx, Y, ...
                                             Phi_fhmm, phi_hist_fhmm, c_hist_fhmm, ...
    if nargin < 10 || isempty(pi0), pi0 = ones(1, S) / S; end

    T = numel(Y);
    eps0 = 1e-15;
    skills_grid = (1:S)';

    %% Outcome-level metrics
    % FHMM per-time log predictive prob (from forward scaling)
    logpred_fhmm = -log(max(c_hist_fhmm, realmin('double')));
    logloss_fhmm = sum(logpred_fhmm);

    % FHMM Pr(Y=1) via predicted joint and win-vector
    pwin_fhmm = nan(1, T);
    Z = build_Z(S, M); % N x M, N=S^M
    Kpow = kron_power_sparse(P, M); % S^M x S^M

    for t = 1:T
        if t == 1
            phi0 = pi0(:);
            for m = 2:M, phi0 = kron(phi0, pi0(:)); end
            psi = Kpow' * phi0;
        else
            psi = Kpow' * phi_hist_fhmm(:, t-1);
        end
        Lt1 = obs_vector(Z, Aidx(t), Bidx(t), 1); % win-likelihood vector
        pwin_fhmm(t) = sum(psi .* Lt1);
    end

    % DHMM via decoupled predictions
    G = obs_matrix_logistic(S, 1);
    pwin_dhmm  = nan(1, T);
    for t = 1:T
        if t == 1
            skills_prev = repmat(pi0, M, 1); % M x S
        else
            skills_prev = squeeze(Phi_dhmm(:, :, t-1));
        end
        skills_pred = skills_prev * P; % M x S
        a = Aidx(t); b = Bidx(t);
        pA = skills_pred(a, :); pB = skills_pred(b, :);
        p_y1 = pA * G * pB';
        pwin_dhmm(t) = max(min(p_y1, 1 - eps0), eps0);
    end

    % Per-time losses
    y = double(Y(:))';
    logpred_dhmm = -( y .* log(pwin_dhmm + eps0) + (1 - y) .* log(1 - pwin_dhmm + eps0) );
    logloss_fhmm = sum(logpred_dhmm);
    brier_fhmm = mean((pwin_fhmm - y).^2);
    brier_dhmm = mean((pwin_dhmm - y).^2);

    %% Posterior-level metrics
    KL_marg = zeros(M, T);
    JS_marg = zeros(M, T);
    TV_marg = zeros(M, T);
    W1_marg = zeros(M, T);

    for t = 1:T
        for m = 1:M
            p = max(squeeze(Phi_fhmm(:, m, t)), eps0);  p = p / sum(p);
            q = max(squeeze(Phi_dhmm(m, :, t))', eps0); q = q / sum(q);

            KL_marg(m, t) = sum(p .* log(p ./ q));

            mavg = 0.5 * (p + q);
            JS_marg(m, t) = 0.5 * sum(p .* log(p ./ mavg)) + 0.5 * sum(q .* log(q ./ mavg));

            TV_marg(m, t) = 0.5 * sum(abs(p - q));

            Fp = cumsum(p); Fq = cumsum(q);
            W1_marg(m, t) = sum(abs(Fp - Fq));         % grid spacing = 1
        end
    end

    %% FHMM multi-information (coupling missed by DHMM)
    N        = S^M;
    MI_joint = zeros(1, T);
    for t = 1:T
        phi = max(phi_hist_fhmm(:, t), eps0); phi = phi / sum(phi);
        denom = ones(N, 1);
        for m = 1:M
            pm = squeeze(Phi_fhmm(:, m, t)); pm = max(pm, eps0); pm = pm / sum(pm);
            denom = denom .* pm(double(Z(:, m)));
        end
        MI_joint(t) = sum(phi .* log(phi ./ denom));
    end

    %% Pack results
    metrics = struct();
    metrics.pwin_fhmm = pwin_fhmm;
    metrics.pwin_dhmm = pwin_dhmm;
    metrics.pred_gap_abs = abs(pwin_fhmm - pwin_dhmm);

    metrics.logloss_fhmm = logloss_fhmm;
    metrics.logloss_dhmm = logloss_dhmm;
    metrics.brier_fhmm = brier_fhmm;
    metrics.brier_dhmm = brier_dhmm;

    metrics.KL_marg = KL_marg;
    metrics.JS_marg = JS_marg;
    metrics.TV_marg = TV_marg;
    metrics.W1_marg = W1_marg;

    metrics.MI_joint = MI_joint;

    metrics.summaries = struct( ...
        'delta_logloss', logloss_dhmm - logloss_fhmm, ...
        'delta_brier', brier_dhmm - brier_fhmm, ...
        'mean_JS', mean(JS_marg, 'all'), ...
        'mean_W1', mean(W1_marg, 'all') ...
    );

    %% Local helpers
    function G = obs_matrix_logistic(S, scale)
        [I, J] = ndgrid(1:S, 1:S);
        G = 1 ./ (1 + exp(-(I - J) / scale));
    end

    function Z = build_Z(S, M)
        N = S^M;
        Z = zeros(N, M, 'uint16');
        for n = 0:N-1
            x = n;
            for mm = M:-1:1
                Z(n+1, mm) = mod(x, S) + 1;
                x = floor(x / S);
            end
        end
    end

    function Pbig = kron_power_sparse(P, M)
        Pbig = sparse(1, 1, 1, 1, 1);
        for k = 1:M
            Pbig = kron(Pbig, sparse(P));
        end
    end

    function Lt = obs_vector(Z, a_idx, b_idx, y)
        xa = double(Z(:, a_idx));
        xb = double(Z(:, b_idx));
        delta = xa - xb;
        pwin = 1 ./ (1 + exp(-delta));
        Lt = pwin;
        if y == 0, Lt = 1 - Lt; end
        Lt = max(Lt, realmin('double'));
    end
end
