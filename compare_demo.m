clear; clc;

%% Shared model & data
S = 10;
M = 4;
h = 0.5;
P = makeP(S, h);
pi0 = ones(1, S) / S;

Aidx = [1 2 1 3 2 3 1 2 1 3 2 3 1 2 1 3 2 3 1 2 1 3 2 3];
Bidx = [2 3 3 2 4 4 2 3 3 2 4 4 2 3 3 2 4 4 2 3 3 2 4 4];
Y    = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
T = numel(Y);

%% Run FHMM (callable)
[Phi_fhmm, loglike_fhmm, phi_hist_fhmm, c_hist_fhmm] = ...
    part1_fhmm(M, S, P, pi0, Aidx, Bidx, Y); % S x M x T

%% Run DHMM (callable)
Phi_dhmm = run_dhmm(S, M, P, Aidx, Bidx, Y, pi0); % M x S x T

%% Compare
metrics = compare_fhmm_dhmm(S, M, P, Aidx, Bidx, Y, ...
                                    Phi_fhmm, phi_hist_fhmm, c_hist_fhmm, ...
                                    Phi_dhmm, pi0);

%% Simple predictive checks
eps0 = 1e-15;
pwin_fhmm = max(min(metrics.pwin_fhmm, 1 - eps0), eps0);
pwin_dhmm = max(min(metrics.pwin_dhmm, 1 - eps0), eps0);

ll_f = -sum( Y .* log(pwin_fhmm) + (1 - Y) .* log(1 - pwin_fhmm) );
ll_d = -sum( Y .* log(pwin_dhmm) + (1 - Y) .* log(1 - pwin_dhmm) );
avg_gap_per_step = (ll_d - ll_f) / T;

acc_f = mean((pwin_fhmm >= 0.5) == Y);
acc_d = mean((pwin_dhmm >= 0.5) == Y);
acc_delta = acc_d - acc_f;

%% End-time marginals
fhmm_img = squeeze(Phi_fhmm(:, :, T));
dhmm_img = squeeze(permute(Phi_dhmm(:, :, T), [2 1 3]));
clims = [min([fhmm_img(:); dhmm_img(:)]), max([fhmm_img(:); dhmm_img(:)])];

tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
imagesc(fhmm_img, clims);
set(gca, 'YDir', 'reverse');
xlabel('Player Index (m)'); ylabel('Skill Level (s)');
title('F-HMM'); axis tight;

nexttile;
imagesc(dhmm_img, clims);
set(gca, 'YDir', 'reverse');
xlabel('Player Index (m)'); ylabel('Skill Level (s)');
title('D-HMM'); axis tight;

cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '';

%% Predictive metrics (per-time & aggregate)
eps0 = 1e-12;
pf = max(min(metrics.pwin_fhmm, 1 - eps0), eps0);
pd = max(min(metrics.pwin_dhmm, 1 - eps0), eps0);
ll_t_f = -( Y .* log(pf) + (1 - Y) .* log(1 - pf) );
ll_t_d = -( Y .* log(pd) + (1 - Y) .* log(1 - pd) );
brier_f = mean((Y - pf).^2);
brier_d = mean((Y - pd).^2);
fprintf('Brier: F = %.4f | D = %.4f | Δ = %.4f\n', brier_f, brier_d, brier_d - brier_f);

figure('Name', 'Cumulative log-loss gap (D - F)');
plot(cumsum(ll_t_d - ll_t_f), 'LineWidth', 1.5); grid on;
xlabel('Time (t)'); ylabel('Cumulative Δ(log-loss)'); title('');

%% Helper: random-walk transition
function P = makeP(S, h)
    P = (1 - h) * eye(S);
    for i = 1:S
        if i > 1, P(i, i-1) = P(i, i-1) + h/2; end
        if i < S, P(i, i+1) = P(i, i+1) + h/2; end
    end
end
