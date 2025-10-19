clear; clc;

% Parameters
S = 10; % number of discrete skill states
M = 3; % number of players
h = 0.5; % random-walk volatility
scale = 1; % logistic scaling
P = makeP(S, h);

% Initial skills (uniform)
skills = (1/S) * ones(M, S);

% Match sequence
Aidx = [1 1 2  1 1 2  1 1 2  1 1 2];
Bidx = [2 3 3  2 3 3  2 3 3  2 3 3];
Y    = [1 1 1  1 1 0  1 1 1  1 1 0];
T = numel(Y);

% Observation matrix
G = obs_matrix_logistic(S, scale);

% Storage
Phi_hist = zeros(M, S, T);

% Main loop
for t = 1:T
    a = Aidx(t); b = Bidx(t); y = Y(t);

    % Prediction
    skills_pred = skills * P;

    % Likelihood for outcome y
    L = G; if y == 0, L = 1 - L; end

    % Decoupled updates for players a,b
    psi_a = skills_pred(a, :);
    psi_b = skills_pred(b, :);

    s_a  = (L * psi_b.').'; % sum_j ψ_b(j) L(i,j)
    phi_a = psi_a .* s_a;  phi_a = phi_a / sum(phi_a);

    s_b  = (psi_a * L); % sum_i ψ_a(i) L(i,j)
    phi_b = psi_b .* s_b;  phi_b = phi_b / sum(phi_b);

    % Write back
    skills = skills_pred;
    skills(a, :) = phi_a;
    skills(b, :) = phi_b;

    Phi_hist(:, :, t) = skills;
end

% Plots
figure('Name','D-HMM Marginals Over Time','NumberTitle','off');

vmin = min(Phi_hist(:));
vmax = max(Phi_hist(:));

for m = 1:M
    subplot(1, M, m);
    imagesc(1:T, 1:S, squeeze(Phi_hist(m, :, :)), [vmin, vmax]);
    axis tight; set(gca, 'YDir', 'reverse');
    xlabel('Time (t)'); ylabel('Skill Level (s)');
end

cb = colorbar('Position', [0.92 0.11 0.02 0.815]);
cb.Label.String = 'Probability';

% Plots — Mean Skill w/ 1 SD
figure('Name','D-HMM: Mean Skill ± 1 SD','NumberTitle','off');

skills_grid = 1:S;
tt          = 1:T;

for m = 1:M
    subplot(1, M, m); hold on;

    Phi_m = squeeze(Phi_hist(m, :, :)); % S x T
    mu    = skills_grid * Phi_m; % 1 x T
    m2    = (skills_grid.^2) * Phi_m; % 1 x T
    sd    = sqrt(max(m2 - mu.^2, 0));

    xx = [tt, fliplr(tt)];
    yy = [mu + sd, fliplr(mu - sd)];
    fill(xx, yy, [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(tt, mu, 'k-', 'LineWidth', 2);

    set(gca, 'YDir', 'reverse');
    xlim([1, T]); ylim([1, S]);
    xlabel('Time (t)'); ylabel('Skill');
    grid on; box on;
end

% Helper functions
function P = makeP(S, h)
    P = (1 - h) * eye(S);
    for i = 1:S
        if i > 1, P(i, i-1) = P(i, i-1) + h/2; end
        if i < S, P(i, i+1) = P(i, i+1) + h/2; end
    end
end

function G = obs_matrix_logistic(S, scale)
    [I, J] = ndgrid(1:S, 1:S);
    G = 1 ./ (1 + exp(-(I - J) / scale));
end

function row = pdf_binomial_row(S, n, p)
    assert(S == n + 1, 'For this helper, use S = n+1.');
    k    = 0:n;
    logC = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1);
    row  = exp(logC + k .* log(p) + (n-k) .* log1p(-p));
    row  = row / sum(row);
end
