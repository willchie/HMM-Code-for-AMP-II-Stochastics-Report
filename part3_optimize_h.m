% Correlation of MeanSkill vs WinRate across h
S     = 500;
scale = 1;
G     = obs_matrix_logistic(S, scale);

% Empirical win-rate for ALL players
M = max(max(Aidx), max(Bidx));
matches_all = zeros(M,1);
wins_all = zeros(M,1);
losses_all = zeros(M,1);
for pid = 1:M
    is_winner = (Tdata{:,2} == pid);
    is_loser  = (Tdata{:,3} == pid);
    wins_all(pid)    = sum(is_winner);
    losses_all(pid)  = sum(is_loser);
    matches_all(pid) = wins_all(pid) + losses_all(pid);
end
win_rate_all = 100 * wins_all ./ max(matches_all,1);
mask = matches_all > 0; % ignore players with no matches in corr

% Sweep h values
h_vals = 0.1:0.1:0.9;
rho_s = zeros(size(h_vals));
rho_p = zeros(size(h_vals));
avg_sd = zeros(size(h_vals));

skills_grid_col  = (1:S)'; % for E[s]
skills_grid2_col = (1:S)'.^2; % for E[s^2]

skills_grid_col = (1:S)'; % column for mean calc
tic;  % start total timer
fprintf('\n--- Running D-HMM correlation sweep over h ---\n');
fprintf('%5s %10s %10s %10s\n','h','Time(s)','ρ_Spearman','ρ_Pearson');
fprintf('---------------------------------------------\n');

for k = 1:numel(h_vals)
    h = h_vals(k);
    t_start = tic;

    P = makeP(S, h);

    % D-HMM forward pass
    skills = (1/S) * ones(M, S);
    for t = 1:numel(Y)
        a = Aidx(t); b = Bidx(t); y = Y(t);

        skills_pred = skills * P;
        L = G; if y==0, L = 1-L; end

        psi_a = skills_pred(a,:);
        psi_b = skills_pred(b,:);
        s_a   = (L * psi_b.').';
        s_b   = (psi_a * L);
        phi_a = psi_a .* s_a; phi_a = phi_a / sum(phi_a);
        phi_b = psi_b .* s_b; phi_b = phi_b / sum(phi_b);

        skills = skills_pred;
        skills(a,:) = phi_a;
        skills(b,:) = phi_b;
    end

    % Final mean & SD per player
    final_mean = skills * skills_grid_col;            
    final_m2   = skills * skills_grid2_col;
    final_var  = max(final_m2 - final_mean.^2, 0); % variance
    final_sd   = sqrt(final_var);
    
    % Average across players
    avg_sd(k) = mean(final_sd(mask));


    % Correlation with empirical win rates
    final_mean = skills * skills_grid_col;
    rho_s(k) = corr(final_mean(mask), win_rate_all(mask), 'Type','Spearman');
    rho_p(k) = corr(final_mean(mask), win_rate_all(mask), 'Type','Pearson');

    fprintf('%5.2f %10.2f %10.3f %10.3f\n', ...
        h, toc(t_start), rho_s(k), rho_p(k));
end

fprintf('---------------------------------------------\n');
fprintf('Total elapsed time: %.2f s\n', toc);


% Plot correlation vs h
figure('Name','Correlation vs h (MeanSkill vs WinRate)','NumberTitle','off');
plot(h_vals, rho_s, '-o', 'LineWidth', 2, 'DisplayName','Spearman'); hold on;
plot(h_vals, rho_p, '-s', 'LineWidth', 2, 'DisplayName','Pearson');
xlabel('Value of h');
ylabel('Correlation Factor Value');
title('');
xlim([0.1 0.9]); ylim([0.75 0.9]);
grid on; box on; legend('Location','best');

figure('Name','Average Final Skill SD vs h','NumberTitle','off');
plot(h_vals, avg_sd, '-o', 'LineWidth', 2);
xlabel('Value of h');
ylabel('Average Final Skill SD');
title('');
xlim([min(h_vals) max(h_vals)]);
ymin = min(avg_sd); ymax = max(avg_sd); pad = 0.05*(ymax - ymin + eps);
ylim([ymin - pad, ymax + pad]);
grid on; box on;

% Helpers
function P = makeP(S,h)
    P = (1-h)*eye(S);
    for i = 1:S
        if i>1, P(i,i-1)=P(i,i-1)+h/2; end
        if i<S, P(i,i+1)=P(i,i+1)+h/2; end
    end
end

function G = obs_matrix_logistic(S,scale)
    [I,J] = ndgrid(1:S,1:S);
    G = 1 ./ (1 + exp(-(I-J)/scale));
end
