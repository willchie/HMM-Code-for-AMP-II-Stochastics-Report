clear; clc;

%  Load and summarise data 
fname = 'tennis_matches.csv';
Tdata = readtable(fname, 'VariableNamingRule','preserve');
disp(' Data summary ');
fprintf('Rows (matches): %d\n', height(Tdata));

% Expect columns: time, winner, loser
time  = Tdata{:,1};
Aidx  = Tdata{:,2}; % winner
Bidx  = Tdata{:,3}; % loser

Y = ones(size(Aidx));
T = numel(Y);

% Get number of players
M = max(max(Aidx), max(Bidx));
fprintf('Number of players: %d\n', M);
fprintf('Distinct time blocks: %d\n', numel(unique(time)));

%  Normalise time indices 
[~, sort_idx] = sortrows(time);
Aidx = Aidx(sort_idx);
Bidx = Bidx(sort_idx);
Y    = Y(sort_idx);

%  Run D-HMM 
S = 500; 
h = 0.3;
scale = 1; % logistic link scale
P = makeP(S, h);

skills = (1/S) * ones(M, S); % uniform initial skill priors
Phi_hist = zeros(M, S, T); % store marginals

G = obs_matrix_logistic(S, scale);

for t = 1:T
    a = Aidx(t);  b = Bidx(t);  y = Y(t);

    % Prediction
    skills_pred = skills * P;

    % Observation likelihood
    L = G; if y==0, L = 1-L; end

    psi_a = skills_pred(a,:);
    psi_b = skills_pred(b,:);

    s_a = (L * psi_b.').';
    phi_a = psi_a .* s_a;  phi_a = phi_a / sum(phi_a);

    s_b = (psi_a * L);
    phi_b = psi_b .* s_b;  phi_b = phi_b / sum(phi_b);

    skills = skills_pred;
    skills(a,:) = phi_a;
    skills(b,:) = phi_b;

    Phi_hist(:,:,t) = skills;

    %  Progress print every 10% to ensure no hang-ups
    if mod(t, round(T/10)) == 0 || t == T
        fprintf('Progress: %.0f% (%d / %d matches)\n', 100*t/T, t, T);
    end
end

fprintf('\nD-HMM filtering complete.\n');

%  Results 
skills_grid = 1:S;

% Mean and variance over time for each player
mu_mat  = zeros(M, T);
var_mat = zeros(M, T);

for m = 1:M
    Phi_m = squeeze(Phi_hist(m,:,:));   % S x T
    mu_mat(m,:)  = skills_grid * Phi_m;
    m2           = (skills_grid.^2) * Phi_m;
    var_mat(m,:) = m2 - mu_mat(m,:).^2;
end

% Final skill estimates and uncertainty
final_mean = mu_mat(:,end);
final_sd   = sqrt(var_mat(:,end));

% Rank players by final mean
[~, rank_order] = sort(final_mean, 'descend');
topN = 10; % number of top players to observe
fprintf('\nTop %d players by final mean skill:\n', topN);
top_players  = rank_order(1:topN);
top_means    = final_mean(top_players);
top_sds      = final_sd(top_players);

Ttop = table(top_players(:), top_means(:), top_sds(:), ...
    'VariableNames', {'Player','MeanSkill','SD'});
disp(Ttop);


% Optional overall summary plot
mean_of_means = mean(mu_mat,1);
std_of_means  = std(mu_mat,[],1);

figure('Name','D-HMM: Overall mean skill trajectory');
fill([1:T, fliplr(1:T)], ...
     [mean_of_means+std_of_means, fliplr(mean_of_means-std_of_means)], ...
     [0.8 0.8 0.8], 'FaceAlpha',0.4,'EdgeColor','none');
hold on;
plot(mean_of_means, 'k', 'LineWidth', 2);

xlabel('Match Index');
ylabel('Mean Skill');
title(sprintf('', h));
grid on;

%  Match statistics for top-10 players 
top_players = Ttop.Player; % extract top player IDs
n_top = numel(top_players);

player_stats = table('Size',[n_top,4], ...
    'VariableTypes',{'double','double','double','double'}, ...
    'VariableNames',{'Player','Matches','Wins','Losses'});

for k = 1:n_top
    pid = top_players(k);
    is_winner = (Tdata{:,2} == pid);
    is_loser  = (Tdata{:,3} == pid);

    wins   = sum(is_winner);
    losses = sum(is_loser);
    matches = wins + losses;

    player_stats(k,:) = {pid, matches, wins, losses};
end

disp(' Match summaries for top 10 players ');
disp(player_stats);

% Win rate
player_stats.WinRate = 100 * player_stats.Wins ./ max(player_stats.Matches,1);
disp(player_stats(:, {'Player','Matches','Wins','Losses','WinRate'}));

%  Compute empirical win rate for ALL players 
all_players = (1:M)';
matches_all = zeros(M,1);
wins_all    = zeros(M,1);
losses_all  = zeros(M,1);

for pid = 1:M
    is_winner = (Tdata{:,2} == pid);
    is_loser  = (Tdata{:,3} == pid);
    wins_all(pid)   = sum(is_winner);
    losses_all(pid) = sum(is_loser);
    matches_all(pid)= wins_all(pid) + losses_all(pid);
end

win_rate_all = 100 * wins_all ./ max(matches_all,1); % avoid divide-by-zero

% Build combined table with D-HMM skill and empirical performance
summary_all = table(all_players, matches_all, wins_all, losses_all, ...
                    win_rate_all, final_mean, final_sd, ...
    'VariableNames', {'Player','Matches','Wins','Losses','WinRate','MeanSkill','SkillSD'});

% -Plot all players
figure('Name','All players: Win rate vs. final D-HMM skill');
scatter(summary_all.MeanSkill, summary_all.WinRate, 50, 'filled');
xlabel('Final Mean Skill');
ylabel('Win Rate (%)');
title(sprintf('', h));
lsline;
xlim([1 S]); % force skill axis 1..S
ylim([0 100]); % force win-rate axis 0..100%
grid on; box on;

% Highlight the top 10 players
hold on;
top_idx = ismember(summary_all.Player, Ttop.Player);
scatter(summary_all.MeanSkill(top_idx), summary_all.WinRate(top_idx), ...
    70, 'r', 'filled', 'MarkerEdgeColor','k');
legend('Players','Trend line','Top 10','Location','best');

% Quantify correlation
[r,p] = corr(summary_all.MeanSkill, summary_all.WinRate, 'Type','Spearman');
fprintf('Spearman correlation (MeanSkill vs WinRate): %.3f  (p = %.3g)\n', r, p);

% Helper functions
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
