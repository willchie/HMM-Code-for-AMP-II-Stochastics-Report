function [Phi_marg, loglike, phi_hist, c_hist] = part1_fhmm(varargin)
    if ~nargin
        M = 3;
        S = 10;
        h = 0.5;
        P = rw_transition(S,h);
        pi0 = ones(1,S)/S; % uniform init priors
    
        % Schedule (a_t vs b_t) and outcomes (Y=1 means a_t wins)
        Aidx = [1 2 3 1 2 3 1 2 3 1 2 3];
        Bidx = [2 3 1 2 3 1 2 3 1 2 3 1];
        Y    = [1 1 1 1 1 1 1 1 1 1 1 1];
    
        % Capture ALL outputs
        [Phi_marg, loglike, phi_hist, c_hist] = fhmm_filter_exact(M,S,P,pi0,Aidx,Bidx,Y);
    
        fprintf('Log-likelihood: %.6f\n', loglike);
        t = numel(Y);
        for m = 1:M
            fprintf('Player %d Φ(:,%d,t=%d) = ', m, m, t);
            disp(round(Phi_marg(:,m,t)',4));
        end
    
        % Embedded consistency checks
        fprintf('\n--- Consistency checks ---\n');
        fprintf('sum(phi_hist(:,end)) = %.6f\n', sum(phi_hist(:,end))); % joint sums to 1
    
        for m = 1:M
            fprintf('sum Φ(:,%d,end) = %.6f\n', m, sum(Phi_marg(:,m,end))); % each marginal sums to 1
        end
    
        fprintf('Any negative probs? %d\n', any(Phi_marg(:) < 0)); % should be 0
    
        % Sanity plot
        figure('Name','FHMM Part 1: Φ_{m,t} heatmaps','NumberTitle','off');
        
        for m = 1:M
            subplot(1,M,m)
            imagesc(squeeze(Phi_marg(:,m,:)), [0, 0.35]); % optional fixed scale
            xlabel('Time (t)');
            ylabel('Skill Level (s)');
            title(sprintf('Player %d',m));
            axis tight
        end
        
        % Add one shared colorbar on the right side of all subplots
        cb = colorbar('Position', [0.93 0.11 0.02 0.815]);
        cb.Label.String = 'Probability';
    
    
        % Mean skill w/ 1 SD
        figure('Name','FHMM: Mean Skill ± 1 SD','NumberTitle','off');
        skills_grid = 1:S;
        t = 1:numel(Y);
        for m = 1:M
            Phi_m = squeeze(Phi_marg(:,m,:)); % S x T
    
            mu = skills_grid * Phi_m; % 1 x T
            m2 = (skills_grid.^2) * Phi_m; % 1 x T
            var_ = m2 - mu.^2; % 1 x T
            sd = sqrt(max(var_, 0)); % guard negatives
    
            subplot(1,M,m); hold on;
            % Shaded band for mean and ± 1 SD
            xx = [t, fliplr(t)];
            yy = [mu + sd, fliplr(mu - sd)];
            fill(xx, yy, [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            plot(t, mu, 'k-', 'LineWidth', 2);
    
            set(gca,'YDir','reverse'); % put low skills at the top
    
            xlim([1, t(end)]); ylim([1, S]);
            xlabel('Time (t)'); ylabel('Skill');
            title(sprintf('Player %d', m));
            grid on; box on;
        end
    else
        % Callable pass-through
        [M,S,P,pi0,Aidx,Bidx,Y] = deal(varargin{:});
        [Phi_marg, loglike, phi_hist, c_hist] = fhmm_filter_exact(M,S,P,pi0,Aidx,Bidx,Y);
        return
    end
end

function [Phi_marg, loglike, phi_hist, c_hist] = fhmm_filter_exact(M,S,P,pi0,Aidx,Bidx,Y)
    T = numel(Y);
    N = S^M;
    
    P_big = kron_power_sparse(P,M);
    
    if isempty(pi0)
        phi = ones(N,1)/N;
    else
        phi = pi0(:);
        for m = 2:M
            phi = kron(phi, pi0(:));
        end
        phi = phi / sum(phi);
    end
    
    % Z(n,m)
    Z = build_Z(S,M);
    
    % Selector matrices A{m}
    A = build_selectors(Z,S,M);

    Phi_marg = zeros(S,M,T);
    phi_hist = zeros(N,T);
    c_hist = zeros(1,T);
    loglike = 0;
    
    for t = 1:T
        % Prediction
        psi = P_big' * phi;
    
        % Observation vector over joint states
        Lt = obs_vector(Z, Aidx(t), Bidx(t), Y(t));
    
        % Unnormalised α and scaqiling
        alpha_tilde = Lt .* psi;
        Zt = sum(alpha_tilde);
        c_t = 1 / Zt;
        phi = alpha_tilde * c_t;
        loglike = loglike - log(c_t);
    
        % Store
        phi_hist(:,t) = phi;
        c_hist(t)     = c_t;
    
        % Player-wise marginals
        for m = 1:M
            Phi_marg(:,m,t) = (phi' * A{m})';
        end
    end
end

function P = rw_transition(S,h)
    % Reflecting-boundary random walk on {1..S}
    P = zeros(S,S);
    for i = 1:S
        P(i,i) = 1 - h;
        if i > 1, P(i,i-1) = P(i,i-1) + (i==S)*h + (i<S)*(h/2); end
        if i < S, P(i,i+1) = P(i,i+1) + (i==1)*h + (i>1)*(h/2); end
    end
    % Exact boundary splits:
    P(1,2)   = h;
    P(S,S-1) = h;
end

function Pbig = kron_power_sparse(P,M)
    % Kronecker power of joint trans matrix
    Pbig = sparse(1,1,1,1,1);
    for k = 1:M
        Pbig = kron(Pbig, sparse(P));
    end
end

function Z = build_Z(S,M)
    N = S^M;
    Z = zeros(N,M,'uint16');
    for n = 0:N-1
        x = n;
        for m = M:-1:1
            Z(n+1,m) = mod(x,S) + 1;
            x = floor(x / S);
        end
    end
end

function A = build_selectors(Z,S,M)
    % A{m} is N x S sparse with (phi' * A{m}) giving marginal over X_m
    N = size(Z,1);
    A = cell(1,M);
    rows = (1:N).';
    onev = ones(N,1);
    for m = 1:M
        cols = double(Z(:,m));
        A{m} = sparse(rows, cols, onev, N, S);
    end
end

function Lt = obs_vector(Z, a_idx, b_idx, y)
    % Logistic observation over joint states: σ(x_a - x_b) for y=1
    xa = double(Z(:,a_idx));
    xb = double(Z(:,b_idx));
    delta = xa - xb;
    pwin  = 1 ./ (1 + exp(-delta));
    Lt = pwin;
    if y==0, Lt = 1 - Lt; end
    Lt = max(Lt, realmin('double')); % guard underflow
end
