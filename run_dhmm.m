function Phi_hist = run_dhmm(S,M,P,Aidx,Bidx,Y,pi0)
% Returns M x S x T marginals over time

if nargin<7 || isempty(pi0), pi0 = ones(1,S)/S; end
T = numel(Y);
scale = 1;

% Init
skills = repmat(pi0, M, 1); % M x S
G = obs_matrix_logistic(S, scale);
Phi_hist = zeros(M,S,T);

for t = 1:T
    a = Aidx(t); b = Bidx(t); y = Y(t);

    % prediction
    skills_pred = skills * P; % M x S

    % likelihood
    L = G; if y==0, L = 1 - L; end

    % update a and b
    psi_a = skills_pred(a,:); psi_b = skills_pred(b,:);
    s_a = (L * psi_b.').'; phi_a = psi_a .* s_a;  phi_a = phi_a / sum(phi_a);
    s_b = (psi_a * L);  phi_b = psi_b .* s_b;  phi_b = phi_b / sum(phi_b);

    skills = skills_pred;
    skills(a,:) = phi_a;
    skills(b,:) = phi_b;
    
    Phi_hist(:,:,t) = skills;  % M x S x T
end
end

% Helpers
function G = obs_matrix_logistic(S,scale)
    [I,J] = ndgrid(1:S,1:S);
    G = 1 ./ (1 + exp(-(I-J)/scale));
end
