function cqr_lambda_sim = cqr_self_tuning(n, XX, tau)
    nsim = 200;
    cqr_lambda_sim = zeros(1, nsim);
    
    for b = 1:nsim
        cqr_lambda_sim(b) = max(abs(XX * cqr_conquer_lambdasim(n, tau)));
    end
    
    cqr_lambda_sim = 2 * cqr_lambda_sim;
end

function cqr_lambda = cqr_conquer_lambdasim(n, tau)
    cqr_lambda = (rand(n, 1) <= tau(1)) - tau(1);
    
    for i = 2:length(tau)
        cqr_lambda = [cqr_lambda; (rand(n, 1) <= tau(i)) - tau(i)];
    end
    
    cqr_lambda = cqr_lambda / (length(tau) * n);
end

function h = cqr_bandwidth(mX, n, tau)
    h0 = (log(length(mX)) / n) ^ 0.25;
    h = max(0.05, (h0 * sqrt(tau - tau^2))^0.5);
end

function cqr_cw = cqr_conquer_weight(x, alpha, tau, h, w)
    cqr_cw = conquer_weight((alpha(1) - x) / h, tau(1), w);
    
    for i = 2:length(tau)
        cqr_cw = [cqr_cw; conquer_weight((alpha(i) - x) / h, tau(i), w)];
    end
    
    cqr_cw = cqr_cw / length(tau);
end

function result = conquer_weight(x, tau, w)
    ker = @(x) 0.5 + 0.5 * sign(x) .* (1 - exp(-abs(x)));
    
    if isempty(w)
        result = (ker(x) - tau) / length(x);
    else
        result = w .* (ker(x) - tau) / length(x);
    end
end

function cqrsc_mean = cqr_smooth_check(x, alpha, tau, h, w)
    cqrsc = zeros(1, length(tau));
    
    for i = 1:length(tau)
        cqrsc(i) = smooth_check(x - alpha(i), tau(i), h, w);
    end
    
    cqrsc_mean = mean(cqrsc);
end

function result = smooth_check(x, tau, h, w)
    loss = @(x) (tau - 0.5) * x + h * (0.25 * (x / h).^2 + 0.25) .* (abs(x) < h) + ...
                0.5 * abs(x) .* (abs(x) >= h);
    result = mean(loss(x));
end

function y = soft_thresh(x, c)
    tmp = abs(x) - c;
    y = sign(x) .* max(0, tmp);
end

function Xstd = standardizeMatrix(X)
    Xcentered = X - mean(X, 1);
    sdX = std(X, 0, 1);
    Xstd = Xcentered ./ sdX;
end

function beta_true = genTrueBeta(p)
    beta_true = zeros(1, p);
    tmp = [1.8; 0; 1.6; 0; 1.4; 0; 1.2; 0; 1; 0; ...
           -1; 0; -1.2; 0; -1.4; 0; -1.6; 0; -1.8];
    lenTmp = length(tmp);
    beta_true(1:lenTmp) = tmp;
end

function metrics = computeMetrics(beta_est, beta_true, Sig, true_set)
    diffVec = (beta_est - beta_true);
    metrics.l1 = norm(diffVec, 1);
    metrics.l2 = norm(diffVec, 2);
    metrics.PE = diffVec * Sig * diffVec';
    
    selected_set = find(beta_est ~= 0);
    falsePos = setdiff(selected_set, true_set);
    truePos = intersect(selected_set, true_set);
    
    if isempty(selected_set)
        metrics.FDP = 0;
    else
        metrics.FDP = numel(falsePos) / numel(selected_set);
    end
    
    if isempty(true_set)
        metrics.TPP = 0;
    else
        metrics.TPP = numel(truePos) / numel(true_set);
    end
end

function Sig = cov_generate(std_vec, corr)
    if nargin < 2
        corr = 0.5;
    end
    p = length(std_vec);
    R = zeros(p, p);
    for j = 1:(p-1)
        R(j, j+1:end) = 1:length(R(j, j+1:end));
    end
    R = R + R';
    Sig = (std_vec' * std_vec) .* (corr * ones(p, p)).^R;
end