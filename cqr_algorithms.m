function result = cqr_irw_EFR(X, Y, tau, sigma)
% CQR_IRW_EFR Composite quantile regression with EFR penalty
%
% This function implements composite quantile regression using the 
% Exponential Family Regularization (EFR) penalty.
%
% Inputs:
%   X - Design matrix (n x p)
%   Y - Response vector (n x 1)  
%   tau - Quantile levels (1 x K)
%   sigma - EFR penalty parameter
%
% Output:
%   result - Structure with fields:
%            .alpha: intercept parameters
%            .beta: coefficient estimates
%            .res: residuals
%            .niter: number of iterations
%            .lambda: penalty parameter
%            .h: bandwidth parameter

    [n, p] = size(X);
    
    % Algorithm parameters
    phi = 0.1;                % Initial step size parameter
    gamma = 1.25;             % Step size adjustment factor
    max_iter = 1e5;           % Maximum iterations
    tol = 1e-5;               % Convergence tolerance
    lambda_param = 2.5;       % Lambda scaling parameter
    
    % Initialize algorithm components
    K = length(tau);
    m_tau = mean(tau);
    XX = repmat(X', 1, K);
    
    % Compute penalty parameter and bandwidth
    Lambda = lambda_param * quantile(compute_lambda_tuning(n, XX, tau), 0.95);
    h = compute_bandwidth(mean(X, 1), n, m_tau);
    
    % Initialize parameters
    beta0 = zeros(1, p);
    alpha0 = zeros(1, K);
    res = Y - X * beta0';
    
    % Create alpha design matrix for quantile intercepts
    alphaX = zeros(K, n * K);
    for i = 1:K
        alphaX(i, (i-1)*n + 1:i*n) = 1;
    end
    
    % Main optimization loop
    count = 0;
    r0 = 1;
    
    while r0 > tol * (sum(beta0.^2) + sum(alpha0.^2)) && count < max_iter
        % Compute gradients using conquer weights
        weight_vec = compute_conquer_weights(res, alpha0, tau, h);
        grad_alpha = alphaX * weight_vec;
        grad_beta = XX * weight_vec;
        
        % Evaluate current loss
        loss_eval0 = compute_smooth_loss(res, alpha0, tau, h);
        
        % Update parameters with EFR penalty
        alpha1 = alpha0 - grad_alpha' / phi;
        beta1 = beta0 - grad_beta' / phi;
        
        % Apply EFR penalty: omega = exp(-(beta/sigma)^2)
        omega = exp(-(beta1 / sigma).^2);
        beta1 = soft_threshold(beta1, Lambda * omega / phi);
        
        % Compute parameter changes
        diff_alpha = alpha1 - alpha0;
        diff_beta = beta1 - beta0;
        r0 = diff_beta * diff_beta' + diff_alpha * diff_alpha';
        res = Y - X * beta1';
        
        % Line search with backtracking
        loss_proxy = loss_eval0 + dot(diff_beta, grad_beta) + dot(diff_alpha, grad_alpha) + 0.5 * phi * r0;
        loss_eval1 = compute_smooth_loss(res, alpha1, tau, h);
        
        while loss_proxy < loss_eval1
            phi = phi * gamma;
            alpha1 = alpha0 - grad_alpha' / phi;
            beta1 = beta0 - grad_beta' / phi;
            omega = exp(-(beta1 / sigma).^2);
            beta1 = soft_threshold(beta1, Lambda * omega / phi);
            diff_alpha = alpha1 - alpha0;
            diff_beta = beta1 - beta0;
            r0 = diff_beta * diff_beta' + diff_alpha * diff_alpha';
            res = Y - X * beta1';
            loss_proxy = loss_eval0 + diff_beta * grad_beta + diff_alpha * grad_alpha + 0.5 * phi * r0;
            loss_eval1 = compute_smooth_loss(res, alpha1, tau, h);
        end
        
        % Update parameters
        alpha0 = alpha1;
        beta0 = beta1;
        count = count + 1;
    end
    
    result = struct('alpha', alpha1, 'beta', beta1, 'res', res, ...
                   'niter', count, 'lambda', Lambda, 'h', h);
end

function y = soft_threshold(x, threshold)
% SOFT_THRESHOLD Apply soft thresholding operator
% This is the proximal operator for L1 penalty
    temp = abs(x) - threshold;
    y = sign(x) .* max(0, temp);
end

function h = compute_bandwidth(mX, n, tau)
% COMPUTE_BANDWIDTH Calculate bandwidth for kernel smoothing
    h0 = (log(length(mX)) / n)^0.25;
    h = max(0.05, (h0 * sqrt(tau - tau^2))^0.5);
end

function lambda_sim = compute_lambda_tuning(n, XX, tau)
% COMPUTE_LAMBDA_TUNING Self-tuning lambda selection using simulation
    nsim = 200;
    lambda_sim = zeros(1, nsim);
    
    for b = 1:nsim
        lambda_sim(b) = max(abs(XX * generate_conquer_lambda(n, tau)));
    end
    
    lambda_sim = 2 * lambda_sim;
end

function conquer_lambda = generate_conquer_lambda(n, tau)
% GENERATE_CONQUER_LAMBDA Generate random lambda for self-tuning
    conquer_lambda = (rand(n, 1) <= tau(1)) - tau(1);
    
    for i = 2:length(tau)
        conquer_lambda = [conquer_lambda; (rand(n, 1) <= tau(i)) - tau(i)];
    end
    
    conquer_lambda = conquer_lambda / (length(tau) * n);
end

function weight_vec = compute_conquer_weights(res, alpha, tau, h)
% COMPUTE_CONQUER_WEIGHTS Calculate conquer weights for all quantiles
    K = length(tau);
    n = length(res);
    weight_vec = zeros(n * K, 1);
    
    for i = 1:K
        start_idx = (i-1) * n + 1;
        end_idx = i * n;
        weight_vec(start_idx:end_idx) = conquer_weight_single((alpha(i) - res) / h, tau(i));
    end
    
    weight_vec = weight_vec / K;
end

function result = conquer_weight_single(x, tau)
% CONQUER_WEIGHT_SINGLE Laplacian kernel weight for single quantile
    ker = @(x) 0.5 + 0.5 * sign(x) .* (1 - exp(-abs(x)));
    result = (ker(x) - tau) / length(x);
end

function loss = compute_smooth_loss(res, alpha, tau, h)
% COMPUTE_SMOOTH_LOSS Calculate smoothed check loss function
    losses = zeros(1, length(tau));
    for i = 1:length(tau)
        x = res - alpha(i);
        losses(i) = mean((tau(i) - 0.5) * x + h * (0.25 * (x / h).^2 + 0.25) .* (abs(x) < h) + ...
                         0.5 * abs(x) .* (abs(x) >= h));
    end
    loss = mean(losses);
end