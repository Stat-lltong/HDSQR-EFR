% CQR_EFR_UTILS Supporting functions for EFR composite quantile regression
%
% This file contains utility functions for data generation, preprocessing,
% and performance evaluation for the EFR method.

function [X, Y] = generate_data(n, Mu, Sig, beta_true, dist_type)
% GENERATE_DATA Generate simulation data with specified error distribution
%
% Inputs:
%   n - Sample size
%   Mu - Mean vector for X
%   Sig - Covariance matrix for X  
%   beta_true - True coefficient vector
%   dist_type - Error distribution: 'Normal', 'Mixture', 't3', 'Cauchy'
%
% Outputs:
%   X - Design matrix (n x p)
%   Y - Response vector (n x 1)

    X = mvnrnd(Mu, Sig, n);
    
    switch lower(dist_type)
        case 'normal'
            de = randn(n, 1);
        case 'mixture'
            % 0.9*N(0,1) + 0.1*N(0,10^2) mixture distribution
            mix_flag = (rand(n, 1) < 0.9);
            de = mix_flag .* randn(n, 1) + (~mix_flag) .* (10 * randn(n, 1));
        case 't3'
            % t-distribution with 3 degrees of freedom
            de = trnd(3, n, 1);
        case 'cauchy'
            % Standard Cauchy distribution
            de = tan(pi * (rand(n, 1) - 0.5));
        otherwise
            error('Unknown distribution type: %s', dist_type);
    end
    
    Y = X * beta_true' + de;
end

function beta_true = generate_true_beta(p)
% GENERATE_TRUE_BETA Generate sparse true coefficient vector
%
% Creates a sparse coefficient vector with a specific pattern
% where the first 19 positions contain alternating non-zero values
    
    beta_true = zeros(1, p);
    pattern = [1.8; 0; 1.6; 0; 1.4; 0; 1.2; 0; 1; 0; ...
               -1; 0; -1.2; 0; -1.4; 0; -1.6; 0; -1.8];
    len_pattern = length(pattern);
    beta_true(1:len_pattern) = pattern;
end

function X_std = standardize_matrix(X)
% STANDARDIZE_MATRIX Center and standardize design matrix
%
% Applies column-wise centering and scaling to unit variance
    
    X_centered = X - mean(X, 1);
    sd_X = std(X, 0, 1);
    X_std = X_centered ./ sd_X;
end

function metrics = compute_metrics(beta_est, beta_true, Sig, true_set)
% COMPUTE_METRICS Calculate performance metrics for coefficient estimation
%
% Inputs:
%   beta_est - Estimated coefficients
%   beta_true - True coefficients
%   Sig - Covariance matrix (for prediction error)
%   true_set - Indices of non-zero true coefficients
%
% Outputs:
%   metrics.l1  - L1 estimation error
%   metrics.l2  - L2 estimation error
%   metrics.PE  - Prediction error
%   metrics.FDP - False discovery proportion
%   metrics.TPP - True positive proportion

    diff_vec = (beta_est - beta_true);
    metrics.l1 = norm(diff_vec, 1);
    metrics.l2 = norm(diff_vec, 2);
    metrics.PE = diff_vec * Sig * diff_vec';
    
    selected_set = find(beta_est ~= 0);
    false_pos = setdiff(selected_set, true_set);
    true_pos = intersect(selected_set, true_set);
    
    % False Discovery Proportion (lower is better)
    if isempty(selected_set)
        metrics.FDP = 0;
    else
        metrics.FDP = numel(false_pos) / numel(selected_set);
    end
    
    % True Positive Proportion (higher is better)
    if isempty(true_set)
        metrics.TPP = 0;
    else
        metrics.TPP = numel(true_pos) / numel(true_set);
    end
end