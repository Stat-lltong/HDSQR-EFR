function demo_efr()
% DEMO_EFR Simple demonstration of EFR composite quantile regression
%
% This script provides a basic example of how to use the EFR method
% for composite quantile regression on simulated data.

    clc; clear; close all;
    fprintf('=== EFR Composite Quantile Regression Demo ===\n\n');
    
    %% Basic setup
    n = 200;        % Sample size
    p = 100;        % Number of variables
    tau = 0.7 * ones(1, 5);  % Multiple quantile levels
    
    % Generate true sparse coefficient
    beta_true = generate_true_beta(p);
    true_set = find(beta_true ~= 0);
    fprintf('True sparsity: %d/%d nonzero coefficients\n', length(true_set), p);
    
    % Generate data with Normal errors
    Mu = zeros(1, p);
    Sig = 0.7.^abs((1:p)' - (1:p));  % AR(1) covariance structure
    [X, Y] = generate_data(n, Mu, Sig, beta_true, 'Normal');
    X = standardize_matrix(X);
    
    fprintf('Data generated: n=%d, p=%d\n\n', n, p);
    
    %% Compare different sigma values for EFR
    sigma_values = [0.2, 0.5, 1.0, 2.0];
    
    fprintf('EFR Method Comparison (different sigma values):\n');
    fprintf('%-8s %8s %8s %8s %8s %8s %8s\n', 'Sigma', 'L1', 'L2', 'PE', 'FDP', 'TPP', 'Time(s)');
    fprintf(repmat('-', 1, 65)); fprintf('\n');
    
    for i = 1:length(sigma_values)
        sigma = sigma_values(i);
        
        % Fit EFR model
        tic;
        result = cqr_irw_EFR(X, Y, tau, sigma);
        elapsed_time = toc;
        
        % Compute metrics
        metrics = compute_metrics(result.beta, beta_true, Sig, true_set);
        
        % Count selected variables
        selected = find(result.beta ~= 0);
        
        % Display results
        fprintf('%-8.1f %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f\n', ...
                sigma, metrics.l1, metrics.l2, metrics.PE, ...
                metrics.FDP, metrics.TPP, elapsed_time);
        
        if i == 1
            fprintf('         (Selected: %d variables, %d iterations)\n', ...
                    length(selected), result.niter);
        end
    end
    
    %% Demonstrate effect of sigma parameter
    fprintf('\n--- Effect of sigma parameter ---\n');
    fprintf('Smaller sigma (e.g., 0.2): More aggressive penalty, sparser solutions\n');
    fprintf('Larger sigma (e.g., 2.0): Milder penalty, less sparse solutions\n');
    
    % Show coefficient patterns for extreme sigma values
    result_small = cqr_irw_EFR(X, Y, tau, 0.2);
    result_large = cqr_irw_EFR(X, Y, tau, 2.0);
    
    fprintf('\nSelected variables:\n');
    fprintf('True nonzeros: '); fprintf('%d ', true_set(1:min(10, end))); 
    if length(true_set) > 10, fprintf('...'); end
    fprintf('\n');
    
    selected_small = find(result_small.beta ~= 0);
    fprintf('Sigma=0.2:     '); fprintf('%d ', selected_small(1:min(10, end)));
    if length(selected_small) > 10, fprintf('...'); end
    fprintf(' (%d total)\n', length(selected_small));
    
    selected_large = find(result_large.beta ~= 0);
    fprintf('Sigma=2.0:     '); fprintf('%d ', selected_large(1:min(10, end)));
    if length(selected_large) > 10, fprintf('...'); end
    fprintf(' (%d total)\n', length(selected_large));
    
    fprintf('\nDemo completed successfully!\n');
    fprintf('Run cqr_efr_main() for comprehensive simulation study.\n');
end