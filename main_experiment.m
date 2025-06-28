function cqr_efr_main()
% CQR_EFR_MAIN Composite quantile regression with EFR penalty
%
% This script demonstrates the EFR (Exponential Family Regularization) 
% method for composite quantile regression across different settings.
%
% Author: [Your Name]
% Date: [Date]

    clc; clear; close all; warning('off');
    
    %% Experimental settings
    n = 500;                              % Sample size
    p_list = [400, 1000];                 % Dimensions to test
    dist_list = {'Normal', 'Mixture', 't3', 'Cauchy'};  % Error distributions
    M = 30;                               % Number of replications
    K = 19;                               % Number of quantiles
    tau = 0.7 * ones(1, K);              % Quantile levels
    sigma_list = [0.2, 1.0];             % EFR penalty parameters
    
    rng(831);  % Set random seed for reproducibility
    
    % Run experiments for each dimension and distribution
    for p = p_list
        fprintf('\n========== Dimension p = %d ==========\n', p);
        
        % Generate true parameters
        beta_true = generate_true_beta(p);
        true_set = find(beta_true ~= 0);
        
        % Generate covariance matrix
        Mu = zeros(1, p);
        Sig = 0.7.^abs((1:p)' - (1:p));
        
        for dist_idx = 1:length(dist_list)
            dist_type = dist_list{dist_idx};
            fprintf('\n--- Distribution: %s ---\n', dist_type);
            
            % Run simulation for both sigma values
            for sigma_idx = 1:length(sigma_list)
                sigma = sigma_list(sigma_idx);
                fprintf('\n  EFR sigma = %.1f\n', sigma);
                
                results = run_efr_simulation(n, p, M, K, tau, sigma, Mu, Sig, beta_true, true_set, dist_type);
                display_efr_results(results, sigma);
                save_efr_results(results, p, dist_type, sigma);
            end
        end
    end
    
    fprintf('\nEFR experiments completed!\n');
end

function results = run_efr_simulation(n, p, M, K, tau, sigma, Mu, Sig, beta_true, true_set, dist_type)
% Run EFR simulation for given parameters
    
    % Initialize result storage
    results = struct('l1', zeros(M, 1), 'l2', zeros(M, 1), ...
                    'PE', zeros(M, 1), 'FDP', zeros(M, 1), 'TPP', zeros(M, 1));
    
    % Main simulation loop
    for rep = 1:M
        if mod(rep, 10) == 0
            fprintf('    Replication %d/%d\n', rep, M);
        end
        
        % Generate data
        [X, Y] = generate_data(n, Mu, Sig, beta_true, dist_type);
        X = standardize_matrix(X);
        
        % Apply EFR method
        result_est = cqr_irw_EFR(X, Y, tau, sigma);
        
        % Compute metrics
        metrics = compute_metrics(result_est.beta, beta_true, Sig, true_set);
        results.l1(rep) = metrics.l1;
        results.l2(rep) = metrics.l2;
        results.PE(rep) = metrics.PE;
        results.FDP(rep) = metrics.FDP;
        results.TPP(rep) = metrics.TPP;
    end
end

function display_efr_results(results, sigma)
% Display mean results for EFR method
    fprintf('    EFR-%.1f Results: L1=%.3f, L2=%.3f, PE=%.3f, FDP=%.3f, TPP=%.3f\n', ...
            sigma, mean(results.l1), mean(results.l2), mean(results.PE), ...
            mean(results.FDP), mean(results.TPP));
end

function save_efr_results(results, p, dist_type, sigma)
% Save EFR results to Excel file
    mean_vals = [mean(results.l1), mean(results.l2), mean(results.PE), ...
                 mean(results.FDP), mean(results.TPP)];
    std_vals = [std(results.l1), std(results.l2), std(results.PE), ...
                std(results.FDP), std(results.TPP)];
    
    col_names = {'Mean_L1', 'Mean_L2', 'Mean_PE', 'Mean_FDP', 'Mean_TPP', ...
                 'Std_L1', 'Std_L2', 'Std_PE', 'Std_FDP', 'Std_TPP'};
    data_table = [mean_vals, std_vals];
    results_table = array2table(data_table, 'VariableNames', col_names, ...
                               'RowNames', {sprintf('EFR_%.1f', sigma)});
    
    filename = 'efr_results.xlsx';
    sheet_name = sprintf('p%d_%s_sigma%.1f', p, dist_type, sigma);
    writetable(results_table, filename, 'Sheet', sheet_name, 'WriteRowNames', true);
end