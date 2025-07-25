function cqr_main_experiment()
    clc; clear; close all; warning('off');
    
    n = 500;
    p_list = [400, 1000];
    dist_list = {'Normal', 'Mixture', 't3', 'Cauchy'};
    M = 30;
    K = 19;
    tau = 0.7 * ones(1, K);
    sigma_list = [0.5, 1.0, 2.0];
    
    rng(831);
    
    for p = p_list
        fprintf('\n========== p = %d ==========\n', p);
        
        beta_true = genTrueBeta(p);
        true_set = find(beta_true ~= 0);
        Mu = zeros(1, p);
        Sig = 0.7.^abs((1:p)' - (1:p));
        
        for dist_idx = 1:length(dist_list)
            dist_type = dist_list{dist_idx};
            fprintf('\n--- Distribution: %s ---\n', dist_type);
            
            % L1 method
            fprintf('\n  L1 Method\n');
            results_l1 = run_simulation_l1(n, p, M, K, tau, Mu, Sig, beta_true, true_set, dist_type);
            display_results(results_l1, 'L1', []);
            save_results(results_l1, p, dist_type, 'L1', []);
            
            % SCAD method
            fprintf('\n  SCAD Method\n');
            results_scad = run_simulation_irw(n, p, M, K, tau, Mu, Sig, beta_true, true_set, dist_type, 'SCAD');
            display_results(results_scad, 'SCAD', []);
            save_results(results_scad, p, dist_type, 'SCAD', []);
            
            % MCP method
            fprintf('\n  MCP Method\n');
            results_mcp = run_simulation_irw(n, p, M, K, tau, Mu, Sig, beta_true, true_set, dist_type, 'MCP');
            display_results(results_mcp, 'MCP', []);
            save_results(results_mcp, p, dist_type, 'MCP', []);
            
            % EFR method with different sigma
            for sigma_idx = 1:length(sigma_list)
                sigma = sigma_list(sigma_idx);
                fprintf('\n  EFR Method (sigma = %.1f)\n', sigma);
                results_efr = run_simulation_efr(n, p, M, K, tau, sigma, Mu, Sig, beta_true, true_set, dist_type);
                display_results(results_efr, 'EFR', sigma);
                save_results(results_efr, p, dist_type, 'EFR', sigma);
            end
        end
    end
    
    fprintf('\nExperiments completed!\n');
end

function results = run_simulation_l1(n, p, M, K, tau, Mu, Sig, beta_true, true_set, dist_type)
    results = struct('l1', zeros(M, 1), 'l2', zeros(M, 1), ...
                    'PE', zeros(M, 1), 'FDP', zeros(M, 1), 'TPP', zeros(M, 1));
    
    for rep = 1:M
        if mod(rep, 10) == 0
            fprintf('    Replication %d/%d\n', rep, M);
        end
        
        [X, Y] = generate_data(n, Mu, Sig, beta_true, dist_type);
        X = standardizeMatrix(X);
        
        Lambda = 2.5 * quantile(cqr_self_tuning(n, repmat(X', 1, K), tau), 0.95);
        result_est = cqr_l1(X, Y, tau, Lambda, n);
        
        metrics = computeMetrics(result_est.beta, beta_true, Sig, true_set);
        results.l1(rep) = metrics.l1;
        results.l2(rep) = metrics.l2;
        results.PE(rep) = metrics.PE;
        results.FDP(rep) = metrics.FDP;
        results.TPP(rep) = metrics.TPP;
    end
end

function results = run_simulation_irw(n, p, M, K, tau, Mu, Sig, beta_true, true_set, dist_type, penalty_type)
    results = struct('l1', zeros(M, 1), 'l2', zeros(M, 1), ...
                    'PE', zeros(M, 1), 'FDP', zeros(M, 1), 'TPP', zeros(M, 1));
    
    for rep = 1:M
        if mod(rep, 10) == 0
            fprintf('    Replication %d/%d\n', rep, M);
        end
        
        [X, Y] = generate_data(n, Mu, Sig, beta_true, dist_type);
        X = standardizeMatrix(X);
        
        result_est = cqr_irw(X, Y, tau, n, penalty_type);
        
        metrics = computeMetrics(result_est.beta, beta_true, Sig, true_set);
        results.l1(rep) = metrics.l1;
        results.l2(rep) = metrics.l2;
        results.PE(rep) = metrics.PE;
        results.FDP(rep) = metrics.FDP;
        results.TPP(rep) = metrics.TPP;
    end
end

function results = run_simulation_efr(n, p, M, K, tau, sigma, Mu, Sig, beta_true, true_set, dist_type)
    results = struct('l1', zeros(M, 1), 'l2', zeros(M, 1), ...
                    'PE', zeros(M, 1), 'FDP', zeros(M, 1), 'TPP', zeros(M, 1));
    
    for rep = 1:M
        if mod(rep, 10) == 0
            fprintf('    Replication %d/%d\n', rep, M);
        end
        
        [X, Y] = generate_data(n, Mu, Sig, beta_true, dist_type);
        X = standardizeMatrix(X);
        
        result_est = cqr_irw_EFR(X, Y, tau, sigma);
        
        metrics = computeMetrics(result_est.beta, beta_true, Sig, true_set);
        results.l1(rep) = metrics.l1;
        results.l2(rep) = metrics.l2;
        results.PE(rep) = metrics.PE;
        results.FDP(rep) = metrics.FDP;
        results.TPP(rep) = metrics.TPP;
    end
end

function display_results(results, method, param)
    if isempty(param)
        fprintf('    %s Results: L1=%.3f, L2=%.3f, PE=%.3f, FDP=%.3f, TPP=%.3f\n', ...
                method, mean(results.l1), mean(results.l2), mean(results.PE), ...
                mean(results.FDP), mean(results.TPP));
    else
        fprintf('    %s-%.1f Results: L1=%.3f, L2=%.3f, PE=%.3f, FDP=%.3f, TPP=%.3f\n', ...
                method, param, mean(results.l1), mean(results.l2), mean(results.PE), ...
                mean(results.FDP), mean(results.TPP));
    end
end

function save_results(results, p, dist_type, method, param)
    mean_vals = [mean(results.l1), mean(results.l2), mean(results.PE), ...
                 mean(results.FDP), mean(results.TPP)];
    std_vals = [std(results.l1), std(results.l2), std(results.PE), ...
                std(results.FDP), std(results.TPP)];
    
    col_names = {'Mean_L1', 'Mean_L2', 'Mean_PE', 'Mean_FDP', 'Mean_TPP', ...
                 'Std_L1', 'Std_L2', 'Std_PE', 'Std_FDP', 'Std_TPP'};
    data_table = [mean_vals, std_vals];
    
    if isempty(param)
        row_name = method;
    else
        row_name = sprintf('%s_%.1f', method, param);
    end
    
    results_table = array2table(data_table, 'VariableNames', col_names, ...
                               'RowNames', {row_name});
    
    filename = 'cqr_results.xlsx';
    sheet_name = sprintf('p%d_%s', p, dist_type);
    writetable(results_table, filename, 'Sheet', sheet_name, 'WriteRowNames', true);
end

function [X, Y] = generate_data(n, Mu, Sig, beta_true, dist_type)
    X = mvnrnd(Mu, Sig, n);
    
    switch lower(dist_type)
        case 'normal'
            de = randn(n, 1);
        case 'mixture'
            mix_flag = (rand(n, 1) < 0.9);
            de = mix_flag .* randn(n, 1) + (~mix_flag) .* (10 * randn(n, 1));
        case 't3'
            de = trnd(3, n, 1);
        case 'cauchy'
            de = tan(pi * (rand(n, 1) - 0.5));
        otherwise
            error('Unknown distribution type: %s', dist_type);
    end
    
    Y = X * beta_true' + de;
end