function demo_hdsqr_methods()
    clc; clear; close all;
    fprintf('=== hdsqr Methods Comparison ===\n\n');
    
    n = 200;
    p = 100;
    tau = 0.7 * ones(1, 5);
    
    beta_true = genTrueBeta(p);
    true_set = find(beta_true ~= 0);
    fprintf('True sparsity: %d/%d nonzero coefficients\n', length(true_set), p);
    
    Mu = zeros(1, p);
    Sig = 0.7.^abs((1:p)' - (1:p));
    [X, Y] = generate_data(n, Mu, Sig, beta_true, 'Normal');
    X = standardizeMatrix(X);
    
    fprintf('Data generated: n=%d, p=%d\n\n', n, p);
    
    fprintf('Method Comparison:\n');
    fprintf('%-8s %8s %8s %8s %8s %8s %8s\n', 'Method', 'L1', 'L2', 'PE', 'FDP', 'TPP', 'Time(s)');
    fprintf(repmat('-', 1, 65)); fprintf('\n');
    
    % L1 method
    tic;
    result_l1 = hdsqr_l1(X, Y, tau, 2.5 * quantile(hdsqr_self_tuning(n, repmat(X', 1, length(tau)), tau), 0.95), n);
    time_l1 = toc;
    metrics_l1 = computeMetrics(result_l1.beta, beta_true, Sig, true_set);
    selected_l1 = find(result_l1.beta ~= 0);
    fprintf('%-8s %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f\n', ...
            'L1', metrics_l1.l1, metrics_l1.l2, metrics_l1.PE, ...
            metrics_l1.FDP, metrics_l1.TPP, time_l1);
    
    % SCAD method
    tic;
    result_scad = hdsqr_irw(X, Y, tau, n, 'SCAD');
    time_scad = toc;
    metrics_scad = computeMetrics(result_scad.beta, beta_true, Sig, true_set);
    selected_scad = find(result_scad.beta ~= 0);
    fprintf('%-8s %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f\n', ...
            'SCAD', metrics_scad.l1, metrics_scad.l2, metrics_scad.PE, ...
            metrics_scad.FDP, metrics_scad.TPP, time_scad);
    
    % MCP method
    tic;
    result_mcp = hdsqr_irw(X, Y, tau, n, 'MCP');
    time_mcp = toc;
    metrics_mcp = computeMetrics(result_mcp.beta, beta_true, Sig, true_set);
    selected_mcp = find(result_mcp.beta ~= 0);
    fprintf('%-8s %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f\n', ...
            'MCP', metrics_mcp.l1, metrics_mcp.l2, metrics_mcp.PE, ...
            metrics_mcp.FDP, metrics_mcp.TPP, time_mcp);
    
    % EFR method
    sigma = 1.0;
    tic;
    result_efr = hdsqr_irw_EFR(X, Y, tau, sigma);
    time_efr = toc;
    metrics_efr = computeMetrics(result_efr.beta, beta_true, Sig, true_set);
    selected_efr = find(result_efr.beta ~= 0);
    fprintf('%-8s %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f\n', ...
            'EFR', metrics_efr.l1, metrics_efr.l2, metrics_efr.PE, ...
            metrics_efr.FDP, metrics_efr.TPP, time_efr);
    
    % EFR different sigma values
    fprintf('\nEFR with different sigma values:\n');
    sigma_values = [0.2, 0.5, 1.0, 2.0];
    fprintf('%-8s %8s %8s %8s %8s %8s\n', 'Sigma', 'L1', 'L2', 'PE', 'FDP', 'TPP');
    fprintf(repmat('-', 1, 55)); fprintf('\n');
    
    for i = 1:length(sigma_values)
        sigma = sigma_values(i);
        result = hdsqr_irw_EFR(X, Y, tau, sigma);
        metrics = computeMetrics(result.beta, beta_true, Sig, true_set);
        fprintf('%-8.1f %8.3f %8.3f %8.3f %8.3f %8.3f\n', ...
                sigma, metrics.l1, metrics.l2, metrics.PE, ...
                metrics.FDP, metrics.TPP);
    end
    
    fprintf('\nSelected variables:\n');
    fprintf('True:    '); fprintf('%d ', true_set(1:min(10, end))); 
    if length(true_set) > 10, fprintf('...'); end
    fprintf('\n');
    
    fprintf('L1:      '); fprintf('%d ', selected_l1(1:min(10, end)));
    if length(selected_l1) > 10, fprintf('...'); end
    fprintf(' (%d total)\n', length(selected_l1));
    
    fprintf('SCAD:    '); fprintf('%d ', selected_scad(1:min(10, end)));
    if length(selected_scad) > 10, fprintf('...'); end
    fprintf(' (%d total)\n', length(selected_scad));
    
    fprintf('MCP:     '); fprintf('%d ', selected_mcp(1:min(10, end)));
    if length(selected_mcp) > 10, fprintf('...'); end
    fprintf(' (%d total)\n', length(selected_mcp));
    
    fprintf('EFR:     '); fprintf('%d ', selected_efr(1:min(10, end)));
    if length(selected_efr) > 10, fprintf('...'); end
    fprintf(' (%d total)\n', length(selected_efr));
    
    fprintf('\nDemo completed.\n');
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
