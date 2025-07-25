function result = hdsqr_irw_EFR(X, Y, tau, sigma)
    [n, p] = size(X); 
    mX = mean(X, 1);
    phi = 0.1;
    r0 = 1;
    gammma1 = 1.25;
    max_iter = 1e5;
    tol = 1e-5;
    lambdaparameter = 2.5;

    K = length(tau);
    m_tau = mean(tau);
    XX = repmat(X', 1, K);
    Lambda = lambdaparameter * quantile(hdsqr_self_tuning(n, XX, tau), 0.95);
    h = hdsqr_bandwidth(mX, n, m_tau);
    beta0 = zeros(1,p);
    count = 0;
    alphaa0 = zeros(1, K);
    res = Y - X * beta0';
    alphaX = zeros(K, n * K);
    for i = 1:K
        for j = (i-1) * n + 1:i * n
            alphaX(i, j) = 1;
        end
    end
    
    count = 0;
    r0 = 1;
    res = Y - X * beta0';

    while r0 > tol * (sum(beta0.^2) + sum(alphaa0.^2)) && count < max_iter
        gradalpha0 = alphaX * hdsqr_conquer_weight(res, alphaa0, tau, h, []);
        gradbeta0 = XX * hdsqr_conquer_weight(res, alphaa0, tau, h, []);
        loss_eval0 = hdsqr_smooth_check(res, alphaa0, tau, h, []);

        alpha1 = alphaa0 - gradalpha0' / phi;
        beta1 = beta0 - gradbeta0' / phi;
        omega = exp(-(beta1 / sigma).^2);
        beta1 = soft_thresh(beta1, Lambda * omega / phi);
      
        diff_alpha = alpha1 - alphaa0;
        diff_beta = beta1 - beta0;
        r0 = diff_beta * diff_beta'+ diff_alpha * diff_alpha';
        res = Y - X * beta1';

        loss_proxy = loss_eval0 + dot(diff_beta, gradbeta0) + dot(diff_alpha, gradalpha0) + 0.5 * phi * r0;
        loss_eval1 = hdsqr_smooth_check(res, alpha1, tau, h, []);

        while loss_proxy < loss_eval1
            phi = phi * gammma1;
            alpha1 = alphaa0 - gradalpha0' / phi;
            beta1 = beta0 - gradbeta0' / phi;
            omega = exp(-(beta1 / sigma).^2);
            beta1 = soft_thresh(beta1, Lambda * omega / phi);
            diff_alpha = alpha1 - alphaa0;
            diff_beta = beta1 - beta0;
            r0 = diff_beta * diff_beta' + diff_alpha * diff_alpha';
            res = Y - X * beta1';
            loss_proxy = loss_eval0 + diff_beta * gradbeta0 + diff_alpha * gradalpha0 + 0.5 * phi * r0;
            loss_eval1 = hdsqr_smooth_check(res, alpha1, tau, h, []);
        end

        alphaa0 = alpha1;
        beta0 = beta1;
        count = count + 1;
    end
    
    result = struct('alpha', alpha1, 'beta', beta1, 'res', res, 'niter', count, 'lambda', Lambda, 'h', h);
end

function result = hdsqr_irw(X, Y, tau, n, penaltyType)
    irw_tol = 1e-6;
    nstep = 5;
    lambdaparameter = 1.6;
    
    K = length(tau);
    XX = repmat(X', 1, K);
    Lambda = lambdaparameter * quantile(hdsqr_self_tuning(n, XX, tau), 0.95);
    model = hdsqr_l1(X, Y, tau, Lambda, n);
    alpha0 = model.alpha;
    beta0 = model.beta;
    res = model.res;
    err = 1;
    count = 1;
    
    while err > irw_tol && count <= nstep
        rw_lambda = concave_weight(beta0 / Lambda, Lambda, penaltyType);
        model = hdsqr_l1(X, Y, tau, rw_lambda, n);
        err = (sum((model.beta - beta0).^2) + sum((model.alpha - alpha0).^2)) / (sum(beta0.^2) + sum(alpha0.^2));
        alpha0 = model.alpha;
        beta0 = model.beta;
        res = model.res;
        count = count + 1;
    end
    
    result = struct('alpha', model.alpha, 'beta', model.beta, 'res', model.res, 'niter', count, 'lambda', Lambda);
end

function result = hdsqr_l1(X, Y, tau, Lambda, ~)
    [n, p] = size(X);
    mX = mean(X, 1);
    phi = 0.1;
    r0 = 1;
    gammma1 = 1.25;
    max_iter = 1e3;
    tol = 1e-5;
    
    K = length(tau);
    m_tau = mean(tau);
    XX = repmat(X', 1, K);
    h = hdsqr_bandwidth(mX, n, m_tau);
    beta0 = zeros(1,p);
    count = 0;
    alphaa0 = zeros(1, K);
    res = Y - X * beta0';
    alphaX = zeros(K, n * K);
    for i = 1:K
        for j = (i-1) * n + 1:i * n
            alphaX(i, j) = 1;
        end
    end
    
    count = 0;
    r0 = 1;
    res = Y - X * beta0';

    while r0 > tol * (sum(beta0.^2) + sum(alphaa0.^2)) && count < max_iter
        gradalpha0 = alphaX * hdsqr_conquer_weight(res, alphaa0, tau, h, []);
        gradbeta0 = XX * hdsqr_conquer_weight(res, alphaa0, tau, h, []);
        loss_eval0 = hdsqr_smooth_check(res, alphaa0, tau, h, []);

        alpha1 = alphaa0 - gradalpha0' / phi;
        beta1 = beta0 - gradbeta0' / phi;
        beta1 = soft_thresh(beta1, Lambda / phi);

        diff_alpha = alpha1 - alphaa0;
        diff_beta = beta1 - beta0;
        r0 = diff_beta * diff_beta'+ diff_alpha * diff_alpha';
        res = Y - X * beta1';

        loss_proxy = loss_eval0 + dot(diff_beta, gradbeta0) + dot(diff_alpha, gradalpha0) + 0.5 * phi * r0;
        loss_eval1 = hdsqr_smooth_check(res, alpha1, tau, h, []);

        while loss_proxy < loss_eval1
            phi = phi * gammma1;
            alpha1 = alphaa0 - gradalpha0' / phi;
            beta1 = beta0 - gradbeta0' / phi;
            beta1 = soft_thresh(beta1, Lambda / phi);
            diff_alpha = alpha1 - alphaa0;
            diff_beta = beta1 - beta0;
            r0 = diff_beta * diff_beta' + diff_alpha * diff_alpha';
            res = Y - X * beta1';
            loss_proxy = loss_eval0 + diff_beta * gradbeta0 + diff_alpha * gradalpha0 + 0.5 * phi * r0;
            loss_eval1 = hdsqr_smooth_check(res, alpha1, tau, h, []);
        end

        alphaa0 = alpha1;
        beta0 = beta1;
        count = count + 1;
    end
    
    result = struct('alpha', alpha1, 'beta', beta1, 'res', res, 'niter', count, 'lambda', Lambda, 'h', h);
end

function w = concave_weight(x, lambda, penaltyType)
    w = zeros(size(x));
    a_scad = 3.7;
    a_mcp = 3.7;

    switch upper(penaltyType)
        case 'SCAD'
            idx1 = (x <= 1);
            w(idx1) = 1;
            
            idx2 = (x > 1 & x <= a_scad);
            w(idx2) = (a_scad - x(idx2)) / (a_scad - 1);
            
            idx3 = (x > a_scad);
            w(idx3) = 0;
            
            w = w * lambda;

        case 'MCP'
            idxA = (x <= a_mcp);
            w(idxA) = (a_mcp - x(idxA)) / a_mcp;
            
            w(~idxA) = 0;
            w = w * lambda;

        otherwise
            error('Unknown penaltyType: %s, please use SCAD or MCP.', penaltyType);
    end
end

function y = soft_thresh(x, c)
    tmp = abs(x) - c;
    y = sign(x) .* max(0, tmp);
end
