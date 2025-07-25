# Usage Guide

## Setup

1. Put all `.m` files in one folder
2. Add to MATLAB path: `addpath('/your/folder')`
3. Test: `demo_efr()`

## Examples

### Simple usage
```matlab
% Basic demo
demo_efr()

% Full experiment  
hdsqr_efr_main()
```

### Your own data
```matlab
% Load your X, Y data
X = standardize_matrix(X);  % Always standardize first

% Set parameters
tau = 0.5;      % Single quantile
sigma = 1.0;    % EFR parameter

% Run
result = hdsqr_irw_EFR(X, Y, tau, sigma);
beta_hat = result.beta;
```

### Try different sigma
```matlab
for sigma = [0.2, 0.5, 1.0, 2.0]
    result = hdsqr_irw_EFR(X, Y, tau, sigma);
    sparsity = sum(result.beta ~= 0);
    fprintf('sigma=%.1f: %d variables\n', sigma, sparsity);
end
```

## Parameters

- `sigma`: Main parameter
  - 0.2 = very sparse
  - 1.0 = balanced  
  - 2.0 = less sparse

## Output

- `result.beta`: Coefficients
- `result.alpha`: Intercepts
- `result.niter`: Number of iterations

## Tips

- Always standardize X first
- Start with sigma = 1.0
- Smaller sigma = more sparse results
- Check `sum(beta ~= 0)` for sparsity

## Files needed

All 4 `.m` files must be in same folder.
