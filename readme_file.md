# EFR Composite Quantile Regression

MATLAB code for composite quantile regression with EFR penalty.

## Files

- `cqr_efr_main.m` - Main simulation
- `demo_efr.m` - Simple example
- `cqr_efr_algorithm.m` - EFR algorithm  
- `cqr_efr_utils.m` - Helper functions

## Usage

### Quick start
```matlab
demo_efr()              % Simple example
cqr_efr_main()          % Full simulation
```

### Basic example
```matlab
% Generate data
n = 100; p = 50;
X = randn(n, p);
beta_true = [2; -1; 1; zeros(p-3, 1)];
Y = X * beta_true + randn(n, 1);

% Standardize X
X = standardize_matrix(X);

% Run EFR
tau = [0.25, 0.5, 0.75];
sigma = 1.0;
result = cqr_irw_EFR(X, Y, tau, sigma);

% Get coefficients
beta_hat = result.beta;
```

## Method

EFR uses penalty weights: `exp(-(beta/sigma)^2)`

- Small sigma → more sparse
- Large sigma → less sparse

## Settings

Default simulation uses:
- n = 500
- p = 400, 1000  
- Error types: Normal, Mixture, t(3), Cauchy
- sigma = 0.2, 1.0

## Output

Results saved to `efr_results.xlsx`

## Requirements

MATLAB with Statistics Toolbox

## Notes

Simple implementation for research use.