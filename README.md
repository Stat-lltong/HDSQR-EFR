MATLAB code for hdsqr with L1, SCAD, MCP, and EFR penalties.

## Files

- `hdsqr_algorithms.m` - Main algorithms for all methods
- `utility_functions.m` - Helper functions
- `demo_example.m` - Demo comparing all methods
- `hdsqr_algorithms.m` - Full simulation study

## Usage

### Quick start

```matlab
demo_hdsqr_methods()          % Compare all methods
hdsqr_main_experiment()       % Full simulation
```

### Basic example

```matlab
% Generate data
n = 100; p = 50;
X = randn(n, p);
beta_true = [2; -1; 1; zeros(p-3, 1)];
Y = X * beta_true + randn(n, 1);
X = standardizeMatrix(X);

% Run methods
tau = [0.25, 0.5, 0.75];
result_l1 = hdsqr_l1(X, Y, tau, 2.5, n);
result_scad = hdsqr_irw(X, Y, tau, n, 'SCAD');
result_mcp = hdsqr_irw(X, Y, tau, n, 'MCP');
result_efr = hdsqr_irw_EFR(X, Y, tau, 1.0);
```

## Methods

- **L1**: Standard LASSO penalty
- **SCAD**: Smoothly clipped absolute deviation
- **MCP**: Minimax concave penalty  
- **EFR**: Exponential family regularization, `exp(-(beta/sigma)^2)`

## Parameters

- EFR sigma: 0.2 (sparse) to 2.0 (less sparse)
- Default: sigma = 1.0

## Output

Results saved to `hdsqr_results.xlsx`

## Requirements

MATLAB with Statistics Toolbox
