<p align="center">
  <h1 align="center">PBOC Latent Monetary Policy Factor Extraction</h1>
  <p align="center">
    <strong>Dynamic Factor Model (DFM) + Kalman Filter</strong><br/>
    <em>Extracting the true liquidity stance from PBOC's multi-dimensional policy toolkit</em>
  </p>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://www.statsmodels.org"><img src="https://img.shields.io/badge/statsmodels-0.14+-4C72B0" alt="statsmodels"/></a>
  <a href="https://plotly.com"><img src="https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly" alt="Plotly"/></a>
  <a href="https://github.com/alpha-xone/xbbg"><img src="https://img.shields.io/badge/Bloomberg-xbbg-FF6600" alt="Bloomberg"/></a>
  <a href="https://www.nber.org/papers/w34626"><img src="https://img.shields.io/badge/NBER-w34626-8B0000" alt="NBER"/></a>
</p>

---

## Overview

This framework implements a **state-space Dynamic Factor Model** to extract a single latent liquidity factor from the People's Bank of China's heterogeneous policy instruments. By applying **Kalman filtering and smoothing**, we distill five observable policy tools into one interpretable signal that captures the PBOC's true monetary policy stance.

The extracted factor enables:
- **Policy regime classification** (tightening / neutral / easing)
- **Divergence detection** between the latent factor and market rates
- **Convexity trading signal generation** via regression residual analysis

> **Methodology:** Based on Helene Rey et al., *"The Ins & Outs of Chinese Monetary Policy Transmission"* ([NBER Working Paper w34626](https://www.nber.org/papers/w34626))

---

## Key Results at a Glance

The following results are from a sample run (2018-01 to 2026-02, 2111 trading days):

| Metric | Value | Interpretation |
|:---|:---:|:---|
| Log-Likelihood | -4222.34 | Model fit quality |
| AIC / BIC | 8476.68 / 8567.16 | Information criteria |
| AR(1) persistence | 0.0416 | Low persistence after differencing |
| Factor skewness | -0.07 | Near-symmetric distribution |
| Factor kurtosis | 0.09 | Near-normal tails |
| IRS_1Y beta (p-value) | 0.034 (0.039) | Significant at 5% level |

### Factor Loadings

| Policy Tool | Loading | Interpretation |
|:---|:---:|:---|
| **DR007** | **0.997** | Strongest signal - interbank rate dominates factor |
| **1Y NCD** | **0.632** | Strong co-movement with interbank liquidity |
| **1Y MLF** | 0.155 | Moderate - administered policy rate |
| **RRR** | -0.146 | Inverse - easing = rate cut = factor down |
| **OMO Net** | ~0.000 | Minimal after rolling normalization |

> Loading > 0: tightening pushes factor up. Loading < 0: easing pushes factor up.

### Policy Regime Distribution

| Regime | Criteria | Description |
|:---|:---:|:---|
| **Tightening** | F(t) > +0.5 | PBOC actively draining liquidity |
| **Neutral** | -0.5 <= F(t) <= +0.5 | Stable liquidity conditions |
| **Easing** | F(t) < -0.5 | PBOC injecting liquidity |

---

## State-Space Formulation

The model is expressed as a standard linear Gaussian state-space system:

```
Observation equation:   Y_t = Lambda * F_t + epsilon_t      epsilon_t ~ N(0, R)
State transition:       F_t = Phi * F_{t-1} + eta_t         eta_t ~ N(0, Q)
Idiosyncratic AR(1):    epsilon_it = rho_i * epsilon_{i,t-1} + v_it
```

| Symbol | Dimension | Description |
|:---:|:---:|:---|
| **Y_t** | k x 1 | Observable PBOC policy tool vector |
| **F_t** | 1 x 1 | **Latent monetary policy factor** |
| **Lambda** | k x 1 | Factor loadings - sensitivity of each tool to the factor |
| **Phi** | 1 x 1 | AR(1) persistence coefficient |
| **R** | k x k | Observation noise covariance (diagonal) |
| **Q** | 1 x 1 | State disturbance variance |
| **rho_i** | 1 x 1 | Idiosyncratic noise AR(1) coefficient per variable |

**Estimation strategy:**
1. **EM initialization** (50 iterations) for stable starting parameters
2. **L-BFGS-B optimization** (MLE) for final estimates, with Powell fallback
3. **Sign identification:** Factor is oriented so F(t) is positively correlated with DR007 (tightening = factor rises)

---

## Data Sources

### Observable Policy Tools (Y_t)

| Indicator | Bloomberg Ticker | Type | Frequency | Stationarity Treatment |
|:---|:---|:---:|:---:|:---|
| **DR007** | `CNFR007 Index` | Price-based | Daily | **Adaptive** - ADF test -> diff if non-stationary |
| **1Y NCD** | `CNAA1Y Index` | Price-based | Daily | **Adaptive** - ADF test -> diff if non-stationary |
| **1Y MLF** | `CHLR12M Index` | Price-based | Daily | **Diff + EMA** (halflife=10d) - smooths discrete jumps |
| **RRR** | `CHRRRP Index` | Quantity-based | Daily | **Diff + EMA** (halflife=10d) - smooths 25bp step changes |
| **OMO Net Injection** | `CNNIOMO Index` | Quantity-based | Daily | **Rolling Z-score** (60d window) |

### Alpha Validation Benchmarks

| Indicator | Bloomberg Ticker | Purpose |
|:---|:---|:---|
| **1Y IRS** | `CNRB1Y Curncy` | Interest rate swap - regression target |
| **10Y CGB** | `GCNY10YR Index` | Government bond yield - regression target |

### Adaptive Stationarity Logic

```
Raw series
  |
  v
ADF test (p < 0.05?)
  |
  +-- YES --> Keep level (stationary)
  |
  +-- NO  --> First difference (Delta)
                |
                v
              ADF test on Delta (p < 0.05?)
                |
                +-- YES --> Use Delta (stationary)
                |
                +-- NO  --> Second difference (Delta^2)
```

For discrete policy rates (MLF, RRR), an **EMA smoothing** step (halflife=10 trading days) is applied after differencing to diffuse sparse jumps into gradual signals, preventing kurtosis-dominated MLE.

### Data Fallback Mechanism

When Bloomberg terminal is unavailable, the framework automatically falls back to a **structural quasi-data generator** based on correlated Ornstein-Uhlenbeck processes with shared innovations:

```
dX_it = kappa_i * (mu_i - X_it) * dt
      + lambda_i * sigma_f * z_t * sqrt(dt)    <-- common factor shock
      + sigma_i * dW_it                         <-- idiosyncratic noise

where z_t ~ N(0,1) is a shared daily innovation driving cross-variable correlation
```

| Indicator | mu (long-run mean) | kappa (mean-reversion) | lambda (factor loading) | sigma (idio.) | Approx. R^2 |
|:---|:---:|:---:|:---:|:---:|:---:|
| DR007 | 2.10% | 0.15 | 0.70 | 0.50 | ~66% |
| 1Y NCD | 2.50% | 0.05 | 0.55 | 0.45 | ~60% |
| 1Y MLF | 2.75% | 0.02 | 0.35 | 0.15 | ~84% |
| RRR | 8.50% | 0.01 | -0.40 | 0.20 | ~80% |
| OMO Net | 0 | 0.20 | 300.0 | 350.0 | ~42% |

> R^2 approximation: `lambda^2 / (lambda^2 + sigma^2)` - represents the proportion of variance explained by the common factor.

---

## Project Structure

```
monetaryPolicy/
|
|-- data_engine.py                  <-- Data engine (Bloomberg + OU fallback)
|   |-- class PBOCDataEngine            Fetch & preprocess 5 policy tools
|   |   |-- fetch()                     Bloomberg-first with auto-fallback
|   |   |-- get_processed()             Adaptive stationarity pipeline
|   |   |-- run_adf_battery()           ADF unit root test suite
|   |   +-- _generate_quasi_data()      Structural DGP v2 (shared innovations)
|   |
|   +-- class BenchmarkDataEngine       Fetch IRS / CGB benchmark rates
|
|-- kalman_model.py                 <-- DFM core (state-space + Kalman filter)
|   |-- class PBOCDynamicFactorModel    MLE estimation, factor extraction
|   |   |-- fit()                       EM init + L-BFGS-B optimization
|   |   +-- _extract_results()          Structured DFMResults output
|   |
|   |-- class AlphaValidator            OLS regression, divergence detection
|   |   |-- run_regression()            HAC-robust (Newey-West, maxlags=10)
|   |   |-- identify_divergence_regimes()  Z-score based signal detection
|   |   +-- get_regression_summary()    Formatted regression output
|   |
|   |-- @dataclass DFMResults           Structured estimation results
|   +-- Utility functions               compute_factor_stats, classify_policy_regime
|
|-- pboc_factor_analysis.ipynb      <-- Main notebook (CIO-grade interactive report)
|-- pboc_factor_analysis.html       <-- Pre-rendered HTML report (static export)
+-- README.md
```

### Design Principles

- **Separation of concerns:** Data acquisition, model estimation, and visualization are fully decoupled
- **Notebook as report:** `.ipynb` contains only high-level calls and charts - zero business logic
- **Dual data source:** Bloomberg real-time with seamless OU fallback, controlled via `force_mock` flag
- **Idiosyncratic AR(1) errors:** Addresses residual autocorrelation flagged by Ljung-Box test

---

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scipy statsmodels plotly jupyter
# For Bloomberg terminal users:
pip install xbbg blpapi
```

### 2. Run the Notebook

```bash
jupyter notebook pboc_factor_analysis.ipynb
```

Or view the pre-rendered report:

```bash
open pboc_factor_analysis.html
# or simply open the HTML file in any browser
```

> No Bloomberg terminal? No configuration needed - the framework automatically falls back to OU quasi-data and the full pipeline executes normally.

### 3. Force Mock Data Mode

```python
engine = PBOCDataEngine(force_mock=True)
raw_data = engine.fetch()
processed = engine.get_processed()
```

---

## Analysis Pipeline

The notebook executes a 5-stage analytical pipeline:

```
+-----------------------------------------------------------------------+
|                                                                       |
|  Stage 1: DATA ACQUISITION                                           |
|  Bloomberg (xbbg) --[connection failed]--> OU quasi-data generator    |
|  Output: 2130 rows x 5 cols (DR007, NCD_1Y, MLF_1Y, RRR, OMO_NET)   |
|                                                                       |
|  Stage 2: STATIONARITY PREPROCESSING                                 |
|  Per-variable adaptive treatment:                                     |
|    DR007    -> diff  (ADF p=0.877 -> diff -> ADF p=0.000)            |
|    NCD_1Y   -> diff  (ADF p=0.432 -> diff -> ADF p=0.000)            |
|    MLF_1Y   -> diff_ema (diff + EMA halflife=10d)                    |
|    RRR      -> diff_ema (diff + EMA halflife=10d)                    |
|    OMO_NET  -> zscore (rolling 60d Z-standardization)                |
|  Output: 2111 rows x 5 cols (all stationary, ADF p=0.0)             |
|                                                                       |
|  Stage 3: DFM ESTIMATION (Kalman Filter)                             |
|  MLE: EM(50 iter) + L-BFGS-B(500 iter)                              |
|  Single factor, AR(1) state, AR(1) idiosyncratic errors              |
|  Output: F(t), Lambda, Phi, R, Q, log-likelihood, AIC, BIC          |
|                                                                       |
|  Stage 4: FACTOR ANALYSIS                                            |
|  - Factor loading interpretation                                      |
|  - Policy regime classification (tightening/neutral/easing)           |
|  - Factor vs. observable tools divergence detection                   |
|  - Descriptive statistics (mean, std, skew, kurtosis, percentiles)   |
|                                                                       |
|  Stage 5: ALPHA VALIDATION                                           |
|  OLS: IRS_1Y = alpha + beta * F(t) + epsilon (HAC standard errors)   |
|  OLS: CGB_10Y = alpha + beta * F(t) + epsilon                       |
|  Residual Z-score > +/-1.5 --> Convexity trading signal              |
|  Output: Regression coefficients, R^2, divergence regime table       |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## Interactive Visualizations

The notebook produces **7 interactive Plotly charts** (dark theme, zoom/hover enabled):

| # | Chart | Description |
|:---:|:---|:---|
| 1 | **Raw Series Overview** | 5-panel subplot - DR007, NCD, MLF, RRR, OMO raw time series |
| 2 | **Factor Loadings Bar Chart** | Lambda values with color coding (positive=cyan, negative=pink) |
| 3 | **Latent Factor Time Series** | F(t) with tightening/easing zone shading and zero line |
| 4 | **Factor vs. Observables** | F(t) overlaid with DR007 & RRR (Z-scored), divergence markers |
| 5 | **Policy Regime Pie Chart** | Distribution of tightening / neutral / easing trading days |
| 6 | **Regression Scatter + OLS Fit** | F(t) vs IRS_1Y and CGB_10Y with R-squared annotation |
| 7 | **CIO Dashboard (3-panel)** | Panel 1: Factor vs IRS / Panel 2: Residual Z-scores / Panel 3: Regime timeline |

> The pre-rendered HTML report (`pboc_factor_analysis.html`) can be viewed in any browser without Jupyter.

---

## API Reference

### `PBOCDataEngine`

```python
from data_engine import PBOCDataEngine

engine = PBOCDataEngine(
    start_date="2018-01-01",    # Data start date
    end_date="2026-02-28",      # Data end date
    omo_rolling_window=20,      # OMO rolling sum window (trading days)
    force_mock=True,            # True = skip Bloomberg, use OU data
)

raw_data = engine.fetch()                # Raw data (auto-fallback)
processed = engine.get_processed()       # Stationarity-processed data
adf_table = engine.run_adf_battery()     # ADF test summary table
adf_single = engine.adf_test(series, "DR007")  # Single series ADF test
```

### `PBOCDynamicFactorModel`

```python
from kalman_model import PBOCDynamicFactorModel

dfm = PBOCDynamicFactorModel(
    data=processed,
    k_factors=1,        # Number of latent factors
    factor_order=1,     # AR(p) order for state transition
    error_order=1,      # AR(1) idiosyncratic errors (mitigates Ljung-Box issues)
    standardize=True,   # Standardize inputs (recommended)
)

results = dfm.fit(maxiter=500, disp=False)

# Access results
results.smoothed_factor       # pd.Series  - Kalman-smoothed factor F(t)
results.latent_factor         # pd.Series  - Filtered factor (one-sided)
results.factor_loadings       # pd.Series  - Loadings Lambda
results.transition_coeff      # float      - Persistence Phi
results.obs_noise_var         # pd.Series  - diag(R)
results.log_likelihood        # float      - Log-likelihood
results.aic                   # float      - AIC
results.bic                   # float      - BIC
results.fitted_values         # pd.DataFrame - Model fitted values
results.residuals             # pd.DataFrame - Observation residuals
results.model_summary         # str        - Full statsmodels summary
```

### `AlphaValidator`

```python
from kalman_model import AlphaValidator

validator = AlphaValidator(
    factor=results.smoothed_factor,
    benchmark=benchmark_data,
)

# Run OLS regression with HAC standard errors (Newey-West, maxlags=10)
reg = validator.run_regression("IRS_1Y")
print(validator.get_regression_summary("IRS_1Y"))

# Identify divergence regimes (|Z| > 1.5 = convexity signal)
regimes = validator.identify_divergence_regimes("IRS_1Y", zscore_threshold=1.5)
# Returns DataFrame with columns: residual Z-score, divergence direction, signal strength
```

### Utility Functions

```python
from kalman_model import compute_factor_stats, classify_policy_regime

# Descriptive statistics
stats = compute_factor_stats(results.smoothed_factor)
# Returns: mean, std, skew, kurtosis, min, max, current, 20d mean, percentiles

# Policy regime classification
regimes = classify_policy_regime(
    results.smoothed_factor,
    tight_threshold=0.5,     # F(t) > 0.5 = tightening
    loose_threshold=-0.5,    # F(t) < -0.5 = easing
)
# Returns pd.Series of "tightening" / "neutral" / "easing"
```

---

## Alpha Validation: Trading Signal Generation

The alpha validation module tests whether the latent factor has predictive power for benchmark interest rates:

```
Regression:  IRS_1Y_t = alpha + beta * F_t + epsilon_t

If beta is significant:
  --> Factor explains market rate movements
  --> Residual epsilon_t captures pricing deviations

Signal generation:
  Z(epsilon_t) > +1.5  -->  "Benchmark overpriced" (short opportunity)
  Z(epsilon_t) < -1.5  -->  "Benchmark underpriced" (long opportunity)
  |Z| <= 1.5           -->  "Equilibrium zone" (no signal)
```

### Sample Regression Output

```
Target: IRS_1Y
  alpha (intercept):    1.5719
  beta  (factor coef):  0.0336  [t=2.06, p=0.039]
  R-squared:            0.0022
  Adj R-squared:        0.0017
  Durbin-Watson:        0.0036

Target: CGB_10Y
  alpha (intercept):    2.6998
  beta  (factor coef): -0.0009  [t=-0.18, p=0.854]
  R-squared:            0.0000
```

> HAC standard errors (Newey-West, maxlags=10) are used to account for serial correlation and heteroskedasticity.

---

## References

- Rey, H., Jiang, Z., & Richmond, R. (2024). *The Ins & Outs of Chinese Monetary Policy Transmission*. NBER Working Paper No. 34626. [[Paper]](https://www.nber.org/papers/w34626)
- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.
- `statsmodels.tsa.statespace.DynamicFactor` [[Documentation]](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor.DynamicFactor.html)

---

## Risk Disclaimer

- OU quasi-data is **for framework validation only** - production deployment requires Bloomberg real-time data
- Single-factor model may not capture structural breaks (e.g., LPR reform, interest rate corridor evolution)
- Residual signals should be evaluated in conjunction with macro fundamentals and market microstructure
- This framework does **not** constitute investment advice
- Past model performance does not guarantee future results

---

<p align="center"><sub>Built for institutional CIO & investment decision review | Data and model outputs are for research purposes only</sub></p>
