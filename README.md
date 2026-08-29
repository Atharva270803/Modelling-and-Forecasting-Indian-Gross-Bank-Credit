# Modelling and Forecasting India's Gross Bank Credit

![Python](https://img.shields.io/badge/Python-Time%20Series-blue)
![Statsmodels](https://img.shields.io/badge/statsmodels-ARIMA%2FVECM-purple)
![PCA](https://img.shields.io/badge/FAVAR-PCA%20Factors-orange)
![Kelly](https://img.shields.io/badge/Forecast%20Accuracy-RMSE%2FMAPE-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Macro%20%26%20Banking-red)

## Project Overview

**A Comparative Study of Exponential Smoothing, ARIMA, ARIMAX, VECM, and FAVAR Models**

Gross Bank Credit (GBC) is the total value of loans and advances extended by banks to individuals, businesses, and government entities, before subtracting provisions or bad loans. It is a core indicator of economic activity: higher GBC signals more investment and consumption, and the Reserve Bank of India uses it directly to monitor credit trends and inform policy.

This project forecasts India's monthly Gross Bank Credit using five progressively more complex time-series frameworks, univariate to high-dimensional multivariate, and compares their forecast accuracy on a common train/test split:

```text
Exponential Smoothing (Holt's Method)
        ↓
ARIMA (univariate)
        ↓
VECM (multivariate, cointegrated sectoral credit)
        ↓
ARIMAX (univariate + exogenous regulatory variables)
        ↓
FAVAR (41 macro variables compressed into latent factors)
```

The comparison is not just "which model fits best" but "how much does adding structure, exogenous variables, or dimensionality reduction actually improve forecast accuracy" — each model is a deliberate step up in the amount of information it's allowed to use.

---

## Objectives

- Compare the predictive accuracy and dynamic responses of Exponential Smoothing, ARIMA, ARIMAX, VAR/VECM, and FAVAR models.
- Examine long-run and short-run relationships among sectoral credit components using ARIMAX and VECM.
- Incorporate high-dimensional macroeconomic information through a FAVAR framework for improved forecasting performance.
- Forecast gross bank credit demand using univariate and multivariate time-series models.

---

## Data

### Main Series and Sectoral Breakdown

The core dependent variable is monthly **Gross Bank Credit**, which decomposes additively into five sectoral components used in the VECM analysis:

```text
Gross Bank Credit = Food Credit + Agriculture & Allied + Industry + Services + Personal Loans
```

### External Variables

**41 external macroeconomic variables** feed the FAVAR model, spanning categories such as:

- Policy and money-market rates (Repo Rate, MSF Rate, Bank Rate, Call Money Rate, T-Bill yields)
- Inflation and prices (CPI, WPI, Primary Articles, Fuel & Power, Manufactured Products)
- External sector (Imports, Exports, INR-USD and INR-EUR spot rates, forward premia)
- Banking system aggregates (Deposits, Reserve Money, Broad Money, CRR, SLR, CD Ratio)
- Real activity (Index of Industrial Production)

**CRR (Cash Reserve Ratio)** and **CDR (Cash-Deposit Ratio)** are used specifically as exogenous regressors in the ARIMAX model.

### Sample

- 114-115 monthly observations
- 75% / 25% chronological train/test split for all models

### Stationarity

Every series was tested with the Augmented Dickey-Fuller (ADF) test at 5% significance. Nearly all variables were non-stationary in levels (p > 0.05) and stationary after first-differencing; exceptions that were already stationary in levels include Exports, Index of Industrial Production, Broad Money, Credit-Deposit Ratio, and the incremental deposit/investment ratios.

---

## Models

### 1. Holt's Exponential Smoothing

A univariate baseline for series with trend but no seasonality. It maintains two smoothing equations, one for the level and one for the trend, weighting recent observations more heavily than older ones.

**Result:** RMSE = 1,408,776.5 Rs Cr, MAPE = 14.66% — the weakest of the five models, as expected from a method that uses no structural or exogenous information at all.

### 2. ARIMA

Combines autoregressive (AR), differencing (I), and moving-average (MA) terms to capture temporal dependence in the univariate series.

- Model order identified via ACF/PACF after first differencing (d = 1)
- Best-fit model: **ARIMA(0,1,1)**, i.e. IMA(1,1), selected on AIC (2160.49) / BIC (2167.78)
- Parameters estimated by Maximum Likelihood Estimation: drift μ = 46,540.40 (p < 0.001), MA(1) coefficient θ₁ = −0.0223 (not significant)

**Result:** RMSE = 309,724 Rs Cr, MAPE = 3.289%

### 3. VECM (Vector Error Correction Model)

Used because the five sectoral credit series are non-stationary but cointegrated — VECM captures both short-run dynamics and long-run equilibrium correction:

$$\Delta X_t = \alpha\beta' X_{t-1} + \sum_i \Gamma_i \Delta X_{t-i} + \mu + \varepsilon_t$$

- VAR lag length selected by AIC/BIC: **p = 1**
- Johansen trace and max-eigenvalue tests: **cointegrating rank r = 3** (H0: r ≤ 3 fails to reject)
- Three cointegrating vectors (β₁, β₂, β₃) estimated as eigenvectors of the reduced-rank regression, giving three stationary linear combinations (ECT1-ECT3) of the five log-credit series
- Loading matrix α estimated by OLS of ΔYt on the three error-correction terms plus a sector-specific trend constant μ

**Result:** RMSE = 383,269 Rs Cr, MAPE = 4.08%

### 4. ARIMAX

An ARIMA model augmented with exogenous regressors — CRR and CDR, both non-stationary in levels and stationary after differencing.

- Model order identified via ACF/PACF of the first-differenced series: **ARIMAX(6,1,1)**, selected on AIC (2001.72) / BIC (2025.41), Ljung-Box p = 0.8751 (no residual autocorrelation)
- Significant coefficients: AR(1) (−0.2354, p = 0.033), AR(5) (−0.1769, p = 0.040), AR(6) (+0.5635, p < 0.001), CRR (−90,001, p < 0.001), intercept (+45,740, p < 0.001)
- CDR was not statistically significant (p = 0.187)

**Result:** RMSE = 98,001 Rs Cr, MAPE = 0.8910% — the best-fit model overall.

### 5. FAVAR (Factor-Augmented Vector Autoregressive)

Motivated by the fact that ARIMA/ARIMAX/VECM each use only a handful of variables, while many macroeconomic indicators are actually relevant — omitting them risks mismeasuring the estimated shocks. FAVAR compresses the 41 external variables into a small number of latent factors via PCA, then places those factors alongside the key observed policy variables into a VAR:

$$X_t = \Lambda_f F_t + \Lambda_y Y_t + e_t \quad \text{(observation equation)}$$
$$[F_t, Y_t]' = \Phi(L)[F_{t-1}, Y_{t-1}]' + u_t \quad \text{(transition equation)}$$

- `Xt`: 41 macro variables (32 retained after dropping variables with 100% missing data or severe collinearity, e.g. SDF Rate, Base Rate, MCLR, Non-Food Credit)
- `Ft`: 12 latent factors, selected as the number needed for cumulative explained variance ≥ 80%
- `Yt`: observed variables — log Gross Credit and Repo Rate
- Preprocessing: log-differencing for Gross Credit/Repo Rate/exchange rates, forward/back-fill for minor gaps, ADF-based differencing for non-stationary series, Z-score standardization before PCA (fit on the training set only)
- VAR lag length selected by AIC: **lag 4** (AIC = −131, max eigenvalue = 0.958)

**Interpreted factors** (top 5 by variance): Financial Tightness (15.9%), Inflation (11.1%), Input Cost (10.2%), Exchange Rate (7.6%), Trade & External Demand (7.1%) — the remaining seven factors account for the rest, covering bank deposit/credit cycle, regulatory liquidity, real activity, lending rate differentials, term premium, banking sector health, and savings/retail credit.

**Result:** RMSE = 163,341 Rs Cr, MAPE = 1.48%

---

## Model Comparison

| Method | RMSE (Rs Crore) | MAPE |
|---|---:|---:|
| Holt's Exponential Smoothing | 1,408,776 | 14.66% |
| ARIMA | 309,724 | 3.289% |
| VECM | 383,269 | 4.08% |
| **ARIMAX** | **98,001** | **0.891%** |
| FAVAR | 163,341 | 1.48% |

**ARIMAX is the best-fit model for forecasting credit demand in this analysis.** Two directly observed exogenous regulatory variables (CRR, CDR) outperformed both the purely univariate ARIMA and the 41-variable, PCA-compressed FAVAR — more information did not translate into better forecasts once it had to be compressed into latent factors, and the sectoral cointegration structure in VECM added complexity without a corresponding accuracy gain.

---

## Limitations

**1. Unmodelled structural breaks** — events like demonetization and COVID-19 were not built into any of the models, so predictions degrade during sudden regime changes.

**2. Limited sample size (N = 115)** — constrains how much the models, particularly the higher-parameter ARIMAX and FAVAR specifications, can learn.

**3. Dependence on past data and subjective factor interpretation** — all five models are backward-looking and struggle with genuinely novel situations; in PCA-based methods (FAVAR), factor interpretation is a judgment call and may not always be fully accurate.

---

## Future Scope

- **Include sudden shocks in models** — intervention dummies or regime-switching components for events like COVID-19 or demonetization.
- **Use more and better data** — longer time spans or higher-frequency data to give the models more to learn from.
- **Use advanced non-linear models** — Neural Networks, Random Forest, XGBoost, or hybrid approaches (e.g. ARIMA combined with machine learning) to capture patterns the linear models miss.

---

## Tech Stack

| Category | Technologies Used |
|---|---|
| Programming language | Python |
| Data manipulation | pandas, NumPy |
| Time-series modelling | statsmodels (ARIMA, Holt, VAR), custom Johansen/VECM implementation |
| Dimensionality reduction | scikit-learn (StandardScaler, PCA) |
| Diagnostics | ADF test, ACF/PACF, Ljung-Box test |
| Visualization | matplotlib, seaborn |
| Data storage | Excel |

---

## Key Takeaways

- Gross Bank Credit forecasting accuracy improved substantially by moving from a naive trend-based method (Holt's) to structured time-series models, but the improvement was not monotonic with model complexity.
- ARIMAX, the simplest multivariate extension (two exogenous variables), outperformed both the purely univariate ARIMA and the far more complex 41-variable FAVAR — adding relevant, targeted exogenous information beat adding a large volume of compressed information.
- VECM's cointegration structure is theoretically well-suited to the additive sectoral relationship in this data, but did not outperform ARIMAX in this sample.
- FAVAR's factor compression (41 → 12 factors) is useful for interpretability (financial tightness, inflation, input cost, etc.) but the dimensionality reduction step appears to have cost some forecast accuracy relative to using a small number of directly relevant exogenous variables.
- All models share the same core limitation: none account for structural breaks (COVID-19, demonetization), and the modest sample size (N = 115) constrains how much the higher-parameter specifications can reliably learn.

---

## Project Contributors

**Atharva Raut** (A040) — M.Sc. Applied Statistics and Analytics

**Vaidehi Thakare** (A041)

**Praanalie Chakraborty** (A042)

**Shreya Sawant** (A044)

**Pruthviraj Deshmukh** (A045)

**Mentor:** Dr. Leena Kulkarni

**Nilkamal School of Mathematics, Applied Statistics & Analytics**
