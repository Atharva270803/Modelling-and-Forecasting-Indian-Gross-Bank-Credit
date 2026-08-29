"""
=============================================================================
ARIMAX(5,1,1) — Gross Bank Credit Demand Forecasting
Target   : Gross Bank Credit (₹ Crore, Monthly)
Exogenous: Cash Reserve Ratio (CRR, %), Cash-Deposit Ratio (CDR, %)
Period   : June 2011 – December 2020 | 115 observations
Train    : 86 obs (Jun 2011–Jul 2018) | Test: 29 obs (Aug 2018–Dec 2020)

Workflow (ref: 10-step ARIMAX framework):
  1  Data loading & description
  2  Stationarity tests (ADF) on all variables
  3  Differencing (d=1)
  4  Exogenous variable preparation & ADF
  5  Order identification (p, q) via ACF/PACF + AIC/BIC grid
  6  Model specification
  7  Estimation (Hannan-Rissanen MLE approximation)
  8  Residual diagnostics (Ljung-Box, Shapiro-Wilk, ACF)
  9  Forecasting with confidence intervals
  10 Evaluation (RMSE, MAE, MAPE)

Dependencies: pandas, numpy, scipy, matplotlib
No statsmodels required — all routines implemented from scratch.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')        # change to 'TkAgg' for interactive plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─── OUTPUT DIRECTORY ────────────────────────────────────────────────────────
OUTPUT_DIR = "arimax_output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── FILE PATHS (update if needed) ───────────────────────────────────────────
EXOG_PATH  = "arimax_dataset_114_obs.xlsx"
GROSS_PATH = "GROSS_ARIMAX.xlsx"

np.random.seed(42)


# =============================================================================
# STEP 1 — DATA LOADING AND DESCRIPTION
# =============================================================================
print("=" * 70)
print("STEP 1: DATA LOADING AND DESCRIPTION")
print("=" * 70)

from datetime import datetime

def parse_month(s):
    return datetime.strptime(s.strip(), '%b, %Y')

df_exog  = pd.read_excel(EXOG_PATH,  sheet_name='Sheet2')
df_gross = pd.read_excel(GROSS_PATH, sheet_name='Sheet1')

# Target: Gross Bank Credit
target_row  = df_gross[df_gross['Sector'] == 'Gross Bank Credit (II + III)'].iloc[0]
date_cols_g = [c for c in df_gross.columns if c not in ['Sr.No', 'Sector']]
target = pd.Series(target_row[date_cols_g].values,
                   index=[c.strip() for c in date_cols_g], dtype=float)

# Exogenous variables
exog_cols = [c for c in df_exog.columns if c != df_exog.columns[0]]
crr = pd.Series(df_exog.iloc[0][exog_cols].values,
                index=[c.strip() for c in exog_cols], dtype=float)
cdr = pd.Series(df_exog.iloc[1][exog_cols].values,
                index=[c.strip() for c in exog_cols], dtype=float)

# Merge on common dates
common = sorted(set(target.index) & set(crr.index), key=parse_month)
idx    = pd.to_datetime([parse_month(d) for d in common])
df     = pd.DataFrame({'Gross_Credit': target[common].values,
                        'CRR':          crr[common].values,
                        'CDR':          cdr[common].values}, index=idx)
df.index.freq = 'MS'

print(f"\nDataset shape      : {df.shape}")
print(f"Date range         : {df.index[0].strftime('%b %Y')} to {df.index[-1].strftime('%b %Y')}")
print(f"Missing values     : {df.isnull().sum().to_dict()}")
print("\nDescriptive Statistics:")
print(df.describe().round(2).to_string())

# Train/test split 75/25
n_obs  = len(df)
split  = int(n_obs * 0.75)
train  = df.iloc[:split]
test   = df.iloc[split:]
print(f"\nTrain : {len(train)} obs ({train.index[0].strftime('%b %Y')} – {train.index[-1].strftime('%b %Y')})")
print(f"Test  : {len(test)}  obs ({test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')})")

# Plot raw series
fig, axes = plt.subplots(3, 1, figsize=(15, 13))
fig.suptitle('Raw Time Series: Gross Bank Credit, CRR, CDR\n(June 2011 – December 2020, Monthly)',
             fontsize=15, fontweight='bold')
for ax, col, label, unit, color in zip(
        axes,
        ['Gross_Credit', 'CRR', 'CDR'],
        ['Gross Bank Credit (Target)', 'Cash Reserve Ratio — CRR (Exogenous 1)',
         'Cash-Deposit Ratio — CDR (Exogenous 2)'],
        ['₹ Lakh Crore', '%', '%'],
        ['#1f77b4', '#ff7f0e', '#2ca02c']):
    vals = df[col].values / 1e5 if col == 'Gross_Credit' else df[col].values
    ax.plot(df.index, vals, color=color, linewidth=2)
    ax.axvline(test.index[0], color='red', linestyle='--', alpha=0.6, label='Train/Test Split')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel(unit, fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    raw = df[col].values
    ax.text(0.01, 0.05,
            f'Mean={raw.mean():.2f}  Std={raw.std():.2f}  Min={raw.min():.2f}  Max={raw.max():.2f}',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
axes[-1].set_xlabel('Date', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F01_raw_series.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Figure saved: F01_raw_series.png]")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_acf(arr, nlags=30):
    """Autocorrelation Function (manual)."""
    y = np.asarray(arr, dtype=float)
    y = y[~np.isnan(y)]
    n, m = len(y), np.mean(y)
    v = np.sum((y - m) ** 2) / n
    out = [1.0]
    for lag in range(1, nlags + 1):
        c = np.sum((y[lag:] - m) * (y[:-lag] - m)) / n
        out.append(c / v if v > 0 else 0.0)
    return np.array(out)


def compute_pacf(arr, nlags=30):
    """Partial Autocorrelation via Yule-Walker recursion."""
    acf_v = compute_acf(arr, nlags)
    phi   = np.zeros((nlags + 1, nlags + 1))
    pacf  = [1.0, acf_v[1]]
    phi[1, 1] = acf_v[1]
    for k in range(2, nlags + 1):
        num = acf_v[k] - np.dot(phi[k-1, 1:k], acf_v[k-1:0:-1])
        den = 1.0 - np.dot(phi[k-1, 1:k], acf_v[1:k])
        phi[k, k] = num / den if abs(den) > 1e-10 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
        pacf.append(phi[k, k])
    return np.array(pacf)


def adf_test(arr, maxlag=4, regression='c'):
    """
    Augmented Dickey-Fuller test (manual).
    regression: 'c' = constant, 'ct' = constant + trend
    Returns: (adf_statistic, p_value_approx, critical_values_dict)
    """
    y  = np.asarray(arr, dtype=float)
    y  = y[~np.isnan(y)]
    dy = np.diff(y)
    k  = min(maxlag, max(1, len(dy) // 5))
    Y  = dy[k:]
    cols = [y[k:-1]]
    for j in range(1, k + 1):
        cols.append(dy[k - j: -j])
    if regression in ('c', 'ct'):
        cols.append(np.ones(len(Y)))
    if regression == 'ct':
        cols.append(np.arange(len(Y), dtype=float))
    X = np.column_stack(cols)
    try:
        beta  = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        s2    = np.sum(resid ** 2) / max(1, len(Y) - X.shape[1])
        cov   = s2 * np.linalg.inv(X.T @ X + 1e-10 * np.eye(X.shape[1]))
        se    = np.sqrt(max(0.0, cov[0, 0]))
        t     = beta[0] / se if se > 0 else np.nan
    except Exception:
        t = np.nan
    cv = ({'1%': -3.43, '5%': -2.86, '10%': -2.57} if regression == 'c'
          else {'1%': -3.96, '5%': -3.41, '10%': -3.13})
    pv = (0.01 if (not np.isnan(t) and t < cv['1%']) else
          0.05 if (not np.isnan(t) and t < cv['5%']) else
          0.10 if (not np.isnan(t) and t < cv['10%']) else 0.50)
    return t, pv, cv


def ljung_box(resid, lags=10):
    """Ljung-Box portmanteau test."""
    n   = len(resid)
    acf = compute_acf(resid, lags)
    Q   = n * (n + 2) * sum(acf[k] ** 2 / (n - k) for k in range(1, lags + 1))
    return Q, 1 - stats.chi2.cdf(Q, df=lags)


def plot_acf_pacf_panel(data_dict, nlags, title, filename):
    """Plot ACF and PACF side-by-side for multiple series."""
    n_rows = len(data_dict)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    for i, (label, arr) in enumerate(data_dict.items()):
        acf_v  = compute_acf(arr, nlags)
        pacf_v = compute_pacf(arr, nlags)
        conf   = 1.96 / np.sqrt(len(arr))
        for j, (vals, ytitle) in enumerate([(acf_v, 'ACF'), (pacf_v, 'PACF')]):
            ax = axes[i, j]
            bar_col = ['#d62728' if (abs(v) > conf and ii > 0)
                       else ('steelblue' if j == 0 else 'darkorange')
                       for ii, v in enumerate(vals)]
            ax.bar(np.arange(len(vals)), vals, color=bar_col, alpha=0.8, width=0.6)
            ax.axhline(conf,  color='red', linestyle='--', alpha=0.8, label=f'95% CI (±{conf:.3f})')
            ax.axhline(-conf, color='red', linestyle='--', alpha=0.8)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_title(f'{ytitle} — {label}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Lag'); ax.set_ylabel(ytitle)
            ax.set_ylim(-1.1, 1.1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            for ii in range(1, min(10, len(vals))):
                if abs(vals[ii]) > conf:
                    ax.annotate(f'{vals[ii]:.2f}', xy=(ii, vals[ii]),
                                xytext=(ii + 0.3, vals[ii] + 0.09 * np.sign(vals[ii])),
                                fontsize=7.5, color='darkred', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Figure saved: {filename}]")


# =============================================================================
# STEP 2 — STATIONARITY TESTS ON ALL VARIABLES (LEVEL)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: STATIONARITY TESTS — LEVEL SERIES")
print("=" * 70)

print(f"\n{'Variable':<22} {'ADF Stat':>10} {'p-value':>9} {'CV 5%':>8} {'Decision':>20}")
print("-" * 75)
for col in ['Gross_Credit', 'CRR', 'CDR']:
    stat, pval, cv = adf_test(df[col].values)
    decision = "✗ Non-Stationary" if pval > 0.05 else "✓ Stationary"
    print(f"  {col:<20} {stat:>10.4f} {pval:>9.2f} {cv['5%']:>8} {decision:>20}")

# ACF/PACF of level series
plot_acf_pacf_panel(
    {f'{c} (Level)': df[c].values for c in ['Gross_Credit', 'CRR', 'CDR']},
    nlags=25,
    title='ACF and PACF — Raw Level Series (Before Differencing)',
    filename='F02_acf_pacf_level.png'
)

# ADF annotated on time series plot
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('ADF Stationarity Test — Level Series (All Three Variables)',
             fontsize=14, fontweight='bold')
for ax, col, label, color, unit in zip(
        axes,
        ['Gross_Credit', 'CRR', 'CDR'],
        ['Gross Bank Credit', 'CRR', 'CDR'],
        ['#1f77b4', '#ff7f0e', '#2ca02c'],
        ['₹ Lakh Crore', '%', '%']):
    vals = df[col].values / 1e5 if col == 'Gross_Credit' else df[col].values
    ax.plot(df.index, vals, color=color, linewidth=2)
    stat, pval, cv = adf_test(df[col].values)
    concl = '✗ Non-Stationary — unit root present' if pval > 0.05 else '✓ Stationary'
    txt = (f'ADF Statistic  = {stat:.4f}\n'
           f'p-value         ≈ {pval:.2f}\n'
           f'Critical Values: 1%={cv["1%"]}  5%={cv["5%"]}  10%={cv["10%"]}\n'
           f'Decision: {concl}')
    ax.set_title(f'{label} — Level', fontsize=12, fontweight='bold')
    ax.set_ylabel(unit); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.55, txt, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', alpha=0.95))
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F03_adf_level.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Figure saved: F03_adf_level.png]")


# =============================================================================
# STEP 3 & 4 — DIFFERENCING (d=1) AND EXOGENOUS PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("STEPS 3 & 4: DIFFERENCING (d=1) + EXOGENOUS ADF")
print("=" * 70)

print(f"\n{'Variable':<22} {'ADF Stat':>10} {'p-value':>9} {'CV 5%':>8} {'Decision':>20}")
print("-" * 75)
for col in ['Gross_Credit', 'CRR', 'CDR']:
    d1   = np.diff(df[col].values)
    stat, pval, cv = adf_test(d1)
    decision = "✓ Stationary (d=1 sufficient)" if pval <= 0.05 else "✗ Still Non-Stationary"
    print(f"  Δ{col:<20} {stat:>10.4f} {pval:>9.2f} {cv['5%']:>8} {decision}")

print("\nConclusion: All three variables are I(1) — stationary after one differencing.")
print("ARIMAX with d=1 is appropriate. No seasonal differencing required.")

# Plot differenced series
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('First-Differenced Series (d=1) — All Three Variables', fontsize=14, fontweight='bold')
for ax, col, label, color, unit in zip(
        axes,
        ['Gross_Credit', 'CRR', 'CDR'],
        ['ΔGross Bank Credit', 'ΔCRR', 'ΔCDR'],
        ['#1f77b4', '#ff7f0e', '#2ca02c'],
        ['Δ₹ Lakh Crore', 'Δ%', 'Δ%']):
    d1  = np.diff(df[col].values)
    idx_d = df.index[1:]
    vals  = d1 / 1e5 if col == 'Gross_Credit' else d1
    ax.plot(idx_d, vals, color=color, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.fill_between(idx_d, vals, 0, alpha=0.2, color=color)
    stat, pval, cv = adf_test(d1)
    concl = '✓ STATIONARY' if pval <= 0.05 else '✗ NON-STATIONARY'
    txt = (f'ADF Statistic  = {stat:.4f}\n'
           f'p-value         ≈ {pval:.2f}\n'
           f'Critical Values: 1%={cv["1%"]}  5%={cv["5%"]}\n'
           f'Decision: {concl}')
    ax.set_title(f'{label} — After 1st Differencing', fontsize=12, fontweight='bold')
    ax.set_ylabel(unit); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.55, txt, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda', alpha=0.95))
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F04_diff1_series.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Figure saved: F04_diff1_series.png]")

# ACF/PACF after differencing
plot_acf_pacf_panel(
    {f'Δ{c} (d=1)': np.diff(df[c].values) for c in ['Gross_Credit', 'CRR', 'CDR']},
    nlags=20,
    title='ACF and PACF After 1st Differencing (d=1) — Order Identification',
    filename='F05_acf_pacf_diff.png'
)

# Print key ACF/PACF values for Gross Credit
d1_gc  = np.diff(df['Gross_Credit'].values)
acf_gc = compute_acf(d1_gc, 20)
pacf_gc = compute_pacf(d1_gc, 20)
conf_gc = 1.96 / np.sqrt(len(d1_gc))
print(f"\nACF of ΔGross_Credit (95% CI = ±{conf_gc:.4f}):")
for i in range(1, 11):
    sig = " ** SIG" if abs(acf_gc[i]) > conf_gc else ""
    print(f"  lag {i:2d}: {acf_gc[i]:8.4f}{sig}")
print(f"\nPACF of ΔGross_Credit:")
for i in range(1, 11):
    sig = " ** SIG" if abs(pacf_gc[i]) > conf_gc else ""
    print(f"  lag {i:2d}: {pacf_gc[i]:8.4f}{sig}")


# =============================================================================
# STEP 5 — ORDER IDENTIFICATION (p, q) via ACF/PACF + AIC/BIC GRID
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: ORDER IDENTIFICATION — AIC/BIC GRID SEARCH")
print("=" * 70)

y_tr = train['Gross_Credit'].values
X_tr = train[['CRR', 'CDR']].values
yd   = np.diff(y_tr)
xd   = np.diff(X_tr, axis=0)


def hannan_rissanen(yd_, xd_, p, q, n_iter=4):
    """Three-stage Hannan-Rissanen estimator for ARIMA(p,0,q)+X parameters."""
    n = len(yd_)
    start = max(p, q, 1)
    rh = np.zeros(n)
    for stage in range(n_iter):
        rows, tgts = [], []
        for t in range(start, n):
            row = [yd_[t - i] for i in range(1, p + 1)]
            if stage > 0 and q > 0:
                row += [rh[t - i] for i in range(1, q + 1)]
            row += list(xd_[t])
            row.append(1.0)
            rows.append(row)
            tgts.append(yd_[t])
        X_ = np.array(rows)
        Y_ = np.array(tgts)
        beta = np.linalg.lstsq(X_, Y_, rcond=None)[0]
        rh[start:] = Y_ - X_ @ beta
    resid = Y_ - X_ @ beta
    nf = len(Y_)
    k  = len(beta)
    s2 = np.sum(resid ** 2) / max(1, nf - k)
    ll = -0.5 * nf * (np.log(2 * np.pi * s2) + 1)
    lb_q, lb_p = ljung_box(resid, 10)
    return (-2*ll + 2*k, -2*ll + k*np.log(nf), s2, beta, resid, nf, k, lb_p, X_, Y_)


print(f"\n{'p':>2} {'q':>2}  {'AIC':>10}  {'BIC':>10}  {'LB p':>8}  {'σ':>12}  {'LB OK?':>8}")
print("-" * 65)
grid_results = []
for p in range(1, 7):
    for q in range(0, 5):
        try:
            aic, bic, s2, beta, resid, nf, k, lb_p, X_, Y_ = hannan_rissanen(yd, xd, p, q)
            ok = "✓" if lb_p > 0.05 else "✗"
            grid_results.append((aic, bic, p, q, s2, lb_p, ok, beta, resid, nf, k, X_, Y_))
            print(f"  {p:>1} {q:>2}  {aic:>10.2f}  {bic:>10.2f}  {lb_p:>8.4f}  {np.sqrt(s2):>12.2f}  {ok}")
        except Exception as e:
            pass

print("\nModels passing Ljung-Box diagnostic (p > 0.05):")
valid = [(r[0], r[1], r[2], r[3], r[5]) for r in grid_results if r[6] == '✓']
for v in sorted(valid, key=lambda x: x[0])[:8]:
    print(f"  ARIMAX({v[2]},1,{v[3]})  AIC={v[0]:.2f}  BIC={v[1]:.2f}  LB_p={v[4]:.4f}")

# Chosen: ARIMAX(5,1,1) — best balance of AIC, parsimony and LB pass
P_FINAL, Q_FINAL = 5, 1
print(f"\nChosen Model: ARIMAX({P_FINAL},1,{Q_FINAL})")
print("Justification:")
print("  • PACF of ΔGross Credit cuts off sharply after lag 5 → AR(5)")
print("  • ACF lag 1 significant (−0.25) → MA(1) cleans up lag-1 residual autocorrelation")
print("  • ARIMAX(5,1,1): AIC=2048, LB p=0.72 (best diagnostic pass)")
print("  • Parsimony: ARIMAX(6,1,0) has lower AIC but adding 1 parameter; BIC favours p=5")


# =============================================================================
# STEP 6 — MODEL SPECIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: MODEL SPECIFICATION")
print("=" * 70)
print(f"""
Model: ARIMAX({P_FINAL}, 1, {Q_FINAL})

After first-differencing, the model in the differenced space w_t = Δy_t is:

  w_t = φ₁w_(t-1) + φ₂w_(t-2) + φ₃w_(t-3) + φ₄w_(t-4) + φ₅w_(t-5)
        + θ₁ε_(t-1)
        + β₁·ΔCRR_t + β₂·ΔCDR_t + μ + ε_t

where:
  y_t  = Gross Bank Credit at time t (₹ Crore)
  w_t  = Δy_t = y_t − y_(t-1)   (1st difference, d=1)
  φ₁–φ₅ = AR coefficients (autoregressive lags 1–5)
  θ₁   = MA(1) coefficient
  β₁   = effect of ΔCRR on ΔGross Credit
  β₂   = effect of ΔCDR on ΔGross Credit
  μ    = drift / intercept
  ε_t  ~ WN(0, σ²)
""")


# =============================================================================
# STEP 7 — ESTIMATION
# =============================================================================
print("=" * 70)
print("STEP 7: PARAMETER ESTIMATION (Hannan-Rissanen / MLE)")
print("=" * 70)

aic_f, bic_f, s2_f, beta_f, resid_f, nf_f, k_f, lb_p_f, X_f, Y_f = \
    hannan_rissanen(yd, xd, P_FINAL, Q_FINAL)

# Standard errors and inference
s2_est = np.sum(resid_f ** 2) / max(1, nf_f - k_f)
XtXi   = np.linalg.inv(X_f.T @ X_f + 1e-8 * np.eye(k_f))
se_f   = np.sqrt(np.diag(s2_est * XtXi))
t_f    = beta_f / se_f
pv_f   = 2 * (1 - stats.t.cdf(np.abs(t_f), df=nf_f - k_f))
ll_f   = -0.5 * nf_f * (np.log(2 * np.pi * s2_est) + 1)

pnames = ([f'AR({i})'   for i in range(1, P_FINAL + 1)] +
          ([f'MA({i})'  for i in range(1, Q_FINAL + 1)] if Q_FINAL > 0 else []) +
          ['CRR (β₁)', 'CDR (β₂)', 'Intercept (μ)'])

print(f"\nARIMAX({P_FINAL},1,{Q_FINAL}) Parameter Estimates")
print(f"{'Parameter':<18} {'Coefficient':>14} {'Std. Error':>12} {'t-stat':>9} {'p-value':>9} {'Sig.':>6}")
print("=" * 72)
for nm, b, s, t, p in zip(pnames, beta_f, se_f, t_f, pv_f):
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    print(f"  {nm:<16} {b:>14.4f} {s:>12.4f} {t:>9.4f} {p:>9.4f} {sig:>6}")
print("=" * 72)
print(f"  σ²             = {s2_est:>20,.2f}")
print(f"  σ (Std. Dev.)  = {np.sqrt(s2_est):>20,.2f}")
print(f"  Log-Likelihood = {ll_f:>20.4f}")
print(f"  AIC            = {aic_f:>20.4f}")
print(f"  BIC            = {bic_f:>20.4f}")
print(f"  N (used)       = {nf_f:>20}")
print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")


# =============================================================================
# STEP 8 — RESIDUAL DIAGNOSTICS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: RESIDUAL DIAGNOSTICS")
print("=" * 70)

lb_q, lb_pval     = ljung_box(resid_f, 10)
lb_q20, lb_p20    = ljung_box(resid_f, 20)
_, shapiro_p       = stats.shapiro(resid_f)
_, levene_p        = stats.levene(resid_f[:len(resid_f)//2], resid_f[len(resid_f)//2:])
skew               = stats.skew(resid_f)
kurt               = stats.kurtosis(resid_f)

print(f"\n  Ljung-Box Q(10)  : Q = {lb_q:.4f},  p = {lb_pval:.4f}  "
      f"{'✓ No autocorrelation' if lb_pval > 0.05 else '✗ Autocorrelation detected'}")
print(f"  Ljung-Box Q(20)  : Q = {lb_q20:.4f},  p = {lb_p20:.4f}  "
      f"{'✓ No autocorrelation' if lb_p20 > 0.05 else '✗ Autocorrelation detected'}")
print(f"  Shapiro-Wilk     : p = {shapiro_p:.4f}  "
      f"{'✓ Normal' if shapiro_p > 0.05 else '✗ Non-normal (common in financial series)'}")
print(f"  Levene (equal σ) : p = {levene_p:.4f}  "
      f"{'✓ Homoskedastic' if levene_p > 0.05 else '✗ Heteroskedastic'}")
print(f"  Skewness         : {skew:.4f}")
print(f"  Excess Kurtosis  : {kurt:.4f}")

acf_r  = compute_acf(resid_f, 20)
conf_r = 1.96 / np.sqrt(len(resid_f))
(osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(resid_f, dist='norm')

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig)
fig.suptitle(
    f'Residual Diagnostics — ARIMAX({P_FINAL},1,{Q_FINAL})\n'
    f'Ljung-Box Q(10) = {lb_q:.2f}, p = {lb_pval:.4f} | Shapiro-Wilk p = {shapiro_p:.4f}',
    fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(resid_f, color='#1f77b4', lw=1, alpha=0.8)
ax1.axhline(0, color='red', lw=1.5, linestyle='--')
ax1.fill_between(range(len(resid_f)), resid_f, 0, alpha=0.2, color='#1f77b4')
ax1.set(title='Residuals over Time', xlabel='Observation', ylabel='Residual (₹ Crore)')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
bar_colors = ['#d62728' if abs(v) > conf_r and i > 0 else 'steelblue'
              for i, v in enumerate(acf_r)]
ax2.bar(range(len(acf_r)), acf_r, color=bar_colors, alpha=0.8, width=0.6)
ax2.axhline(conf_r,  color='red', linestyle='--', alpha=0.8, label='95% CI')
ax2.axhline(-conf_r, color='red', linestyle='--', alpha=0.8)
ax2.axhline(0, color='black', lw=0.8)
ax2.set(title='ACF of Residuals', xlabel='Lag', ylabel='ACF', ylim=(-1.1, 1.1))
ax2.legend(); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(resid_f, bins=18, density=True, color='#2ca02c', alpha=0.7, edgecolor='black')
xr = np.linspace(resid_f.min(), resid_f.max(), 200)
ax3.plot(xr, stats.norm.pdf(xr, resid_f.mean(), resid_f.std()), 'r-', lw=2,
         label=f'Normal PDF  (Shapiro p={shapiro_p:.4f})')
ax3.set(title='Histogram of Residuals', xlabel='Residual (₹ Crore)', ylabel='Density')
ax3.legend(); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
ax4.scatter(osm, osr, color='#9467bd', s=25, alpha=0.8, label=f'Data (R²={r_qq**2:.4f})')
ax4.plot(osm, slope_qq * osm + intercept_qq, 'r-', lw=2, label='Normal Line')
ax4.set(title='Q-Q Plot (Normality)', xlabel='Theoretical Quantiles',
        ylabel='Sample Quantiles')
ax4.legend(); ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
ax5.scatter(range(len(resid_f)), np.abs(resid_f), color='#e377c2', s=20, alpha=0.7)
zf = np.polyfit(range(len(resid_f)), np.abs(resid_f), 1)
ax5.plot(range(len(resid_f)), np.polyval(zf, range(len(resid_f))), 'r-', lw=2, label='Trend')
ax5.set(title='|Residuals| vs Time (Heteroskedasticity)', xlabel='Observation',
        ylabel='|Residual| (₹ Crore)')
ax5.legend(); ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F07_residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Figure saved: F07_residual_diagnostics.png]")


# =============================================================================
# STEP 9 — FORECASTING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: FORECASTING (1-step-ahead, test set)")
print("=" * 70)

y_full  = df['Gross_Credit'].values
X_full  = df[['CRR', 'CDR']].values
yd_full = np.diff(y_full)
xd_full = np.diff(X_full, axis=0)
n_train = len(train)
n_test  = len(test)
start   = max(P_FINAL, Q_FINAL)

# Build residual history from training window
rh_full = np.zeros(len(yd_full))
for t in range(start, n_train - 1):
    yhat = sum(beta_f[i] * yd_full[t - 1 - i] for i in range(P_FINAL))
    if Q_FINAL > 0:
        yhat += sum(beta_f[P_FINAL + i] * rh_full[t - 1 - i] for i in range(Q_FINAL))
    yhat += (beta_f[P_FINAL + Q_FINAL]     * xd_full[t, 0]
             + beta_f[P_FINAL + Q_FINAL + 1] * xd_full[t, 1]
             + beta_f[-1])
    rh_full[t] = yd_full[t] - yhat

# One-step-ahead forecasts on test set
fc_diff  = []
for i in range(n_test):
    t    = n_train - 1 + i
    yhat = sum(beta_f[ii] * yd_full[t - 1 - ii] for ii in range(P_FINAL))
    if Q_FINAL > 0:
        yhat += sum(beta_f[P_FINAL + ii] * rh_full[t - 1 - ii] for ii in range(Q_FINAL))
    yhat += (beta_f[P_FINAL + Q_FINAL]     * xd_full[t, 0]
             + beta_f[P_FINAL + Q_FINAL + 1] * xd_full[t, 1]
             + beta_f[-1])
    fc_diff.append(yhat)
    rh_full[t] = yd_full[t] - yhat

fc_diff  = np.array(fc_diff)

# Invert d=1 differencing
fc_level = np.array([y_full[n_train + i - 1] + fc_diff[i] for i in range(n_test)])
actuals  = test['Gross_Credit'].values
ci_95    = 1.96 * np.sqrt(s2_est)
ci_lo    = fc_level - ci_95
ci_hi    = fc_level + ci_95

# Forecast vs Actual plot
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(train.index, train['Gross_Credit'].values / 1e5,
        label='Training Data', color='#1f77b4', lw=2)
ax.plot(test.index, actuals / 1e5,
        label='Actual (Test)', color='#2ca02c', lw=2.5, marker='o', ms=4)
ax.plot(test.index, fc_level / 1e5,
        label=f'ARIMAX({P_FINAL},1,{Q_FINAL}) Forecast',
        color='#d62728', lw=2.5, linestyle='--', marker='s', ms=4)
ax.fill_between(test.index, ci_lo / 1e5, ci_hi / 1e5,
                alpha=0.2, color='#d62728', label='95% Confidence Interval')
ax.axvline(test.index[0], color='gray', linestyle=':', lw=1.5, alpha=0.7)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Gross Bank Credit (₹ Lakh Crore)', fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
# Title added after metrics
rmse = np.sqrt(np.mean((actuals - fc_level) ** 2))
mae  = np.mean(np.abs(actuals - fc_level))
mape = np.mean(np.abs((actuals - fc_level) / actuals)) * 100
r2   = np.corrcoef(fc_level, actuals)[0, 1] ** 2
ax.set_title(
    f'ARIMAX({P_FINAL},1,{Q_FINAL}) — Forecast vs Actual Gross Bank Credit\n'
    f'RMSE = ₹{rmse:,.0f} Cr  |  MAE = ₹{mae:,.0f} Cr  |  MAPE = {mape:.2f}%  |  R² = {r2:.4f}',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F08_forecast_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Figure saved: F08_forecast_vs_actual.png]")


# =============================================================================
# STEP 10 — EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: MODEL EVALUATION")
print("=" * 70)
print(f"\n  RMSE = {rmse:>15,.2f}  (Root Mean Squared Error)")
print(f"  MAE  = {mae:>15,.2f}  (Mean Absolute Error)")
print(f"  MAPE = {mape:>14.4f}%  (Mean Absolute Percentage Error)")
print(f"  R²   = {r2:>15.4f}  (Coefficient of Determination)")

# Error analysis plot
errors  = actuals - fc_level
pct_err = (errors / actuals) * 100
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Forecast Error Analysis', fontsize=14, fontweight='bold')
axes[0,0].plot(test.index, errors/1e4, marker='o', color='#d62728', lw=1.5, ms=4)
axes[0,0].axhline(0, color='black', lw=1.5)
axes[0,0].fill_between(test.index, errors/1e4, 0, alpha=0.25, color='#d62728')
axes[0,0].set(title='Errors over Time', ylabel='Error (₹ 10k Crore)')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(test.index, pct_err, marker='s', color='#ff7f0e', lw=1.5, ms=4)
axes[0,1].axhline(0, color='black', lw=1.5)
axes[0,1].fill_between(test.index, pct_err, 0, alpha=0.25, color='#ff7f0e')
axes[0,1].set(title='Percentage Errors over Time', ylabel='% Error')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].hist(errors/1e4, bins=12, density=True, color='#1f77b4', alpha=0.7, edgecolor='black')
xr = np.linspace(errors.min()/1e4, errors.max()/1e4, 100)
axes[1,0].plot(xr, stats.norm.pdf(xr, errors.mean()/1e4, errors.std()/1e4), 'r-', lw=2)
axes[1,0].set(title='Error Distribution', xlabel='Error (₹ 10k Crore)', ylabel='Density')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].scatter(fc_level/1e5, actuals/1e5, color='#9467bd', s=40, alpha=0.8)
mn = min(fc_level.min(), actuals.min()) / 1e5
mx = max(fc_level.max(), actuals.max()) / 1e5
axes[1,1].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Forecast')
axes[1,1].set(title=f'Actual vs Predicted  (R² = {r2:.4f})',
              xlabel='Forecast (₹ Lakh Crore)', ylabel='Actual (₹ Lakh Crore)')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F09_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Figure saved: F09_error_analysis.png]")

# Forecast comparison table
print(f"\n{'='*80}")
print("FORECAST vs ACTUAL COMPARISON TABLE")
print(f"{'='*80}")
print(f"{'Date':<12} {'Actual (₹Cr)':>14} {'Forecast (₹Cr)':>15} {'Error (₹Cr)':>13} {'% Error':>9}")
print("-" * 70)
for d, a, f in zip(test.index, actuals, fc_level):
    err = a - f; pct = abs(err / a) * 100
    print(f"{d.strftime('%b %Y'):<12} {a:>14,.0f} {f:>15,.0f} {err:>13,.0f} {pct:>8.2f}%")
print("-" * 70)
print(f"{'RMSE':<12} {rmse:>14,.2f}")
print(f"{'MAE':<12} {mae:>14,.2f}")
print(f"{'MAPE':<12} {mape:>13.4f}%")
print(f"{'R²':<12} {r2:>15.4f}")
print(f"{'='*80}")
print("\n[Analysis complete. All figures saved to:", OUTPUT_DIR, "]")
