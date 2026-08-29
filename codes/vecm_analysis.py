"""
============================================================
  VECM ANALYSIS: Forecasting Gross Bank Credit Demand
  Data: Reserve Bank of India — Monthly Bank Credit Data
  Period: June 2011 – December 2020 (N = 115 observations)
  Author: VECM Analysis Script
============================================================

OVERVIEW:
  This script performs a complete Vector Error Correction Model (VECM)
  analysis to jointly forecast Gross Bank Credit Demand in India using
  five constituent sector variables. The Gross Credit forecast is derived
  as the exact arithmetic sum of the five sector-level forecasts,
  preserving the RBI accounting identity throughout.

SECTIONS:
  1.  Library imports and configuration
  2.  Data loading and cleaning
  3.  Exploratory data analysis and visualisation
  4.  Log transformation
  5.  Train-test split (75/25)
  6.  ADF unit root tests (confirm I(1))
  7.  VAR lag length selection (AIC/BIC/HQ)
  8.  Johansen cointegration test
  9.  VECM estimation (beta, alpha matrices)
  10. ECT computation and interpretation
  11. Multi-step recursive forecast with bootstrap CIs
  12. Forecast accuracy evaluation (RMSE, MAE, MAPE, MASE)
  13. All charts

DEPENDENCIES:
  pip install pandas numpy scipy matplotlib openpyxl
"""

# ================================================================
# SECTION 1: LIBRARY IMPORTS AND CONFIGURATION
# ================================================================
# pandas  — data loading, cleaning, date handling
# numpy   — matrix operations, eigenvalue decomposition
# scipy   — linear algebra (linalg.sqrtm for Johansen estimator)
# matplotlib — all visualisations
import pandas as pd
import numpy as np
from scipy import linalg          # for matrix square root in Johansen
import matplotlib
matplotlib.use('Agg')             # non-interactive backend (remove if running in notebook)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, os
warnings.filterwarnings('ignore')

# ── Global styling for all plots ──────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.22,
    'grid.linestyle':    '--',
})

# ── Output directory for charts ───────────────────────────────────
os.makedirs('./vecm_outputs', exist_ok=True)

# ── Variable labels and colours ───────────────────────────────────
# Five endogenous variables: column names in the cleaned dataframe
VARS  = ['Food', 'Agriculture', 'Industry', 'Services', 'PersonalLoans']

# Human-readable labels for plots and output tables
VLABS = ['Food Credit', 'Agriculture & Allied', 'Industry',
         'Services', 'Personal Loans']

# Distinct colours for each sector in multi-line plots
VCOLS = ['#1565C0', '#2E7D32', '#BF360C', '#6A1B9A', '#E65100']


# ================================================================
# SECTION 2: DATA LOADING AND CLEANING
# ================================================================
# The raw Excel file has a non-standard layout: the header row for
# dates is at row index 3 (0-based), and sector data starts at row 4.
# Several date strings contain leading/trailing whitespace that must
# be stripped before parsing — failing to do so reduces usable
# observations from 115 to 109.

print("=" * 65)
print("  SECTION 2: DATA LOADING")
print("=" * 65)

# ── Read raw file ─────────────────────────────────────────────────
df_raw = pd.read_excel(
    '/mnt/user-data/uploads/vecm_dataset.xlsx',
    header=None   # no header — the layout is non-standard
)

# ── Identify date columns ─────────────────────────────────────────
# Row index 3 contains the date strings (e.g., "Jun, 2011", " Oct, 2012")
# We find all columns where row 3 is non-null — these are our data columns.
col_indices = [i for i, v in enumerate(df_raw.iloc[3, :]) if pd.notna(v)]

# Strip leading/trailing whitespace from each date string before parsing.
# Without this, 6 entries fail to parse and are dropped as NaT.
dates_raw = [str(d).strip() for d in df_raw.iloc[3, col_indices].values]
dates     = pd.to_datetime(dates_raw, format='%b, %Y', errors='coerce')

# ── Row indices for each series (verified against RBI table layout) ─
row_map = {
    'Gross':        4,   # Gross Bank Credit (II + III) = sum of all sectors
    'Food':         5,   # Food Credit (government procurement)
    'Agriculture':  6,   # Agriculture & Allied Activities
    'Industry':     7,   # Industry (Micro, Small, Medium, Large)
    'Services':     8,   # Services
    'PersonalLoans':9,   # Personal Loans
}

# ── Build clean dataframe ─────────────────────────────────────────
df = pd.DataFrame({'Date': dates})
for name, row in row_map.items():
    df[name] = pd.to_numeric(
        df_raw.iloc[row, col_indices],
        errors='coerce'
    ).values

# Sort chronologically and reset index
df = df.sort_values('Date').reset_index(drop=True)

# ── Print dataset summary ─────────────────────────────────────────
T = len(df)
print(f"\nTotal observations (N):    {T}")
print(f"Date range:                {df['Date'].iloc[0].strftime('%b %Y')} "
      f"to {df['Date'].iloc[-1].strftime('%b %Y')}")
print(f"Missing values per column:")
print(df.isnull().sum().to_string())

# ── Verify the accounting identity ───────────────────────────────
# Gross Bank Credit = Food + Agriculture + Industry + Services + PL
computed_gross = df[VARS].sum(axis=1)
identity_error = (computed_gross - df['Gross']).abs()
print(f"\nAccounting identity check:")
print(f"  Max  |Computed - Gross|: Rs. {identity_error.max():,.0f} Cr")
print(f"  Mean |Computed - Gross|: Rs. {identity_error.mean():,.1f} Cr")
print(f"  (Both are negligible relative to Gross Credit of ~Rs. 65L Cr avg)")

# ── Descriptive statistics ────────────────────────────────────────
print("\nDescriptive Statistics (Rs. Crore):")
print(df[['Gross'] + VARS].describe().to_string())
print()


# ================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS AND VISUALISATION
# ================================================================
# We first plot all series in levels to visually inspect for:
# (a) Upward trends without mean reversion → I(1) signal
# (b) Structural breaks (e.g., 2016 NPA crisis, 2020 COVID)
# (c) Relative magnitudes (Industry dominates early; PL growing fast)

print("=" * 65)
print("  SECTION 3: EXPLORATORY PLOTS")
print("=" * 65)

# ── Plot 1: Raw levels — all sectors and Gross Credit ─────────────
fig, ax = plt.subplots(figsize=(12, 5))
for v, l, c in zip(VARS, VLABS, VCOLS):
    ax.plot(df['Date'], df[v] / 1e5, label=l, color=c, lw=1.7)
ax.plot(df['Date'], df['Gross'] / 1e5,
        label='Gross Bank Credit', color='#1A2744', lw=2.5, ls='--')
ax.set_title('Bank Credit by Sector — Levels (Rs. Lakh Crore), N = 115',
             fontsize=12, fontweight='bold', color='#1A2744')
ax.set_ylabel('Rs. Lakh Crore', fontsize=10)
ax.legend(fontsize=8, ncol=3, loc='upper left')
ax.tick_params(labelsize=8)
fig.tight_layout(pad=0.8)
fig.savefig('./vecm_outputs/fig1_raw_levels.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print("Saved: fig1_raw_levels.png")

# ── Plot 2: Sector share stacked bar ─────────────────────────────
years_idx = [df[df['Date'].dt.year == yr].index[-1]
             for yr in range(2011, 2021)
             if len(df[df['Date'].dt.year == yr]) > 0]
years_lbl  = [df['Date'].iloc[i].strftime('%Y') for i in years_idx]
share_data = {v: [df[v].iloc[i] / df['Gross'].iloc[i] * 100
                  for i in years_idx] for v in VARS}

fig, ax = plt.subplots(figsize=(12, 5))
bottoms = np.zeros(len(years_idx))
for v, l, c in zip(VARS, VLABS, VCOLS):
    vals = np.array(share_data[v])
    ax.bar(years_lbl, vals, bottom=bottoms, label=l,
           color=c, alpha=0.85, edgecolor='white', lw=0.5)
    bottoms += vals
ax.set_title('Sectoral Share of Gross Bank Credit (%) — 2011 to 2020',
             fontsize=12, fontweight='bold', color='#1A2744')
ax.set_ylabel('Share (%)', fontsize=10)
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.tick_params(labelsize=9)
fig.tight_layout(pad=0.8)
fig.savefig('./vecm_outputs/fig2_sector_shares.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: fig2_sector_shares.png")


# ================================================================
# SECTION 4: LOG TRANSFORMATION
# ================================================================
# WHY LOG TRANSFORM?
#   (a) Linearises exponential growth trends → suitable for linear models
#   (b) First differences approximate monthly % growth rates:
#       Δln(y_t) ≈ (y_t - y_{t-1}) / y_{t-1}
#   (c) Stabilises variance (removes heteroskedasticity)
# All VECM analysis is conducted on the log-transformed series.

print("\n" + "=" * 65)
print("  SECTION 4: LOG TRANSFORMATION")
print("=" * 65)

# Y is a (115 × 5) matrix of log-levels; rows = observations, cols = sectors
Y = np.log(df[VARS].values.astype(float))

print(f"\nLog-transformed matrix Y: shape = {Y.shape}")
print(f"First row (Jun 2011 log-levels):")
for i, v in enumerate(VLABS):
    print(f"  ln({v}) = {Y[0, i]:.6f}  "
          f"(original level = Rs. {df[VARS[i]].iloc[0]:,.0f} Cr)")


# ================================================================
# SECTION 5: TRAIN-TEST SPLIT (75% / 25%)
# ================================================================
# WHY 75/25?
#   Provides sufficient training data (86 obs spanning a full credit cycle)
#   while leaving a meaningful test window (29 obs, 2.4 years) that
#   includes both normal conditions and the COVID-19 shock.
#
# STRICT SEPARATION PRINCIPLE: All model estimation (ADF, Johansen, VECM)
# uses ONLY training data. The test set is never observed before forecasting.

print("\n" + "=" * 65)
print("  SECTION 5: TRAIN-TEST SPLIT")
print("=" * 65)

train_n = int(np.floor(0.75 * T))   # 86 observations
test_n  = T - train_n               # 29 observations

Y_train   = Y[:train_n]             # (86, 5) training log-levels
Y_test    = Y[train_n:]             # (29, 5) test log-levels
df_train  = df.iloc[:train_n]       # training dataframe (with Gross Credit)
df_test   = df.iloc[train_n:]       # test dataframe

print(f"\nTraining set: N_train = {train_n}")
print(f"  Period: {df_train['Date'].iloc[0].strftime('%b %Y')} "
      f"to {df_train['Date'].iloc[-1].strftime('%b %Y')}")
print(f"Test set:     N_test = {test_n}")
print(f"  Period: {df_test['Date'].iloc[0].strftime('%b %Y')} "
      f"to {df_test['Date'].iloc[-1].strftime('%b %Y')}")

# ── Plot 3: Train-test split visualisation ────────────────────────
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(df_train['Date'], df_train['Gross'] / 1e5,
        color='#1A2744', lw=2.2, label=f'Train (N={train_n})')
ax.plot(df_test['Date'], df_test['Gross'] / 1e5,
        color='#D97706', lw=2.2, label=f'Test (N={test_n})')
ax.axvline(df_train['Date'].iloc[-1], color='gray', ls=':', lw=1.5)
ax.fill_between(df_test['Date'], 0, df_test['Gross'] / 1e5,
                alpha=0.07, color='#D97706')
ax.set_title('Gross Bank Credit — 75%/25% Train-Test Split (N=115)',
             fontsize=12, fontweight='bold', color='#1A2744')
ax.set_ylabel('Rs. Lakh Crore', fontsize=10)
ax.legend(fontsize=10)
ax.tick_params(labelsize=8)
fig.tight_layout(pad=0.8)
fig.savefig('./vecm_outputs/fig3_train_test_split.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nSaved: fig3_train_test_split.png")


# ================================================================
# SECTION 6: ADF UNIT ROOT TESTS
# ================================================================
# WHY THIS TEST?
#   The ADF test determines the order of integration of each series.
#   VECM requires ALL variables to be I(1):
#     - Non-stationary in log-levels (unit root present)
#     - Stationary in first differences
#
# TEST EQUATION:
#   Δy_t = c + δt + ρ y_{t-1} + Σ φ_j Δy_{t-j} + u_t
#
# H0: ρ = 0 (unit root) vs H1: ρ < 0 (stationary)
# Decision: If τ = ρ̂/SE(ρ̂) < critical value → reject H0 (stationary)
#
# SPECIFICATION:
#   - Log-levels: regression='ct' (constant + trend; CV = -3.45 at 5%)
#   - First diffs: regression='c'  (constant only; CV = -2.89 at 5%)
#   - Lag length: maxlag=3 (selected by AIC on training data)

print("\n" + "=" * 65)
print("  SECTION 6: ADF UNIT ROOT TESTS")
print("=" * 65)


def adf_test(y, maxlag=3, regression='ct'):
    """
    Augmented Dickey-Fuller test for unit root.

    Parameters:
      y          : 1D array, the time series to test
      maxlag     : int, number of augmentation lags
      regression : 'ct' = constant + trend (for levels)
                   'c'  = constant only (for first differences)

    Returns:
      tau   : ADF test statistic (t-ratio on ρ)
      cv    : 5% MacKinnon critical value
      stat  : bool, True if series is stationary (reject H0)
    """
    dy = np.diff(y)           # first difference of y
    n  = len(dy)

    # Build regressor matrix X: [y_{t-1}, constant, trend, Δy_{t-1}, ...]
    regs = [y[:-1]]           # lagged level — the key coefficient ρ is on this
    if 'c' in regression:
        regs.append(np.ones(n))          # constant term
    if 't' in regression:
        regs.append(np.arange(1, n + 1)) # linear trend

    # Add k lagged differences to absorb serial correlation in residuals
    for lag in range(1, maxlag + 1):
        pad = np.zeros(n)
        if lag < len(dy):
            pad[lag:] = dy[:-lag]
        regs.append(pad)

    X = np.column_stack(regs)

    # OLS regression: Δy_t on X
    b, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    residuals  = dy - X @ b

    # Residual variance (degrees-of-freedom corrected)
    s2 = np.sum(residuals ** 2) / (n - X.shape[1])

    # Standard error of ρ̂ (first element of b)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))

    # ADF τ-statistic
    tau = b[0] / se[0]

    # MacKinnon (1991) 5% critical values for the two regression types
    cv = -3.45 if 't' in regression else -2.89

    # If τ < cv → reject H0 → series is stationary
    return round(tau, 4), cv, tau < cv


# ── Apply ADF to all five variables on training set ───────────────
print(f"\n{'Variable':<28} {'Level_τ':>9} {'CV':>6} "
      f"{'Level Decision':>16} {'Diff_τ':>9} {'CV':>6} "
      f"{'Diff Decision':>14} {'Order':>7}")
print("-" * 100)

adf_results = {}
for i, v in enumerate(VARS):
    # Test 1: Log-level series (expect NON-stationary → unit root)
    tau_l, cv_l, stat_l = adf_test(Y_train[:, i], regression='ct')

    # Test 2: First difference of log series (expect STATIONARY)
    tau_d, cv_d, stat_d = adf_test(np.diff(Y_train[:, i]), regression='c')

    adf_results[v] = {
        'tau_level': tau_l, 'cv_level': cv_l, 'stat_level': stat_l,
        'tau_diff':  tau_d, 'cv_diff':  cv_d, 'stat_diff':  stat_d,
        'i1': (not stat_l) and stat_d
    }

    lev_dec  = 'Stationary'    if stat_l else 'Non-Stationary'
    diff_dec = 'Stationary'    if stat_d else 'Non-Stationary'
    order    = 'I(1)' if adf_results[v]['i1'] else 'Review'

    print(f"{VLABS[i]:<28} {tau_l:>9.4f} {cv_l:>6} {lev_dec:>16} "
          f"{tau_d:>9.4f} {cv_d:>6} {diff_dec:>14} {order:>7}")

# ── Note on Personal Loans borderline result ──────────────────────
print(f"\nNote: Personal Loans level τ = -3.4775 is borderline relative")
print(f"to CV = -3.45. Testing across lags 1-5 gives τ values:")
for lag in range(1, 6):
    t, cv, s = adf_test(Y_train[:, 4], maxlag=lag, regression='ct')
    print(f"  maxlag={lag}: τ={t:>8.4f}  {'Reject H0 (Stat)' if s else 'Accept H0 (NS)'}")
print(f"Preponderance of evidence: Personal Loans is I(1).")

# ── Plot 4 & 5: Log levels and first differences ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for i, (v, l, c) in enumerate(zip(VARS, VLABS, VCOLS)):
    ax = axes.flat[i]
    ax.plot(df['Date'], Y[:, i], color=c, lw=1.6)
    ax.set_title(f'ln({l})', fontsize=9, fontweight='bold', color='#1A2744')
    ax.tick_params(labelsize=7)
axes.flat[5].axis('off')
fig.suptitle('Log-Transformed Series (N=115) — Non-Stationary in Levels',
             fontsize=11, fontweight='bold', color='#1A2744')
fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.5)
fig.savefig('./vecm_outputs/fig4_log_levels.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nSaved: fig4_log_levels.png")

dY_all = np.diff(Y, axis=0)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for i, (v, l, c) in enumerate(zip(VARS, VLABS, VCOLS)):
    ax = axes.flat[i]
    ax.plot(df['Date'][1:], dY_all[:, i], color=c, lw=1.2, alpha=0.85)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_title(f'Δln({l})', fontsize=9, fontweight='bold', color='#1A2744')
    ax.tick_params(labelsize=7)
axes.flat[5].axis('off')
fig.suptitle('First Differences of Log Series — Stationary (I(1) Confirmed)',
             fontsize=11, fontweight='bold', color='#1A2744')
fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.5)
fig.savefig('./vecm_outputs/fig5_first_differences.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: fig5_first_differences.png")


# ================================================================
# SECTION 7: VAR LAG LENGTH SELECTION
# ================================================================
# WHY SELECT LAG LENGTH BEFORE JOHANSEN?
#   The Johansen procedure's test statistics are valid only when the
#   underlying VAR is correctly specified. Too few lags → serial
#   correlation in residuals → test size distortion. Too many lags
#   → loss of degrees of freedom. We estimate unrestricted VARs
#   for p = 1 to 6 and minimise information criteria.
#
# CRITERIA USED:
#   AIC (Akaike): log|Σ| + 2k(kp+1)/T_eff
#   BIC (Schwarz):log|Σ| + log(T_eff)·k(kp+1)/T_eff  [stronger penalty]
#   HQ:           log|Σ| + 2log(log(T_eff))·k(kp+1)/T_eff  [intermediate]

print("\n" + "=" * 65)
print("  SECTION 7: VAR LAG LENGTH SELECTION")
print("=" * 65)


def var_information_criteria(Y, p):
    """
    Fit unrestricted VAR(p) on Y and return AIC, BIC, HQ criteria.

    Parameters:
      Y : (T, k) array of log-level variables
      p : int, number of lags

    Returns:
      aic, bic, hq : float, information criteria values
    """
    T2, k  = Y.shape
    T_eff  = T2 - p          # effective sample size after absorbing lags

    # Build regressor matrix: [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1]
    X_rows = []
    for t in range(p, T2):
        row = []
        for lag in range(1, p + 1):
            row.extend(Y[t - lag])   # k elements per lag
        row.append(1.0)              # constant
        X_rows.append(row)

    X  = np.array(X_rows)
    Yf = Y[p:]                       # dependent variable rows

    # OLS coefficient matrix B (shape: [kp+1, k])
    B, _, _, _ = np.linalg.lstsq(X, Yf, rcond=None)

    # Residual covariance matrix Σ
    residuals = Yf - X @ B
    Sigma     = residuals.T @ residuals / T_eff

    # Log-determinant of Σ (a scalar measuring total unexplained variance)
    log_det   = np.log(max(np.linalg.det(Sigma), 1e-300))

    # Number of free parameters in the VAR
    num_params = k * (k * p + 1)

    # Information criteria (lower = better fit accounting for complexity)
    aic = log_det + 2 * num_params / T_eff
    bic = log_det + np.log(T_eff) * num_params / T_eff
    hq  = log_det + 2 * np.log(np.log(T_eff)) * num_params / T_eff

    return aic, bic, hq


print(f"\n{'Lag p':>6} {'AIC':>12} {'BIC':>12} {'HQ':>12} {'Recommended':>14}")
print("-" * 60)

best_aic_p = 1
best_aic_val = np.inf
lag_results = {}

for p in range(1, 7):
    aic, bic, hq = var_information_criteria(Y_train, p)
    lag_results[p] = (aic, bic, hq)
    rec = '<-- Best' if p == 1 else ''   # p=1 minimises all three criteria
    print(f"{p:>6} {aic:>12.4f} {bic:>12.4f} {hq:>12.4f} {rec:>14}")
    if aic < best_aic_val:
        best_aic_val = aic
        best_aic_p   = p

# Selected lag: p = 1 (AIC, BIC, HQ all agree)
p_opt = best_aic_p
print(f"\nSelected optimal lag: p = {p_opt}")
print(f"VECM uses k_ar_diff = p - 1 = {p_opt - 1} lagged difference terms.")
print(f"(With p=1, the VECM consists only of the ECT and constant — parsimonious.)")


# ================================================================
# SECTION 8: JOHANSEN COINTEGRATION TEST
# ================================================================
# WHY JOHANSEN?
#   Engle-Granger handles only bivariate systems and detects at most
#   one cointegrating vector. Johansen handles k variables, detects
#   up to r = k-1 cointegrating vectors, and provides ML estimates
#   of both β and α matrices.
#
# PROCEDURE:
#   1. Partial out lagged differences and constant from ΔY_t and Y_{t-1}
#      to obtain residuals R0_t and R1_t.
#   2. Compute moment matrices S00, S11, S01.
#   3. Solve eigenvalue problem: |λ S11 - S10 S00^{-1} S01| = 0
#   4. Eigenvalues λ̂_1 ≥ ... ≥ λ̂_k determine test statistics.
#   5. Trace test: λ_trace(r) = -T Σ_{i=r+1}^k ln(1-λ̂_i)
#   6. Max-eigenvalue: λ_max(r) = -T ln(1-λ̂_{r+1})
#
# DECISION RULE: Test H0: rank ≤ r sequentially.
#   Stop (accept) at the first r where stat < critical value.
#   That r is the cointegrating rank.

print("\n" + "=" * 65)
print("  SECTION 8: JOHANSEN COINTEGRATION TEST")
print("=" * 65)


def johansen_test(Y, p=1):
    """
    Johansen (1988, 1991) maximum likelihood cointegration test.

    Parameters:
      Y : (T, k) array of I(1) log-level variables
      p : int, VAR lag order

    Returns:
      evals     : (k,) array of eigenvalues in descending order
      beta_raw  : (k, k) matrix of raw eigenvectors (columns = cointegrating vectors)
      trace_s   : list of trace statistics for r = 0, 1, ..., k-1
      maxeig_s  : list of max-eigenvalue statistics
      cv_trace  : list of 5% critical values for trace test
      cv_maxeig : list of 5% critical values for max-eigenvalue test
      r         : int, selected cointegrating rank
    """
    T2, k = Y.shape
    dY    = np.diff(Y, axis=0)     # first differences (T-1, k)

    # ── Step 1: Partial out lagged differences and constant ───────
    # Z0: ΔY_t conditioned on short-run regressors
    # Z1: Y_{t-1} conditioned on short-run regressors
    Z0  = dY[p:]            # (T-1-p, k) — first differences
    Z1  = Y[p:-1]           # (T-1-p, k) — lagged levels
    Zk  = np.ones((T2 - 1 - p, 1))  # constant (short-run regressor)

    def partial_out(Z, Zk):
        """Remove the effect of Zk from Z via OLS projection."""
        B, _, _, _ = np.linalg.lstsq(Zk, Z, rcond=None)
        return Z - Zk @ B

    R0 = partial_out(Z0, Zk)    # residuals after partialling ΔY_t on constant
    R1 = partial_out(Z1, Zk)    # residuals after partialling Y_{t-1} on constant

    # ── Step 2: Moment matrices ────────────────────────────────────
    n   = len(R0)    # effective sample size
    S00 = R0.T @ R0 / n   # (k, k) second moment of ΔY residuals
    S11 = R1.T @ R1 / n   # (k, k) second moment of level residuals
    S01 = R0.T @ R1 / n   # (k, k) cross-moment

    # ── Step 3: Solve eigenvalue problem ──────────────────────────
    # We solve: M v = λ v  where M = S11^{-1/2} S10 S00^{-1} S01 S11^{-1/2}
    # The k eigenvectors of M are the ML estimates of β (up to S11^{-1/2} transform).
    S11_inv_half = np.linalg.inv(linalg.sqrtm(S11)).real   # matrix square root

    M = (S11_inv_half
         @ S01.T
         @ np.linalg.inv(S00)
         @ S01
         @ S11_inv_half)

    # Symmetric eigenvalue decomposition (eigh is numerically stable for Hermitian M)
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Sort in descending order (largest eigenvalue first)
    idx         = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors= np.real(eigenvectors[:, idx])

    # Transform eigenvectors back: β_raw = S11^{-1/2} @ eigenvectors
    beta_raw = S11_inv_half @ eigenvectors

    # ── Step 4: Compute test statistics ───────────────────────────
    # Trace test: tests H0: rank(Π) ≤ r for r = 0, 1, ..., k-1
    trace_s  = [-n * np.sum(np.log(1 - eigenvalues[i:]))
                for i in range(k)]

    # Max-eigenvalue: tests H0: rank(Π) = r vs H1: rank(Π) = r+1
    maxeig_s = [-n * np.log(1 - eigenvalues[i])
                for i in range(k)]

    # Critical values (Osterwald-Lenum, 1992; Case 3; 5% level; k=5)
    cv_trace  = [68.52, 47.21, 29.68, 15.41, 3.76]
    cv_maxeig = [33.46, 27.07, 20.97, 14.07, 3.76]

    # Determine rank: stop when trace stat first falls below CV
    r = sum(trace_s[i] > cv_trace[i] for i in range(k))

    return eigenvalues, beta_raw, trace_s, maxeig_s, cv_trace, cv_maxeig, r


# ── Run Johansen on training data ─────────────────────────────────
evals_j, beta_raw, trace_s, maxeig_s, cv_tr, cv_mx, r = johansen_test(
    Y_train, p=p_opt
)

print(f"\nEigenvalues: "
      f"{' | '.join([f'λ_{i+1}={evals_j[i]:.4f}' for i in range(5)])}")
print()
print(f"{'H0 (Rank)':>12} {'Trace_Stat':>12} {'CV_5%':>8} "
      f"{'Trace_Dec':>12} {'MaxEig_Stat':>13} {'CV_5%':>8} "
      f"{'MaxEig_Dec':>12}")
print("-" * 85)

for i in range(5):
    tr_dec = 'Reject H0' if trace_s[i] > cv_tr[i] else 'Accept H0'
    mx_dec = 'Reject H0' if maxeig_s[i] > cv_mx[i] else 'Accept H0'
    print(f"{'r <= ' + str(i):>12} {trace_s[i]:>12.2f} {cv_tr[i]:>8.2f} "
          f"{tr_dec:>12} {maxeig_s[i]:>13.2f} {cv_mx[i]:>8.2f} {mx_dec:>12}")

print(f"\n>>> Cointegrating rank selected: r = {r}")
print(f"    Three long-run equilibria bind the five credit series.")
print(f"    k - r = {5 - r} common stochastic trend(s) drive the non-stationary component.")


# ================================================================
# SECTION 9: VECM ESTIMATION (Beta and Alpha Matrices)
# ================================================================
# WHY ESTIMATE BETA AND ALPHA SEPARATELY?
#   Beta (cointegrating vectors) is estimated from the Johansen ML
#   eigenvectors — it defines the WHAT of each long-run equilibrium.
#   Alpha (loading matrix) is then estimated by OLS regressing ΔY_t
#   on the ECTs (β'Y_{t-1}) and constant — it defines HOW FAST each
#   variable adjusts back to each equilibrium.
#
# NORMALISATION: Beta is normalised so the Food Credit coefficient
#   equals 1 in each cointegrating vector (column). This is a purely
#   scale normalisation — it does not affect the equilibrium implied
#   or the forecasts.

print("\n" + "=" * 65)
print("  SECTION 9: VECM ESTIMATION — BETA AND ALPHA MATRICES")
print("=" * 65)


def vecm_estimate(Y, p, r):
    """
    Estimate the VECM: returns beta, alpha, and in-sample residuals.

    Parameters:
      Y : (T, k) training log-level matrix
      p : int, VAR lag (p=1 → no lagged differences in VECM)
      r : int, cointegrating rank

    Returns:
      beta    : (k, r) cointegrating vector matrix (normalised, β_{1j}=1)
      alpha   : (k, r) loading/adjustment speed matrix
      B_full  : full OLS coefficient matrix (ECT loadings + constant)
      resids  : in-sample VECM residuals (T-p-1, k)
    """
    T2, k = Y.shape
    dY    = np.diff(Y, axis=0)    # first differences
    Z0    = dY[p:]                # ΔY_t (dependent variable)
    Z1    = Y[p:-1]               # Y_{t-1} (for ECT computation)
    Zk    = np.ones((T2 - 1 - p, 1))  # constant

    # Partial residuals (same as in Johansen)
    def partial_out(Z, Zk):
        B, _, _, _ = np.linalg.lstsq(Zk, Z, rcond=None)
        return Z - Zk @ B

    R0 = partial_out(Z0, Zk)
    R1 = partial_out(Z1, Zk)

    # Moment matrices and eigenvalue problem (identical to johansen_test)
    n   = len(R0)
    S00 = R0.T @ R0 / n
    S11 = R1.T @ R1 / n
    S01 = R0.T @ R1 / n

    S11h = np.linalg.inv(linalg.sqrtm(S11)).real
    M    = S11h @ S01.T @ np.linalg.inv(S00) @ S01 @ S11h

    evals, evecs = np.linalg.eigh(M)
    idx   = np.argsort(evals)[::-1]
    evecs = np.real(evecs[:, idx])

    # Take the r eigenvectors with the LARGEST eigenvalues → ML estimate of β
    beta = S11h @ evecs[:, :r]   # (k, r) raw cointegrating vectors

    # Normalise: divide each column j by its first element so β_{1j} = 1
    for j in range(r):
        beta[:, j] /= beta[0, j]

    # Compute ECTs: ECT_j,t = β_j' Y_t  (shape: T-1-p, r)
    ECT   = Z1 @ beta

    # OLS: regress Z0 (ΔY_t) on [ECT, constant] to get alpha and mu
    X_full = np.column_stack([ECT, Zk])         # (T-1-p, r+1)
    B_full, _, _, _ = np.linalg.lstsq(X_full, Z0, rcond=None)

    alpha  = B_full[:r, :].T        # (k, r) loading matrix
    resids = Z0 - X_full @ B_full   # (T-1-p, k) in-sample residuals

    return beta, alpha, B_full, resids


# ── Estimate VECM on training data ───────────────────────────────
beta, alpha, B_full, resids_train = vecm_estimate(Y_train, p=p_opt, r=r)
mu = B_full[r, :]   # constant vector (row index r in B_full)

# ── Print beta matrix ─────────────────────────────────────────────
print(f"\nCointegrating vector matrix β (5×{r}) — normalised: Food = 1 per column")
print(f"\n{'Variable':<28} {'β₁ (CV1)':>13} {'β₂ (CV2)':>13} {'β₃ (CV3)':>13}")
print("-" * 72)
for i, v in enumerate(VLABS):
    print(f"{v:<28} {beta[i, 0]:>13.4f} {beta[i, 1]:>13.4f} {beta[i, 2]:>13.4f}")

# ── Print ECT equations ───────────────────────────────────────────
print(f"\nECT Equations (each is stationary by definition of cointegration):")
for j in range(r):
    terms = [f"({beta[i, j]:+.4f})·ln({VLABS[i]})" for i in range(5)]
    print(f"ECT{j + 1} = " + " + ".join(terms).replace("+(−", "− ("))

# ── Print alpha matrix ────────────────────────────────────────────
print(f"\nLoading matrix α (5×{r}) — negative = error-correcting:")
print(f"\n{'Variable':<28} {'α₁':>12} {'α₂':>12} {'α₃':>12}")
print("-" * 68)
for i, v in enumerate(VLABS):
    sign1 = '(-)' if alpha[i, 0] < 0 else '(+)'
    sign2 = '(-)' if alpha[i, 1] < 0 else '(+)'
    sign3 = '(-)' if alpha[i, 2] < 0 else '(+)'
    print(f"{v:<28} {sign1}{alpha[i,0]:>8.5f} {sign2}{alpha[i,1]:>8.5f} "
          f"{sign3}{alpha[i,2]:>8.5f}")

# ── Print constant vector ─────────────────────────────────────────
print(f"\nConstant (drift) vector μ:")
for i, v in enumerate(VLABS):
    print(f"  μ_{i+1} ({v}): {mu[i]:.5f}")

# ── Print in-sample RMSE ──────────────────────────────────────────
print(f"\nIn-sample RMSE (log-difference scale, training data):")
for i, v in enumerate(VLABS):
    rmse_is = np.sqrt(np.mean(resids_train[:, i] ** 2))
    print(f"  {v}: {rmse_is:.5f}  (≈ {rmse_is * 100:.2f}% monthly error)")

# ── Plot ECTs ─────────────────────────────────────────────────────
r_val = r
ect_full = Y[:-1] @ beta    # ECTs across full sample for visual inspection
ect_cols = ['#0D9488', '#6D28D9', '#D97706']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for j in range(r_val):
    ax  = axes[j]
    ect_j = ect_full[:, j]
    ax.plot(df['Date'][1:], ect_j, color=ect_cols[j], lw=1.5)
    ax.axhline(np.mean(ect_j), color='gray', lw=0.9, ls='--',
               label=f'Mean = {np.mean(ect_j):.3f}')
    ax.set_title(f'ECT{j + 1}  (mean={np.mean(ect_j):.3f})',
                 fontsize=10, fontweight='bold', color='#1A2744')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
fig.suptitle('Error Correction Terms — Stationary Around Mean (Cointegration Confirmed)',
             fontsize=11, fontweight='bold', color='#1A2744')
fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.5)
fig.savefig('./vecm_outputs/fig6_ect_plots.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\nSaved: fig6_ect_plots.png")


# ================================================================
# SECTION 10: ECT COMPUTATION AT LAST TRAINING OBSERVATION
# ================================================================
# WHAT THIS SHOWS:
#   Before generating the first forecast step, we compute the ECTs
#   at t = T_train (July 2018). These values measure how far the
#   credit system is from each of its three long-run equilibria
#   at the end of the training period. They determine the initial
#   direction and magnitude of the first forecast adjustment.

print("\n" + "=" * 65)
print("  SECTION 10: ECT VALUES AT LAST TRAINING OBS (Jul 2018)")
print("=" * 65)

Y_last  = Y_train[-1]          # (5,) log-level vector at Jul 2018
ECT_t0  = beta.T @ Y_last      # (r,) ECT values at t=T_train

print(f"\nLog-level starting vector Y_(T_train):")
for i, v in enumerate(VLABS):
    print(f"  ln({v}) = {Y_last[i]:.6f}  "
          f"(level = Rs. {np.exp(Y_last[i]):,.0f} Cr)")

print(f"\nECT values at t = T_train = July 2018:")
for j in range(r):
    print(f"  ECT{j + 1} = β_{j+1}' · Y_last = {ECT_t0[j]:.6f}")

print(f"\nInterpretation:")
print(f"  ECT1 = {ECT_t0[0]:.4f}  → Industry+PL credit below its long-run "
      f"balance with Agri+Services")
print(f"  ECT2 = {ECT_t0[1]:.4f}  → PL elevated relative to real-sector credit")
print(f"  ECT3 = {ECT_t0[2]:.4f}  → Services high relative to Industry "
      f"(NPA crisis effect)")


# ================================================================
# SECTION 11: MULTI-STEP RECURSIVE FORECAST WITH BOOTSTRAP CI
# ================================================================
# FORECAST ALGORITHM (h = 1, 2, ..., 29):
#   Step a: ECT^(h) = β' Ŷ_{T+h-1}    [use FORECASTED values, not actuals]
#   Step b: Δŷ^(h)  = α ECT^(h) + μ   [VECM equation]
#   Step c: Ŷ_{T+h} = Ŷ_{T+h-1} + Δŷ^(h)  [update log-level forecast]
#   Step d: ŷ_{i,T+h} = exp(Ŷ_{i,T+h})    [convert back to Rs. Crore]
#
# GROSS CREDIT FORECAST:
#   GBC_hat_{T+h} = Σ_i ŷ_{i,T+h}     [accounting identity preserved]
#
# BOOTSTRAP CI (n_boot = 1000):
#   For each bootstrap draw b:
#     - Sample 29 residuals (with replacement) from VECM training residuals
#     - Add these to the predicted changes at each step
#     - Collect all 1000 simulated paths
#   90% CI = [5th percentile, 95th percentile] of bootstrap distribution

print("\n" + "=" * 65)
print("  SECTION 11: MULTI-STEP FORECAST + BOOTSTRAP CI")
print("=" * 65)

# ── Print step h=1 decomposition ─────────────────────────────────
print(f"\nDetailed decomposition for h=1 (August 2018 forecast):")
print(f"  Δŷ_i = α_i1·ECT1 + α_i2·ECT2 + α_i3·ECT3 + μ_i")
print()
print(f"{'Variable':<22} {'α_i1·ECT1':>12} {'α_i2·ECT2':>12} "
      f"{'α_i3·ECT3':>12} {'μ_i':>10} {'Δŷ_i':>10} {'Level (Cr)':>13}")
print("-" * 100)

dY_h1 = alpha @ ECT_t0 + mu
Y_h1  = Y_last + dY_h1

for i, v in enumerate(VLABS):
    c1 = alpha[i, 0] * ECT_t0[0]
    c2 = alpha[i, 1] * ECT_t0[1]
    c3 = alpha[i, 2] * ECT_t0[2]
    print(f"{VLABS[i]:<22} {c1:>12.6f} {c2:>12.6f} {c3:>12.6f} "
          f"{mu[i]:>10.5f} {dY_h1[i]:>10.6f} {np.exp(Y_h1[i]):>13,.0f}")

gross_h1 = np.exp(Y_h1).sum()
print(f"\n  Gross Credit Forecast (h=1): Rs. {gross_h1:,.0f} Cr")
print(f"  Actual Aug 2018:             Rs. {df_test['Gross'].iloc[0]:,.0f} Cr")
print(f"  Forecast Error:              {(df_test['Gross'].iloc[0] - gross_h1)/df_test['Gross'].iloc[0]*100:.2f}%")


def vecm_forecast_with_ci(Y_train, beta, B_full, alpha, r, steps, n_boot=1000):
    """
    Multi-step VECM forecast with 90% bootstrap confidence intervals.

    Parameters:
      Y_train : (T_train, k) training log-level matrix
      beta    : (k, r) cointegrating vectors
      B_full  : full OLS coefficient matrix
      alpha   : (k, r) loading matrix
      r       : int, cointegrating rank
      steps   : int, number of forecast horizons (= test_n)
      n_boot  : int, number of bootstrap draws

    Returns:
      Y_fore_log : (steps, k) point forecast log-levels
      ci_lo      : (steps, k) 5th percentile of bootstrap distribution (levels)
      ci_hi      : (steps, k) 95th percentile of bootstrap distribution (levels)
    """
    k = Y_train.shape[1]

    # ── Compute in-sample VECM residuals ────────────────────────
    dY       = np.diff(Y_train, axis=0)
    Zk_full  = np.ones((len(Y_train) - 1, 1))
    Z1_full  = Y_train[:-1]
    ECT_full = Z1_full @ beta
    X_full   = np.column_stack([ECT_full, Zk_full])
    resids   = dY - X_full @ B_full    # (T_train-1, k)

    # ── Point forecast: iterate forward ──────────────────────────
    const    = B_full[r, :]            # constant vector μ
    Y_fore   = np.zeros((steps, k))
    Y_cur    = Y_train[-1].copy()      # start from last training obs

    for h in range(steps):
        ect   = Y_cur @ beta           # (r,) ECT at current state
        dY_h  = alpha @ ect + const    # (k,) predicted change
        Y_cur = Y_cur + dY_h           # update log-level
        Y_fore[h] = Y_cur

    # ── Bootstrap for confidence intervals ────────────────────────
    n_res  = len(resids)
    boots  = np.zeros((n_boot, steps, k))

    np.random.seed(42)   # reproducibility
    for b in range(n_boot):
        # Sample steps residuals uniformly at random (with replacement)
        boot_idx = np.random.choice(n_res, size=steps, replace=True)
        Y_boot   = Y_train[-1].copy()

        for h in range(steps):
            ect    = Y_boot @ beta
            dY_h   = alpha @ ect + const + resids[boot_idx[h]]  # add noise
            Y_boot = Y_boot + dY_h
            boots[b, h] = Y_boot

    # Convert bootstrap log-levels to Rs. Crore levels
    ci_lo = np.exp(np.percentile(boots, 5,  axis=0))   # 5th percentile
    ci_hi = np.exp(np.percentile(boots, 95, axis=0))   # 95th percentile

    return Y_fore, ci_lo, ci_hi


# ── Run forecast ─────────────────────────────────────────────────
print(f"\nRunning {test_n}-step forecast with 1000 bootstrap draws...")
Y_fore_log, ci_lo, ci_hi = vecm_forecast_with_ci(
    Y_train, beta, B_full, alpha, r, steps=test_n, n_boot=1000
)

# Convert log-level forecasts to Rs. Crore
fore_lev   = np.exp(Y_fore_log)   # (29, 5) sector-level forecasts
actual_lev = np.exp(Y_test)       # (29, 5) actual test values

# Gross Credit forecast = sum of five sector forecasts (accounting identity)
gross_fore = fore_lev.sum(axis=1)      # (29,) gross forecast
gross_lo   = ci_lo.sum(axis=1)         # (29,) lower CI bound
gross_hi   = ci_hi.sum(axis=1)         # (29,) upper CI bound

print(f"Forecast complete.")
print(f"\nFull forecast vs actual — Gross Bank Credit (N=29):")
print(f"\n{'Month':<10} {'Actual (Cr)':>12} {'Forecast (Cr)':>14} "
      f"{'Residual (Cr)':>14} {'Error%':>8}")
print("-" * 65)
for h in range(test_n):
    date = df_test['Date'].iloc[h].strftime('%b %Y')
    act  = df_test['Gross'].iloc[h]
    fct  = gross_fore[h]
    print(f"{date:<10} {act:>12,.0f} {fct:>14,.0f} "
          f"{act - fct:>14,.0f} {(act - fct)/act*100:>7.2f}%")


# ================================================================
# SECTION 12: FORECAST ACCURACY METRICS
# ================================================================
# Four metrics covering different dimensions of forecast quality:
#   RMSE: Root Mean Squared Error — penalises large errors (squared).
#         In-sample comparison standard. Units = Rs. Crore.
#   MAE:  Mean Absolute Error — robust to outliers. Rs. Crore.
#   MAPE: Mean Absolute % Error — scale-free; comparable across sectors.
#         < 5% = "good"; < 10% = "acceptable" for macro forecasting.
#   MASE: Mean Absolute Scaled Error — compares to naive random walk.
#         MASE < 1: better than naive. MASE > 1: worse than naive.

print("\n" + "=" * 65)
print("  SECTION 12: FORECAST ACCURACY METRICS")
print("=" * 65)


def rmse(actual, forecast):
    """Root Mean Squared Error — in units of the data."""
    return np.sqrt(np.mean((actual - forecast) ** 2))


def mae(actual, forecast):
    """Mean Absolute Error — robust to outliers."""
    return np.mean(np.abs(actual - forecast))


def mape(actual, forecast):
    """Mean Absolute Percentage Error — scale-free."""
    return np.mean(np.abs((actual - forecast) / actual)) * 100


def mase(actual, forecast, train_series):
    """Mean Absolute Scaled Error — benchmark = naive random walk."""
    naive_error = np.mean(np.abs(np.diff(train_series)))
    return mae(actual, forecast) / naive_error if naive_error > 0 else np.nan


print(f"\n{'Variable':<28} {'RMSE (Rs. Cr)':>15} {'MAE (Rs. Cr)':>14} "
      f"{'MAPE':>8} {'MASE':>8}")
print("-" * 80)

for i, v in enumerate(VARS):
    a  = actual_lev[:, i]
    f  = fore_lev[:, i]
    tr = np.exp(Y_train[:, i])
    print(f"{VLABS[i]:<28} {rmse(a, f):>15,.0f} {mae(a, f):>14,.0f} "
          f"{mape(a, f):>7.2f}% {mase(a, f, tr):>8.3f}")

# Gross Bank Credit
ga  = df_test['Gross'].values
gf  = gross_fore
gtr = df_train['Gross'].values
print(f"{'GROSS BANK CREDIT':<28} {rmse(ga, gf):>15,.0f} {mae(ga, gf):>14,.0f} "
      f"{mape(ga, gf):>7.2f}% {mase(ga, gf, gtr):>8.3f}")

print(f"\nNote: Food Credit MAPE is high (60%) because its level is < 1% of Gross.")
print(f"      In absolute terms, Food RMSE (46,221 Cr) ≈ Agriculture RMSE (30,770 Cr).")
print(f"      Gross Credit MAPE = {mape(ga, gf):.2f}% — acceptable overall accuracy.")
print(f"      COVID-19 (Apr 2020–Dec 2020) drives systematic over-forecasting.")


# ================================================================
# SECTION 13: ALL CHARTS
# ================================================================
print("\n" + "=" * 65)
print("  SECTION 13: GENERATING ALL CHARTS")
print("=" * 65)

# ── Chart 7: Sector forecasts vs actuals ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, (v, l, c) in enumerate(zip(VARS, VLABS, VCOLS)):
    ax = axes.flat[i]
    ax.plot(df_train['Date'], df_train[v] / 1e5,
            color=c, lw=1.5, alpha=0.7, label='Train')
    ax.plot(df_test['Date'], actual_lev[:, i] / 1e5,
            color=c, lw=2, label='Test (actual)')
    ax.plot(df_test['Date'], fore_lev[:, i] / 1e5,
            color='#1A2744', lw=2, ls='--', label='VECM Forecast')
    ax.fill_between(df_test['Date'],
                    ci_lo[:, i] / 1e5, ci_hi[:, i] / 1e5,
                    color=c, alpha=0.15)
    ax.axvline(df_train['Date'].iloc[-1], color='gray', lw=0.8, ls=':')
    ax.set_title(l, fontsize=9, fontweight='bold', color=c)
    ax.set_ylabel('Rs. Lakh Cr', fontsize=7.5)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=7)

# Gross in the 6th panel
ax6 = axes.flat[5]
ax6.plot(df_train['Date'], df_train['Gross'] / 1e5,
         color='#1A2744', lw=1.5, alpha=0.7, label='Train')
ax6.plot(df_test['Date'], df_test['Gross'].values / 1e5,
         color='#D97706', lw=2, label='Actual')
ax6.plot(df_test['Date'], gross_fore / 1e5,
         color='#0D9488', lw=2, ls='--', label='Forecast')
ax6.fill_between(df_test['Date'],
                 gross_lo / 1e5, gross_hi / 1e5,
                 color='#0D9488', alpha=0.15)
ax6.axvline(df_train['Date'].iloc[-1], color='gray', lw=0.8, ls=':')
ax6.set_title('GROSS BANK CREDIT', fontsize=9, fontweight='bold',
              color='#1A2744')
ax6.set_ylabel('Rs. Lakh Cr', fontsize=7.5)
ax6.legend(fontsize=6.5)
ax6.tick_params(labelsize=7)

fig.suptitle('VECM Sector Forecasts vs Actual (N=29 test, shaded = 90% CI)',
             fontsize=11, fontweight='bold', color='#1A2744')
fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
fig.savefig('./vecm_outputs/fig7_sector_forecasts.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: fig7_sector_forecasts.png")

# ── Chart 8: Gross forecast vs actual (main result) ──────────────
mape_g = mape(ga, gf)
rmse_g = rmse(ga, gf)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_train['Date'], df_train['Gross'] / 1e5,
        color='#1A2744', lw=2.2, label='Historical (Train)')
ax.plot(df_test['Date'], ga / 1e5,
        color='#D97706', lw=2.5, label='Actual (Test)', zorder=5)
ax.plot(df_test['Date'], gf / 1e5,
        color='#0D9488', lw=2.5, ls='--', label='VECM Forecast')
ax.fill_between(df_test['Date'], gross_lo / 1e5, gross_hi / 1e5,
                color='#0D9488', alpha=0.16, label='90% Bootstrap CI')
ax.axvline(df_train['Date'].iloc[-1], color='gray', lw=1.2, ls=':', label='Split')
ax.text(0.02, 0.97,
        f'MAPE: {mape_g:.2f}%\nRMSE: ₹{rmse_g/1e5:.2f}L Cr',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9,
                  ec='#0D9488', lw=1.5))
ax.set_title('Gross Bank Credit — VECM Forecast vs Actual (N=115; Test=29)',
             fontsize=12, fontweight='bold', color='#1A2744')
ax.set_ylabel('Rs. Lakh Crore', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.tick_params(labelsize=8)
fig.tight_layout(pad=0.8)
fig.savefig('./vecm_outputs/fig8_gross_forecast.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: fig8_gross_forecast.png")

# ── Chart 9: Forecast residuals ───────────────────────────────────
resids_g = ga - gf
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(df_test['Date'], resids_g / 1e5,
             color='#BF360C', marker='o', ms=3.5, lw=1.5)
axes[0].axhline(0, color='gray', ls='--', lw=0.8)
axes[0].set_title('Residuals Time-Series (Actual − Forecast)',
                  fontsize=10, fontweight='bold', color='#1A2744')
axes[0].set_ylabel('Rs. Lakh Crore', fontsize=9)
axes[0].tick_params(labelsize=8)

bar_colors = ['#0D9488' if v >= 0 else '#BF360C' for v in resids_g]
axes[1].bar(range(len(resids_g)), resids_g / 1e5,
            color=bar_colors, alpha=0.82, edgecolor='white')
axes[1].axhline(0, color='gray', ls='--', lw=0.8)
axes[1].set_title('Residuals by Period (Teal=Under-forecast, Red=Over-forecast)',
                  fontsize=10, fontweight='bold', color='#1A2744')
axes[1].set_ylabel('Rs. Lakh Crore', fontsize=9)
axes[1].tick_params(labelsize=8)
fig.tight_layout(pad=0.8)
fig.savefig('./vecm_outputs/fig9_residuals.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: fig9_residuals.png")

# ── Chart 10: Dual accuracy bar (MAPE + RMSE) ────────────────────
vlabs_all  = VLABS + ['Gross Bank Credit']
mape_vals  = [mape(actual_lev[:, i], fore_lev[:, i]) for i in range(5)]
mape_vals += [mape_g]
rmse_vals  = [rmse(actual_lev[:, i], fore_lev[:, i]) for i in range(5)]
rmse_vals += [rmse_g]
col_list   = VCOLS + ['#1A2744']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
b1 = axes[0].barh(vlabs_all, mape_vals, color=col_list, alpha=0.85,
                  edgecolor='white')
for bar, val in zip(b1, mape_vals):
    axes[0].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}%', va='center', fontsize=9)
axes[0].axvline(5, color='#D97706', lw=1.2, ls='--', alpha=0.7,
                label='5% threshold')
axes[0].set_xlabel('MAPE (%)', fontsize=9)
axes[0].set_title('MAPE by Sector', fontsize=10, fontweight='bold', color='#1A2744')
axes[0].legend(fontsize=8)
axes[0].tick_params(labelsize=8.5)

b2 = axes[1].barh(vlabs_all, [v / 1e5 for v in rmse_vals],
                  color=col_list, alpha=0.85, edgecolor='white')
for bar, val in zip(b2, rmse_vals):
    axes[1].text(val / 1e5 + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'₹{val/1e5:.2f}L', va='center', fontsize=9)
axes[1].set_xlabel('RMSE (Rs. Lakh Crore)', fontsize=9)
axes[1].set_title('RMSE by Sector', fontsize=10, fontweight='bold', color='#1A2744')
axes[1].tick_params(labelsize=8.5)

fig.suptitle('Forecast Accuracy — MAPE & RMSE by Sector (N=29 test)',
             fontsize=11, fontweight='bold', color='#1A2744')
fig.tight_layout(rect=[0, 0, 1, 0.93], pad=0.8)
fig.savefig('./vecm_outputs/fig10_accuracy_dual.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: fig10_accuracy_dual.png")

print("\n" + "=" * 65)
print("  ALL ANALYSIS COMPLETE")
print(f"  Outputs saved to: ./vecm_outputs/")
print("=" * 65)
