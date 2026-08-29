import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import itertools

# ── Color Palette ──────────────────────────────────────────────
NAVY   = '#1B3A6B'
TEAL   = '#0B7285'
CORAL  = '#E64980'
AMBER  = '#F59F00'
GREEN  = '#2F9E44'
PURPLE = '#7048E8'
LTBLUE = '#74C0FC'
GRAY   = '#868E96'
BG     = '#F8F9FA'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.titlecolor': NAVY,
    'axes.labelcolor': NAVY,
    'xtick.color': GRAY,
    'ytick.color': GRAY,
    'grid.color': '#DEE2E6',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
})

# ────────────────────────────────────────────────────────────────
# 1. DATA PREPARATION
# ────────────────────────────────────────────────────────────────
df = pd.read_excel('/mnt/user-data/uploads/merged_arimax_dataset.xlsx')
df.columns = ['Date', 'GrossCredit']
df = df.iloc[:114].copy()
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%b, %Y')
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)
df.index.freq = 'MS'

series = df['GrossCredit'].astype(float)

n = len(series)
train_size = int(n * 0.75)
train = series.iloc[:train_size]
test  = series.iloc[train_size:]

print(f"Total obs  : {n}")
print(f"Train      : {len(train)}  ({train.index[0].strftime('%b %Y')} - {train.index[-1].strftime('%b %Y')})")
print(f"Test       : {len(test)}   ({test.index[0].strftime('%b %Y')} - {test.index[-1].strftime('%b %Y')})")
print("\nDescriptive Statistics (full series):")
print(series.describe().round(2))

# ────────────────────────────────────────────────────────────────
# FIGURE 1 - Original Time Series
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1.2]})

ax = axes[0]
ax.fill_between(train.index, train/1e5, alpha=0.15, color=TEAL)
ax.fill_between(test.index,  test/1e5,  alpha=0.15, color=CORAL)
ax.plot(train.index, train/1e5, color=TEAL,  lw=2,   label=f'Training ({len(train)} obs)')
ax.plot(test.index,  test/1e5,  color=CORAL, lw=2,   label=f'Test ({len(test)} obs)')
ax.axvline(test.index[0], color=AMBER, lw=1.5, ls='--', alpha=0.9, label='Train/Test Boundary')
ax.set_title('Gross Credit Demand - Monthly Time Series (India, Rs Crore)', pad=12)
ax.set_ylabel('Gross Credit (Rs Lakh Crore)', labelpad=8)
ax.legend(framealpha=0.9, fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}L'))
ax.grid(True, alpha=0.5)
ax.set_facecolor(BG)

ax2 = axes[1]
rm  = series.rolling(12).mean()/1e5
rs  = series.rolling(12).std()/1e5
ax2.plot(rm.index, rm, color=PURPLE, lw=1.8, label='12-Month Rolling Mean')
ax2.fill_between(rs.index, rm-rs, rm+rs, alpha=0.2, color=PURPLE, label='+-1 Std Dev Band')
ax2.set_title('12-Month Rolling Mean +/- 1 Std Dev', pad=8)
ax2.set_ylabel('Rs Lakh Crore', labelpad=8)
ax2.legend(framealpha=0.9, fontsize=9)
ax2.grid(True, alpha=0.5)
ax2.set_facecolor(BG)

plt.suptitle('Figure 1: Exploratory Analysis - Gross Credit Demand', 
             fontsize=14, fontweight='bold', color=NAVY, y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/fig1_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved.")

# ────────────────────────────────────────────────────────────────
# FIGURE 2 - ACF & PACF of Original Series
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(train, lags=30, ax=axes[0], color=TEAL, vlines_kwargs={'colors': TEAL},
         title='ACF - Original Series (Training)')
plot_pacf(train, lags=30, ax=axes[1], color=CORAL, vlines_kwargs={'colors': CORAL},
          title='PACF - Original Series (Training)', method='ywm')
for ax in axes:
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.5)
    ax.axhline(0, color=NAVY, lw=0.8)

plt.suptitle('Figure 2: ACF & PACF - Original (Non-Stationary) Series', 
             fontsize=13, fontweight='bold', color=NAVY)
plt.tight_layout()
plt.savefig('/home/claude/fig2_acf_pacf_original.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved.")

# ────────────────────────────────────────────────────────────────
# 3. STATIONARITY CHECK
# ────────────────────────────────────────────────────────────────
def adf_report(ts, label):
    r = adfuller(ts.dropna(), autolag='AIC')
    print(f"\n  ADF Test - {label}")
    print(f"  Test Statistic : {r[0]:.4f}")
    print(f"  p-value        : {r[1]:.6f}")
    print(f"  Lags Used      : {r[2]}")
    print(f"  Obs Used       : {r[3]}")
    for k,v in r[4].items():
        print(f"  Critical ({k}) : {v:.4f}")
    print(f"  => {'STATIONARY' if r[1]<0.05 else 'NON-STATIONARY'}")
    return r

print("\n=== STATIONARITY TESTS ===")
r0 = adf_report(train,     "Level (Original)")
train_d1 = train.diff().dropna()
r1 = adf_report(train_d1,  "1st Difference")

adf_level_stat = r0[0]
adf_level_pval = r0[1]
adf_diff_stat  = r1[0]
adf_diff_pval  = r1[1]
adf_level_crit = r0[4]
adf_diff_crit  = r1[4]

# ────────────────────────────────────────────────────────────────
# FIGURE 3 - Differenced Series + ACF/PACF
# ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax0 = fig.add_subplot(gs[0, :])
ax0.plot(train_d1.index, train_d1/1e3, color=GREEN, lw=1.8, label='First Difference (wt = delta_yt)')
ax0.axhline(0, color=NAVY, lw=0.8, ls='--')
ax0.fill_between(train_d1.index, train_d1/1e3, alpha=0.15, color=GREEN)
ax0.set_title('w_t = Delta(y_t) = y_t - y_{t-1}   [First-Differenced Series, d=1]', pad=10)
ax0.set_ylabel("Delta Gross Credit (Rs '000 Crore)", labelpad=8)
ax0.legend(fontsize=10)
ax0.grid(True, alpha=0.5)
ax0.set_facecolor(BG)

ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
plot_acf(train_d1,  lags=25, ax=ax1, color=TEAL,  vlines_kwargs={'colors': TEAL},
         title='ACF - Differenced Series (d=1)')
plot_pacf(train_d1, lags=25, ax=ax2, color=CORAL, vlines_kwargs={'colors': CORAL},
          title='PACF - Differenced Series (d=1)', method='ywm')
for ax in [ax1, ax2]:
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.5)
    ax.axhline(0, color=NAVY, lw=0.8)

plt.suptitle('Figure 3: First-Differenced Series - Stationarity & Correlation Structure',
             fontsize=13, fontweight='bold', color=NAVY)
plt.savefig('/home/claude/fig3_differenced.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved.")

# ────────────────────────────────────────────────────────────────
# 4. MODEL IDENTIFICATION & SELECTION
# ────────────────────────────────────────────────────────────────
# From ACF/PACF of d=1 series: significant spikes at lags 1,2 => MA(1), MA(2)
# PACF cuts off after lag 1,2 => AR(1), AR(2)
# Candidate models with d=1
candidates = [
    (1,1,0),(2,1,0),(1,1,1),(2,1,1),(1,1,2),(2,1,2),(0,1,1),(0,1,2)
]

results_table = []
print("\n=== MODEL SELECTION ===")
print(f"{'Model':<18} {'AIC':>10} {'BIC':>10} {'LogLik':>12} {'Converged':>10}")
print("-"*60)
for p,d,q in candidates:
    try:
        m = ARIMA(train, order=(p,d,q), trend='t').fit()
        results_table.append({'Model': f'ARIMA({p},{d},{q})', 'p':p,'d':d,'q':q,
                              'AIC': round(m.aic,2), 'BIC': round(m.bic,2),
                              'LogLik': round(m.llf,2), 'Conv': True, 'fit': m})
        print(f"ARIMA({p},{d},{q}){'':<8} {m.aic:>10.2f} {m.bic:>10.2f} {m.llf:>12.2f} {'Yes':>10}")
    except Exception as e:
        print(f"ARIMA({p},{d},{q}){'':<8} {'FAILED':>10} -- {e}")

res_df = pd.DataFrame([{k:v for k,v in r.items() if k != 'fit'} for r in results_table])
best   = min(results_table, key=lambda x: x['AIC'])
best2  = min(results_table, key=lambda x: x['BIC'])
print(f"\nBest by AIC: {best['Model']}  AIC={best['AIC']}")
print(f"Best by BIC: {best2['Model']} BIC={best2['BIC']}")

# ────────────────────────────────────────────────────────────────
# FIGURE 4 - AIC/BIC Model Comparison
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models_names = [r['Model'] for r in results_table]
aics = [r['AIC'] for r in results_table]
bics = [r['BIC'] for r in results_table]

colors_aic = [CORAL if r['Model']==best['Model']  else LTBLUE for r in results_table]
colors_bic = [AMBER if r['Model']==best2['Model'] else LTBLUE for r in results_table]

bars1 = axes[0].barh(models_names, aics, color=colors_aic, edgecolor='white', height=0.6)
axes[0].set_title('AIC Comparison (lower = better)', pad=10)
axes[0].set_xlabel('AIC Value')
axes[0].invert_xaxis()
for bar, val in zip(bars1, aics):
    axes[0].text(val + (max(aics)-min(aics))*0.01, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=9, color=NAVY)

bars2 = axes[1].barh(models_names, bics, color=colors_bic, edgecolor='white', height=0.6)
axes[1].set_title('BIC Comparison (lower = better)', pad=10)
axes[1].set_xlabel('BIC Value')
axes[1].invert_xaxis()
for bar, val in zip(bars2, bics):
    axes[1].text(val + (max(bics)-min(bics))*0.01, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=9, color=NAVY)

for ax in axes:
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.4, axis='x')

plt.suptitle('Figure 4: Model Selection - AIC & BIC Comparison', 
             fontsize=13, fontweight='bold', color=NAVY)
plt.tight_layout()
plt.savefig('/home/claude/fig4_model_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved.")

# ────────────────────────────────────────────────────────────────
# 5. MODEL ESTIMATION
# ────────────────────────────────────────────────────────────────
final_model = best['fit']
final_order = (best['p'], best['d'], best['q'])

print(f"\n=== PARAMETER ESTIMATES: {best['Model']} ===")
print(final_model.summary())

# Parameter table
params    = final_model.params
se        = final_model.bse
tstat     = final_model.tvalues
pval      = final_model.pvalues
conf      = final_model.conf_int()

param_df = pd.DataFrame({
    'Parameter': params.index,
    'Coefficient': params.values.round(4),
    'Std Error':   se.values.round(4),
    't-Statistic': tstat.values.round(4),
    'p-Value':     pval.values.round(6),
    '95% CI Lower': conf.iloc[:,0].values.round(4),
    '95% CI Upper': conf.iloc[:,1].values.round(4),
    'Significance': ['***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else '' 
                     for p in pval.values]
})
print("\nParameter Table:")
print(param_df.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# 6. DIAGNOSTIC CHECKING
# ────────────────────────────────────────────────────────────────
residuals = final_model.resid.dropna()
lb_test   = acorr_ljungbox(residuals, lags=[10, 15, 20], return_df=True)

print("\n=== LJUNG-BOX TEST ===")
print(lb_test)

# ────────────────────────────────────────────────────────────────
# FIGURE 5 - Residual Diagnostics
# ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# Residual plot
ax0 = fig.add_subplot(gs[0, :2])
ax0.plot(residuals.index, residuals/1e3, color=TEAL, lw=1.5, label='Residuals')
ax0.axhline(0, color=CORAL, lw=1.2, ls='--')
ax0.fill_between(residuals.index, residuals/1e3, alpha=0.12, color=TEAL)
ax0.set_title('Residuals Over Time', pad=8)
ax0.set_ylabel("Residuals (Rs '000 Crore)")
ax0.legend(fontsize=9)
ax0.grid(True, alpha=0.5)
ax0.set_facecolor(BG)

# Histogram
ax1 = fig.add_subplot(gs[0, 2])
n_r, bins, patches = ax1.hist(residuals/1e3, bins=18, color=PURPLE, alpha=0.75,
                                edgecolor='white', density=True)
mu, sigma = residuals.mean()/1e3, residuals.std()/1e3
x  = np.linspace(bins[0], bins[-1], 200)
ax1.plot(x, stats.norm.pdf(x, mu, sigma), color=CORAL, lw=2, label='Normal PDF')
ax1.set_title('Residual Histogram', pad=8)
ax1.set_xlabel("Residuals (Rs '000 Crore)")
ax1.set_ylabel('Density')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.5)
ax1.set_facecolor(BG)

# ACF of residuals
ax2 = fig.add_subplot(gs[1, 0])
plot_acf(residuals, lags=20, ax=ax2, color=GREEN, vlines_kwargs={'colors': GREEN},
         title='ACF of Residuals')
ax2.set_facecolor(BG)
ax2.grid(True, alpha=0.5)

# PACF of residuals
ax3 = fig.add_subplot(gs[1, 1])
plot_pacf(residuals, lags=20, ax=ax3, color=AMBER, vlines_kwargs={'colors': AMBER},
          title='PACF of Residuals', method='ywm')
ax3.set_facecolor(BG)
ax3.grid(True, alpha=0.5)

# QQ Plot
ax4 = fig.add_subplot(gs[1, 2])
qq = stats.probplot(residuals, dist='norm')
ax4.scatter(qq[0][0], qq[0][1]/1e3, color=LTBLUE, s=30, alpha=0.8, zorder=3)
m_qq, b_qq = np.polyfit(qq[0][0], qq[0][1]/1e3, 1)
x_line = np.linspace(qq[0][0].min(), qq[0][0].max(), 100)
ax4.plot(x_line, m_qq*x_line + b_qq, color=CORAL, lw=2, label='Normal Reference')
ax4.set_title('QQ Plot of Residuals', pad=8)
ax4.set_xlabel('Theoretical Quantiles')
ax4.set_ylabel("Sample Quantiles (Rs '000 Crore)")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.5)
ax4.set_facecolor(BG)

plt.suptitle(f'Figure 5: Residual Diagnostics - {best["Model"]}',
             fontsize=13, fontweight='bold', color=NAVY)
plt.savefig('/home/claude/fig5_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved.")

# ────────────────────────────────────────────────────────────────
# 7. FORECASTING
# ────────────────────────────────────────────────────────────────
n_test   = len(test)
fc_res   = final_model.get_forecast(steps=n_test)
fc_mean  = fc_res.predicted_mean
fc_ci    = fc_res.conf_int(alpha=0.05)

forecast_df = pd.DataFrame({
    'Month':    test.index.strftime('%b %Y'),
    'Actual':   test.values.round(0).astype(int),
    'Forecast': fc_mean.values.round(0).astype(int),
    'CI_Lower': fc_ci.iloc[:,0].values.round(0).astype(int),
    'CI_Upper': fc_ci.iloc[:,1].values.round(0).astype(int),
    'Abs_Error': np.abs(test.values - fc_mean.values).round(0).astype(int),
    'APE_%':    (np.abs(test.values - fc_mean.values)/test.values*100).round(2)
})

print("\n=== FORECAST TABLE ===")
print(forecast_df.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# FIGURE 6 - Actual vs Forecast
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]
ax.plot(train.index,   train/1e5,    color=TEAL,   lw=1.8, alpha=0.8, label='Training Data')
ax.plot(test.index,    test/1e5,     color=NAVY,   lw=2.2, label='Actual (Test)')
ax.plot(fc_mean.index, fc_mean/1e5,  color=CORAL,  lw=2.2, ls='--', label=f'{best["Model"]} Forecast')
ax.fill_between(fc_ci.index, fc_ci.iloc[:,0]/1e5, fc_ci.iloc[:,1]/1e5,
                color=CORAL, alpha=0.15, label='95% Confidence Interval')
ax.axvline(test.index[0], color=AMBER, lw=1.5, ls=':', alpha=0.9)
ax.set_title(f'Gross Credit Demand - Actual vs {best["Model"]} Forecast', pad=12)
ax.set_ylabel('Gross Credit (Rs Lakh Crore)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}L'))
ax.legend(framealpha=0.9, fontsize=10)
ax.grid(True, alpha=0.5)
ax.set_facecolor(BG)

# Forecast error panel
ax2 = axes[1]
err = (test.values - fc_mean.values)/1e3
colors_err = [GREEN if e >= 0 else CORAL for e in err]
ax2.bar(test.index, err, color=colors_err, width=20, alpha=0.8)
ax2.axhline(0, color=NAVY, lw=0.8)
ax2.set_title('Forecast Errors (Actual - Forecast)', pad=8)
ax2.set_ylabel("Error (Rs '000 Crore)")
ax2.grid(True, alpha=0.5)
ax2.set_facecolor(BG)

plt.suptitle('Figure 6: Actual vs Forecasted Values with 95% Confidence Interval',
             fontsize=13, fontweight='bold', color=NAVY)
plt.tight_layout()
plt.savefig('/home/claude/fig6_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6 saved.")

# ────────────────────────────────────────────────────────────────
# 8. MODEL EVALUATION
# ────────────────────────────────────────────────────────────────
rmse = np.sqrt(np.mean((test.values - fc_mean.values)**2))
mae  = np.mean(np.abs(test.values - fc_mean.values))
mape = np.mean(np.abs(test.values - fc_mean.values)/test.values) * 100

print(f"\n=== FORECAST ACCURACY ===")
print(f"RMSE : {rmse:,.2f} Rs Crore")
print(f"MAE  : {mae:,.2f} Rs Crore")
print(f"MAPE : {mape:.4f}%")

# ────────────────────────────────────────────────────────────────
# Save all metrics and tables for report
# ────────────────────────────────────────────────────────────────
np.save('/home/claude/analysis_data.npy', {
    'train_size': len(train), 'test_size': len(test), 'n': n,
    'train_start': str(train.index[0]), 'train_end': str(train.index[-1]),
    'test_start':  str(test.index[0]),  'test_end':   str(test.index[-1]),
    'adf_level_stat': adf_level_stat, 'adf_level_pval': adf_level_pval,
    'adf_level_crit': adf_level_crit,
    'adf_diff_stat':  adf_diff_stat,  'adf_diff_pval':  adf_diff_pval,
    'adf_diff_crit':  adf_diff_crit,
    'best_model': best['Model'], 'best_aic': best['AIC'], 'best_bic': best['BIC'],
    'rmse': rmse, 'mae': mae, 'mape': mape,
    'results_table': [(r['Model'], r['AIC'], r['BIC']) for r in results_table],
    'param_table': param_df.to_dict('records'),
    'forecast_df': forecast_df.to_dict('records'),
    'lb_table': lb_test.to_dict(),
    'final_order': final_order,
    'final_summary': str(final_model.summary()),
    'series_mean': series.mean(), 'series_std': series.std(),
    'series_min': series.min(), 'series_max': series.max(),
}, allow_pickle=True)

print("\nAll figures and data saved successfully!")
print(f"Best model: {best['Model']}  |  AIC={best['AIC']}  |  BIC={best['BIC']}")
print(f"RMSE={rmse:,.0f}  MAPE={mape:.3f}%")
