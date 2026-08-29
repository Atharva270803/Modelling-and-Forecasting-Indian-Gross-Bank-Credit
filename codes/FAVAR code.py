# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:28:57 2026

@author: Lenovo
"""

# ============================================================
# FAVAR — IMPROVED & CORRECTED VERSION
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

# -------------------------
# 1. LOAD DATA
# -------------------------
FILE_PATH = r"C:\Users\Lenovo\Desktop\Downloads\FAVAR_DATA.xlsx"

df = pd.read_excel(FILE_PATH, header=0, index_col=0).T

df.index = df.index.str.strip()
df.index = pd.to_datetime(df.index, format='%b, %Y', errors='coerce')
df = df[df.index.notnull()].sort_index()

df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(method='ffill').fillna(method='bfill')

# -------------------------
# 2. DROP REDUNDANT VARIABLES
# -------------------------
DROP_COLS = [
    'Standing Deposit Facility(SDF) Rate','Base Rate','MCLR(Overnight)',
    'Term Deposit Rate >1 Year','Incremental Credit-Deposit Ratio',
    'Incremental Investment- Deposit Ratio','Reserve Repo Rate',
    'Bank Rate','Non-Food Credit'
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# -------------------------
# 3. TRANSFORMATION
# -------------------------
for col in ['INR-US Spot Rate', 'INR-Euro Spot Rate']:
    if col in df.columns:
        df[col] = np.log(df[col]).diff() * 100

df = df.dropna()

# -------------------------
# 4. STATIONARITY
# -------------------------
for col in df.columns:
    result = adfuller(df[col].dropna())
    if result[1] > 0.05:
        df[col] = df[col].diff()

df = df.dropna()

# -------------------------
# 5. SPLIT
# -------------------------
split_idx = int(len(df) * 0.75)
train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

# -------------------------
# 6. DEFINE Y and X
# -------------------------
Y_VARS = ['Credit', 'Policy Repo Rate', 'All India Consumer Price Index']
X_VARS = [c for c in df.columns if c not in Y_VARS]

# -------------------------
# 7. PCA (FIXED)
# -------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train_df[X_VARS])
X_all_scaled   = scaler.transform(df[X_VARS])

pca = PCA()
pca.fit(X_train_scaled)

cumvar = np.cumsum(pca.explained_variance_ratio_)

# 🔥 CONTROL FACTORS (important fix)
n_factors = min(3, np.argmax(cumvar >= 0.8) + 1)

print("Number of factors:", n_factors)

F_all = pca.transform(X_all_scaled)[:, :n_factors]

F_train = F_all[:split_idx]
F_test  = F_all[split_idx:]

factor_names = [f'F{i+1}' for i in range(n_factors)]

# -------------------------
# 8. LOADINGS (FIXED)
# -------------------------
loadings = pd.DataFrame(
    pca.components_.T[:, :n_factors],
    index=X_VARS,
    columns=factor_names
)

print("\nTop variables per factor:")
for f in factor_names:
    print("\n", f)
    print(loadings[f].abs().sort_values(ascending=False).head(5))

# -------------------------
# 9. BUILD FAVAR
# -------------------------
F_train_df = pd.DataFrame(F_train, index=train_df.index, columns=factor_names)
F_test_df  = pd.DataFrame(F_test, index=test_df.index, columns=factor_names)

favar_train = pd.concat([F_train_df, train_df[Y_VARS]], axis=1)
favar_test  = pd.concat([F_test_df,  test_df[Y_VARS]], axis=1)

# -------------------------
# 10. VAR MODEL (FIXED)
# -------------------------
model = VAR(favar_train)

lag_selection = model.select_order(4)
lag = lag_selection.aic

print("Selected lag:", lag)

results = model.fit(lag)

# -------------------------
# 11. FORECAST
# -------------------------
lag_order = results.k_ar

forecast = results.forecast(
    favar_train.values[-lag_order:], steps=len(test_df)
)

forecast_df = pd.DataFrame(
    forecast, index=test_df.index, columns=favar_train.columns
)

credit_forecast = forecast_df['Credit']
credit_actual   = test_df['Credit']

# -------------------------
# 12. METRICS (FIXED)
# -------------------------
actual = credit_actual.replace(0, 1e-6)

rmse = np.sqrt(np.mean((actual - credit_forecast) ** 2))
mae  = np.mean(np.abs(actual - credit_forecast))
mape = np.mean(np.abs((actual - credit_forecast) / actual)) * 100

ss_res = np.sum((actual - credit_forecast) ** 2)
ss_tot = np.sum((actual - actual.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("\n=== RESULTS ===")
print("RMSE :", rmse)
print("MAE  :", mae)
print("MAPE :", mape)
print("R²   :", r2)

# -------------------------
# 13. PLOT
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(train_df.index, train_df['Credit'], label='Train', color='blue')
plt.plot(test_df.index, credit_actual, label='Actual', color='black')
plt.plot(test_df.index, credit_forecast, label='Forecast', color='red')
plt.legend()
plt.title("Improved FAVAR Credit Forecast")
plt.show()