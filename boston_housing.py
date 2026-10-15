# ============================================================
# BOSTON HOUSING - MULTIPLE LINEAR REGRESSION
# CRISP-DM Machine Learning Project
# ============================================================

# PHASE 1: BUSINESS UNDERSTANDING
# Goal: Predict Boston median house value (MEDV in $1000s)
# based on structural, neighborhood, and socioeconomic attributes.

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# PHASE 2: DATA UNDERSTANDING
csv_path = "BostonHousing.csv"
if not os.path.exists(csv_path):
    csv_path = "../DIC-2/BostonHousing.csv"

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCorrelation with MEDV (Median Home Value):")
print(df.corr()["medv"].sort_values(ascending=False))

# Plot 1: RM (Rooms) vs MEDV
plt.figure(figsize=(7, 5))
plt.scatter(df["rm"], df["medv"], alpha=0.7, color="#06b6d4")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median Home Value ($1000s) (MEDV)")
plt.title("Rooms (RM) vs Median Home Value (MEDV)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("boston_rm_vs_medv.png", dpi=300)
plt.close()
print("\nSaved plot: boston_rm_vs_medv.png")

# PHASE 3: DATA PREPARATION
df = df.drop_duplicates().copy()

# Features & Target
# Sweet Spot Features: rm (Rooms), ptratio (Pupil-Teacher Ratio), lstat (% Lower Status)
X = df[["rm", "ptratio", "lstat"]]
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# PHASE 4: MODELING
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed.")
print(f"Intercept (beta_0): {model.intercept_:.4f}")
for feat, coef in zip(X.columns, model.coef_):
    print(f"Coefficient ({feat.upper()}): {coef:+.4f}")

# PHASE 5: EVALUATION
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n================ MODEL EVALUATION ================")
print(f"MAE:  ${mae:,.2f}k (${mae*1000:,.0f})")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}k (${rmse*1000:,.0f})")
print(f"R2:   {r2:.4f}")

# Plot 2: Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.8, color="#10b981")
lower = min(y_test.min(), y_pred.min())
upper = max(y_test.max(), y_pred.max())
plt.plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
plt.xlabel("Actual MEDV ($1000s)")
plt.ylabel("Predicted MEDV ($1000s)")
plt.title("Actual vs Predicted Home Values (Boston)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("boston_actual_vs_predicted.png", dpi=300)
plt.close()
print("Saved plot: boston_actual_vs_predicted.png")

# Plot 3: Residuals
residuals = y_test.to_numpy() - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.8, color="#8b5cf6")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted MEDV ($1000s)")
plt.ylabel("Residuals ($1000s)")
plt.title("Residual Analysis (Boston Housing)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("boston_residual_analysis.png", dpi=300)
plt.close()
print("Saved plot: boston_residual_analysis.png")

# PHASE 6: DEPLOYMENT
# Predict new house scenario: 6.5 rooms, 16.0 pupil-teacher ratio, 8.5% lower status
new_house = pd.DataFrame({
    "rm": [6.5],
    "ptratio": [16.0],
    "lstat": [8.5]
})

pred_price = model.predict(new_house)[0]
print("\n================ NEW HOUSE PREDICTION ================")
print(f"Input: Rooms=6.5, PTRATIO=16.0, LSTAT=8.5%")
print(f"Predicted Median Value: ${pred_price:,.2f}k (${pred_price*1000:,.0f})")

joblib.dump(model, "boston_housing_lr.pkl")
print("\nModel saved as boston_housing_lr.pkl")
