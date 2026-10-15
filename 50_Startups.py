
# ============================================================
# 50 STARTUPS - MULTIPLE LINEAR REGRESSION
# CRISP-DM Machine Learning Project
# ============================================================

# PHASE 1: BUSINESS UNDERSTANDING
# Goal: Predict startup profit from business expenditures
# and the state in which the startup operates.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PHASE 2: DATA UNDERSTANDING
# Load data directly from GitHub

url = (
    "https://github.com/harimittapalli/"
    "Mulitple-Linear-Reggression/raw/master/50_Startups.csv"
)

df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nState distribution:")
print(df["State"].value_counts())

# Explore correlations among numerical variables
print("\nCorrelation matrix:")
print(df.select_dtypes(include="number").corr())

# R&D Spend vs Profit
plt.figure(figsize=(7, 5))
plt.scatter(df["R&D Spend"], df["Profit"])
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("R&D Spend vs Profit")
plt.grid(alpha=0.3)
plt.show()

# PHASE 3: DATA PREPARATION

# Remove duplicate records, if any
df = df.drop_duplicates().copy()

# Define input features and target
X = df.drop(columns=["Profit"])
y = df["Profit"]

numeric_features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend"
]
categorical_features = ["State"]

# One-hot encode State; drop first to avoid redundancy
preprocessor = ColumnTransformer(
    transformers=[
        ("state", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ), categorical_features),
        ("numeric", "passthrough", numeric_features)
    ]
)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# PHASE 4: MODELING
# Build a preprocessing + regression pipeline

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

print("\nModel training completed.")

# Extract model coefficients
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.named_steps[
        "regressor"
    ].coef_
})

print("\nIntercept:")
print(model.named_steps["regressor"].intercept_)

print("\nRegression coefficients:")
print(coefficients)

# PHASE 5: EVALUATION

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print(f"MAE:  {mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2:   {r2:.4f}")

# Compare actual and predicted profits
results = pd.DataFrame({
    "Actual Profit": y_test.to_numpy(),
    "Predicted Profit": y_pred,
    "Residual": y_test.to_numpy() - y_pred
})

print("\nActual vs Predicted:")
print(results.round(2))

# Visualization 1: Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.8)

lower = min(y_test.min(), y_pred.min())
upper = max(y_test.max(), y_pred.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    label="Perfect prediction"
)

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Visualization 2: Residual analysis
residuals = y_test.to_numpy() - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Profit")
plt.ylabel("Residual")
plt.title("Residual Analysis")
plt.grid(alpha=0.3)
plt.show()

# PHASE 6: DEPLOYMENT
# Predict profit for a new startup

new_startup = pd.DataFrame({
    "R&D Spend": [100000],
    "Administration": [120000],
    "Marketing Spend": [150000],
    "State": ["California"]
})

predicted_profit = model.predict(new_startup)[0]

print("\nNEW STARTUP PREDICTION")
print(f"Predicted profit: ${predicted_profit:,.2f}")

# Save trained pipeline for reuse
import joblib

joblib.dump(model, "startup_profit_lr.pkl")

print("\nModel saved as startup_profit_lr.pkl")
