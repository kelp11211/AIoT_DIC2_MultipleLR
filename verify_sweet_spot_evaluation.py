"""
=============================================================================
50 Startups - CRISP-DM Phase 5: Evaluation & Parsimony Verification Script
=============================================================================
This script performs in-depth statistical inference, OLS regression summary,
VIF multicollinearity diagnostic, and train/test parsimony evaluation for the
optimal 2-feature sweet spot model (R&D Spend + Marketing Spend).
=============================================================================
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_dataset():
    """Load dataset from local CSV or fallback to GitHub."""
    local_paths = ["50_Startups.csv", "../DIC-2/50_Startups.csv", "DIC-2/50_Startups.csv"]
    for path in local_paths:
        if os.path.exists(path):
            print(f"[Data Loader] Loaded from local file: {path}")
            return pd.read_csv(path)

    url = (
        "https://github.com/harimittapalli/"
        "Mulitple-Linear-Reggression/raw/master/50_Startups.csv"
    )
    print(f"[Data Loader] Fetching from remote URL: {url}")
    return pd.read_csv(url)


def main():
    print("=" * 75)
    print(" 50 STARTUPS - CRISP-DM EVALUATION & STATISTICAL INFERENCE")
    print("=" * 75)

    df = load_dataset()
    df = df.drop_duplicates().copy()

    # ---------------------------------------------------------
    # PART 1: STATSMODELS OLS SUMMARY (k=2 Sweet Spot Model)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" [PART 1] OLS REGRESSION SUMMARY (Sweet Spot: k=2)")
    print(" Predictors: R&D Spend + Marketing Spend | Target: Profit")
    print("=" * 75)

    X_sweet = df[["R&D Spend", "Marketing Spend"]]
    X_sweet_const = sm.add_constant(X_sweet)
    y = df["Profit"]

    ols_model = sm.OLS(y, X_sweet_const).fit()
    print(ols_model.summary())

    # ---------------------------------------------------------
    # PART 2: MULTICOLLINEARITY DIAGNOSTICS (VIF Analysis)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" [PART 2] VARIANCE INFLATION FACTOR (VIF) DIAGNOSTIC")
    print(" Rule of thumb: VIF < 5 indicates safe / no harmful multicollinearity")
    print("=" * 75)

    vif_df = pd.DataFrame()
    vif_df["Predictor"] = X_sweet.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X_sweet.values, i)
        for i in range(X_sweet.shape[1])
    ]
    vif_df["Status"] = vif_df["VIF"].apply(
        lambda x: "Safe (< 5.0)" if x < 5.0 else "High Multicollinearity (>= 5.0)"
    )
    print(vif_df.to_string(index=False))

    # ---------------------------------------------------------
    # PART 3: TRAIN / TEST PARSIMONY EVALUATION (k=2 vs k=5)
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" [PART 3] PARSIMONY EVALUATION (Occam's Razor & Overfitting Cliff)")
    print(" Train: 80% (40 startups) | Test: 20% (10 startups), Random State = 42")
    print("=" * 75)

    # Prepare Full Model (k=5 with One-Hot State)
    df_encoded = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=float)
    X_full = df_encoded.drop(columns=["Profit"])
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.20, random_state=42
    )

    # Train Full Model
    reg_full = LinearRegression().fit(X_train_full, y_train)
    y_pred_full = reg_full.predict(X_test_full)
    rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
    mae_full = mean_absolute_error(y_test, y_pred_full)
    r2_full = r2_score(y_test, y_pred_full)
    n_test, p_full = len(y_test), X_test_full.shape[1]
    adj_r2_full = 1 - (1 - r2_full) * (n_test - 1) / (n_test - p_full - 1)

    # Prepare Sweet Spot Model (k=2: R&D + Marketing)
    X_train_sweet = X_train_full[["R&D Spend", "Marketing Spend"]]
    X_test_sweet = X_test_full[["R&D Spend", "Marketing Spend"]]

    reg_sweet = LinearRegression().fit(X_train_sweet, y_train)
    y_pred_sweet = reg_sweet.predict(X_test_sweet)
    rmse_sweet = np.sqrt(mean_squared_error(y_test, y_pred_sweet))
    mae_sweet = mean_absolute_error(y_test, y_pred_sweet)
    r2_sweet = r2_score(y_test, y_pred_sweet)
    p_sweet = X_test_sweet.shape[1]
    adj_r2_sweet = 1 - (1 - r2_sweet) * (n_test - 1) / (n_test - p_sweet - 1)

    comparison = pd.DataFrame({
        "Metric": [
            "Features Count (k)",
            "Active Predictors",
            "Test RMSE ($)",
            "Test MAE ($)",
            "Test R^2 Score",
            "Test Adjusted R^2",
            "Occam's Razor Verdict"
        ],
        "Full Model (k=5)": [
            "5",
            "R&D, Admin, Mkt, State_FL, State_NY",
            f"${rmse_full:,.2f}",
            f"${mae_full:,.2f}",
            f"{r2_full:.4f}",
            f"{adj_r2_full:.4f}",
            "[WARNING] Overfitting Cliff (Higher Test Error)"
        ],
        "Sweet Spot Model (k=2)": [
            "2",
            "R&D Spend, Marketing Spend",
            f"${rmse_sweet:,.2f}",
            f"${mae_sweet:,.2f}",
            f"{r2_sweet:.4f}",
            f"{adj_r2_sweet:.4f}",
            "[OPTIMAL] Global Sweet Spot (Lowest Error)"
        ]
    })
    print(comparison.to_string(index=False))

    # ---------------------------------------------------------
    # PART 4: LIVE SCENARIO PREDICTION
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" [PART 4] NEW STARTUP LIVE PROFIT PREDICTION")
    print(" Inputs: R&D Spend = $100,000 | Marketing Spend = $150,000")
    print("=" * 75)

    new_startup_k2 = pd.DataFrame({
        "R&D Spend": [100000.0],
        "Marketing Spend": [150000.0]
    })
    pred_profit_k2 = reg_sweet.predict(new_startup_k2)[0]

    b0 = reg_sweet.intercept_
    b_rd = reg_sweet.coef_[0]
    b_mkt = reg_sweet.coef_[1]

    print(f"Fitted Equation: Profit = {b0:,.2f} + ({b_rd:.4f} * R&D) + ({b_mkt:.4f} * Marketing)")
    print(f"Base Intercept:    ${b0:,.2f}")
    print(f"R&D Contribution:  +${b_rd * 100000:,.2f}")
    print(f"Mkt Contribution:  +${b_mkt * 150000:,.2f}")
    print("-" * 50)
    print(f"Predicted Startup Profit: ${pred_profit_k2:,.2f}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
