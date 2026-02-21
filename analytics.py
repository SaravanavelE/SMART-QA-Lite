"""
analytics.py
------------
SMART-QA Lite – Core Analytics Module
Implements:
  1. Rejection Rate Analysis  (daily / machine / operator / weekly)
  2. Pareto Analysis          (80/20 defect breakdown)
  3. Correlation & Regression (process parameters vs rejection rate)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1B4F72",
    "accent":    "#E74C3C",
    "ok":        "#27AE60",
    "warning":   "#F39C12",
    "light_bg":  "#F4F6F9",
    "grid":      "#D5D8DC",
}


def _style_ax(ax, title, xlabel="", ylabel=""):
    """Apply consistent minimal styling to an Axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["primary"], pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color=PALETTE["grid"])
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REJECTION RATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rejection_summary(df: pd.DataFrame) -> dict:
    """
    Compute aggregated rejection rates across multiple dimensions.

    Formula
    -------
    Rejection_Rate (%) = (Σ Rejected_Qty / Σ Produced_Qty) × 100

    Returns dict with keys: daily, machine, operator, weekly.
    """

    def agg_rate(group_col):
        return (
            df.groupby(group_col)
            .apply(lambda g: (g["Rejected_Qty"].sum() / g["Produced_Qty"].sum()) * 100)
            .reset_index(name="Rejection_Rate_Pct")
            .sort_values("Rejection_Rate_Pct", ascending=False)
        )

    # Daily trend
    daily = (
        df.groupby("Date")
        .apply(lambda g: (g["Rejected_Qty"].sum() / g["Produced_Qty"].sum()) * 100)
        .reset_index(name="Rejection_Rate_Pct")
        .sort_values("Date")
    )

    # Weekly trend
    weekly = (
        df.groupby("Week")
        .apply(lambda g: (g["Rejected_Qty"].sum() / g["Produced_Qty"].sum()) * 100)
        .reset_index(name="Rejection_Rate_Pct")
        .sort_values("Week")
    )

    return {
        "daily":    daily,
        "machine":  agg_rate("Machine_ID"),
        "operator": agg_rate("Operator_ID"),
        "weekly":   weekly,
    }


def plot_rejection_trends(summary: dict) -> plt.Figure:
    """
    2×2 grid of rejection trend charts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(PALETTE["light_bg"])
    fig.suptitle("Rejection Rate Analysis", fontsize=16, fontweight="bold",
                 color=PALETTE["primary"], y=1.01)

    # ── Daily trend (line) ────────────────────────────────────────────────────
    ax = axes[0, 0]
    daily = summary["daily"]
    ax.plot(daily["Date"], daily["Rejection_Rate_Pct"],
            color=PALETTE["primary"], linewidth=2, marker="o", markersize=4)
    ax.fill_between(daily["Date"], daily["Rejection_Rate_Pct"],
                    alpha=0.12, color=PALETTE["primary"])
    _style_ax(ax, "Daily Rejection Rate (%)", ylabel="Rate (%)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # ── Machine-wise (horizontal bar) ─────────────────────────────────────────
    ax = axes[0, 1]
    mach = summary["machine"]
    colors = [PALETTE["accent"] if r == mach["Rejection_Rate_Pct"].max()
              else PALETTE["primary"] for r in mach["Rejection_Rate_Pct"]]
    bars = ax.barh(mach["Machine_ID"], mach["Rejection_Rate_Pct"],
                   color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    _style_ax(ax, "Machine-wise Rejection (%)", xlabel="Rate (%)")

    # ── Operator-wise (horizontal bar) ────────────────────────────────────────
    ax = axes[1, 0]
    op = summary["operator"]
    colors = [PALETTE["accent"] if r == op["Rejection_Rate_Pct"].max()
              else "#5D8AA8" for r in op["Rejection_Rate_Pct"]]
    bars = ax.barh(op["Operator_ID"], op["Rejection_Rate_Pct"],
                   color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    _style_ax(ax, "Operator-wise Rejection (%)", xlabel="Rate (%)")

    # ── Weekly trend (bar) ────────────────────────────────────────────────────
    ax = axes[1, 1]
    wk = summary["weekly"]
    ax.bar(wk["Week"].astype(str), wk["Rejection_Rate_Pct"],
           color=PALETTE["primary"], edgecolor="white", width=0.6)
    _style_ax(ax, "Weekly Rejection Rate (%)", xlabel="Week No.", ylabel="Rate (%)")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARETO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pareto analysis of defect types.

    Logic
    -----
    1. Count total rejected units per defect category.
    2. Sort descending.
    3. Compute % contribution and cumulative %.
    4. Flag categories in the "vital few" (cumulative ≤ 80 %).
    """
    pareto = (
        df.groupby("Defect_Type")["Rejected_Qty"]
        .sum()
        .reset_index(name="Total_Rejected")
        .sort_values("Total_Rejected", ascending=False)
    )
    total = pareto["Total_Rejected"].sum()
    pareto["Pct_Contribution"] = (pareto["Total_Rejected"] / total) * 100
    pareto["Cumulative_Pct"] = pareto["Pct_Contribution"].cumsum()
    pareto["Vital_Few"] = pareto["Cumulative_Pct"] <= 80.0
    # Include the first item that pushes cumulative past 80
    first_over = pareto[pareto["Cumulative_Pct"] > 80].index
    if len(first_over):
        pareto.loc[first_over[0], "Vital_Few"] = True

    return pareto.reset_index(drop=True)


def plot_pareto(pareto_df: pd.DataFrame) -> plt.Figure:
    """
    Classic Pareto chart: bars (count) + line (cumulative %).
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["light_bg"])

    bar_colors = [PALETTE["accent"] if vf else PALETTE["primary"]
                  for vf in pareto_df["Vital_Few"]]

    bars = ax1.bar(pareto_df["Defect_Type"], pareto_df["Total_Rejected"],
                   color=bar_colors, edgecolor="white", zorder=2)
    ax1.bar_label(bars, padding=3, fontsize=9)
    ax1.set_ylabel("Total Rejected Qty", fontsize=10, color=PALETTE["primary"])
    ax1.tick_params(axis="x", rotation=20, labelsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(pareto_df["Defect_Type"], pareto_df["Cumulative_Pct"],
             color=PALETTE["warning"], linewidth=2.5, marker="D",
             markersize=6, zorder=3, label="Cumulative %")
    ax2.axhline(80, linestyle="--", color=PALETTE["accent"], linewidth=1.5,
                alpha=0.8, label="80% Line")
    ax2.set_ylabel("Cumulative Contribution (%)", fontsize=10, color=PALETTE["warning"])
    ax2.set_ylim(0, 115)
    ax2.spines[["top"]].set_visible(False)

    # Legend
    vital_patch = mpatches.Patch(color=PALETTE["accent"], label="Vital Few (≤80%)")
    minor_patch = mpatches.Patch(color=PALETTE["primary"], label="Trivial Many (>80%)")
    ax1.legend(handles=[vital_patch, minor_patch], loc="upper left",
               fontsize=9, framealpha=0.8)

    ax1.set_title("Pareto Chart – Defect Type Analysis (80/20 Rule)",
                  fontsize=13, fontweight="bold", color=PALETTE["primary"], pad=12)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CORRELATION & REGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

PROCESS_PARAMS = ["Temperature", "Pressure", "Speed"]


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix between process parameters and Rejection_Rate.
    """
    cols = PROCESS_PARAMS + ["Rejection_Rate"]
    return df[cols].corr()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """
    Annotated heatmap of the correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(PALETTE["light_bg"])

    mask = np.zeros_like(corr_matrix, dtype=bool)  # show full matrix
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap=cmap,
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax
    )
    ax.set_title("Correlation Matrix – Process Params vs Rejection Rate",
                 fontsize=12, fontweight="bold", color=PALETTE["primary"], pad=12)
    plt.tight_layout()
    return fig


def run_regression(df: pd.DataFrame) -> dict:
    """
    Ordinary Least Squares regression:
        Rejection_Rate ~ Temperature + Pressure + Speed

    Returns
    -------
    dict with keys: model, scaler, coefficients, r2, rmse, predictions.
    """
    X = df[PROCESS_PARAMS].copy()
    y = df["Rejection_Rate"].copy()

    # Drop rows with NaN in features or target
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    coefficients = pd.DataFrame({
        "Feature":     PROCESS_PARAMS,
        "Coefficient": model.coef_,
        "Abs_Impact":  np.abs(model.coef_),
    }).sort_values("Abs_Impact", ascending=False)

    return {
        "model":        model,
        "scaler":       scaler,
        "coefficients": coefficients,
        "r2":           round(r2, 4),
        "rmse":         round(rmse, 4),
        "intercept":    round(model.intercept_, 4),
        "predictions":  y_pred,
        "actuals":      y.values,
    }


def plot_regression_results(reg_result: dict) -> plt.Figure:
    """
    Side-by-side: feature importance bar + actual vs predicted scatter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["light_bg"])

    # ── Feature importance ────────────────────────────────────────────────────
    ax = axes[0]
    coeff = reg_result["coefficients"]
    bar_colors = [PALETTE["accent"] if c > 0 else PALETTE["primary"]
                  for c in coeff["Coefficient"]]
    bars = ax.barh(coeff["Feature"], coeff["Coefficient"],
                   color=bar_colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    _style_ax(ax, "Regression Coefficients\n(Standardised)",
              xlabel="Coefficient Value")
    ax.invert_yaxis()

    # ── Actual vs predicted ────────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(reg_result["actuals"], reg_result["predictions"],
               alpha=0.5, color=PALETTE["primary"], edgecolors="white",
               linewidth=0.5, s=40)
    mn = min(reg_result["actuals"].min(), reg_result["predictions"].min()) - 0.5
    mx = max(reg_result["actuals"].max(), reg_result["predictions"].max()) + 0.5
    ax.plot([mn, mx], [mn, mx], "--", color=PALETTE["accent"], linewidth=1.5)
    _style_ax(ax, f"Actual vs Predicted  (R²={reg_result['r2']})",
              xlabel="Actual Rejection Rate (%)", ylabel="Predicted (%)")

    plt.tight_layout()
    return fig
