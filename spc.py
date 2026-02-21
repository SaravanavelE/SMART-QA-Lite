"""
spc.py
------
SMART-QA Lite – Statistical Process Control (SPC) Module
Implements a p-chart for monitoring the rejection (defect) rate.

Mathematical Background
-----------------------
A p-chart monitors the proportion of non-conforming items over time.

For each subgroup i with sample size nᵢ and rejection count dᵢ:

    pᵢ  = dᵢ / nᵢ                      (observed rejection proportion)

Centre Line (p̄):
    p̄   = ΣdᵢΣnᵢ                       (weighted average proportion)

Control Limits for subgroup i:
    UCLᵢ = p̄ + 3 × √(p̄(1 − p̄) / nᵢ)
    LCLᵢ = max(0, p̄ − 3 × √(p̄(1 − p̄) / nᵢ))

The factor 3 represents ±3 standard deviations, giving ~99.73% coverage
assuming the process is in statistical control (binomial approximation).

Out-of-Control Signals (Western Electric Rules applied)
-------------------------------------------------------
  Rule 1: Any single point beyond UCL or LCL.
  Rule 2: 8 consecutive points on the same side of the centre line.
  Rule 3: 6 consecutive points trending strictly up or down.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "primary":  "#1B4F72",
    "accent":   "#E74C3C",
    "ok":       "#27AE60",
    "warning":  "#F39C12",
    "ucl":      "#C0392B",
    "lcl":      "#2471A3",
    "cl":       "#1E8449",
    "light_bg": "#F4F6F9",
    "grid":     "#D5D8DC",
}


# ═══════════════════════════════════════════════════════════════════════════════
# P-CHART COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pchart(df: pd.DataFrame, group_by: str = "Date") -> pd.DataFrame:
    """
    Compute p-chart statistics grouped by `group_by` column.

    Parameters
    ----------
    df       : cleaned production DataFrame (from data_loader)
    group_by : column to use as the x-axis subgroup (default "Date")

    Returns
    -------
    pd.DataFrame with columns:
        group, n, d, p, p_bar, UCL, LCL, OOC (out-of-control flag)
    """
    grouped = (
        df.groupby(group_by)
        .agg(n=("Produced_Qty", "sum"), d=("Rejected_Qty", "sum"))
        .reset_index()
    )
    grouped.rename(columns={group_by: "group"}, inplace=True)

    # Observed proportion
    grouped["p"] = grouped["d"] / grouped["n"]

    # Centre line (grand weighted average)
    p_bar = grouped["d"].sum() / grouped["n"].sum()
    grouped["p_bar"] = p_bar

    # Control limits (variable width – one per subgroup)
    grouped["sigma"] = np.sqrt(p_bar * (1 - p_bar) / grouped["n"])
    grouped["UCL"]   = p_bar + 3 * grouped["sigma"]
    grouped["LCL"]   = (p_bar - 3 * grouped["sigma"]).clip(lower=0)

    # ── Out-of-control detection ───────────────────────────────────────────────
    grouped["OOC_R1"]  = (grouped["p"] > grouped["UCL"]) | (grouped["p"] < grouped["LCL"])
    grouped["OOC_R2"]  = _run_rule(grouped["p"], grouped["p_bar"], n=8)
    grouped["OOC_R3"]  = _trend_rule(grouped["p"], n=6)
    grouped["OOC"]     = grouped["OOC_R1"] | grouped["OOC_R2"] | grouped["OOC_R3"]

    grouped["OOC_Reason"] = grouped.apply(_ooc_reason, axis=1)

    return grouped.drop(columns=["sigma"])


def _run_rule(p_series: pd.Series, center: pd.Series, n: int = 8) -> pd.Series:
    """Rule 2: n consecutive points on the same side of centre line."""
    above = (p_series > center).astype(int)
    flags = pd.Series(False, index=p_series.index)
    for i in range(n - 1, len(p_series)):
        window = above.iloc[i - n + 1: i + 1]
        if window.sum() == n or window.sum() == 0:
            flags.iloc[i - n + 1: i + 1] = True
    return flags


def _trend_rule(p_series: pd.Series, n: int = 6) -> pd.Series:
    """Rule 3: n consecutive points trending strictly up or down."""
    flags = pd.Series(False, index=p_series.index)
    vals  = p_series.values
    for i in range(n - 1, len(vals)):
        window = vals[i - n + 1: i + 1]
        diffs  = np.diff(window)
        if np.all(diffs > 0) or np.all(diffs < 0):
            flags.iloc[i - n + 1: i + 1] = True
    return flags


def _ooc_reason(row) -> str:
    reasons = []
    if row["OOC_R1"]:
        if row["p"] > row["UCL"]:
            reasons.append("Above UCL")
        else:
            reasons.append("Below LCL")
    if row["OOC_R2"]:
        reasons.append("Run of 8")
    if row["OOC_R3"]:
        reasons.append("Trend of 6")
    return ", ".join(reasons) if reasons else ""


# ═══════════════════════════════════════════════════════════════════════════════
# P-CHART PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pchart(pc_df: pd.DataFrame, title: str = "p-Chart: Rejection Rate") -> plt.Figure:
    """
    Draw a standard p-chart with UCL, LCL, centre line, and OOC highlights.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["light_bg"])

    x     = range(len(pc_df))
    x_lbl = pc_df["group"].astype(str).tolist()

    # ── Control limit bands ────────────────────────────────────────────────────
    ax.fill_between(x, pc_df["LCL"], pc_df["UCL"],
                    alpha=0.07, color=PALETTE["primary"], label="Control Band")

    # ── UCL / LCL / CL lines ──────────────────────────────────────────────────
    ax.plot(x, pc_df["UCL"],  "--", color=PALETTE["ucl"],  linewidth=1.5, label="UCL")
    ax.plot(x, pc_df["LCL"],  "--", color=PALETTE["lcl"],  linewidth=1.5, label="LCL")
    ax.plot(x, pc_df["p_bar"], "-", color=PALETTE["cl"],   linewidth=1.8, label="CL (p̄)")

    # ── p-values ───────────────────────────────────────────────────────────────
    in_ctrl  = ~pc_df["OOC"]
    ax.plot(np.array(list(x))[in_ctrl],  pc_df["p"][in_ctrl],
            "o-", color=PALETTE["primary"], markersize=5, linewidth=1.5, label="In Control")
    ax.plot(np.array(list(x))[pc_df["OOC"]], pc_df["p"][pc_df["OOC"]],
            "^", color=PALETTE["accent"], markersize=9, zorder=5,
            label=f"Out-of-Control ({pc_df['OOC'].sum()} pts)")

    # ── Annotate OOC points ────────────────────────────────────────────────────
    for i, row in pc_df[pc_df["OOC"]].iterrows():
        idx = list(pc_df.index).index(i)
        ax.annotate(
            f"{row['p']*100:.1f}%",
            xy=(idx, row["p"]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=8, color=PALETTE["accent"], ha="center"
        )

    # ── Axes formatting ────────────────────────────────────────────────────────
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_lbl, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.set_ylabel("Rejection Proportion (p)", fontsize=10)
    ax.set_xlabel("Subgroup (Date)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["primary"], pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # ── Summary annotation box ─────────────────────────────────────────────────
    p_bar_pct = pc_df["p_bar"].iloc[0] * 100
    ucl_pct   = pc_df["UCL"].mean() * 100
    lcl_pct   = pc_df["LCL"].mean() * 100
    ooc_count = int(pc_df["OOC"].sum())

    summary_txt = (
        f"p̄ = {p_bar_pct:.2f}%\n"
        f"UCL ≈ {ucl_pct:.2f}%\n"
        f"LCL ≈ {lcl_pct:.2f}%\n"
        f"OOC points: {ooc_count}"
    )
    ax.text(
        0.01, 0.97, summary_txt,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor=PALETTE["grid"])
    )

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# OOC SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def get_ooc_summary(pc_df: pd.DataFrame) -> pd.DataFrame:
    """Return a filtered table of out-of-control subgroups with details."""
    ooc = pc_df[pc_df["OOC"]].copy()
    ooc["p_pct"]   = (ooc["p"]   * 100).round(2)
    ooc["UCL_pct"] = (ooc["UCL"] * 100).round(2)
    ooc["LCL_pct"] = (ooc["LCL"] * 100).round(2)
    return ooc[["group", "n", "d", "p_pct", "p_bar", "UCL_pct", "LCL_pct", "OOC_Reason"]].rename(
        columns={"group": "Date", "n": "Produced", "d": "Rejected",
                 "p_pct": "Rate(%)", "p_bar": "CL", "UCL_pct": "UCL(%)", "LCL_pct": "LCL(%)"}
    )
