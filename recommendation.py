"""
recommendation.py
-----------------
SMART-QA Lite – Rule-Based Recommendation Engine

Generates plain-language, actionable recommendations based on:
  • Machine-level rejection rates
  • Operator-level rejection rates
  • SPC out-of-control signals
  • Process parameter correlations
  • Pareto vital-few defect types
  • Material batch quality

All rules are transparent and interpretable — no black-box logic.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "rejection_rate_high":    8.0,   # % — trigger machine/operator alert
    "rejection_rate_medium":  5.0,   # % — watch threshold
    "corr_strong":            0.5,   # |r| — strong predictor
    "corr_moderate":          0.3,   # |r| — moderate predictor
    "ooc_min_points":         2,     # ≥ N OOC pts triggers SPC alert
    "pareto_vital_few":       80.0,  # cumulative % threshold
}

PRIORITY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "🔵"}


@dataclass
class Recommendation:
    category:    str          # Machine / Operator / SPC / Parameter / Defect / Batch
    priority:    str          # HIGH / MEDIUM / LOW / INFO
    subject:     str          # e.g.  "Machine M1"
    finding:     str          # What was observed
    action:      str          # What to do
    metric:      str = ""     # Supporting metric string

    def to_dict(self) -> dict:
        return {
            "Priority":   f"{PRIORITY_EMOJI[self.priority]} {self.priority}",
            "Category":   self.category,
            "Subject":    self.subject,
            "Finding":    self.finding,
            "Action":     self.action,
            "Metric":     self.metric,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(
    df: pd.DataFrame,
    rejection_summary: dict,
    pareto_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    pc_df: pd.DataFrame,
    reg_result: dict,
) -> List[Recommendation]:
    """
    Master function — runs all rule sets and returns a list of Recommendations.
    """
    recs: List[Recommendation] = []

    recs += _machine_rules(rejection_summary["machine"])
    recs += _operator_rules(rejection_summary["operator"])
    recs += _spc_rules(pc_df)
    recs += _pareto_rules(pareto_df)
    recs += _correlation_rules(corr_matrix)
    recs += _batch_rules(df)
    recs += _shift_rules(df)

    # Sort by priority weight
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    recs.sort(key=lambda r: priority_order[r.priority])

    return recs


# ── Machine rules ──────────────────────────────────────────────────────────────

def _machine_rules(machine_df: pd.DataFrame) -> List[Recommendation]:
    recs = []
    overall_avg = machine_df["Rejection_Rate_Pct"].mean()

    for _, row in machine_df.iterrows():
        mid  = row["Machine_ID"]
        rate = row["Rejection_Rate_Pct"]

        if rate >= THRESHOLDS["rejection_rate_high"]:
            recs.append(Recommendation(
                category = "Machine",
                priority = "HIGH",
                subject  = f"Machine {mid}",
                finding  = f"Rejection rate is {rate:.1f}% — exceeds high threshold ({THRESHOLDS['rejection_rate_high']}%).",
                action   = (
                    "1. Schedule immediate preventive maintenance.\n"
                    "2. Inspect tooling, fixtures, and alignment.\n"
                    "3. Review last maintenance log.\n"
                    "4. Monitor next 3 shifts closely."
                ),
                metric   = f"Rate: {rate:.1f}% | Threshold: {THRESHOLDS['rejection_rate_high']}%"
            ))
        elif rate >= THRESHOLDS["rejection_rate_medium"]:
            recs.append(Recommendation(
                category = "Machine",
                priority = "MEDIUM",
                subject  = f"Machine {mid}",
                finding  = f"Rejection rate is {rate:.1f}% — above medium threshold.",
                action   = (
                    "1. Add to next scheduled maintenance.\n"
                    "2. Check process parameters for drift.\n"
                    "3. Compare output against a well-performing machine."
                ),
                metric   = f"Rate: {rate:.1f}% | Avg: {overall_avg:.1f}%"
            ))
        else:
            recs.append(Recommendation(
                category = "Machine",
                priority = "LOW",
                subject  = f"Machine {mid}",
                finding  = f"Rejection rate is {rate:.1f}% — within acceptable limits.",
                action   = "Continue routine monitoring. No action required.",
                metric   = f"Rate: {rate:.1f}%"
            ))
    return recs


# ── Operator rules ─────────────────────────────────────────────────────────────

def _operator_rules(operator_df: pd.DataFrame) -> List[Recommendation]:
    recs = []
    avg = operator_df["Rejection_Rate_Pct"].mean()
    max_rate = operator_df["Rejection_Rate_Pct"].max()

    for _, row in operator_df.iterrows():
        oid  = row["Operator_ID"]
        rate = row["Rejection_Rate_Pct"]

        # Flag operators >1.5× the group average
        if rate >= THRESHOLDS["rejection_rate_high"] or rate > avg * 1.5:
            recs.append(Recommendation(
                category = "Operator",
                priority = "HIGH" if rate >= THRESHOLDS["rejection_rate_high"] else "MEDIUM",
                subject  = f"Operator {oid}",
                finding  = (
                    f"Rejection rate {rate:.1f}% is "
                    f"{'above high threshold' if rate >= THRESHOLDS['rejection_rate_high'] else '50% above group average'}."
                ),
                action   = (
                    "1. Arrange targeted skills retraining.\n"
                    "2. Pair with a top-performing operator for knowledge transfer.\n"
                    "3. Review Standard Operating Procedure compliance.\n"
                    "4. Investigate if workload or shift timing is a factor."
                ),
                metric   = f"Rate: {rate:.1f}% | Group Avg: {avg:.1f}%"
            ))
    return recs


# ── SPC rules ─────────────────────────────────────────────────────────────────

def _spc_rules(pc_df: pd.DataFrame) -> List[Recommendation]:
    recs = []
    ooc_count = int(pc_df["OOC"].sum())

    if ooc_count == 0:
        recs.append(Recommendation(
            category = "SPC",
            priority = "INFO",
            subject  = "Process Stability",
            finding  = "All subgroups are within control limits. Process is stable.",
            action   = "No immediate SPC action required. Continue monitoring.",
            metric   = "OOC points: 0"
        ))
        return recs

    # Summarise OOC reasons
    ooc_reasons = pc_df[pc_df["OOC"]]["OOC_Reason"].value_counts().to_dict()
    reason_str  = ", ".join([f"{r}: {c}×" for r, c in ooc_reasons.items()])

    priority = "HIGH" if ooc_count >= THRESHOLDS["ooc_min_points"] else "MEDIUM"

    recs.append(Recommendation(
        category = "SPC",
        priority = priority,
        subject  = "Process Stability – p-Chart",
        finding  = (
            f"{ooc_count} out-of-control point(s) detected.\n"
            f"Reasons: {reason_str}"
        ),
        action   = (
            "1. Investigate root cause for each flagged date.\n"
            "2. Check for machine setup changes, operator changes, or material issues.\n"
            "3. If 'Run of 8' detected → look for systematic process shift.\n"
            "4. If 'Trend of 6' detected → investigate gradual drift (tool wear, temperature).\n"
            "5. Document corrective actions and re-sample."
        ),
        metric   = f"OOC points: {ooc_count} of {len(pc_df)}"
    ))
    return recs


# ── Pareto rules ───────────────────────────────────────────────────────────────

def _pareto_rules(pareto_df: pd.DataFrame) -> List[Recommendation]:
    recs = []
    vital = pareto_df[pareto_df["Vital_Few"]]["Defect_Type"].tolist()

    defect_actions = {
        "Porosity":       "Check raw material moisture, melting temperature, and degassing procedure.",
        "Dimensional":    "Verify tooling wear, fixture alignment, and machine calibration.",
        "Surface Finish": "Inspect cutting speed, feed rate, coolant flow, and tool condition.",
        "Crack":          "Review thermal cycle, material composition, and cooling rate.",
        "Burr":           "Check cutting tool sharpness and feed parameters.",
        "Warpage":        "Audit clamping pressure, material handling, and cooling uniformity.",
    }

    for defect in vital:
        contrib = pareto_df[pareto_df["Defect_Type"] == defect]["Pct_Contribution"].values[0]
        default_action = (
            "Convene a quality team root-cause analysis session for this defect type."
        )
        action = defect_actions.get(defect, default_action)

        recs.append(Recommendation(
            category = "Defect / Pareto",
            priority = "HIGH" if contrib >= 30 else "MEDIUM",
            subject  = f"Defect: {defect}",
            finding  = (
                f"'{defect}' is a VITAL FEW defect contributing {contrib:.1f}% of total rejections."
            ),
            action   = f"Priority reduction target.\n{action}",
            metric   = f"Contribution: {contrib:.1f}%"
        ))
    return recs


# ── Correlation / parameter rules ─────────────────────────────────────────────

def _correlation_rules(corr_matrix: pd.DataFrame) -> List[Recommendation]:
    recs = []
    param_corrs = corr_matrix["Rejection_Rate"].drop("Rejection_Rate")

    param_guidance = {
        "Temperature": "Implement tighter temperature control / PID tuning. Review thermocouple calibration.",
        "Pressure":    "Check pressure regulators, seals, and hydraulic system for drift.",
        "Speed":       "Optimise feed/speed parameters. Consider DOE (Design of Experiments) to find optimal range.",
    }

    for param, corr in param_corrs.items():
        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative"

        if abs_corr >= THRESHOLDS["corr_strong"]:
            recs.append(Recommendation(
                category = "Process Parameter",
                priority = "HIGH",
                subject  = f"Parameter: {param}",
                finding  = (
                    f"{param} has a STRONG {direction} correlation with rejection rate (r = {corr:.2f}).\n"
                    f"{'Higher' if corr > 0 else 'Lower'} {param} → higher rejections."
                ),
                action   = (
                    f"1. {param_guidance.get(param, 'Investigate and control this parameter.')}\n"
                    "2. Define and enforce SPC control limits for this parameter.\n"
                    "3. Log parameter values per shift for trend monitoring."
                ),
                metric   = f"Pearson r = {corr:.3f}"
            ))
        elif abs_corr >= THRESHOLDS["corr_moderate"]:
            recs.append(Recommendation(
                category = "Process Parameter",
                priority = "MEDIUM",
                subject  = f"Parameter: {param}",
                finding  = (
                    f"{param} shows a MODERATE {direction} correlation (r = {corr:.2f})."
                ),
                action   = (
                    f"1. Monitor {param} closely and log deviations.\n"
                    "2. Investigate in combination with other high-correlation parameters."
                ),
                metric   = f"Pearson r = {corr:.3f}"
            ))
    return recs


# ── Material batch rules ───────────────────────────────────────────────────────

def _batch_rules(df: pd.DataFrame) -> List[Recommendation]:
    recs = []
    batch_rate = (
        df.groupby("Material_Batch")
        .apply(lambda g: (g["Rejected_Qty"].sum() / g["Produced_Qty"].sum()) * 100)
        .reset_index(name="Rate")
        .sort_values("Rate", ascending=False)
    )
    overall_avg = (df["Rejected_Qty"].sum() / df["Produced_Qty"].sum()) * 100

    for _, row in batch_rate.iterrows():
        if row["Rate"] > overall_avg * 1.3:  # 30% above average
            recs.append(Recommendation(
                category = "Material Batch",
                priority = "MEDIUM",
                subject  = f"Batch {row['Material_Batch']}",
                finding  = (
                    f"Batch {row['Material_Batch']} has rejection rate {row['Rate']:.1f}%, "
                    f"which is {row['Rate'] - overall_avg:.1f}% above average."
                ),
                action   = (
                    "1. Quarantine remaining stock from this batch for re-inspection.\n"
                    "2. Raise a supplier quality concern.\n"
                    "3. Request material test certificates.\n"
                    "4. Compare with accepted batches for physical/chemical properties."
                ),
                metric   = f"Batch Rate: {row['Rate']:.1f}% | Avg: {overall_avg:.1f}%"
            ))
    return recs


# ── Shift rules ────────────────────────────────────────────────────────────────

def _shift_rules(df: pd.DataFrame) -> List[Recommendation]:
    recs = []
    shift_rate = (
        df.groupby("Shift")
        .apply(lambda g: (g["Rejected_Qty"].sum() / g["Produced_Qty"].sum()) * 100)
        .reset_index(name="Rate")
    )
    max_shift = shift_rate.loc[shift_rate["Rate"].idxmax()]
    min_shift = shift_rate.loc[shift_rate["Rate"].idxmin()]
    spread = max_shift["Rate"] - min_shift["Rate"]

    if spread >= 3.0:   # >3 percentage point spread across shifts
        recs.append(Recommendation(
            category = "Shift Analysis",
            priority = "MEDIUM",
            subject  = f"Shift Variation ({max_shift['Shift']} vs {min_shift['Shift']})",
            finding  = (
                f"'{max_shift['Shift']}' shift has {max_shift['Rate']:.1f}% rejection rate "
                f"vs '{min_shift['Shift']}' shift at {min_shift['Rate']:.1f}%. "
                f"Spread = {spread:.1f} percentage points."
            ),
            action   = (
                "1. Audit shift handover procedures.\n"
                "2. Check if night/weekend shifts have different supervision levels.\n"
                "3. Standardise pre-shift machine setup checks.\n"
                "4. Compare SOPs adherence across shifts."
            ),
            metric   = f"Max: {max_shift['Rate']:.1f}% | Min: {min_shift['Rate']:.1f}% | Spread: {spread:.1f}%"
        ))
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def recommendations_to_df(recs: List[Recommendation]) -> pd.DataFrame:
    """Convert recommendation list to a DataFrame for display."""
    return pd.DataFrame([r.to_dict() for r in recs])


def get_summary_counts(recs: List[Recommendation]) -> dict:
    """Count recommendations by priority."""
    from collections import Counter
    counts = Counter(r.priority for r in recs)
    return {"HIGH": counts.get("HIGH", 0), "MEDIUM": counts.get("MEDIUM", 0),
            "LOW": counts.get("LOW", 0), "INFO": counts.get("INFO", 0)}
