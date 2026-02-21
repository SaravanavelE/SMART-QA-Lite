"""
data_loader.py
--------------
SMART-QA Lite – Data Loading & Validation Module
Handles CSV/Excel ingestion, schema validation, and basic cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Required column schema ────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Date", "Shift", "Machine_ID", "Operator_ID",
    "Produced_Qty", "Rejected_Qty", "Defect_Type",
    "Material_Batch", "Temperature", "Pressure", "Speed",
]

NUMERIC_COLUMNS = ["Produced_Qty", "Rejected_Qty", "Temperature", "Pressure", "Speed"]


# ── Loader ─────────────────────────────────────────────────────────────────────
def load_data(filepath) -> pd.DataFrame:
    """
    Load production data from a CSV or Excel file.

    Parameters
    ----------
    filepath : str | Path | file-like object
        Path to .csv or .xlsx file, or a Streamlit UploadedFile object.

    Returns
    -------
    pd.DataFrame  – cleaned, validated production DataFrame.

    Raises
    ------
    ValueError  – if required columns are missing or data is unusable.
    """
    filepath_str = str(getattr(filepath, "name", filepath))

    # ── Read file ──────────────────────────────────────────────────────────────
    try:
        if filepath_str.endswith(".xlsx") or filepath_str.endswith(".xls"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    # ── Strip whitespace from column names ─────────────────────────────────────
    df.columns = df.columns.str.strip()

    # ── Validate required columns ──────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Parse dates ────────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        print(f"[WARNING] {invalid_dates} rows have invalid dates and will be dropped.")
        df = df.dropna(subset=["Date"])

    # ── Coerce numeric columns ─────────────────────────────────────────────────
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop rows with non-positive production qty ─────────────────────────────
    df = df[df["Produced_Qty"] > 0].copy()

    # ── Clamp Rejected_Qty: cannot exceed Produced_Qty ────────────────────────
    df["Rejected_Qty"] = df["Rejected_Qty"].clip(lower=0)
    df.loc[df["Rejected_Qty"] > df["Produced_Qty"], "Rejected_Qty"] = df["Produced_Qty"]

    # ── Fill missing numeric params with column median ─────────────────────────
    for col in ["Temperature", "Pressure", "Speed"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # ── Derive core metric ─────────────────────────────────────────────────────
    df["Rejection_Rate"] = (df["Rejected_Qty"] / df["Produced_Qty"]) * 100

    # ── Derive calendar features ───────────────────────────────────────────────
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day_of_Week"] = df["Date"].dt.day_name()

    # ── Strip string columns ───────────────────────────────────────────────────
    for col in ["Shift", "Machine_ID", "Operator_ID", "Defect_Type", "Material_Batch"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    return df


# ── Summary helper ─────────────────────────────────────────────────────────────
def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a quick summary dict for display purposes."""
    return {
        "total_rows": len(df),
        "date_range": f"{df['Date'].min().date()} → {df['Date'].max().date()}",
        "machines": sorted(df["Machine_ID"].unique().tolist()),
        "operators": sorted(df["Operator_ID"].unique().tolist()),
        "defect_types": sorted(df["Defect_Type"].unique().tolist()),
        "total_produced": int(df["Produced_Qty"].sum()),
        "total_rejected": int(df["Rejected_Qty"].sum()),
        "overall_rejection_rate": round(
            (df["Rejected_Qty"].sum() / df["Produced_Qty"].sum()) * 100, 2
        ),
    }
