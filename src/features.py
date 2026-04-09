from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "incident_date" in out.columns:
        incident_dt = pd.to_datetime(out["incident_date"], errors="coerce")
        out["incident_month"] = incident_dt.dt.month.fillna(0).astype(int)
        out["incident_day_of_week"] = incident_dt.dt.dayofweek.fillna(0).astype(int)
        out["incident_is_weekend"] = incident_dt.dt.dayofweek.isin([5, 6]).astype(int)

    if {"total_claim_amount", "policy_annual_premium"}.issubset(out.columns):
        premium = out["policy_annual_premium"].replace(0, np.nan)
        out["claim_to_premium_ratio"] = (out["total_claim_amount"] / premium).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "total_claim_amount" in out.columns and "claim_severity" not in out.columns:
        out["claim_severity"] = pd.cut(
            out["total_claim_amount"],
            bins=3,
            labels=[0, 1, 2],
        )

    if "number_of_vehicles_involved" in out.columns:
        out["multi_vehicle"] = (out["number_of_vehicles_involved"] > 1).astype(int)

    return out
