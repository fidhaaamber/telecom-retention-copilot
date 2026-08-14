from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COLUMN = "CustomerID"


def checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, na_values=["", " "])
    data[TARGET] = data[TARGET].map({"Yes": 1, "No": 0}).astype("Int64")
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create only contemporaneous features; no outcomes or post-contact fields."""
    df = data.copy()
    num = lambda name: pd.to_numeric(df.get(name, 0), errors="coerce").fillna(0)
    minutes = num("MonthlyMinutes").clip(lower=0)
    subs = num("UniqueSubs").clip(lower=1)
    df["RevenuePerMinute"] = num("MonthlyRevenue") / minutes.replace(0, np.nan)
    df["OverageRatio"] = num("OverageMinutes") / minutes.replace(0, np.nan)
    df["DroppedBlockedRate"] = (num("DroppedCalls") + num("BlockedCalls")) / (num("PeakCallsInOut") + num("OffPeakCallsInOut")).replace(0, np.nan)
    df["CareCallIntensity"] = num("CustomerCareCalls") / (minutes / 100 + 1)
    df["InactiveSubscriptionRatio"] = (subs - num("ActiveSubs").clip(lower=0)) / subs
    df["TenureBand"] = pd.cut(num("MonthsInService"), [-1, 6, 12, 24, 60, np.inf], labels=["0-6", "7-12", "13-24", "25-60", "60+"]).astype(str)
    df["EquipmentAgeBand"] = pd.cut(num("CurrentEquipmentDays"), [-1, 180, 365, 730, np.inf], labels=["0-6m", "6-12m", "1-2y", "2y+"]).astype(str)
    df["UsageChangeAbs"] = num("PercChangeMinutes").abs()
    # Additional leakage-safe behavioural ratios. Small denominators are guarded
    # so an inactive account is not mistaken for a data-quality extreme.
    total_calls = (num("OutboundCalls") + num("InboundCalls") + num("ReceivedCalls")).clip(lower=0)
    total_attempts = (total_calls + num("DroppedCalls") + num("BlockedCalls") + num("UnansweredCalls")).clip(lower=0)
    df["TotalCallVolume"] = total_calls
    df["ServiceFailureRate"] = (num("DroppedCalls") + num("BlockedCalls")) / total_attempts.replace(0, np.nan)
    df["UnansweredRate"] = num("UnansweredCalls") / total_attempts.replace(0, np.nan)
    df["CareCallsPer100Calls"] = 100 * num("CustomerCareCalls") / total_attempts.replace(0, np.nan)
    df["RoamingCallRatio"] = num("RoamingCalls") / total_attempts.replace(0, np.nan)
    df["RevenueToRecurringRatio"] = num("MonthlyRevenue") / num("TotalRecurringCharge").replace(0, np.nan)
    df["RevenueChangeRatio"] = num("PercChangeRevenues") / num("MonthlyRevenue").abs().replace(0, np.nan)
    df["MinutesPerSubscription"] = minutes / subs
    df["HandsetsPerSubscription"] = num("Handsets") / subs
    df["ActiveSubscriptionRatio"] = num("ActiveSubs").clip(lower=0) / subs
    df["EquipmentAgeToTenure"] = num("CurrentEquipmentDays") / (num("MonthsInService").clip(lower=0) * 30 + 1)
    df["TenureLog"] = np.log1p(num("MonthsInService").clip(lower=0))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    # Explicitly exclude target, identifier, and post-risk retention outcomes (leakage).
    leakage = {TARGET, ID_COLUMN, "RetentionOffersAccepted", "MadeCallToRetentionTeam", "RetentionCalls"}
    return [c for c in df.columns if c not in leakage]
