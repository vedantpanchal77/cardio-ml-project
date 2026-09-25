"""Loading, cleaning and feature engineering for the cardiovascular dataset.

The same ``engineer_features`` function is used while training and inside the
live ``/api/predict`` endpoint, so a patient typed into the web page goes
through exactly the same transformation as a row from the CSV. That avoids
"train/serve skew", a classic reason for a model that looks good in a notebook
and misbehaves in an application.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "cardio"
DAYS_PER_YEAR = 365.25

# Order matters: the weight vector of the model lines up with this list.
FEATURES = [
    "age_years", "gender", "height", "weight", "bmi", "ap_hi",
    "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active",
]

FEATURE_LABELS = {
    "age_years": "Age",
    "gender": "Sex",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
    "ap_hi": "Systolic BP",
    "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol",
    "gluc": "Glucose",
    "smoke": "Smoking",
    "alco": "Alcohol",
    "active": "Activity",
}

# Single source of truth for the cleaning step. The API and the web page read
# this list too, so what the page shows is what the code really does.
CLEANING_RULES = [
    {"key": "ap_hi", "label": "Systolic (ap_hi)", "low": 90, "high": 200, "unit": "mmHg",
     "why": "Outside this range it is a typing error, not a patient"},
    {"key": "ap_lo", "label": "Diastolic (ap_lo)", "low": 60, "high": 140, "unit": "mmHg",
     "why": "Same idea: negative and five-digit readings exist in the file"},
    {"key": "bp_order", "label": "Diastolic below systolic", "low": None, "high": None, "unit": "",
     "why": "A diastolic reading can never reach the systolic one"},
    {"key": "height", "label": "Height", "low": 140, "high": 200, "unit": "cm",
     "why": "Removes 55 cm and 250 cm entries"},
    {"key": "weight", "label": "Weight", "low": 40, "high": 160, "unit": "kg",
     "why": "Removes 10 kg adults"},
    {"key": "bmi", "label": "BMI", "low": 15, "high": 50, "unit": "",
     "why": "Guards against impossible height and weight pairs"},
]


# --------------------------------------------------------------------------
# Loading and feature engineering
# --------------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the Kaggle file (semicolon separated) and convert age from days to years."""
    df = pd.read_csv(path, sep=";")
    df["age_years"] = df["age"] / DAYS_PER_YEAR
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived BMI column. Works on any frame with height and weight."""
    out = df.copy()
    out["bmi"] = out["weight"] / (out["height"] / 100.0) ** 2
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the model input as a float array in the fixed FEATURES order."""
    return df[FEATURES].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def _rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean mask per rule: True where the row FAILS the rule."""
    return {
        "ap_hi": ~df["ap_hi"].between(90, 200),
        "ap_lo": ~df["ap_lo"].between(60, 140),
        "bp_order": df["ap_lo"] >= df["ap_hi"],
        "height": ~df["height"].between(140, 200),
        "weight": ~df["weight"].between(40, 160),
        "bmi": ~df["bmi"].between(15, 50),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drop physically impossible rows.

    Returns the cleaned frame and, for each rule, how many rows failed it.
    A row can fail several rules, so the per-rule counts overlap and do not
    add up to the total number of removed rows.
    """
    masks = _rule_masks(df)
    bad = np.zeros(len(df), dtype=bool)
    report = []
    for rule in CLEANING_RULES:
        failing = masks[rule["key"]].to_numpy()
        bad |= failing
        report.append({**rule, "rows_failing": int(failing.sum())})
    return df.loc[~bad].reset_index(drop=True), report


def load_clean_dataset(path: str | Path):
    """Full pipeline used by training: raw CSV -> cleaned frame + cleaning report."""
    raw = load_raw(path)
    engineered = engineer_features(raw)
    cleaned, rule_report = clean(engineered)
    return raw, cleaned, rule_report


# --------------------------------------------------------------------------
# Standardisation, written by hand
# --------------------------------------------------------------------------
class Standardizer:
    """z = (x - mean) / std, with mean and std learned from the training rows only.

    Fitting on training rows only matters: computing them on the full dataset
    would let information about the test set leak into training.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)  # constant column -> avoid dividing by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Standardizer":
        obj = cls()
        obj.mean = np.asarray(data["mean"], dtype=float)
        obj.std = np.asarray(data["std"], dtype=float)
        return obj
