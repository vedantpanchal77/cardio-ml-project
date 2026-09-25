"""Serving layer: loads the trained artifacts once and answers the API's questions.

Flask (app.py) only handles HTTP. Everything that involves the model lives here,
so it can be unit-tested without starting a web server.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import metrics as M
from .logistic_regression import LogisticRegressionScratch
from .preprocessing import FEATURES, FEATURE_LABELS, Standardizer, engineer_features, feature_matrix
from .training import MODEL_NAMES, PRIMARY, artifacts_present, train_all

# name -> (kind, allowed values or (low, high)).  These are HARD limits that reject nonsense such as
# a 900 kg patient. Values that are merely outside what the model saw while training are accepted
# but flagged with a warning (see ``out_of_range`` in predict()).
INPUT_FIELDS = {
    "age": ("number", (18, 100)),          # years
    "gender": ("choice", (1, 2)),          # 1 = woman, 2 = man (the dataset's coding)
    "height": ("number", (100, 250)),      # cm
    "weight": ("number", (20, 300)),       # kg
    "ap_hi": ("number", (60, 260)),        # systolic mmHg
    "ap_lo": ("number", (30, 200)),        # diastolic mmHg
    "cholesterol": ("choice", (1, 2, 3)),  # 1 normal, 2 above normal, 3 well above
    "gluc": ("choice", (1, 2, 3)),
    "smoke": ("choice", (0, 1)),
    "alco": ("choice", (0, 1)),
    "active": ("choice", (0, 1)),
}


class ValidationError(ValueError):
    def __init__(self, errors: dict):
        super().__init__("invalid input")
        self.errors = errors


def validate_payload(payload) -> dict:
    """Check a JSON body from the browser; return clean floats or raise ValidationError."""
    if not isinstance(payload, dict):
        raise ValidationError({"body": "expected a JSON object"})
    errors, values = {}, {}
    for name, (kind, spec) in INPUT_FIELDS.items():
        if name not in payload:
            errors[name] = "missing"
            continue
        try:
            v = float(payload[name])
        except (TypeError, ValueError):
            errors[name] = "must be a number"
            continue
        if not math.isfinite(v):
            errors[name] = "must be a finite number"
        elif kind == "number" and not spec[0] <= v <= spec[1]:
            errors[name] = f"must be between {spec[0]} and {spec[1]}"
        elif kind == "choice" and v not in spec:
            errors[name] = "must be one of " + ", ".join(str(s) for s in spec)
        else:
            values[name] = v
    if not errors and values["ap_lo"] >= values["ap_hi"]:
        errors["ap_lo"] = "diastolic pressure must be lower than systolic pressure"
    if errors:
        raise ValidationError(errors)
    return values


def parse_threshold(value, default: float = 0.5) -> float:
    if value is None or value == "":
        return default
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValidationError({"threshold": "must be a number"})
    if not (math.isfinite(t) and 0.0 <= t <= 1.0):
        raise ValidationError({"threshold": "must be between 0 and 1"})
    return t


class ModelService:
    def __init__(self, models_dir: str | Path):
        d = Path(models_dir)
        self.report = json.loads((d / "report.json").read_text())
        doc = json.loads((d / "logreg_scratch.json").read_text())
        self.doc = doc
        self.model = LogisticRegressionScratch.from_dict(doc["model"])
        self.scaler = Standardizer.from_dict(doc["standardizer"])
        
        try:
            self.sk_models = joblib.load(d / "sklearn_models.joblib")
        except Exception:
            self.sk_models = {}

        try:
            with np.load(d / "test_scores.npz") as z:
                self.y_test = z["y_true"]
                self.test_scores = {k: z[k] for k in z.files if k != "y_true"}
        except Exception:
            self.y_test = np.array([])
            self.test_scores = {}

        if len(self.y_test) > 0:
            self._auc = {k: M.roc_auc(self.y_test, s) for k, s in self.test_scores.items()}
            self._roc = {k: M.roc_points(self.y_test, s) for k, s in self.test_scores.items()}
        else:
            self._auc = {}
            self._roc = {}

    # ---------------------------------------------------------------- loading
    @classmethod
    def load_or_train(cls, models_dir: str | Path, data_path: str | Path, verbose: bool = True):
        """Load saved artifacts; train first if they are missing."""
        models_dir = Path(models_dir)
        if artifacts_present(models_dir):
            try:
                return cls(models_dir)
            except Exception as exc:  # corrupt or unreadable artifacts
                if verbose:
                    print(f"Could not load saved models ({exc!r})")
        try:
            train_all(data_path, models_dir, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"Training skipped or read-only filesystem ({exc!r})")
        return cls(models_dir)



    # ------------------------------------------------------------ information
    def _name(self, key: str) -> str:
        return MODEL_NAMES[key][0]

    def summary(self) -> dict:
        r = self.report
        return {"dataset": r["dataset"], "split": r["split"], "trained_at": r["trained_at"]}

    def model_info(self) -> dict:
        m, s = self.model, self.scaler
        features = [
            {"key": k, "label": FEATURE_LABELS[k], "weight": float(m.weights[i]),
             "mean": float(s.mean[i]), "std": float(s.std[i])}
            for i, k in enumerate(FEATURES)
        ]
        return {
            "key": PRIMARY,
            "name": self._name(PRIMARY),
            "hyperparameters": {"learning_rate": m.learning_rate, "epochs": m.epochs, "l2": m.l2},
            "bias": float(m.bias),
            "features": features,
            "loss_history": [round(v, 6) for v in m.loss_history],
            "training_seconds": self.doc["training_seconds"],
            "trained_at": self.doc["trained_at"],
            "train_rows": self.report["split"]["train_rows"],
            "test_rows": self.report["split"]["test_rows"],
            "cross_validation": self.report["cross_validation"],
        }

    def compare(self) -> dict:
        rows = self.report["models"]
        best = max(rows, key=lambda r: r["accuracy"])["key"]
        return {"models": rows, "best_accuracy_key": best,
                "cross_validation": self.report["cross_validation"],
                "sklearn_check": self.report["sklearn_check"]}

    def _resolve(self, key: str | None) -> str:
        key = key or PRIMARY
        if key not in self.test_scores:
            raise ValidationError({"model": "unknown model; choose one of " + ", ".join(self.test_scores)})
        return key

    def evaluate(self, threshold: float, model_key: str | None = None) -> dict:
        key = self._resolve(model_key)
        out = M.classification_metrics(self.y_test, self.test_scores[key], threshold)
        out.update({"model": key, "name": self._name(key), "auc": self._auc[key]})
        return out

    def roc(self, model_key: str | None = None) -> dict:
        key = self._resolve(model_key)
        return {"model": key, "name": self._name(key), "auc": self._auc[key], **self._roc[key]}

    # -------------------------------------------------------------- inference
    def predict(self, payload: dict) -> dict:
        threshold = parse_threshold(payload.get("threshold") if isinstance(payload, dict) else None)
        v = validate_payload(payload)

        # Same feature code path as training (train/serve consistency).
        frame = pd.DataFrame([{**v, "age_years": v["age"]}])
        x = feature_matrix(engineer_features(frame))            # shape (1, 12)
        z = self.scaler.transform(x)[0]                          # standardised with TRAINING mean/std
        contributions = self.model.weights * z                   # w_j * z_j : each feature's push on the log-odds
        logit = float(self.model.bias + contributions.sum())
        p = float(self.model.predict_proba(z[None, :])[0])

        lo, hi = np.array(self.doc["feature_min"]), np.array(self.doc["feature_max"])
        out_of_range = [
            {"feature": k, "label": FEATURE_LABELS[k], "value": round(float(x[0, i]), 2),
             "min": round(float(lo[i]), 2), "max": round(float(hi[i]), 2)}
            for i, k in enumerate(FEATURES) if x[0, i] < lo[i] - 1e-9 or x[0, i] > hi[i] + 1e-9
        ]
        parts = sorted(
            ({"feature": k, "label": FEATURE_LABELS[k], "value": round(float(x[0, i]), 2),
              "z": round(float(z[i]), 4), "weight": round(float(self.model.weights[i]), 4),
              "contribution": round(float(contributions[i]), 4)}
             for i, k in enumerate(FEATURES)),
            key=lambda d: abs(d["contribution"]), reverse=True)

        if p < 0.35:
            band = {"key": "low", "text": "the model leans healthy"}
        elif p < 0.65:
            band = {"key": "mid", "text": "close to the decision boundary"}
        else:
            band = {"key": "high", "text": "the model leans towards disease"}

        others = [
            {"key": k, "name": self._name(k), "probability": round(float(pipe.predict_proba(x)[0, 1]), 4)}
            for k, pipe in self.sk_models.items()
        ]
        return {
            "model": PRIMARY,
            "probability": p,
            "logit": logit,
            "intercept": float(self.model.bias),
            "threshold": threshold,
            "label": int(p >= threshold),
            "band": band,
            "bmi": round(float(x[0, FEATURES.index("bmi")]), 2),
            "contributions": parts,
            "out_of_range": out_of_range,
            "other_models": others,
        }
