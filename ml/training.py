"""Training pipeline: clean -> split -> train every model -> evaluate -> save artifacts.

Run it with ``python train.py``. The web server never trains while handling a
request; it only loads the files written here.

Artifacts written to ``models/``:
    logreg_scratch.json      weights, bias, scaler, loss curve of the from-scratch model
    sklearn_models.joblib    the comparison models (scikit-learn pipelines)
    test_scores.npz          true labels + predicted probabilities on the held-out test set
    report.json              dataset facts, cleaning report, metrics of every model, cross-validation
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import metrics as M
from .logistic_regression import LogisticRegressionScratch
from .preprocessing import (
    FEATURE_LABELS, FEATURES, TARGET, Standardizer, feature_matrix, load_clean_dataset,
)

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PRIMARY = "logreg_scratch"
SCRATCH_PARAMS = {"learning_rate": 0.1, "epochs": 1000, "l2": 0.01}

ARTIFACT_FILES = ("report.json", "logreg_scratch.json", "test_scores.npz", "sklearn_models.joblib")

# Order of rows in the comparison table.
MODEL_NAMES = {
    "logreg_scratch": ("Logistic regression (from scratch)", "NumPy only, batch gradient descent - the SOP algorithm"),
    "logreg_sklearn": ("Logistic regression (scikit-learn)", "library reference; should agree with the scratch version"),
    "naive_bayes": ("Gaussian naive Bayes", "assumes the features are independent"),
    "knn": ("k-nearest neighbours", "k = 25, euclidean distance on standardised features"),
    "decision_tree": ("Decision tree", "max depth 6, at least 50 rows per leaf"),
    "random_forest": ("Random forest", "150 trees, max depth 10"),
    "gradient_boosting": ("Gradient boosting", "150 shallow trees, each correcting the last"),
}


def _sklearn_models() -> dict:
    """Every comparison model is a Pipeline(StandardScaler, classifier), fitted on training rows only."""
    return {
        "logreg_sklearn": make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=1000)),
        "naive_bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "decision_tree": make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED)),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20,
                                   random_state=SEED, n_jobs=-1)),
        "gradient_boosting": make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                                       random_state=SEED)),
    }


def artifacts_present(models_dir: str | Path) -> bool:
    return all((Path(models_dir) / name).exists() for name in ARTIFACT_FILES)


def _cross_validate_scratch(X: np.ndarray, y: np.ndarray) -> dict:
    """Stratified k-fold on the TRAINING rows only; the test set is never touched here."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for tr, va in skf.split(X, y):
        scaler = Standardizer().fit(X[tr])
        model = LogisticRegressionScratch(**SCRATCH_PARAMS).fit(scaler.transform(X[tr]), y[tr])
        p = model.predict_proba(scaler.transform(X[va]))
        accs.append(M.classification_metrics(y[va], p)["accuracy"])
        aucs.append(M.roc_auc(y[va], p))
    return {
        "folds": CV_FOLDS,
        "fold_accuracy": [round(a, 5) for a in accs],
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }


def train_all(data_path: str | Path, models_dir: str | Path, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- 1. data
    log("1/6  loading and cleaning the dataset ...")
    raw, cleaned, rule_report = load_clean_dataset(data_path)
    X = feature_matrix(cleaned)
    y = cleaned[TARGET].to_numpy(dtype=int)
    log(f"     {len(raw):,} raw rows -> {len(cleaned):,} after cleaning "
        f"({len(raw) - len(cleaned):,} removed)")

    # ------------------------------------------------------------ 2. split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    log(f"2/6  split: {len(ytr):,} train / {len(yte):,} test (stratified, seed {SEED})")

    # ------------------------------------------- 3. logistic regression from scratch
    log("3/6  training logistic regression from scratch ...")
    scaler = Standardizer().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    t0 = time.perf_counter()
    scratch = LogisticRegressionScratch(**SCRATCH_PARAMS).fit(Ztr, ytr)
    scratch_seconds = time.perf_counter() - t0
    scratch_test = scratch.predict_proba(Zte)
    log(f"     {scratch.epochs} epochs in {scratch_seconds:.2f}s, "
        f"final loss {scratch.loss_history[-1]:.4f}")

    log(f"4/6  {CV_FOLDS}-fold cross-validation of the scratch model (training rows only) ...")
    cv = _cross_validate_scratch(Xtr, ytr)
    log(f"     accuracy {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f}")

    # ------------------------------------------------ 4. comparison models
    log("5/6  training the comparison models ...")
    test_scores = {PRIMARY: scratch_test}
    fit_seconds = {PRIMARY: scratch_seconds}
    fitted = {}
    for key, pipe in _sklearn_models().items():
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_seconds[key] = time.perf_counter() - t0
        test_scores[key] = pipe.predict_proba(Xte)[:, 1]
        fitted[key] = pipe
        log(f"     {key:<18} fitted in {fit_seconds[key]:6.2f}s")

    # ------------------------------------------------------ 5. evaluation
    rows = []
    for key, (name, note) in MODEL_NAMES.items():
        m = M.classification_metrics(yte, test_scores[key], 0.5)
        rows.append({
            "key": key, "name": name, "note": note,
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "auc": M.roc_auc(yte, test_scores[key]),
            "train_seconds": round(fit_seconds[key], 3),
        })

    # Does the hand-written model agree with the library implementation?
    sk = fitted["logreg_sklearn"]
    sk_w = sk.named_steps["logisticregression"].coef_[0]
    sk_train_p = sk.predict_proba(Xtr)[:, 1]
    scratch_train_p = scratch.predict_proba(Ztr)
    sklearn_check = {
        "max_abs_weight_diff": float(np.max(np.abs(sk_w - scratch.weights))),
        "max_abs_probability_diff": float(np.max(np.abs(test_scores["logreg_sklearn"] - scratch_test))),
        "accuracy_diff": float(abs(rows[0]["accuracy"] - rows[1]["accuracy"])),
        "train_log_loss_scratch": M.log_loss(ytr, scratch_train_p),
        "train_log_loss_sklearn": M.log_loss(ytr, sk_train_p),
    }

    # ---------------------------------------------------------- 6. save
    log("6/6  saving artifacts ...")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scratch_doc = {
        "primary_key": PRIMARY,
        "features": FEATURES,
        "feature_labels": [FEATURE_LABELS[f] for f in FEATURES],
        "standardizer": scaler.to_dict(),
        "model": scratch.to_dict(),
        # Range seen while training: used to warn when a prediction extrapolates.
        "feature_min": Xtr.min(axis=0).tolist(),
        "feature_max": Xtr.max(axis=0).tolist(),
        "training_seconds": round(scratch_seconds, 3),
        "trained_at": now,
    }
    (models_dir / "logreg_scratch.json").write_text(json.dumps(scratch_doc))
    np.savez_compressed(models_dir / "test_scores.npz", y_true=yte, **test_scores)
    joblib.dump(fitted, models_dir / "sklearn_models.joblib", compress=3)

    report = {
        "trained_at": now,
        "seed": SEED,
        "versions": {"scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "dataset": {
            "raw_rows": int(len(raw)),
            "clean_rows": int(len(cleaned)),
            "removed_rows": int(len(raw) - len(cleaned)),
            "removed_pct": float((len(raw) - len(cleaned)) / len(raw)),
            "rules": rule_report,
            "positive_rate": float(y.mean()),
            "mean_age_years": float(cleaned["age_years"].mean()),
            "mean_bmi": float(cleaned["bmi"].mean()),
            "mean_systolic": float(cleaned["ap_hi"].mean()),
            "n_features": len(FEATURES),
        },
        "split": {"train_rows": int(len(ytr)), "test_rows": int(len(yte)),
                  "test_size": TEST_SIZE, "stratified": True},
        "primary_model": PRIMARY,
        "scratch": {**{k: v for k, v in SCRATCH_PARAMS.items()},
                    "train_seconds": round(scratch_seconds, 3),
                    "final_loss": scratch.loss_history[-1]},
        "models": rows,
        "cross_validation": cv,
        "sklearn_check": sklearn_check,
    }
    (models_dir / "report.json").write_text(json.dumps(report, indent=1))
    log("done.")
    return report
