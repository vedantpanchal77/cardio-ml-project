"""Unit tests. Run with:  python -m pytest tests/ -q"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

from ml import metrics as M
from ml.logistic_regression import LogisticRegressionScratch
from ml.preprocessing import Standardizer, clean, engineer_features


# --------------------------------------------------------------- preprocessing
def test_engineer_features_bmi():
    df = pd.DataFrame({"height": [200.0], "weight": [80.0]})
    out = engineer_features(df)
    assert out["bmi"].iloc[0] == pytest.approx(20.0)


def test_clean_removes_impossible_rows():
    df = pd.DataFrame({
        "ap_hi": [120, 5000], "ap_lo": [80, 90], "height": [170, 170],
        "weight": [70, 70], "bmi": [24.2, 24.2],
    })
    cleaned, report = clean(df)
    assert len(cleaned) == 1
    assert any(r["rows_failing"] >= 1 for r in report)


def test_clean_enforces_diastolic_below_systolic():
    df = pd.DataFrame({"ap_hi": [110], "ap_lo": [120], "height": [170], "weight": [70], "bmi": [24.2]})
    cleaned, _ = clean(df)
    assert len(cleaned) == 0


def test_standardizer_zero_mean_unit_variance():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    s = Standardizer().fit(X)
    Z = s.transform(X)
    assert np.allclose(Z.mean(axis=0), [0, 0], atol=1e-9)
    assert np.allclose(Z.std(axis=0), [1, 1], atol=1e-9)


def test_standardizer_handles_constant_column():
    X = np.array([[5.0], [5.0], [5.0]])
    s = Standardizer().fit(X)
    assert not np.isnan(s.transform(X)).any()


def test_standardizer_round_trip_dict():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    s = Standardizer().fit(X)
    s2 = Standardizer.from_dict(s.to_dict())
    assert np.allclose(s.transform(X), s2.transform(X))


# --------------------------------------------------------- logistic regression
def test_scratch_logreg_separates_linearly_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (200, 2)), rng.normal(2, 0.5, (200, 2))])
    y = np.array([0] * 200 + [1] * 200)
    model = LogisticRegressionScratch(learning_rate=0.5, epochs=500, l2=0.0).fit(X, y)
    acc = (model.predict(X) == y).mean()
    assert acc > 0.95


def test_scratch_logreg_loss_decreases_monotonically_on_average():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = (rng.random(300) > 0.5).astype(float)
    model = LogisticRegressionScratch(learning_rate=0.1, epochs=200).fit(X, y)
    assert model.loss_history[-1] < model.loss_history[0]


def test_scratch_logreg_matches_sklearn_closely():
    sklearn = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 5))
    true_w = np.array([1.0, -0.5, 0.3, 0.0, 2.0])
    y = (1 / (1 + np.exp(-(X @ true_w))) > rng.random(2000)).astype(float)
    scaler = Standardizer().fit(X)
    Z = scaler.transform(X)
    scratch = LogisticRegressionScratch(learning_rate=0.5, epochs=2000, l2=0.001).fit(Z, y)
    sk = sklearn.LogisticRegression(C=1000, max_iter=2000).fit(Z, y)
    assert np.max(np.abs(scratch.weights - sk.coef_[0])) < 0.05
    assert np.max(np.abs(scratch.predict_proba(Z) - sk.predict_proba(Z)[:, 1])) < 0.02


def test_scratch_logreg_persistence_round_trip():
    X = np.random.default_rng(3).normal(size=(50, 3))
    y = (np.random.default_rng(4).random(50) > 0.5).astype(float)
    model = LogisticRegressionScratch(epochs=50).fit(X, y)
    restored = LogisticRegressionScratch.from_dict(model.to_dict())
    assert np.allclose(model.predict_proba(X), restored.predict_proba(X))


def test_sigmoid_is_numerically_stable_for_large_inputs():
    z = np.array([-1e6, 0.0, 1e6])
    p = LogisticRegressionScratch.sigmoid(z)
    assert np.all(np.isfinite(p))
    assert p[0] == pytest.approx(0.0, abs=1e-9)
    assert p[2] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------- metrics
def test_classification_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = M.classification_metrics(y, p, 0.5)
    assert m["accuracy"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0


def test_classification_metrics_all_wrong():
    y = np.array([0, 1])
    p = np.array([0.9, 0.1])
    m = M.classification_metrics(y, p, 0.5)
    assert m["accuracy"] == 0.0


def test_roc_auc_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    y = (rng.random(500) > 0.5).astype(int)
    p = np.clip(y + rng.normal(0, 0.4, 500), 0, 1)
    mine = M.roc_auc(y, p)
    theirs = sklearn_metrics.roc_auc_score(y, p)
    assert mine == pytest.approx(theirs, abs=1e-9)


def test_roc_auc_random_scores_near_half():
    rng = np.random.default_rng(6)
    y = (rng.random(5000) > 0.5).astype(int)
    p = rng.random(5000)
    assert M.roc_auc(y, p) == pytest.approx(0.5, abs=0.03)


# ------------------------------------------------------------------------ app
@pytest.fixture(scope="module")
def client():
    import sys
    sys.path.insert(0, str(ROOT))
    from app import create_app
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_summary_endpoint_shape(client):
    d = client.get("/api/summary").get_json()
    assert d["dataset"]["clean_rows"] > 0
    assert d["split"]["train_rows"] + d["split"]["test_rows"] == d["dataset"]["clean_rows"]


def test_evaluate_endpoint_threshold_zero_catches_everyone(client):
    d = client.get("/api/evaluate?threshold=0").get_json()
    assert d["recall"] == pytest.approx(1.0)


def test_evaluate_rejects_bad_threshold(client):
    r = client.get("/api/evaluate?threshold=7")
    assert r.status_code == 422


def test_predict_valid_patient(client):
    body = dict(age=55, gender=1, height=165, weight=74, ap_hi=120, ap_lo=80,
                cholesterol=1, gluc=1, smoke=0, alco=0, active=1)
    r = client.post("/api/predict", json=body)
    d = r.get_json()
    assert r.status_code == 200
    assert 0.0 <= d["probability"] <= 1.0
    assert len(d["contributions"]) == 12
    assert len(d["other_models"]) >= 5


def test_predict_rejects_missing_field(client):
    body = dict(age=55, gender=1, height=165, weight=74, ap_hi=120, ap_lo=80,
                cholesterol=1, gluc=1, smoke=0, alco=0)  # 'active' missing
    r = client.post("/api/predict", json=body)
    assert r.status_code == 422
    assert "active" in r.get_json()["details"]


def test_predict_rejects_diastolic_above_systolic(client):
    body = dict(age=55, gender=1, height=165, weight=74, ap_hi=110, ap_lo=120,
                cholesterol=1, gluc=1, smoke=0, alco=0, active=1)
    r = client.post("/api/predict", json=body)
    assert r.status_code == 422
    assert "ap_lo" in r.get_json()["details"]


def test_predict_flags_out_of_training_range(client):
    body = dict(age=95, gender=1, height=165, weight=74, ap_hi=250, ap_lo=80,
                cholesterol=1, gluc=1, smoke=0, alco=0, active=1)
    r = client.post("/api/predict", json=body)
    assert r.status_code in (200, 422)  # 250 systolic is inside the hard limit but outside training range
    if r.status_code == 200:
        assert len(r.get_json()["out_of_range"]) > 0


def test_predict_rejects_non_json_body(client):
    r = client.post("/api/predict", data="not json")
    assert r.status_code == 400


def test_unknown_route_returns_json_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.get_json()["error"]


def test_compare_endpoint_has_every_model(client):
    d = client.get("/api/compare").get_json()
    keys = {m["key"] for m in d["models"]}
    assert {"logreg_scratch", "logreg_sklearn", "random_forest"}.issubset(keys)
