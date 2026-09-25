# Cardiovascular Disease Prediction

A machine-learning mini-project with a real backend: a Flask API trains and
serves a logistic-regression model written from scratch in NumPy, compares it
against six scikit-learn models, and a single web page drives it all.

## What changed from the single-HTML version

The original was one static HTML file that trained a model in the visitor's
browser on page load. This version moves everything that matters onto a
Python backend:

- **Training happens once, on the server** (`train.py`), not on every page
  load in JavaScript. The browser only ever sees the results.
- **The core algorithm is still written from scratch** — `ml/logistic_regression.py`
  is plain NumPy, no scikit-learn — but it is checked against scikit-learn's
  own `LogisticRegression` in `ml/training.py` and in `tests/`.
- **Six extra models** (scikit-learn logistic regression, Gaussian naive
  Bayes, k-nearest neighbours, a decision tree, a random forest, and gradient
  boosting) are trained on the same split for a fair comparison — no k-NN
  row is truncated to a small sample the way the in-browser version had to.
- **5-fold cross-validation** on the training data, not just a single split.
- **A real `/api/predict` endpoint** does the prediction, with server-side
  validation (impossible values are rejected, out-of-training-range values
  are flagged, not silently extrapolated).
- **A test suite** (`tests/test_project.py`, 26 tests) checks the cleaning
  rules, the from-scratch model against scikit-learn, the metrics against
  scikit-learn's, and every API endpoint.

## Project layout

```
app.py                  Flask app: HTTP routes only
train.py                Entry point: python train.py
ml/
  preprocessing.py       load, clean, engineer features, hand-written Standardizer
  logistic_regression.py the from-scratch model (NumPy only)
  metrics.py              accuracy/precision/recall/F1/ROC/AUC, written by hand
  training.py             the full training pipeline; trains all 7 models
  serving.py              loads artifacts, validates input, answers API questions
data/
  cardio_train.csv        the Kaggle dataset (semicolon-separated)
models/                   written by train.py: weights, scaler, metrics, comparison models
templates/index.html      the page (Jinja, served by Flask)
static/app.js             fetches from the API and draws the charts
static/style.css          the visual design
tests/test_project.py     unit + API tests
```

## Running it

```bash
pip install -r requirements.txt
python train.py     # trains all 7 models, writes models/ (~30s on a laptop)
python app.py        # http://127.0.0.1:5000
```

If `models/` is missing or was built with a different scikit-learn version,
`app.py` runs `train.py` automatically on start-up.

Run the tests with:

```bash
pip install pytest
python -m pytest tests/ -q
```

## API

| Method | Path              | Returns                                                        |
|--------|-------------------|-----------------------------------------------------------------|
| GET    | `/api/health`     | Liveness check                                                  |
| GET    | `/api/summary`    | Dataset size, cleaning rules, what they removed, split sizes     |
| GET    | `/api/model`      | Hyper-parameters, learned weights, scaler, loss curve, CV scores |
| GET    | `/api/evaluate?threshold=0.5` | Confusion matrix, accuracy, precision, recall, F1, AUC |
| GET    | `/api/roc`        | ROC curve points and AUC                                         |
| GET    | `/api/compare`    | All 7 models side by side, plus the scratch-vs-scikit-learn check |
| POST   | `/api/predict`    | Probability, log-odds, per-feature contributions for one patient |

## Honest limitations

This predicts whether the disease was *present at the time of the
examination*, not a future risk. Accuracy across every model sits around
72–74%: the ceiling is in the available features (self-reported blood
pressure, no genetic or dietary data), not in the choice of algorithm. It is
a student project trained on a public research dataset, not a diagnostic
tool.

Dataset: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
(Svetlana Ulianova, Kaggle), 70,000 examination records.
