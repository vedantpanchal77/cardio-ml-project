"""Flask backend for the cardiovascular disease prediction project.

    python app.py          ->  http://127.0.0.1:5000

The server never trains while handling a request. ``train.py`` writes the model
files once; this app loads them at start-up (and trains first if they are missing).

Endpoints (JSON, except "/")
    GET  /                 the single-page front end
    GET  /api/health       liveness check
    GET  /api/summary      dataset size, cleaning rules, what the rules removed, train/test split
    GET  /api/model        hyper-parameters, learned weights, scaler, loss curve, cross-validation
    GET  /api/evaluate     test-set metrics at ?threshold=0.5  (confusion matrix, F1, AUC ...)
    GET  /api/roc          ROC curve points + AUC
    GET  /api/compare      scratch model vs scikit-learn models on the same split
    POST /api/predict      score one patient; returns probability + per-feature contributions
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ml.serving import ModelService, ValidationError, parse_threshold

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cardio_train.csv"
MODELS_DIR = ROOT / "models"


def create_app(service: ModelService | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT / "static"), template_folder=str(ROOT / "templates"))
    svc = service or ModelService.load_or_train(MODELS_DIR, DATA_PATH)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.after_request
    def add_header(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


    @app.get("/learned")
    def page_learned():
        return render_template("learned.html")

    @app.get("/comparison")
    def page_comparison():
        return render_template("compare.html")

    @app.get("/try-model")
    def page_try_model():
        return render_template("predict.html")


    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model=svc.model_info()["name"], trained_at=svc.report["trained_at"])

    @app.get("/api/summary")
    def summary():
        return jsonify(svc.summary())

    @app.get("/api/model")
    def model():
        return jsonify(svc.model_info())

    @app.get("/api/evaluate")
    def evaluate():
        threshold = parse_threshold(request.args.get("threshold"))
        return jsonify(svc.evaluate(threshold, request.args.get("model")))

    @app.get("/api/roc")
    def roc():
        return jsonify(svc.roc(request.args.get("model")))

    @app.get("/api/compare")
    def compare():
        return jsonify(svc.compare())

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="request body must be JSON"), 400
        return jsonify(svc.predict(payload))

    # ---- errors: always JSON under /api so the front end can show a useful message
    @app.errorhandler(ValidationError)
    def bad_input(err: ValidationError):
        return jsonify(error="invalid input", details=err.errors), 422

    @app.errorhandler(404)
    def not_found(_err):
        if request.path.startswith("/api/"):
            return jsonify(error="unknown endpoint"), 404
        return "Not found", 404

    @app.errorhandler(405)
    def wrong_method(_err):
        return jsonify(error="method not allowed"), 405

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)

