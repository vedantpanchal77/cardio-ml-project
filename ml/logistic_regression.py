"""Logistic regression implemented from scratch with NumPy.

No machine-learning library is used: only array maths. This is the algorithm
the project's SOP requires to be written by hand.

The model
---------
    z = X @ w + b                       (linear score, also called the logit or log-odds)
    p = sigmoid(z) = 1 / (1 + e^-z)     (probability that cardio == 1)

The loss (binary cross-entropy with an L2 penalty on the weights)
-----------------------------------------------------------------
    L = -mean( y*log(p) + (1-y)*log(1-p) )  +  lambda/(2n) * sum(w^2)

Its gradient, which is what gradient descent follows downhill
-------------------------------------------------------------
    dL/dw = X^T (p - y) / n  +  lambda * w / n
    dL/db = mean(p - y)

Training repeats: predict -> measure error -> move every weight a small step
(the learning rate) against its gradient. The intercept ``b`` is not penalised.
"""
from __future__ import annotations

import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, l2: float = 0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------ maths
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid: never computes exp() of a large positive number."""
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        e = np.exp(z[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def _loss(self, p: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        p = np.clip(p, 1e-12, 1 - 1e-12)  # log(0) would be -inf
        cross_entropy = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        penalty = self.l2 / (2 * n) * np.sum(self.weights ** 2)
        return float(cross_entropy + penalty)

    # --------------------------------------------------------------- training
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.weights = np.zeros(d)   # start from "no opinion": every weight and the bias are 0
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            p = self.predict_proba(X)                 # 1. predict
            error = p - y                             # 2. how wrong is each row?
            self.loss_history.append(self._loss(p, y))
            grad_w = X.T @ error / n + self.l2 * self.weights / n   # 3. gradient
            grad_b = error.mean()
            self.weights -= self.learning_rate * grad_w             # 4. step downhill
            self.bias -= self.learning_rate * grad_b
        return self

    # ------------------------------------------------------------- prediction
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """The log-odds z = X @ w + b."""
        return np.asarray(X, dtype=float) @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of class 1 (cardiovascular disease present)."""
        return self.sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "loss_history": [float(v) for v in self.loss_history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LogisticRegressionScratch":
        obj = cls(data["learning_rate"], data["epochs"], data["l2"])
        obj.weights = np.asarray(data["weights"], dtype=float)
        obj.bias = float(data["bias"])
        obj.loss_history = list(data.get("loss_history", []))
        return obj
