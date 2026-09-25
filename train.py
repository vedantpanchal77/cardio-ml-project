"""Train every model and write the artifacts to models/.

    python train.py

Takes roughly a minute on a laptop. Run it again whenever you change the data,
the cleaning rules or the hyper-parameters in ml/training.py.
"""
from pathlib import Path

from ml.training import train_all

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    train_all(ROOT / "data" / "cardio_train.csv", ROOT / "models")
