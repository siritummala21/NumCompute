"""
Generate ``demo/data/sensor_stream.csv`` — a synthetic streaming classification
dataset used by the demo, the benchmarks and some tests.

Design (so every part of the framework gets exercised):

* 6 000 rows in time order, 5 numeric sensor readings + 1 categorical
  ``site`` column, 3 classes (``0 = normal``, ``1 = warning``, ``2 = fault``).
* ~3 % of numeric values are missing (empty cells) -> SimpleImputer, NaN rules.
* ``humidity`` is constant for the first 500 rows -> zero-variance chunks.
* A new site ``"D"`` only appears after row 4 000 -> OneHotEncoder growth.
* **Concept drift** at row 3 000: the rule mapping readings to the label
  changes, so a model that never forgets would degrade -> buffered retraining.
* 4 % label noise.

Run:  python demo/data/generate_dataset.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numcompute_stream.io import write_csv  # noqa: E402

N, DRIFT_AT, NEW_SITE_AT = 6000, 3000, 4000


def make(seed: int = 42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(60, 12, N)
    vib = rng.gamma(2.0, 1.5, N)
    pressure = rng.normal(100, 15, N)
    humidity = rng.uniform(20, 80, N)
    humidity[:500] = 45.0                       # zero-variance start
    load = rng.uniform(0, 1, N)
    sites = np.where(np.arange(N) < NEW_SITE_AT,
                     rng.choice(["A", "B", "C"], N),
                     rng.choice(["A", "B", "C", "D"], N, p=[.25, .25, .2, .3]))

    site_bias = np.select([sites == "B", sites == "D"], [5.0, 12.0], 0.0)
    risk_before = 0.06 * (temp - 60) + 0.7 * (vib - 3) + 1.5 * (load - 0.5) + 0.05 * site_bias
    risk_after = (0.05 * (pressure - 100) - 0.6 * (vib - 3) + 0.04 * (humidity - 50)
                  + 0.1 * site_bias)
    risk = np.where(np.arange(N) < DRIFT_AT, risk_before, risk_after)
    y = np.digitize(risk, [0.6, 1.8])           # 3 classes
    flip = rng.random(N) < 0.04
    y[flip] = rng.integers(0, 3, flip.sum())

    num = np.column_stack([temp, vib, pressure, humidity, load]).round(3)
    num[rng.random(num.shape) < 0.03] = np.nan
    X = np.empty((N, 6), dtype=object)
    X[:, :5] = num
    X[:, 5] = sites
    return X, y


if __name__ == "__main__":
    X, y = make()
    out = os.path.join(os.path.dirname(__file__), "sensor_stream.csv")
    write_csv(out, X, ["temperature", "vibration", "pressure", "humidity", "load", "site"],
              y, target_name="status")
    print(f"wrote {out}: {X.shape[0]} rows, class counts {np.bincount(y).tolist()}")
