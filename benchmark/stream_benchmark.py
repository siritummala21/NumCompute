"""Benchmarks for NumCompute-Stream.

(a) Loop vs vectorised: each vectorised function against a plain Python
    version that gives the same result (best split, tree fit, predict,
    running mean/variance, confusion matrix, and a whole streaming run).
(b) buffer_size: accuracy, number of retrains, retrain time and memory for
    buffer sizes 100 to 2000 on the drifting sensor stream (3 seeds).
(c) Retraining strategy: a tree trained once and never updated, the buffered
    tree, and the buffered tree with DDM drift detection.

Timing uses the group's ``numcompute.benchmarking.timeit`` (one warm-up run,
then the best of several runs). Results are written to benchmark/stream_results/.

Run:  python benchmark/stream_benchmark.py            (about 30 s)
      python benchmark/stream_benchmark.py --quick    (smaller sizes)
"""

import argparse
import csv
import datetime
import json
import os
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo", "data"))

from generate_dataset import make as make_sensor_data  # noqa: E402
from numcompute.benchmarking import timeit  # noqa: E402
from numcompute_stream import (DDM, ColumnTransformer, DecisionTreeClassifier,  # noqa: E402
                               OneHotEncoder, Pipeline, SimpleImputer, StandardScaler,
                               StreamTrainer, chunk_stream)
from numcompute_stream import visualise as vis  # noqa: E402
from numcompute_stream.metrics import StreamingConfusionMatrix, confusion_matrix_loop  # noqa: E402
from numcompute_stream.stats import RunningStats, welford_loop  # noqa: E402
from numcompute_stream.tree import best_split_loop, best_split_vectorised  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "stream_results")
SEEDS = (42, 1, 2)                     # seed 42 is the CSV used in the demo


def time_best(fn, repeat):
    """(best, mean) seconds over ``repeat`` runs after one warm-up run."""
    result = timeit(fn, repeat=repeat, number=1, warmup=1)
    return float(result["best"]), float(result["mean"])


def compare(operation, n, loop_fn, vec_fn, loop_repeat=3):
    loop_best, loop_mean = time_best(loop_fn, loop_repeat)
    vec_best, vec_mean = time_best(vec_fn, 7)
    return {"operation": operation, "n": n, "loop_best_s": loop_best,
            "vectorised_best_s": vec_best, "speedup": loop_best / vec_best,
            "loop_mean_s": loop_mean, "vectorised_mean_s": vec_mean}


def synthetic(n, d=6, seed=0):
    """Random 3-class data with a non-linear rule."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] ** 2 - X[:, 2] > 0.5).astype(int) + (X[:, 3] > 1)
    return X, y


def make_model(buffer_size, splitter="vectorised"):
    """The same pipeline as the demo: impute and scale numbers, one-hot the site."""
    numeric = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer("most_frequent")),
                            ("onehot", OneHotEncoder())])
    pre = ColumnTransformer([("num", numeric, [0, 1, 2, 3, 4]), ("cat", categorical, [5])])
    tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, buffer_size=buffer_size,
                                  splitter=splitter)
    return Pipeline([("pre", pre), ("tree", tree)])


class TrainOnce:
    """Baseline: train on the first buffer only, then never update the tree."""

    def __init__(self, buffer_size):
        self.model = make_model(buffer_size)
        self.final_estimator = self.model["tree"]

    def partial_fit(self, X, y):
        if not self.model.is_fitted:
            self.model.partial_fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def is_fitted(self):
        return self.model.is_fitted

    def memory_bytes(self):
        return self.model.memory_bytes()


def run_stream(model, X, y, **kwargs):
    trainer = StreamTrainer(model, rolling_window=500, **kwargs)
    trainer.run(chunk_stream(X, y, chunk_size=100))
    return trainer


def write_table(name, rows):
    """Save rows as CSV and as a Markdown table, and print them."""
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                                       for v in r.values()) + " |")
    with open(os.path.join(OUT, name + ".md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n{name}\n" + "\n".join(lines))


def bench_loop_vs_vectorised(quick):
    rows = []
    for n in ([250, 500, 1000] if quick else [250, 500, 1000, 2000]):
        X, y = synthetic(n)
        f = np.arange(X.shape[1])
        rows.append(compare("best split (one node)", n, lambda: best_split_loop(X, y, 3, f),
                            lambda: best_split_vectorised(X, y, 3, f)))
    for n in ([500, 1000] if quick else [500, 1000, 2000]):
        X, y = synthetic(n)
        rows.append(compare("tree fit (depth 6)", n,
                            lambda: DecisionTreeClassifier(max_depth=6, splitter="loop").fit(X, y),
                            lambda: DecisionTreeClassifier(max_depth=6).fit(X, y)))
    X, y = synthetic(20000)
    tree = DecisionTreeClassifier(max_depth=8).fit(X[:2000], y[:2000])
    assert np.array_equal(tree.predict(X), tree.predict_loop(X))
    rows.append(compare("tree predict", 20000, lambda: tree.predict_loop(X),
                        lambda: tree.predict(X)))
    rows.append(compare("running mean/var (40 chunks)", 20000, lambda: welford_loop(X),
                        lambda: [RunningStats().update(c) for c in np.array_split(X, 40)]))
    yt = np.random.default_rng(0).integers(0, 5, 200000)
    yp = np.random.default_rng(1).integers(0, 5, 200000)
    assert np.array_equal(confusion_matrix_loop(yt, yp, list(range(5))),
                          StreamingConfusionMatrix().update(yt, yp).result())
    rows.append(compare("confusion matrix", 200000,
                        lambda: confusion_matrix_loop(yt, yp, list(range(5))),
                        lambda: StreamingConfusionMatrix().update(yt, yp)))

    # a whole streaming run with each splitter: same accuracy, different time
    X, y = make_sensor_data(42)
    n = 2000 if quick else 3000
    accuracy = {}

    def stream(splitter):
        trainer = run_stream(make_model(500, splitter), X[:n], y[:n])
        accuracy[splitter] = trainer.summary()["final_cumulative_accuracy"]

    rows.append(compare("stream run, buffer 500", n, lambda: stream("loop"),
                        lambda: stream("vectorised"), loop_repeat=1))
    assert accuracy["loop"] == accuracy["vectorised"]
    write_table("loop_vs_vectorised", rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(range(len(rows)), [r["speedup"] for r in rows], color=vis.COLOURS[0])
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [f"{r['operation']} (n={r['n']})" for r in rows],
                  fontsize=8)
    ax.invert_yaxis()
    for i, r in enumerate(rows):
        ax.text(r["speedup"] * 1.08, i, f"{r['speedup']:.0f}x", va="center", fontsize=8)
    ax.set_title("Vectorised speed-up over the loop version (best times)", loc="left")
    ax.set_xlabel("Speed-up (log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loop_vs_vectorised.png"), dpi=150)
    plt.close(fig)
    return rows


def bench_buffer_sizes(quick):
    sizes = [100, 250, 500, 1000] if quick else [100, 250, 500, 1000, 2000]
    streams = {seed: make_sensor_data(seed) for seed in SEEDS}
    rows, curves = [], {}
    for b in sizes:
        runs = [run_stream(make_model(b), X, y) for X, y in streams.values()]
        acc = [r.summary()["final_cumulative_accuracy"] for r in runs]
        rows.append({"buffer_size": b,
                     "accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc)),
                     "n_retrains": runs[0].summary()["n_retrains"],
                     "retrain_seconds": float(np.mean([r.summary()["total_fit_seconds"]
                                                       for r in runs])),
                     "peak_memory_kib": runs[0].summary()["peak_memory_bytes"] / 1024})
        curves[f"buffer {b}"] = runs[0].history_arrays()["rolling_accuracy"]
    write_table("buffer_size", rows)
    vis.plot_buffer_tradeoff([r["buffer_size"] for r in rows],
                             [r["accuracy_mean"] for r in rows],
                             [r["retrain_seconds"] for r in rows],
                             save_path=os.path.join(OUT, "buffer_tradeoff.png"), show=False)
    vis.compare_models(curves, title="Rolling accuracy (500 samples), seed 42",
                       save_path=os.path.join(OUT, "buffer_rolling_accuracy.png"), show=False)
    plt.close("all")
    return rows


def bench_strategies():
    strategies = {
        "train once": lambda: (TrainOnce(500), {}),
        "buffered (500)": lambda: (make_model(500), {}),
        "buffered + DDM (500)": lambda: (make_model(500),
                                         {"drift_detector": DDM(min_samples=200)}),
    }
    rows, curves = [], {}
    for name, build in strategies.items():
        acc = []
        for seed in SEEDS:
            model, kwargs = build()
            trainer = run_stream(model, *make_sensor_data(seed), **kwargs)
            acc.append(trainer.summary()["final_cumulative_accuracy"])
            if seed == 42:
                curves[name] = trainer.history_arrays()["rolling_accuracy"]
                drift_retrains = trainer.n_drift_retrains_
        rows.append({"strategy": name, "accuracy_mean": float(np.mean(acc)),
                     "accuracy_std": float(np.std(acc)), "drift_retrains_seed42": drift_retrains})
    write_table("strategies", rows)
    fig = vis.compare_models(curves, title="Rolling accuracy by strategy, seed 42 "
                             "(drift at chunk 30)", show=False)
    fig.axes[0].axvline(30, color="grey", linestyle="--", linewidth=1)
    fig.savefig(os.path.join(OUT, "strategies.png"), dpi=150, bbox_inches="tight")
    plt.close("all")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    quick = parser.parse_args().quick
    os.makedirs(OUT, exist_ok=True)
    env = {"run": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "python": platform.python_version(), "numpy": np.__version__,
           "matplotlib": matplotlib.__version__, "os": f"{platform.system()} {platform.release()}",
           "machine": platform.machine(), "cpu_count": os.cpu_count(), "quick": quick}
    loop_rows = bench_loop_vs_vectorised(quick)
    buffer_rows = bench_buffer_sizes(quick)
    strategy_rows = bench_strategies()

    # the numbers quoted in the README and the report
    best = max(buffer_rows, key=lambda r: r["accuracy_mean"])
    split = [r for r in loop_rows if r["operation"].startswith("best split")]
    summary = dict(env)
    summary.update({
        "split_speedup_smallest": split[0]["speedup"], "split_n_smallest": split[0]["n"],
        "split_speedup_largest": split[-1]["speedup"], "split_n_largest": split[-1]["n"],
        "tree_fit_speedup": [r for r in loop_rows if r["operation"].startswith("tree fit")][-1]
        ["speedup"],
        "predict_speedup": next(r["speedup"] for r in loop_rows
                                if r["operation"] == "tree predict"),
        "stream_speedup": loop_rows[-1]["speedup"],
        "best_buffer_size": best["buffer_size"], "best_buffer_accuracy": best["accuracy_mean"],
        "buffer_500_accuracy": next(r["accuracy_mean"] for r in buffer_rows
                                    if r["buffer_size"] == 500),
        "strategy_accuracy": {r["strategy"]: r["accuracy_mean"] for r in strategy_rows},
    })
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
