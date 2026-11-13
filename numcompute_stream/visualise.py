"""Matplotlib plots for streaming runs.

The three plots asked for in the brief are ``plot_metric_over_time``,
``compare_models`` and ``plot_predictions_vs_ground_truth``. The others
(confusion matrix, feature importances, dashboard, buffer trade-off) are used
in the demo and the benchmark.

Every function takes ``ax`` (draw into an existing axis), ``save_path``
(write a PNG) and ``show``, and returns the figure. NaN values, e.g. chunks
before the first retrain, are left as gaps in the lines.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _axis(ax, size=(8, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        return fig, ax
    return ax.figure, ax


def _labels(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    return fig


def _values(values, name):
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError(f"{name} is empty, nothing to plot.")
    return values


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value", x=None,
                          xlabel="Chunk", retrain_points=None, smooth=None, ax=None,
                          save_path=None, show=True):
    """One metric per chunk, with optional retrain markers and a moving average."""
    y = _values(metric_values, "metric_values")
    x = np.arange(y.size) if x is None else np.asarray(x)
    fig, ax = _axis(ax)
    ax.plot(x, y, color=COLOURS[0], marker="o", markersize=3, label=ylabel)
    if smooth and 1 < smooth <= y.size:
        # moving average that skips NaN: average of the valid values in each window
        ok = ~np.isnan(y)
        kernel = np.ones(smooth)
        total = np.convolve(np.where(ok, y, 0.0), kernel, mode="same")
        count = np.convolve(ok.astype(float), kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            ax.plot(x, total / count, "k--", linewidth=1.2, label=f"{smooth}-chunk average")
    for i, p in enumerate(retrain_points if retrain_points is not None else []):
        ax.axvline(x[p], color="grey", linestyle=":", linewidth=1,
                   label="retrain" if i == 0 else None)
    _labels(ax, title, xlabel, ylabel)
    ax.legend(frameon=False)
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2=None, labels=("Model A", "Model B"),
                   title="Model comparison", ylabel="Accuracy", xlabel="Chunk", ax=None,
                   save_path=None, show=True):
    """Metric histories of two models; pass a dict ``{label: values}`` for more."""
    if isinstance(metric1, dict):
        series = metric1
    elif metric2 is None:
        series = dict(zip(labels, metric1))
    else:
        series = {labels[0]: metric1, labels[1]: metric2}
    if len(series) > len(COLOURS):
        raise ValueError(f"At most {len(COLOURS)} models fit in one plot.")
    fig, ax = _axis(ax)
    for i, (label, values) in enumerate(series.items()):
        values = _values(values, label)
        ax.plot(values, color=COLOURS[i], marker=MARKERS[i], markersize=4,
                markevery=max(1, values.size // 15), label=label)
    _labels(ax, title, xlabel, ylabel)
    ax.legend(frameon=False)
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred,
                                     title="Predictions vs ground truth (latest chunk)",
                                     max_points=200, class_names=None, ax=None,
                                     save_path=None, show=True):
    """True labels (circles) and predictions (dots) per sample; errors get a red cross."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different lengths.")
    if y_true.size == 0:
        raise ValueError("No labels to plot.")
    y_true, y_pred = y_true[:max_points], y_pred[:max_points]
    classes = np.unique(np.concatenate([y_true, y_pred]))
    t, p = np.searchsorted(classes, y_true), np.searchsorted(classes, y_pred)
    idx, wrong = np.arange(t.size), t != p
    fig, ax = _axis(ax, size=(9, 3.5))
    ax.scatter(idx, t, s=60, facecolors="none", edgecolors=COLOURS[0], label="ground truth")
    ax.scatter(idx, p, s=14, color=COLOURS[1], label="prediction")
    if wrong.any():
        ax.scatter(idx[wrong], p[wrong], s=70, marker="x", color="red",
                   label=f"error ({wrong.sum()})")
    ax.set_yticks(range(classes.size))
    ax.set_yticklabels(class_names if class_names is not None else [str(c) for c in classes])
    _labels(ax, f"{title}, accuracy {1 - wrong.mean():.1%}", "Sample in chunk", "Class")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return _finish(fig, save_path, show)


def plot_confusion_matrix(matrix, labels=None, title="Confusion matrix", normalise=False,
                          ax=None, save_path=None, show=True):
    """Confusion matrix as a heat map with the counts written in the cells."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square.")
    if normalise:
        rows = m.sum(axis=1, keepdims=True)
        m = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)
    fig, ax = _axis(ax, size=(4.5, 4))
    image = ax.imshow(m, cmap="Blues")
    labels = list(range(m.shape[0])) if labels is None else labels
    ax.set_xticks(range(m.shape[0]), [str(v) for v in labels])
    ax.set_yticks(range(m.shape[0]), [str(v) for v in labels])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            text = f"{m[i, j]:.2f}" if normalise else str(int(m[i, j]))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_title(title, loc="left")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _finish(fig, save_path, show)


def plot_feature_importances(importances, feature_names=None, top_k=15,
                             title="Feature importances", ax=None, save_path=None, show=True):
    """Horizontal bars for the ``top_k`` most important features."""
    imp = _values(importances, "importances")
    names = list(feature_names) if feature_names is not None else \
        [f"x{i}" for i in range(imp.size)]
    order = np.argsort(imp)[::-1][:top_k][::-1]
    fig, ax = _axis(ax, size=(7, 0.35 * len(order) + 1))
    ax.barh(range(len(order)), imp[order], color=COLOURS[0])
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    _labels(ax, title, "Share of the impurity decrease", "")
    return _finish(fig, save_path, show)


def plot_buffer_tradeoff(buffer_sizes, accuracy, retrain_seconds,
                         title="Effect of buffer_size", save_path=None, show=True):
    """Accuracy and total retrain time against buffer_size, side by side."""
    b = np.asarray(buffer_sizes)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    left.plot(b, accuracy, marker="o", color=COLOURS[0])
    _labels(left, "Prequential accuracy", "buffer_size", "Accuracy")
    right.plot(b, retrain_seconds, marker="s", color=COLOURS[1])
    _labels(right, "Total retraining time", "buffer_size", "Seconds")
    for a in (left, right):
        a.set_xscale("log")
        a.set_xticks(b, [str(v) for v in b])
    fig.suptitle(title)
    return _finish(fig, save_path, show)


def plot_stream_dashboard(history, title="Streaming run", save_path=None, show=True):
    """Accuracy, fit time and memory per chunk from ``StreamTrainer.history_arrays()``."""
    x = history["chunk"]
    retrains = np.flatnonzero(history["retrained"].astype(bool))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, key in enumerate(("chunk_accuracy", "cumulative_accuracy", "rolling_accuracy")):
        axes[0].plot(x, history[key], color=COLOURS[i], marker=MARKERS[i], markersize=3,
                     label=key.replace("_", " "))
    _labels(axes[0], "Accuracy", "", "Accuracy")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].bar(x, history["fit_seconds"] * 1000, color=COLOURS[0])
    _labels(axes[1], "partial_fit time per chunk (spikes are retrains)", "", "ms")
    axes[2].plot(x, history["memory_bytes"] / 1024, color=COLOURS[0])
    _labels(axes[2], "Tree and buffer memory", "Chunk", "KiB")
    for p in retrains:
        for a in axes:
            a.axvline(x[p], color="grey", linestyle=":", linewidth=1)
    fig.suptitle(title)
    return _finish(fig, save_path, show)


__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth",
           "plot_confusion_matrix", "plot_feature_importances", "plot_buffer_tradeoff",
           "plot_stream_dashboard"]
