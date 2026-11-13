"""StreamTrainer: runs a model over a stream of chunks and logs every chunk.

Evaluation is prequential (test-then-train): each chunk is first used to
score the current model, which has not seen those rows yet, and only then to
train it. So every logged accuracy is an out-of-sample number.

Logged per chunk: chunk, n_samples, chunk_accuracy, cumulative_accuracy,
rolling_accuracy, cumulative_f1, auc (binary only), chunk_error, retrained,
n_retrains, buffer_count, fit_seconds, predict_seconds, memory_bytes,
samples_seen, and drift when a drift detector is used.

Drift-triggered retraining
--------------------------
With ``drift_detector=DDM()`` the detector sees each chunk's errors. When it
signals drift, the tree is retrained at once with ``flush`` on the rows seen
since the detector's warning started, instead of waiting for the buffer to
fill. Older rows in the buffer are from the old concept and are dropped.
"""

import csv
import logging
import time

import numpy as np

from ._validation import NotFittedError, check_labels, check_scalar
from .metrics import Accuracy, F1Score, RollingMetric, StreamingAUC, \
    StreamingConfusionMatrix

logger = logging.getLogger("numcompute_stream")


def _final(model):
    """The tree inside a pipeline, or the model itself."""
    return getattr(model, "final_estimator", model)


class StreamTrainer:
    """Train and evaluate a model chunk by chunk.

    Parameters
    ----------
    model : estimator or Pipeline with ``partial_fit`` and ``predict``
    rolling_window : int
        Number of recent samples for the rolling accuracy.
    name : str, optional
        Label for logs and plots.
    log_every : int
        Write a log line every ``log_every`` chunks (0 = never).
    drift_detector : DDM or None
    drift_min_samples : int
        Smallest number of buffered rows for a drift retrain.

    Example
    -------
    >>> trainer = StreamTrainer(pipe)                      # doctest: +SKIP
    >>> for X_chunk, y_chunk in stream:                    # doctest: +SKIP
    ...     trainer.process_chunk(X_chunk, y_chunk)
    >>> trainer.summary()["final_cumulative_accuracy"]     # doctest: +SKIP
    """

    def __init__(self, model, rolling_window=500, name=None, log_every=0,
                 drift_detector=None, drift_min_samples=100):
        if not (callable(getattr(model, "partial_fit", None))
                and callable(getattr(model, "predict", None))):
            raise TypeError("model must have partial_fit() and predict().")
        if drift_detector is not None and not callable(getattr(_final(model), "flush", None)):
            raise TypeError("drift_detector needs a model with flush() (the tree has it).")
        self.model = model
        self.rolling_window = check_scalar(rolling_window, "rolling_window", kind="int",
                                           min_val=1)
        self.name = name or type(_final(model)).__name__
        self.log_every = check_scalar(log_every, "log_every", kind="int", min_val=0)
        self.drift_detector = drift_detector
        self.drift_min_samples = check_scalar(drift_min_samples, "drift_min_samples",
                                              kind="int", min_val=1)
        self.reset_metrics()

    def reset_metrics(self):
        """Clear the log and the metrics (the model is not touched)."""
        self.history = []
        self.accuracy = Accuracy()
        self.f1 = F1Score(average="macro")
        self.confusion = StreamingConfusionMatrix()
        self.rolling = RollingMetric(Accuracy(), window=self.rolling_window)
        self.auc = StreamingAUC()
        self.n_drift_retrains_ = 0
        self._samples_seen = 0
        self._since_warning = 0
        self.last_predictions = None
        self.last_truth = None
        return self

    def _score(self, X, y):
        """Predict the chunk and update the metrics (NaN if no tree is trained yet)."""
        nan = float("nan")
        out = {"chunk_accuracy": nan, "chunk_error": nan, "cumulative_accuracy": nan,
               "rolling_accuracy": nan, "cumulative_f1": nan, "auc": nan,
               "predict_seconds": 0.0}
        if not getattr(self.model, "is_fitted", True):
            return out, None
        start = time.perf_counter()
        try:
            y_pred = self.model.predict(X)
        except NotFittedError:
            return out, None
        out["predict_seconds"] = time.perf_counter() - start
        for metric in (self.accuracy, self.f1, self.confusion, self.rolling):
            metric.update(y, y_pred)
        acc = float(np.mean(y_pred == y))
        out.update(chunk_accuracy=acc, chunk_error=1.0 - acc,
                   cumulative_accuracy=self.accuracy.result(),
                   rolling_accuracy=self.rolling.result(), cumulative_f1=self.f1.result())
        classes = getattr(_final(self.model), "classes_", None)
        if classes is not None and len(classes) == 2 and hasattr(self.model, "predict_proba"):
            self.auc.pos_label = classes[1]
            self.auc.update(y, self.model.predict_proba(X)[:, 1])
            out["auc"] = self.auc.result()
        self.last_predictions, self.last_truth = y_pred, y
        return out, y_pred

    def process_chunk(self, X, y):
        """Score the chunk, train on it and log it. Returns the log record.

        An empty chunk is skipped and returns None.

        Complexity: one predict and one partial_fit on the chunk, plus O(chunk + k^2)
        for the metrics.
        """
        y = check_labels(y, allow_empty=True)
        if y.shape[0] == 0:
            return None
        record = {"chunk": len(self.history), "n_samples": int(y.shape[0])}
        scores, y_pred = self._score(X, y)
        record.update(scores)

        drift = False
        if self.drift_detector is not None and y_pred is not None:
            self.drift_detector.update(y, y_pred)
            drift = self.drift_detector.drift_detected
            # count the rows since the warning started: they show the new concept
            if self.drift_detector.state in ("warning", "drift"):
                self._since_warning += y.shape[0]
            else:
                self._since_warning = 0

        tree = _final(self.model)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        retrained = bool(getattr(tree, "last_partial_fit_retrained_", False))
        if drift and not retrained:
            before = tree.n_retrains_
            tree.flush(min_samples=self.drift_min_samples, last_n=self._since_warning)
            if tree.n_retrains_ > before:
                retrained = True
                self.n_drift_retrains_ += 1
        if drift:
            self._since_warning = 0
        record["fit_seconds"] = time.perf_counter() - start
        record["retrained"] = retrained
        record["n_retrains"] = int(getattr(tree, "n_retrains_", 0))
        record["buffer_count"] = int(getattr(tree, "buffer_count", 0))
        if self.drift_detector is not None:
            record["drift"] = bool(drift)

        self._samples_seen += y.shape[0]
        record["samples_seen"] = self._samples_seen
        memory = getattr(self.model, "memory_bytes", None)
        record["memory_bytes"] = int(memory()) if callable(memory) else 0
        self.history.append(record)
        if self.log_every and record["chunk"] % self.log_every == 0:
            logger.info("[%s] chunk %d | acc=%.3f cum=%.3f | retrained=%s | mem=%d B",
                        self.name, record["chunk"], record["chunk_accuracy"],
                        record["cumulative_accuracy"], retrained, record["memory_bytes"])
        return record

    def run(self, stream, max_chunks=None):
        """Process every ``(X_chunk, y_chunk)`` pair of ``stream``; returns the history."""
        for i, (X, y) in enumerate(stream):
            if max_chunks is not None and i >= max_chunks:
                break
            self.process_chunk(X, y)
        return self.history

    def history_arrays(self):
        """The log as ``{field: array with one value per chunk}``."""
        if not self.history:
            return {}
        return {k: np.array([r.get(k, np.nan) for r in self.history])
                for k in self.history[0]}

    def summary(self):
        """Final numbers for the whole stream."""
        h = self.history_arrays()
        tree = _final(self.model)
        return {"name": self.name,
                "chunks": len(self.history),
                "samples_seen": self._samples_seen,
                "final_cumulative_accuracy": self.accuracy.result(),
                "final_macro_f1": self.f1.result(),
                "final_rolling_accuracy": self.rolling.result(),
                "auc": self.auc.result(),
                "n_retrains": int(getattr(tree, "n_retrains_", 0)),
                "n_drift_retrains": self.n_drift_retrains_,
                "total_fit_seconds": float(np.sum(h.get("fit_seconds", 0.0))),
                "total_predict_seconds": float(np.sum(h.get("predict_seconds", 0.0))),
                "peak_memory_bytes": int(np.max(h.get("memory_bytes", [0])))}

    def to_csv(self, path):
        """Write the per-chunk log to a CSV file."""
        if not self.history:
            raise ValueError("The history is empty.")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)


__all__ = ["StreamTrainer"]
