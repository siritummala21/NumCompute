"""Streaming classification metrics.

Every metric has ``update(y_true_chunk, y_pred_chunk)``, ``result()`` and
``reset()``. Accuracy, precision, recall and F1 are all read from one
accumulated confusion matrix, so streaming the labels in chunks gives exactly
the same numbers as computing them on all labels at once. Each chunk is
counted with the group's ``numcompute.metrics.confusion_matrix``.
"""

from collections import deque

import numpy as np

from numcompute import metrics as _group

from ._validation import check_labels, check_scalar


def _check_pair(y_true, y_pred):
    y_true = check_labels(y_true, "y_true", allow_empty=True)
    y_pred = check_labels(y_pred, "y_pred", allow_empty=True)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true has {y_true.shape[0]} labels but y_pred has "
                         f"{y_pred.shape[0]}.")
    return y_true, y_pred


class StreamingConfusionMatrix:
    """Confusion matrix that grows when new labels appear.

    Row i / column j count samples with true label ``labels_[i]`` and
    predicted label ``labels_[j]``. Memory is O(k^2) for k classes,
    independent of the stream length.
    """

    def __init__(self, labels=None):
        self.labels = labels
        self.reset()

    def reset(self):
        """Zero the counts."""
        self.labels_ = np.array([]) if self.labels is None else np.unique(self.labels)
        k = self.labels_.size
        self.matrix_ = np.zeros((k, k), dtype=np.int64)
        self.n_samples_ = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of (true, predicted) labels; an empty chunk is ignored.

        Complexity: O(chunk + k^2) for k classes; memory O(k^2).
        """
        y_true, y_pred = _check_pair(y_true_chunk, y_pred_chunk)
        if y_true.size == 0:
            return self
        new = np.union1d(np.unique(y_true), np.unique(y_pred))
        if self.labels_.size and (self.labels_.dtype.kind in "iufb") != (new.dtype.kind in "iufb"):
            raise TypeError("Label type changed mid-stream (numbers vs strings).")
        labels = np.union1d(self.labels_, new) if self.labels_.size else new
        if labels.size > self.labels_.size:               # new class: grow the matrix
            bigger = np.zeros((labels.size, labels.size), dtype=np.int64)
            pos = np.searchsorted(labels, self.labels_)
            bigger[np.ix_(pos, pos)] = self.matrix_
            self.labels_, self.matrix_ = labels, bigger
        self.matrix_ += _group.confusion_matrix(y_true, y_pred, labels=self.labels_)
        self.n_samples_ += y_true.size
        return self

    def result(self):
        """Copy of the (k, k) count matrix."""
        return self.matrix_.copy()


class _ConfusionMetric:
    """Base class for metrics computed from a confusion matrix."""

    def __init__(self, average="macro", pos_label=1):
        if average not in ("binary", "micro", "macro", "weighted"):
            raise ValueError("average must be 'binary', 'micro', 'macro' or 'weighted'.")
        self.average = average
        self.pos_label = pos_label
        self.reset()

    def reset(self):
        """Zero the counts."""
        self.confusion = StreamingConfusionMatrix()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of labels.

        Complexity: O(chunk + k^2); result() is O(k^2).
        """
        self.confusion.update(y_true_chunk, y_pred_chunk)
        return self

    def _per_class(self):
        """True positives, predicted positives and actual positives per class."""
        cm = self.confusion.matrix_
        return np.diag(cm).astype(float), cm.sum(axis=0).astype(float), cm.sum(axis=1)

    def _average(self, scores, support):
        if self.average == "binary":
            labels = self.confusion.labels_
            if self.pos_label not in labels:
                return 0.0
            return float(scores[np.searchsorted(labels, self.pos_label)])
        if self.average == "weighted":
            return float(np.sum(scores * support) / max(support.sum(), 1))
        return float(np.mean(scores)) if scores.size else 0.0


def _safe_div(a, b):
    """a / b with 0 where b == 0 (sklearn's zero_division=0)."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


class Accuracy(_ConfusionMetric):
    """Fraction of correct predictions so far."""

    def __init__(self):
        super().__init__()

    def result(self):
        """Accuracy so far (NaN before any update)."""
        cm = self.confusion.matrix_
        return float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")


class Precision(_ConfusionMetric):
    """Precision with binary / micro / macro / weighted averaging."""

    def result(self):
        """Precision so far, averaged as set by ``average``."""
        tp, pred, actual = self._per_class()
        if self.average == "micro":
            return float(_safe_div(tp.sum(), pred.sum()))
        return self._average(_safe_div(tp, pred), actual)


class Recall(_ConfusionMetric):
    """Recall with binary / micro / macro / weighted averaging."""

    def result(self):
        """Recall so far, averaged as set by ``average``."""
        tp, _, actual = self._per_class()
        if self.average == "micro":
            return float(_safe_div(tp.sum(), actual.sum()))
        return self._average(_safe_div(tp, actual.astype(float)), actual)


class F1Score(_ConfusionMetric):
    """F1 score (harmonic mean of precision and recall)."""

    def result(self):
        """F1 so far, averaged as set by ``average``."""
        tp, pred, actual = self._per_class()
        if self.average == "micro":
            p = _safe_div(tp.sum(), pred.sum())
            r = _safe_div(tp.sum(), actual.sum())
            return float(_safe_div(2 * p * r, p + r))
        p, r = _safe_div(tp, pred), _safe_div(tp, actual.astype(float))
        return self._average(_safe_div(2 * p * r, p + r), actual)


class StreamingAUC:
    """ROC AUC over the stream.

    The labels and scores of every chunk are kept, and ``result()`` calls the
    group's ``numcompute.metrics.roc_auc_score``. This is exact; memory grows
    with the number of samples (two numbers per sample). For more than two
    classes pass ``classes`` and a score matrix: the result is the macro
    one-vs-rest AUC.
    """

    def __init__(self, pos_label=1, classes=None):
        self.pos_label = pos_label
        self.classes = classes
        self.reset()

    def reset(self):
        """Forget all scores."""
        self._y, self._scores = [], []
        return self

    def update(self, y_true_chunk, y_score_chunk):
        """Add labels and scores (shape (n,) or (n, n_classes))."""
        y = check_labels(y_true_chunk, "y_true", allow_empty=True)
        scores = np.asarray(y_score_chunk, dtype=float)
        if scores.shape[0] != y.shape[0]:
            raise ValueError("y_true and y_score have different lengths.")
        if not np.isfinite(scores).all():
            raise ValueError("y_score contains NaN or infinite values.")
        if y.size:
            self._y.append(y)
            self._scores.append(scores)
        return self

    def result(self):
        """AUC so far; NaN until both classes have been seen.

        Complexity: O(n log n) over all n samples seen (memory O(n)).
        """
        if not self._y:
            return float("nan")
        y, s = np.concatenate(self._y), np.concatenate(self._scores)
        if s.ndim == 1:
            positive = y == self.pos_label
            if positive.all() or not positive.any():
                return float("nan")
            return float(_group.roc_auc_score(positive.astype(int), s))
        aucs = []
        for k, c in enumerate(self.classes):
            positive = y == c
            if positive.any() and not positive.all():
                aucs.append(_group.roc_auc_score(positive.astype(int), s[:, k]))
        return float(np.mean(aucs)) if aucs else float("nan")


class RollingMetric:
    """A metric over the most recent ``window`` samples only.

    Useful to see how the model does *now*, not averaged over the whole
    stream (which hides concept drift).
    """

    def __init__(self, metric=None, window=500):
        self.metric = metric if metric is not None else Accuracy()
        self.window = check_scalar(window, "window", kind="int", min_val=1)
        self.reset()

    def reset(self):
        """Forget the stored samples."""
        self._recent = deque()
        self._count = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add a chunk and drop samples that fall outside the window.

        Complexity: O(chunk); memory O(window + chunk).
        """
        y_true, y_pred = _check_pair(y_true_chunk, y_pred_chunk)
        if y_true.size:
            self._recent.append((y_true, y_pred))
            self._count += y_true.size
        # drop whole chunks while the rest still covers the window
        while self._recent and self._count - self._recent[0][0].size >= self.window:
            self._count -= self._recent.popleft()[0].size
        return self

    def result(self):
        """Metric value over the last ``window`` samples."""
        if not self._recent:
            return float("nan")
        y_true = np.concatenate([t for t, _ in self._recent])[-self.window:]
        y_pred = np.concatenate([p for _, p in self._recent])[-self.window:]
        self.metric.reset()
        return self.metric.update(y_true, y_pred).result()


def confusion_matrix_loop(y_true, y_pred, labels):
    """Confusion matrix with a Python loop (benchmark baseline only)."""
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm


__all__ = ["StreamingConfusionMatrix", "Accuracy", "Precision", "Recall", "F1Score",
           "StreamingAUC", "RollingMetric", "confusion_matrix_loop"]
