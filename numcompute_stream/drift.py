"""DDM drift detector (Gama et al., 2004), updated one chunk at a time.

With a fixed ``buffer_size`` the tree only reacts to a change in the data at
the next scheduled retrain. DDM watches the error rate of the predictions.
When it rises clearly above the best level seen so far, ``StreamTrainer``
retrains the tree straight away on the most recent rows.

With ``p = (errors + 1) / (n + 2)`` and ``s = sqrt(p (1 - p) / n)``:

* the reference point is the smallest ``p + s`` seen (``p_min``, ``s_min``);
* warning when ``p + s >= p_min + 2 * s_min``;
* drift when ``p + s >= p_min + 3 * s_min``.

The +1 / +2 (Laplace smoothing) keeps ``s`` above zero. Without it, a model
with no errors yet has ``s_min = 0`` and its first mistake already counts as
drift. After a drift everything is reset to learn the new error level.
"""

import numpy as np

from ._validation import check_labels, check_scalar


class DDM:
    """Drift Detection Method.

    Parameters
    ----------
    min_samples : int
        Predictions to see before drift can be signalled.
    warning_level, drift_level : float
        Multiples of ``s_min`` for the two thresholds (default 2 and 3).

    Example
    -------
    >>> ddm = DDM(min_samples=50)
    >>> ddm.update_counts(n_errors=5, n=100).state
    'stable'
    >>> ddm.update_counts(n_errors=60, n=100).state
    'drift'
    """

    def __init__(self, min_samples=200, warning_level=2.0, drift_level=3.0):
        self.min_samples = check_scalar(min_samples, "min_samples", kind="int", min_val=1)
        self.warning_level = check_scalar(warning_level, "warning_level", min_val=1e-12)
        self.drift_level = check_scalar(drift_level, "drift_level", min_val=1e-12)
        if not self.warning_level < self.drift_level:
            raise ValueError("warning_level must be smaller than drift_level.")
        self.n_detections_ = 0
        self.reset()

    def reset(self):
        """Forget the error counts and the reference level."""
        self.n_ = 0
        self.n_errors_ = 0
        self.p_min_ = np.inf
        self.s_min_ = np.inf
        self.state = "stable"
        return self

    def update_counts(self, n_errors, n):
        """Add ``n`` predictions, ``n_errors`` of them wrong."""
        n = check_scalar(n, "n", kind="int", min_val=0)
        n_errors = check_scalar(n_errors, "n_errors", kind="int", min_val=0, max_val=n)
        if n == 0:
            return self
        self.n_ += n
        self.n_errors_ += n_errors
        p = (self.n_errors_ + 1.0) / (self.n_ + 2.0)
        s = np.sqrt(p * (1.0 - p) / self.n_)
        if self.n_ < self.min_samples:
            self.state = "stable"
            return self
        if p + s < self.p_min_ + self.s_min_:
            self.p_min_, self.s_min_ = p, s
        if p + s >= self.p_min_ + self.drift_level * self.s_min_:
            self.n_detections_ += 1
            self.reset()
            self.state = "drift"
        elif p + s >= self.p_min_ + self.warning_level * self.s_min_:
            self.state = "warning"
        else:
            self.state = "stable"
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add a chunk of true and predicted labels.

        Complexity: O(chunk) time, O(1) memory.
        """
        y_true = check_labels(y_true_chunk, "y_true", allow_empty=True)
        y_pred = check_labels(y_pred_chunk, "y_pred", allow_empty=True)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length.")
        return self.update_counts(int(np.sum(y_true != y_pred)), y_true.size)

    @property
    def drift_detected(self):
        """True if the last update signalled drift."""
        return self.state == "drift"


__all__ = ["DDM"]
