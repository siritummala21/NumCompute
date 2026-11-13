"""Chunk-based statistics: mean/variance, quantiles and histograms.

Every class has ``update(X_chunk)`` and ``result()``. Statistics are per
feature (along axis 0) and NaN values are skipped. ``RunningStats`` and
``StreamingHistogram`` take an optional ``window`` (number of recent chunks to
keep), which turns them into sliding-window statistics.

``RunningStats`` builds on the group's ``numcompute.stats.RunningStats`` and
the histograms are counted with ``numcompute.stats.histogram``. The tests
check that streaming gives the same answer as the group's batch functions.
"""

import warnings
from collections import deque

import numpy as np

from numcompute import stats as _group_stats

from ._validation import check_array, check_random_state, check_scalar


def _power_of_two_scale(X):
    """Per-feature power of two close to the largest absolute value (1 if there is none).

    Dividing by a power of two only changes the exponent of a float, so no
    precision is lost.
    """
    with np.errstate(divide="ignore"):
        biggest = np.nanmax(np.where(np.isnan(X), 0.0, np.abs(X)), axis=0)
        exponent = np.where(biggest > 0, np.floor(np.log2(np.where(biggest > 0, biggest, 1.0))), 0)
    return np.ldexp(1.0, exponent.astype(int))


class RunningStats:
    """Running count, mean, variance, min and max per feature.

    This wraps the group's ``numcompute.stats.RunningStats``, which already
    merges chunks with Chan's formula and skips NaN. What it adds:

    * a sliding ``window`` (statistics over the last ``window`` chunks only),
      done by keeping one group ``RunningStats`` per chunk and combining them
      with the group's ``merge()``;
    * input checks, a missing-value count and a ``result()`` dictionary;
    * values are divided by a power of two (one per feature, chosen from the
      first chunk) before the squares are taken. Dividing by a power of two
      is exact, and it stops ``(x - mean)^2`` from overflowing for values
      near 1e308, where the variance came out wrong before.

    Complexity: ``update`` O(chunk * d); memory O(d), or O(window * d).
    """

    def __init__(self, window=None):
        self.window = check_scalar(window, "window", kind="int", min_val=1, allow_none=True)
        self.reset()

    def reset(self):
        """Forget everything."""
        self.n_features_ = None
        self.n_missing_ = None
        self._scale = None
        self._total = _group_stats.RunningStats()
        self._chunks = deque(maxlen=self.window)
        return self

    def update(self, X_chunk):
        """Add one chunk of shape (n_samples, n_features); an empty chunk is ignored."""
        X = check_array(X_chunk, "X_chunk", allow_empty=True)
        if X.shape[0] == 0:
            return self
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self.n_missing_ = np.zeros(X.shape[1], dtype=int)
        elif X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}.")
        self.n_missing_ += np.isnan(X).sum(axis=0)
        if self._scale is None:
            self._scale = _power_of_two_scale(X)
        Xs = X / self._scale
        if self.window is None:
            self._total.update(Xs)
        else:
            self._chunks.append(_group_stats.RunningStats().update(Xs))
        return self

    def _state(self):
        if self.n_features_ is None:
            raise RuntimeError("RunningStats has not seen any data yet.")
        if self.window is None:
            return self._total
        merged = _group_stats.RunningStats()
        for part in self._chunks:
            merged.merge(part)
        return merged

    @property
    def count_(self):
        """Non-missing values seen per feature."""
        return self._state().count

    @property
    def min_(self):
        """Smallest value per feature (inf if a column is all NaN)."""
        return self._state().min * self._scale

    @property
    def max_(self):
        """Largest value per feature (-inf if a column is all NaN)."""
        return self._state().max * self._scale

    def mean(self):
        """Mean per feature, shape (n_features,); NaN for an all-NaN column."""
        state = self._state()
        return np.where(state.count > 0, state.mean * self._scale, np.nan)

    def var(self, ddof=0):
        """Variance per feature; NaN where there are not enough values.

        For values near 1e308 the variance itself is larger than the biggest
        float and comes out as inf, but ``std`` is still finite.
        """
        with np.errstate(over="ignore"):
            return self._state().var(ddof) * self._scale ** 2

    def std(self, ddof=0):
        """Standard deviation per feature (taken before scaling back, so it does not overflow)."""
        return np.sqrt(self._state().var(ddof)) * self._scale

    def result(self):
        """Dictionary with count, mean, var, std, min, max and n_missing."""
        return {"count": self.count_, "mean": self.mean(), "var": self.var(),
                "std": self.std(), "min": self.min_, "max": self.max_,
                "n_missing": self.n_missing_.copy()}


def welford_loop(X):
    """Element-by-element Welford in pure Python (benchmark baseline only)."""
    X = np.asarray(X, dtype=float)
    means, variances = [], []
    for j in range(X.shape[1]):
        n, mean, m2 = 0, 0.0, 0.0
        for x in X[:, j]:
            if x != x:          # NaN
                continue
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)
        means.append(mean if n else np.nan)
        variances.append(m2 / n if n else np.nan)
    return np.array(means), np.array(variances)


class StreamingQuantile:
    """Per-feature quantiles from a reservoir sample.

    Reservoir sampling keeps a uniform random sample of ``capacity`` rows
    from the whole stream, so memory stays fixed. With ``capacity=None``
    every row is kept and the quantiles are exact.
    """

    def __init__(self, capacity=1000, random_state=None):
        self.capacity = check_scalar(capacity, "capacity", kind="int", min_val=1,
                                     allow_none=True)
        self.random_state = random_state
        self.reset()

    def reset(self):
        """Forget everything."""
        self._rng = check_random_state(self.random_state)
        self._sample = None
        self.n_seen_ = 0
        return self

    def update(self, X_chunk):
        """Add one chunk; an empty chunk is ignored.

        Complexity: O(chunk * d); memory O(capacity * d).
        """
        X = check_array(X_chunk, "X_chunk", allow_empty=True)
        if X.shape[0] == 0:
            return self
        if self._sample is None:
            self._sample = X[:0].copy()
        if self.capacity is None:
            self._sample = np.vstack([self._sample, X])
        else:
            room = self.capacity - self._sample.shape[0]
            if room > 0:                                   # still filling the reservoir
                self._sample = np.vstack([self._sample, X[:room]])
            rest = X[max(room, 0):]
            if rest.shape[0]:
                # row number t (0-based) replaces slot j ~ U{0..t} if j < capacity
                t = self.n_seen_ + max(room, 0) + np.arange(rest.shape[0])
                j = self._rng.integers(0, t + 1)
                keep = j < self.capacity
                self._sample[j[keep]] = rest[keep]
        self.n_seen_ += X.shape[0]
        return self

    def result(self, q=0.5):
        """Quantile(s) per feature; shape (n_features,) or (len(q), n_features)."""
        if self._sample is None:
            raise RuntimeError("StreamingQuantile has not seen any data yet.")
        q = np.asarray(q, dtype=float)
        if np.any((q < 0) | (q > 1)):
            raise ValueError("q must be within [0, 1].")
        with warnings.catch_warnings():                   # all-NaN column -> NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanquantile(self._sample, q, axis=0)


class StreamingHistogram:
    """Per-feature histograms with fixed bins.

    The bin range is given with ``range=(lo, hi)`` or taken from the first
    chunk. Later values outside the range are counted in the first or last
    bin, so no value is lost. Counting uses the group's
    ``numcompute.stats.histogram``.

    Parameters
    ----------
    bins : int
    range : (float, float) or None
    window : int or None
        If set, counts cover only the last ``window`` chunks.
    """

    def __init__(self, bins=10, range=None, window=None):
        self.bins = check_scalar(bins, "bins", kind="int", min_val=1)
        if range is not None and not range[0] < range[1]:
            raise ValueError(f"range must be (lo, hi) with lo < hi, got {range}.")
        self.range = range
        self.window = check_scalar(window, "window", kind="int", min_val=1, allow_none=True)
        self.reset()

    def reset(self):
        """Forget all counts."""
        self.edges_ = None
        self._counts = None
        self._recent = deque(maxlen=self.window)
        return self

    def update(self, X_chunk):
        """Add one chunk; NaN values are skipped and an empty chunk is ignored.

        Complexity: O(chunk * d); memory O(d * bins).
        """
        X = check_array(X_chunk, "X_chunk", allow_empty=True)
        if X.shape[0] == 0:
            return self
        if self.edges_ is None:
            if self.range is not None:
                lo = np.full(X.shape[1], float(self.range[0]))
                hi = np.full(X.shape[1], float(self.range[1]))
            else:
                lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
                lo, hi = np.nan_to_num(lo), np.nan_to_num(hi)
                same = hi <= lo                            # constant column: widen by 0.5
                lo, hi = np.where(same, lo - 0.5, lo), np.where(same, hi + 0.5, hi)
            # built from half values so hi - lo cannot overflow (exact for powers of two)
            self.edges_ = (np.linspace(lo / 2, hi / 2, self.bins + 1) * 2).T   # (d, bins + 1)
            self._counts = np.zeros((X.shape[1], self.bins), dtype=np.int64)
        counts = np.zeros_like(self._counts)
        for j in range(X.shape[1]):
            lo, hi = self.edges_[j, 0], self.edges_[j, -1]
            col = np.clip(X[:, j], lo, hi)
            # count the halves: same bins, but no overflow for ranges near 1e308
            counts[j] = _group_stats.histogram(col / 2, bins=self.bins,
                                               range=(lo / 2, hi / 2))[0]
        if self.window is None:
            self._counts += counts
        else:
            self._recent.append(counts)
        return self

    def result(self):
        """``(counts, edges)`` with shapes (d, bins) and (d, bins + 1)."""
        if self.edges_ is None:
            raise RuntimeError("StreamingHistogram has not seen any data yet.")
        counts = self._counts if self.window is None else sum(self._recent)
        return counts.copy(), self.edges_.copy()


class StreamStats:
    """Running moments, quantiles and histograms together (``update_stats``)."""

    def __init__(self, window=None, quantile_capacity=1000, bins=10, random_state=0):
        self.moments = RunningStats(window=window)
        self.quantiles = StreamingQuantile(quantile_capacity, random_state=random_state)
        self.histogram = StreamingHistogram(bins=bins, window=window)

    def update(self, X_chunk):
        """Feed one chunk to every statistic.

        Complexity: O(chunk * d).
        """
        X = check_array(X_chunk, "X_chunk", allow_empty=True)
        self.moments.update(X)
        self.quantiles.update(X)
        self.histogram.update(X)
        return self

    update_stats = update

    def result(self):
        """Moments plus quartiles and the histogram, as one dictionary."""
        out = self.moments.result()
        q25, q50, q75 = self.quantiles.result([0.25, 0.5, 0.75])
        out.update({"q25": q25, "median": q50, "q75": q75})
        out["hist_counts"], out["hist_edges"] = self.histogram.result()
        return out


def update_stats(X_chunk, state=None, **kwargs):
    """Fold one chunk into ``state`` (a :class:`StreamStats`, created if None)."""
    if state is None:
        state = StreamStats(**kwargs)
    return state.update(X_chunk)


__all__ = ["RunningStats", "StreamingQuantile", "StreamingHistogram", "StreamStats",
           "update_stats", "welford_loop"]
