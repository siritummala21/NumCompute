"""Streaming versions of the group's preprocessing classes.

Each class subclasses the matching class in ``numcompute.preprocessing`` and
adds ``partial_fit``, so the statistics are updated chunk by chunk instead of
being computed once on the whole data set.

The constant-column test is exact (``max == min``). This is the fix for the
2.1 feedback, where a column like [1e-16, 2e-16, 3e-16] was wrongly treated
as constant because its std was below a fixed threshold.
"""

import warnings

import numpy as np

import numcompute.preprocessing as _group

from ._validation import NotFittedError, check_array, check_n_features, check_scalar
from .base import TransformerMixin
from .stats import RunningStats, StreamingQuantile


def _is_missing(X):
    """Missing mask for numeric or object arrays (None, NaN or '')."""
    if X.dtype != object:
        return np.isnan(X)
    return np.array([[v is None or v == "" or (isinstance(v, float) and v != v) for v in row]
                     for row in X], dtype=bool).reshape(X.shape)


def _as_2d_object(X):
    """2-D object array (used for categorical columns)."""
    arr = np.asarray(X, dtype=object)
    if arr.size == 0:
        return arr.reshape(0, arr.shape[1] if arr.ndim == 2 else 0)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {arr.shape}.")
    return arr


def _check_for_transform(X, n_features, owner):
    """Validate X for transform; an empty chunk returns an empty (0, d) array."""
    if n_features is None:
        raise NotFittedError(f"{owner} is not fitted; call partial_fit first.")
    X = check_array(X, allow_empty=True)
    if X.shape[0] == 0:
        return np.empty((0, n_features))
    check_n_features(X, n_features, owner)
    return X


class StandardScaler(TransformerMixin, _group.StandardScaler):
    """Standardise features with a running mean and standard deviation.

    The mean and variance come from :class:`RunningStats` (Chan's merge), so
    streaming the data in chunks gives the same result as fitting it at once.
    A column is constant only if its max equals its min; then its scale is 1
    and it transforms to exactly 0.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.reset()

    def reset(self):
        """Forget the running statistics."""
        self._stats = RunningStats()
        return self

    def partial_fit(self, X_chunk, y=None):
        """Update the running mean/variance with one chunk.

        Complexity: O(chunk * d) time, O(d) memory.
        """
        self._stats.update(check_array(X_chunk, "X_chunk", allow_empty=True))
        return self

    @property
    def n_features_in_(self):
        """Number of columns seen in partial_fit (None before)."""
        return self._stats.n_features_

    @property
    def mean_(self):
        """Per-feature mean (the exact value for constant columns)."""
        constant = self._stats.max_ == self._stats.min_
        return np.where(constant, self._stats.min_, np.nan_to_num(self._stats.mean()))

    @property
    def scale_(self):
        """Per-feature std, or 1 for constant / empty columns."""
        std = np.nan_to_num(self._stats.std())
        constant = ~(self._stats.max_ > self._stats.min_)
        return np.where(constant | (std == 0), 1.0, std)

    def transform(self, X):
        """Return (X - mean) / std, shape (n_samples, n_features).

        Complexity: O(n * d).
        """
        X = _check_for_transform(X, self.n_features_in_, "StandardScaler")
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def inverse_transform(self, X):
        """Undo :meth:`transform`."""
        X = _check_for_transform(X, self.n_features_in_, "StandardScaler")
        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X


class MinMaxScaler(TransformerMixin, _group.MinMaxScaler):
    """Scale features into ``feature_range`` using the running min and max.

    Values outside the range seen so far map outside ``feature_range``
    unless ``clip=True``. Constant columns map to the lower bound.
    """

    def __init__(self, feature_range=(0.0, 1.0), clip=False):
        lo = check_scalar(feature_range[0], "feature_range[0]")
        hi = check_scalar(feature_range[1], "feature_range[1]")
        if not lo < hi:
            raise ValueError(f"feature_range must be (min, max) with min < max, "
                             f"got {feature_range}.")
        self.feature_range = (lo, hi)
        self.clip = clip
        self.reset()

    def reset(self):
        """Forget the running min/max."""
        self._stats = RunningStats()
        return self

    def partial_fit(self, X_chunk, y=None):
        """Update the running min and max with one chunk.

        Complexity: O(chunk * d) time, O(d) memory.
        """
        self._stats.update(check_array(X_chunk, "X_chunk", allow_empty=True))
        return self

    def _range(self):
        lo = np.where(np.isfinite(self._stats.min_), self._stats.min_, 0.0)
        hi = np.where(np.isfinite(self._stats.max_), self._stats.max_, lo)
        span = np.where(hi > lo, hi - lo, 1.0)            # exact constant test
        return lo, span

    def transform(self, X):
        """Scale X into ``feature_range``.

        Complexity: O(n * d).
        """
        X = _check_for_transform(X, self._stats.n_features_, "MinMaxScaler")
        lo, span = self._range()
        a, b = self.feature_range
        out = (X - lo) / span * (b - a) + a
        return np.clip(out, a, b) if self.clip else out

    def inverse_transform(self, X):
        """Map scaled values back to the original units."""
        X = _check_for_transform(X, self._stats.n_features_, "MinMaxScaler")
        lo, span = self._range()
        a, b = self.feature_range
        return (X - a) / (b - a) * span + lo


class SimpleImputer(TransformerMixin, _group.SimpleImputer):
    """Fill missing values with a running estimate per column.

    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}
        'mean' uses the running mean, 'median' a reservoir sample, and
        'most_frequent' running value counts. 'most_frequent' and 'constant'
        also work on categorical (object) columns.
    fill_value : scalar, optional
        Used by 'constant', and for columns with no observed value yet.
    """

    STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy="mean", fill_value=None):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}.")
        if isinstance(fill_value, float) and not np.isfinite(fill_value):
            raise ValueError("fill_value must be finite.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.reset()

    def reset(self):
        """Forget all running estimates."""
        self.n_features_in_ = None
        self._is_object = False
        self._moments = RunningStats()
        self._sample = StreamingQuantile(capacity=1000, random_state=0)
        self._counts = []
        return self

    def _prepare(self, X):
        arr = np.asarray(X)
        if arr.dtype == object or arr.dtype.kind in "US":
            if self.strategy in ("mean", "median"):
                raise TypeError(f"strategy={self.strategy!r} needs numeric data; use "
                                "'most_frequent' or 'constant' for categorical columns.")
            return _as_2d_object(arr)
        return check_array(arr, allow_empty=True)

    def partial_fit(self, X_chunk, y=None):
        """Update the fill values with one chunk (missing entries are skipped).

        Complexity: O(chunk * d); 'most_frequent' also keeps one count per distinct value.
        """
        X = self._prepare(X_chunk)
        if X.shape[0] == 0:
            return self
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self._is_object = X.dtype == object
            self._counts = [{} for _ in range(X.shape[1])]
        check_n_features(X, self.n_features_in_, "SimpleImputer")
        if self.strategy == "mean":
            self._moments.update(X)
        elif self.strategy == "median":
            self._sample.update(X)
        elif self.strategy == "most_frequent":
            missing = _is_missing(X)
            for j in range(X.shape[1]):                   # one np.unique per column
                values, counts = np.unique(X[~missing[:, j], j].astype(str)
                                           if self._is_object else X[~missing[:, j], j],
                                           return_counts=True)
                for v, c in zip(values.tolist(), counts.tolist()):
                    self._counts[j][v] = self._counts[j].get(v, 0) + c
        return self

    @property
    def statistics_(self):
        """Current fill value per column."""
        if self.n_features_in_ is None:
            raise NotFittedError("SimpleImputer is not fitted; call partial_fit first.")
        default = self.fill_value if self.fill_value is not None else (
            "missing" if self._is_object else 0.0)
        if self.strategy == "constant":
            return np.array([default] * self.n_features_in_, dtype=object)
        if self.strategy == "most_frequent":
            # ties go to the smallest value so the result is reproducible
            stats = [min(c, key=lambda k: (-c[k], k)) if c else default for c in self._counts]
            return np.array(stats, dtype=object if self._is_object else float)
        stats = self._moments.mean() if self.strategy == "mean" else self._sample.result(0.5)
        if np.isnan(stats).any():
            warnings.warn(f"Some columns have no values yet; filling them with {default!r}.",
                          RuntimeWarning)
        return np.where(np.isnan(stats), default, stats)

    def transform(self, X):
        """Return a copy of X with missing entries replaced.

        Complexity: O(n * d).
        """
        if self.n_features_in_ is None:
            raise NotFittedError("SimpleImputer is not fitted; call partial_fit first.")
        X = self._prepare(X)
        if X.shape[0] == 0:
            return X.reshape(0, self.n_features_in_)
        check_n_features(X, self.n_features_in_, "SimpleImputer")
        out = X.copy()
        rows, cols = np.nonzero(_is_missing(X))
        out[rows, cols] = self.statistics_[cols]
        return out


class OneHotEncoder(TransformerMixin, _group.OneHotEncoder):
    """One-hot encoder that learns new categories as the stream goes on.

    A new category is added as the *last* output column, so arrays encoded
    earlier keep their column positions (older rows just have 0 in the new
    column). The tree relies on this when its input gets wider. Missing values
    (None/NaN) and, with ``handle_unknown='ignore'``, unseen categories
    encode as all zeros.
    """

    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")
        self.handle_unknown = handle_unknown
        self.reset()

    def reset(self):
        """Forget all categories."""
        self.n_features_in_ = None
        self.categories_ = []          # per input column, in order of discovery
        self._column_of = []           # per input column: category -> output column
        self._names = []
        return self

    def partial_fit(self, X_chunk, y=None):
        """Register categories not seen before.

        Complexity: O(chunk * d) time; memory O(number of categories).
        """
        X = _as_2d_object(X_chunk)
        if X.shape[0] == 0:
            return self
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.categories_ = [[] for _ in range(X.shape[1])]
            self._column_of = [{} for _ in range(X.shape[1])]
        check_n_features(X, self.n_features_in_, "OneHotEncoder")
        missing = _is_missing(X)
        for j in range(X.shape[1]):
            for v in dict.fromkeys(X[~missing[:, j], j].tolist()):   # keeps first-seen order
                if v not in self._column_of[j]:
                    self._column_of[j][v] = len(self._names)
                    self.categories_[j].append(v)
                    self._names.append((j, v))
        return self

    @property
    def n_features_out_(self):
        """Number of one-hot output columns."""
        return len(self._names)

    def get_feature_names_out(self, input_features=None):
        """Output column names such as ``'site=A'``."""
        names = input_features or [f"x{j}" for j in range(self.n_features_in_ or 0)]
        return [f"{names[j]}={v}" for j, v in self._names]

    get_feature_names = get_feature_names_out     # name used by numcompute

    def transform(self, X):
        """Encode X into shape (n_samples, n_features_out_).

        Complexity: O(n * d) to look up categories, O(n * n_out) for the output.
        """
        if self.n_features_in_ is None:
            raise NotFittedError("OneHotEncoder is not fitted; call partial_fit first.")
        X = _as_2d_object(X)
        out = np.zeros((X.shape[0], self.n_features_out_))
        if X.shape[0] == 0:
            return out
        check_n_features(X, self.n_features_in_, "OneHotEncoder")
        missing = _is_missing(X)
        for j in range(X.shape[1]):
            cols = np.array([self._column_of[j].get(v, -1) for v in X[:, j].tolist()])
            unknown = (cols < 0) & ~missing[:, j]
            if unknown.any() and self.handle_unknown == "error":
                raise ValueError(f"Unknown categories {sorted(set(X[unknown, j]))} "
                                 f"in column {j}.")
            rows = np.nonzero(cols >= 0)[0]
            out[rows, cols[rows]] = 1.0
        return out

    def inverse_transform(self, X):
        """Map one-hot rows back to categories (all-zero blocks give None)."""
        X = np.asarray(X, dtype=float)
        out = np.full((X.shape[0], self.n_features_in_), None, dtype=object)
        for j, cats in enumerate(self.categories_):
            if not cats:
                continue
            block = X[:, [self._column_of[j][c] for c in cats]]
            best = block.argmax(axis=1)
            hit = block.max(axis=1) > 0
            out[hit, j] = np.array(cats, dtype=object)[best[hit]]
        return out


#: Name used in the assignment brief.
Imputer = SimpleImputer

__all__ = ["StandardScaler", "MinMaxScaler", "SimpleImputer", "Imputer", "OneHotEncoder"]
