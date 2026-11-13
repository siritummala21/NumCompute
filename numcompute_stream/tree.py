"""Decision tree classifier (CART).

How a split is found (vectorised)
---------------------------------
For each candidate feature the rows are sorted once. The labels in that
order are one-hot encoded and summed cumulatively, so row i of the result
holds the class counts of the left child if we split after row i. The right
child is ``total - left``. The Gini or entropy of every possible split is then
one array expression, and ``argmax`` picks the best one. This is
O(n log n) per feature instead of O(n^2) for trying every threshold in a loop
(``best_split_loop``, kept for the benchmark).

Edge cases
----------
* Only positions where the sorted value changes are valid thresholds, so equal
  values are never split apart (ties).
* Ties between splits: the lowest feature index, then the lowest threshold.
* NaN features: the split is scored on the non-missing rows, and NaN rows go
  to the child that received more training rows.
* A chunk with one class or constant features gives a single leaf.
"""

import time

import numpy as np

from ._validation import NotFittedError, check_array, check_random_state, check_scalar, \
    check_X_y
from .base import BaseEstimator, ClassifierMixin


def gini(counts):
    """Gini impurity 1 - sum(p^2) along the last axis (0 for an empty node)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(total > 0, total, 1.0)
    return 1.0 - np.sum(p * p, axis=-1) * (total[..., 0] > 0)


def entropy(counts):
    """Entropy in bits along the last axis, with 0 * log 0 = 0."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(total > 0, total, 1.0)
    return -np.sum(p * np.log2(np.where(p > 0, p, 1.0)), axis=-1)


CRITERIA = {"gini": gini, "entropy": entropy}


def _threshold(lo, hi):
    """Mid-point between two sorted values, guaranteed to satisfy lo <= t < hi."""
    mid = lo + (hi - lo) / 2.0
    return mid if lo <= mid < hi else lo        # rounding can land on hi


def best_split_vectorised(X, y, n_classes, features, criterion="gini", min_samples_leaf=1):
    """Best (feature, threshold) for the rows ``X``, ``y`` (labels 0..C-1).

    Returns a dict ``{'feature', 'threshold', 'gain', 'nan_left'}``, or None
    when no split improves the impurity.

    Complexity: O(k n log n + k n C) time and O(k n C) memory for n rows, k features, C classes.
    """
    impurity = CRITERIA[criterion]
    n = X.shape[0]
    if n < 2 * min_samples_leaf:
        return None
    Xf = X[:, features]                                    # (n, k)
    order = np.argsort(Xf, axis=0, kind="stable")          # NaN sorts last
    xs = np.take_along_axis(Xf, order, axis=0)
    onehot = np.eye(n_classes)[y[order]]                   # (n, k, C)
    left = np.cumsum(onehot, axis=0)                       # left counts after row i

    n_valid = (~np.isnan(xs)).sum(axis=0)                  # non-NaN rows per feature
    total = left[np.maximum(n_valid - 1, 0), np.arange(len(features))]
    total = total * (n_valid > 0)[:, None]                 # (k, C)
    n_left = np.arange(1, n)[:, None]                      # (n-1, 1)
    n_right = n_valid[None, :] - n_left                    # (n-1, k)
    right = np.maximum(total[None] - left[:-1], 0)         # clamp rows inside the NaN tail

    with np.errstate(invalid="ignore"):
        changes = xs[1:] > xs[:-1]                         # False across NaNs and ties
    valid = changes & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    if not valid.any():
        return None

    weighted = (n_left * impurity(left[:-1]) + np.maximum(n_right, 0) * impurity(right)) \
        / np.maximum(n_valid, 1)
    gain = np.where(valid, impurity(total)[None, :] - weighted, -np.inf)
    # argmax over the transposed array = lowest feature first, then lowest threshold
    j, i = divmod(int(np.argmax(gain.T)), n - 1)
    if gain[i, j] <= 1e-12:
        return None
    return {"feature": int(features[j]),
            "threshold": float(_threshold(xs[i, j], xs[i + 1, j])),
            "gain": float(gain[i, j]),
            "nan_left": bool(i + 1 >= n_valid[j] - (i + 1))}


def best_split_loop(X, y, n_classes, features, criterion="gini", min_samples_leaf=1):
    """Same result as :func:`best_split_vectorised`, with plain Python loops.

    Tries every threshold of every feature and recounts the classes each time
    (O(n^2) per feature). Only used as the benchmark baseline.

    Complexity: O(k n^2) time (every threshold recounts the classes).
    """
    impurity = CRITERIA[criterion]
    best, best_gain = None, 1e-12
    for f in features:
        col = X[:, f]
        ok = ~np.isnan(col)
        xv, yv = col[ok], y[ok]
        if xv.size < 2:
            continue
        parent = impurity(np.bincount(yv, minlength=n_classes))
        values = sorted(set(xv.tolist()))
        for lo, hi in zip(values[:-1], values[1:]):
            t = _threshold(lo, hi)
            go_left = xv <= t
            nl, nr = int(go_left.sum()), int((~go_left).sum())
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            child = (nl * impurity(np.bincount(yv[go_left], minlength=n_classes))
                     + nr * impurity(np.bincount(yv[~go_left], minlength=n_classes))) / xv.size
            if parent - child > best_gain + 1e-15:
                best_gain = parent - child
                best = {"feature": int(f), "threshold": float(t),
                        "gain": float(parent - child), "nan_left": nl >= nr}
    return best


SPLITTERS = {"vectorised": best_split_vectorised, "loop": best_split_loop}


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Decision tree classifier.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth (the root is depth 0); None grows until leaves are pure.
    min_samples_split : int
        A node needs at least this many samples to be split.
    min_samples_leaf : int
        Each child must keep at least this many samples.
    max_features : int, float, 'sqrt', 'log2' or None
        Number of features tried at each node (random subset), None = all.
    criterion : {'gini', 'entropy'}
    splitter : {'vectorised', 'loop'}
        'loop' is the slow reference version used in the benchmark.
    random_state : int or None
        Seed for the feature subsets.

    Example
    -------
    >>> tree = DecisionTreeClassifier(max_depth=5).fit(X, y)   # doctest: +SKIP
    >>> tree.predict(X_new)                                  # doctest: +SKIP
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, criterion="gini", splitter="vectorised", random_state=None):
        self.max_depth = check_scalar(max_depth, "max_depth", kind="int", min_val=0,
                                      allow_none=True)
        self.min_samples_split = check_scalar(min_samples_split, "min_samples_split",
                                              kind="int", min_val=2)
        self.min_samples_leaf = check_scalar(min_samples_leaf, "min_samples_leaf",
                                             kind="int", min_val=1)
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}.")
        if splitter not in SPLITTERS:
            raise ValueError(f"splitter must be 'vectorised' or 'loop', got {splitter!r}.")
        if isinstance(max_features, str) and max_features not in ("sqrt", "log2"):
            raise ValueError("max_features must be an int, a float in (0, 1], 'sqrt', "
                             "'log2' or None.")
        if isinstance(max_features, float):
            check_scalar(max_features, "max_features", min_val=1e-12, max_val=1.0)
        elif not isinstance(max_features, str):
            check_scalar(max_features, "max_features", kind="int", min_val=1, allow_none=True)
        self.max_features = max_features
        self.criterion = criterion
        self.splitter = splitter
        self.random_state = random_state
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        """Drop the tree."""
        self._rng = check_random_state(self.random_state)
        self.classes_ = None
        self.n_features_in_ = None
        self._tree = None
        self.n_retrains_ = 0
        self.retrain_history_ = []
        return self

    def _add_classes(self, y):
        if self.classes_ is not None and (self.classes_.dtype.kind in "iufb") != \
                (y.dtype.kind in "iufb"):
            raise TypeError("Label type changed mid-stream (numbers vs strings).")
        self.classes_ = np.unique(y) if self.classes_ is None else \
            np.union1d(self.classes_, np.unique(y))

    @property
    def is_fitted(self):
        """True once a tree has been trained."""
        return self._tree is not None

    def _check_fitted(self):
        if self._tree is None:
            raise NotFittedError("No tree trained yet: call fit() first.")

    # ------------------------------------------------------------- training
    def fit(self, X, y):
        """Train a tree from scratch on X (n_samples, n_features) and y (n_samples,).

        Complexity: about O(depth * d * n log n) for n rows and d features.
        """
        X, y = check_X_y(X, y)
        self.reset()
        self._add_classes(y)
        self._build(X, y)
        return self

    def _n_try(self, d):
        """How many features to try at each node."""
        m = self.max_features
        if m is None:
            return d
        if m == "sqrt":
            return max(1, int(np.sqrt(d)))
        if m == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(m, float):
            return max(1, int(m * d))
        return min(m, d)

    def _build(self, X, y):
        """Grow the tree and store it in flat arrays (fast to predict with)."""
        start = time.perf_counter()
        n, d = X.shape
        y_codes = np.searchsorted(self.classes_, y)
        n_classes = self.classes_.size
        split_fn = SPLITTERS[self.splitter]
        k = self._n_try(d)
        nodes = {"feature": [], "threshold": [], "left": [], "right": [], "nan_left": [],
                 "counts": [], "depth": []}
        importance = np.zeros(d)

        def grow(rows, depth):
            """Make a node for ``rows`` and split it if allowed; returns its index."""
            node = len(nodes["feature"])
            counts = np.bincount(y_codes[rows], minlength=n_classes)
            for key, value in (("feature", -1), ("threshold", np.nan), ("left", -1),
                               ("right", -1), ("nan_left", True), ("counts", counts),
                               ("depth", depth)):
                nodes[key].append(value)
            pure = np.count_nonzero(counts) <= 1
            if pure or rows.size < self.min_samples_split or \
                    (self.max_depth is not None and depth >= self.max_depth):
                return node
            features = np.arange(d) if k == d else np.sort(self._rng.choice(d, k, replace=False))
            split = split_fn(X[rows], y_codes[rows], n_classes, features, self.criterion,
                             self.min_samples_leaf)
            if split is None:
                return node
            col = X[rows, split["feature"]]
            go_left = np.where(np.isnan(col), split["nan_left"], col <= split["threshold"])
            importance[split["feature"]] += rows.size / n * split["gain"]
            nodes["feature"][node] = split["feature"]
            nodes["threshold"][node] = split["threshold"]
            nodes["nan_left"][node] = split["nan_left"]
            nodes["left"][node] = grow(rows[go_left], depth + 1)
            nodes["right"][node] = grow(rows[~go_left], depth + 1)
            return node

        grow(np.arange(n), 0)
        self._tree = {key: np.array(value) for key, value in nodes.items()}
        self._tree["classes"] = self.classes_.copy()
        self.feature_importances_ = importance / importance.sum() if importance.sum() else \
            importance
        self.n_features_in_ = d
        self.n_retrains_ += 1
        self.retrain_history_.append({"retrain": self.n_retrains_, "n_samples": n,
                                      "seconds": time.perf_counter() - start,
                                      "depth": self.get_depth(),
                                      "n_leaves": self.get_n_leaves()})

    # ----------------------------------------------------------- prediction
    def _check_X(self, X):
        self._check_fitted()
        X = check_array(X, allow_empty=True)
        if X.shape[0] == 0:
            return np.empty((0, self.n_features_in_))
        if X.shape[1] < self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the tree uses "
                             f"{self.n_features_in_}.")
        return X

    def apply(self, X):
        """Leaf index for each sample. All samples move down one level per loop step.

        Complexity: O(n * depth) with one NumPy step per level.
        """
        X = self._check_X(X)
        t = self._tree
        node = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.get_depth()):
            inner = t["left"][node] >= 0
            if not inner.any():
                break
            rows, nd = np.nonzero(inner)[0], node[inner]
            x = X[rows, t["feature"][nd]]
            go_left = np.where(np.isnan(x), t["nan_left"][nd], x <= t["threshold"][nd])
            node[rows] = np.where(go_left, t["left"][nd], t["right"][nd])
        return node

    def predict_proba(self, X):
        """Class probabilities, shape (n_samples, n_classes)."""
        counts = self._tree["counts"][self.apply(X)].astype(float)
        proba = counts / counts.sum(axis=1, keepdims=True)
        if self._tree["classes"].size == self.classes_.size:
            return proba
        # classes first seen after the last retrain get probability 0
        full = np.zeros((proba.shape[0], self.classes_.size))
        full[:, np.searchsorted(self.classes_, self._tree["classes"])] = proba
        return full

    def predict(self, X):
        """Predicted labels, shape (n_samples,).

        Complexity: O(n * depth).
        """
        counts = self._tree["counts"][self.apply(X)]
        return self._tree["classes"][np.argmax(counts, axis=1)]

    def predict_loop(self, X):
        """One sample at a time in Python (benchmark baseline only)."""
        X = self._check_X(X)
        t, out = self._tree, []
        for row in X:
            node = 0
            while t["left"][node] >= 0:
                x = row[t["feature"][node]]
                left = t["nan_left"][node] if np.isnan(x) else x <= t["threshold"][node]
                node = t["left"][node] if left else t["right"][node]
            out.append(t["classes"][np.argmax(t["counts"][node])])
        return np.array(out)

    # ------------------------------------------------------------ inspection
    def get_depth(self):
        """Depth of the latest tree (a single leaf has depth 0)."""
        self._check_fitted()
        return int(self._tree["depth"].max())

    def get_n_leaves(self):
        """Number of leaves of the latest tree."""
        self._check_fitted()
        return int((self._tree["left"] < 0).sum())

    def memory_bytes(self):
        """Bytes used by the tree arrays."""
        return int(sum(a.nbytes for a in self._tree.values())) if self._tree else 0

    def export_text(self, feature_names=None, decimals=3):
        """The tree as indented text."""
        self._check_fitted()
        t = self._tree
        names = feature_names or [f"x{i}" for i in range(self.n_features_in_)]
        lines = []

        def show(node, indent):
            """Add the lines for ``node`` and its children."""
            pad = "|   " * indent
            if t["left"][node] < 0:
                label = t["classes"][np.argmax(t["counts"][node])]
                lines.append(f"{pad}|--- class: {label} (n={t['counts'][node].sum()})")
                return
            name, thr = names[t["feature"][node]], round(float(t["threshold"][node]), decimals)
            lines.append(f"{pad}|--- {name} <= {thr}")
            show(t["left"][node], indent + 1)
            lines.append(f"{pad}|--- {name} >  {thr}")
            show(t["right"][node], indent + 1)

        show(0, 0)
        return "\n".join(lines)


#: Name used in the brief.
DecisionTree = DecisionTreeClassifier

__all__ = ["DecisionTreeClassifier", "DecisionTree", "gini", "entropy",
           "best_split_vectorised", "best_split_loop"]
