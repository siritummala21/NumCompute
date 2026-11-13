"""Pipeline and ColumnTransformer with ``partial_fit``.

Both classes extend the group versions in ``numcompute.pipeline``. The group
``Pipeline`` already handles ``fit``, ``predict``, ``predict_proba``,
passthrough steps and steps whose ``fit`` takes only ``X`` (the 2.1 fix).
What is added here is the streaming part:

``partial_fit(X_chunk, y_chunk)`` goes through the steps in order. Each step
is updated with the chunk and then transforms it for the next step:

* a step with ``partial_fit`` is updated with every chunk;
* a step with only ``fit`` (e.g. from another library) is fitted on the first
  chunk and then kept fixed;
* a step with only ``transform`` has nothing to learn.

Example
-------
>>> pipe = Pipeline([("scale", StandardScaler()),
...                  ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=500))])
>>> pipe.partial_fit(X_chunk, y_chunk)          # doctest: +SKIP
"""

import numpy as np

from numcompute import pipeline as _group
from numcompute.pipeline import _accepts_y, _is_passthrough


def _has(obj, method):
    return callable(getattr(obj, method, None))


def _call(method, X, y=None):
    """Call ``method(X, y)`` or ``method(X)``, depending on what it accepts."""
    return method(X, y) if _accepts_y(method) else method(X)


def _update(step, X, y, fitted_once, key):
    """Update one step with a chunk (see the rules in the module docstring)."""
    if _has(step, "partial_fit"):
        _call(step.partial_fit, X, y)
    elif _has(step, "fit") and key not in fitted_once:
        _call(step.fit, X, y)
        fitted_once.add(key)


class Pipeline(_group.Pipeline):
    """The group ``Pipeline`` plus ``partial_fit`` and ``reset``.

    It also accepts intermediate steps that only have ``transform`` (the
    group version needs ``fit`` as well), because the module notes above
    promise that such steps work.
    """

    def __init__(self, steps):
        super().__init__(steps)
        self._fitted_once = set()

    def _validate(self):
        if not self.steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [n for n, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique, got {names}.")
        for name, step in self.steps[:-1]:
            if not (_is_passthrough(step) or _has(step, "transform")):
                raise TypeError(f"Intermediate step {name!r} ({type(step).__name__}) needs "
                                "a transform() method.")
        name, last = self.steps[-1]
        if not (_is_passthrough(last) or _has(last, "fit") or _has(last, "partial_fit")):
            raise TypeError(f"Final step {name!r} needs fit() or partial_fit().")

    def _fit_transform_head(self, X, y):
        """Used by the group ``fit``: transform-only steps are applied without fitting."""
        Xt = X
        for _, step in self.steps[:-1]:
            if _is_passthrough(step):
                continue
            Xt = _group._fit_transform(step, Xt, y) if _has(step, "fit") else step.transform(Xt)
        return Xt

    @property
    def final_estimator(self):
        """The last step (the tree in this project)."""
        return self.steps[-1][1]

    def reset(self):
        """Reset every step that can be reset."""
        for _, step in self.steps:
            if _has(step, "reset"):
                step.reset()
        self._fitted_once = set()
        return self

    def partial_fit(self, X_chunk, y_chunk=None):
        """Update every step with one chunk; an empty chunk is ignored.

        Complexity: the sum of the steps' partial_fit and transform costs.
        """
        if np.shape(X_chunk)[0] == 0:
            return self
        final = self.final_estimator
        if not _has(final, "partial_fit"):
            raise TypeError(f"The last step ({type(final).__name__}) has no partial_fit(); "
                            "use fit() instead.")
        Xt = X_chunk
        for name, step in self.steps[:-1]:
            if _is_passthrough(step):
                continue
            _update(step, Xt, y_chunk, self._fitted_once, name)
            Xt = step.transform(Xt)
        _call(final.partial_fit, Xt, y_chunk)
        return self

    def score(self, X, y):
        """Accuracy of ``predict(X)``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def is_fitted(self):
        """False while the final tree is still waiting for its first full buffer."""
        return bool(getattr(self.final_estimator, "is_fitted", True))

    def memory_bytes(self):
        """Memory reported by the steps (only the tree reports it here)."""
        return sum(s.memory_bytes() for _, s in self.steps if _has(s, "memory_bytes"))


def _is_categorical(t):
    """True if ``t`` is, or contains, a OneHotEncoder (it needs the raw values)."""
    from .preprocessing import OneHotEncoder
    if isinstance(t, Pipeline):
        return any(isinstance(s, OneHotEncoder) for _, s in t.steps)
    return isinstance(t, OneHotEncoder)


class ColumnTransformer(_group.ColumnTransformer):
    """Different transformers for different columns, with ``partial_fit``.

    ``transformers`` is a list of ``(name, transformer, column_indices)``.
    The input may be an object array that mixes numbers and strings: numeric
    blocks are converted to float, blocks for a OneHotEncoder are passed as
    they are. Put the OneHotEncoder last, so the columns it adds for new
    categories end up at the end of the output.
    """

    def __init__(self, transformers, remainder="drop"):
        super().__init__(transformers, remainder)
        for name, t, _ in self.transformers:
            if not _has(t, "transform"):
                raise TypeError(f"Transformer {name!r} has no transform().")
        self._fitted_once = set()

    def _check(self, X):
        X = np.asarray(X)
        X = X.astype(object) if X.dtype.kind in "OUS" else X.astype(float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} columns, expected {self.n_features_in_}.")
        return X

    def _block(self, X, t, cols):
        block = X[:, self._columns(cols, X.shape[1])]
        return block if _is_categorical(t) else block.astype(float)

    def reset(self):
        """Reset every inner transformer."""
        for _, t, _ in self.transformers:
            if _has(t, "reset"):
                t.reset()
        self._fitted_once = set()
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_
        return self

    def partial_fit(self, X_chunk, y=None):
        """Update each transformer with its columns of the chunk."""
        if np.shape(X_chunk)[0] == 0:
            return self
        X = self._check(X_chunk)
        for name, t, cols in self.transformers:
            _update(t, self._block(X, t, cols), y, self._fitted_once, name)
        return self

    def fit(self, X, y=None):
        """Reset, then learn from X as one chunk."""
        return self.reset().partial_fit(X, y)

    def transform(self, X):
        """Transform each block and stack the results into one float array.

        Complexity: the transformers' costs plus O(n * d_out) to stack.
        """
        X = self._check(X)
        blocks = []
        for _, t, cols in self.transformers:
            out = np.asarray(t.transform(self._block(X, t, cols)), dtype=float)
            blocks.append(out.reshape(-1, 1) if out.ndim == 1 else out)
        if self.remainder == "passthrough":
            blocks.append(X[:, self._remainder(X.shape[1])].astype(float))
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))

    def fit_transform(self, X, y=None):
        """fit(X) then transform(X)."""
        return self.fit(X, y).transform(X)

    def _remainder(self, d):
        used = {int(c) for _, _, cols in self.transformers for c in cols}
        return [c for c in range(d) if c not in used]

    def get_feature_names_out(self, input_features):
        """Output column names, given the input column names."""
        names = []
        for _, t, cols in self.transformers:
            sub = [input_features[c] for c in cols]
            last = t.final_estimator if isinstance(t, Pipeline) else t
            names += last.get_feature_names_out(sub) if _has(last, "get_feature_names_out") \
                else sub
        if self.remainder == "passthrough":
            names += [input_features[c] for c in self._remainder(len(input_features))]
        return names


def make_pipeline(*steps):
    """Pipeline with step names taken from the class names."""
    return Pipeline(_group.make_pipeline(*steps).steps)


__all__ = ["Pipeline", "ColumnTransformer", "make_pipeline"]
