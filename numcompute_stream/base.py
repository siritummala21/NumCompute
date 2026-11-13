"""Base classes, taken from the group package and extended for streaming.

``BaseEstimator`` (``get_params`` / ``set_params`` / ``__repr__``) and
``TransformerMixin`` come from ``numcompute.pipeline``. Streaming adds:

* ``partial_fit(X_chunk, y_chunk)`` - update the state with one chunk;
* ``reset()`` - forget everything learned;
* ``fit(X, y)`` - ``reset()`` followed by one ``partial_fit`` on all of X.
"""

import numpy as np

from numcompute.pipeline import BaseEstimator
from numcompute.pipeline import TransformerMixin as _GroupTransformerMixin

__all__ = ["BaseEstimator", "TransformerMixin", "ClassifierMixin", "set_params_checked"]


def set_params_checked(est, **params):
    """``set_params`` that runs the same checks as ``__init__``.

    The group ``set_params`` only sets the attribute, so a value like
    ``max_depth=-5`` was accepted without an error. Here the object is built
    again with the new values, which checks them and forgets what was
    learned (old statistics may not fit the new settings).
    """
    current = est.get_params()
    unknown = sorted(set(params) - set(current))
    if unknown:
        raise ValueError(f"Invalid parameter(s) {unknown} for {type(est).__name__}. "
                         f"Valid: {sorted(current)}")
    fresh = type(est)(**{**current, **params})     # raises before anything changes
    est.__dict__.update(fresh.__dict__)
    return est


class TransformerMixin(_GroupTransformerMixin):
    """Group ``TransformerMixin`` plus the streaming ``fit`` rule."""

    def set_params(self, **params):
        """Change parameters (checked like in ``__init__``); this resets the state."""
        return set_params_checked(self, **params)

    def fit(self, X, y=None):
        """Reset, then learn from ``X`` as a single chunk."""
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit_transform(self, X, y=None):
        """Update with the chunk, then transform it."""
        return self.partial_fit(X, y).transform(X)


class ClassifierMixin:
    """Adds ``score`` (accuracy) and a checked ``set_params`` to classifiers."""

    def set_params(self, **params):
        """Change parameters (checked like in ``__init__``); this resets the tree."""
        return set_params_checked(self, **params)

    def score(self, X, y):
        """Mean accuracy of ``predict(X)`` against ``y``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
