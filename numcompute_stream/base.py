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

__all__ = ["BaseEstimator", "TransformerMixin", "ClassifierMixin"]


class TransformerMixin(_GroupTransformerMixin):
    """Group ``TransformerMixin`` plus the streaming ``fit`` rule."""

    def fit(self, X, y=None):
        """Reset, then learn from ``X`` as a single chunk."""
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit_transform(self, X, y=None):
        """Update with the chunk, then transform it."""
        return self.partial_fit(X, y).transform(X)


class ClassifierMixin:
    """Adds ``score`` (accuracy) to classifiers."""

    def score(self, X, y):
        """Mean accuracy of ``predict(X)`` against ``y``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
