"""Input checks shared by every module.

Conventions: ``X`` is 2-D with shape (n_samples, n_features), samples along
axis 0; ``y`` is 1-D with shape (n_samples,).
"""

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when a method needs a fitted model but none is available."""


def check_array(X, name="X", allow_empty=False, dtype=float):
    """Return ``X`` as a 2-D array of shape (n_samples, n_features).

    A 1-D array is rejected because it is ambiguous (one sample or one
    feature). NaN is allowed; infinity is not. With ``allow_empty=True`` a
    chunk with 0 rows is accepted (streaming methods ignore it).
    """
    if X is None:
        raise TypeError(f"{name} must be array-like, got None.")
    try:
        arr = np.asarray(X, dtype=dtype) if dtype is not None else np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} could not be converted to a numeric array: {exc}") from exc
    if allow_empty and arr.size == 0 and arr.ndim <= 2:
        return arr.reshape(0, arr.shape[1] if arr.ndim == 2 else 0)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features); got shape {arr.shape}. "
                         f"Use X.reshape(-1, 1) for one feature or X.reshape(1, -1) "
                         f"for one sample.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty (0 samples).")
    if np.issubdtype(arr.dtype, np.floating) and np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values.")
    return arr


def check_labels(y, name="y", allow_empty=False):
    """Return ``y`` as a 1-D label array; NaN/inf labels are rejected."""
    if y is None:
        raise TypeError(f"{name} must be array-like, got None.")
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,); got shape {arr.shape}.")
    if arr.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty (0 samples).")
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite labels.")
    return arr


def check_X_y(X, y, allow_empty=False):
    """Validate ``X`` and ``y`` together and check they have the same length."""
    X = check_array(X, allow_empty=allow_empty)
    y = check_labels(y, allow_empty=allow_empty)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}.")
    return X, y


def check_n_features(X, expected, owner):
    """Raise if ``X`` does not have ``expected`` columns."""
    if X.shape[1] != expected:
        raise ValueError(f"{owner} was fitted with {expected} features but got {X.shape[1]}.")


def check_scalar(value, name, kind="float", min_val=None, max_val=None, allow_none=False):
    """Validate a numeric hyper-parameter (rejects bool, NaN, inf, wrong type, out of range)."""
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number, got bool.")
    if kind == "int":
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
        value = int(value)
    else:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}.")
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}.")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}.")
    return value


def check_random_state(seed):
    """Turn None, an int or a Generator into a ``np.random.Generator``."""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    raise TypeError(f"random_state must be None, int or Generator; got {type(seed).__name__}.")
