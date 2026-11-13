
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence, float, int]

__all__ = [
    "check_array",
    "check_consistent_length",
    "pairwise_distances",
    "logsumexp",
    "softmax",
    "log_softmax",
    "sigmoid",
    "relu",
    "leaky_relu",
    "tanh",
    "topk_mask",
    "batch_iterator",
]



def check_array(
    X: ArrayLike,
    *,
    ensure_2d: bool = True,
    allow_nan: bool = True,
    dtype: Optional[type] = float,
    allow_empty: bool = False,
    name: str = "X",
) -> np.ndarray:
    
    try:
        arr = np.asarray(X, dtype=dtype) if dtype is not None else np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} could not be converted to dtype {getattr(dtype, '__name__', dtype)}: {exc}"
        ) from exc

    if ensure_2d:
        if arr.ndim == 1:
            raise ValueError(
                f"Expected 2D array for {name}, got 1D array of shape {arr.shape}. "
                "Reshape your data with X.reshape(-1, 1) for a single feature "
                "or X.reshape(1, -1) for a single sample."
            )
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for {name}, got {arr.ndim}D array of shape {arr.shape}.")
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} is empty (shape {arr.shape}); at least one element is required.")
    if not allow_nan and np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return arr


def check_consistent_length(*arrays: np.ndarray) -> None:
    
    lengths = [len(a) for a in arrays if a is not None]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent numbers of samples: {lengths}")



def pairwise_distances(
    A: ArrayLike, B: Optional[ArrayLike] = None, metric: str = "euclidean"
) -> np.ndarray:
    
    A = check_array(A, name="A")
    B = A if B is None else check_array(B, name="B")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have the same number of columns, got {A.shape[1]} and {B.shape[1]}.")

    if metric in ("euclidean", "sqeuclidean"):
        # Direct differences, not the Gram identity |a|^2 + |b|^2 - 2 a.b: that
        # identity cancels catastrophically for nearby points with large
        # coordinates (1e9 vs 1e9 + 1 gave 0 instead of 1). Differences are
        # formed in halves (exact) and scaled by their largest component, so
        # neither huge nor tiny coordinates overflow or underflow. Rows of A
        # are processed in blocks to bound memory at ~32 MB.
        # Time O(n*m*d); extra space O(block*m*d).
        D = np.empty((A.shape[0], B.shape[0]))
        block = max(1, (1 << 22) // max(1, B.shape[0] * A.shape[1]))
        for start in range(0, A.shape[0], block):
            half = A[start:start + block, None, :] * 0.5 - B[None, :, :] * 0.5
            scale = np.abs(half).max(axis=-1, initial=0.0)
            safe = np.where(scale > 0, scale, 1.0)
            r = half / safe[..., None]
            with np.errstate(over="ignore"):
                D[start:start + block] = 2.0 * scale * np.sqrt(np.einsum("ijk,ijk->ij", r, r))
        if metric == "euclidean":
            return D
        with np.errstate(over="ignore"):
            return D * D
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    if metric == "cosine":
        na = np.linalg.norm(A, axis=1)[:, None]
        nb = np.linalg.norm(B, axis=1)[None, :]
        denom = na * nb
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(denom > 0, (A @ B.T) / np.where(denom > 0, denom, 1.0), 0.0)
        return 1.0 - np.clip(sim, -1.0, 1.0)
    raise ValueError(f"Unknown metric '{metric}'. Use 'euclidean', 'sqeuclidean', 'manhattan' or 'cosine'.")



def logsumexp(x: ArrayLike, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("logsumexp of an empty array is undefined.")
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    # +inf inputs: exp(inf)=inf -> log inf = inf, which is correct already.
    if not keepdims:
        s = np.squeeze(s, axis=axis) if axis is not None else s.reshape(())
    return s[()] if np.ndim(s) == 0 else s


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("softmax of an empty array is undefined.")
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    
    x = np.asarray(x, dtype=float)
    return x - logsumexp(x, axis=axis, keepdims=True)


def sigmoid(x: ArrayLike) -> np.ndarray:
    
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out[()] if out.ndim == 0 else out


def relu(x: ArrayLike) -> np.ndarray:
    
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, np.where(np.isnan(x), np.nan, 0.0))


def leaky_relu(x: ArrayLike, alpha: float = 0.01) -> np.ndarray:
    
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)


def tanh(x: ArrayLike) -> np.ndarray:
    
    return np.tanh(np.asarray(x, dtype=float))



def topk_mask(values: ArrayLike, k: int, largest: bool = True, axis: int = -1) -> np.ndarray:
    
    a = np.asarray(values)
    n = a.shape[axis] if a.ndim else 1
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, {n}], got {k}.")
    mask = np.zeros(a.shape, dtype=bool)
    if k == 0:
        return mask
    if largest:
        idx = np.take(np.argpartition(a, n - k, axis=axis), np.arange(n - k, n), axis=axis)
    else:
        idx = np.take(np.argpartition(a, k - 1, axis=axis), np.arange(k), axis=axis)
    np.put_along_axis(mask, idx, True, axis=axis)
    return mask


def batch_iterator(
    *arrays: ArrayLike, batch_size: int = 32, shuffle: bool = False, seed: Optional[int] = None,
    drop_last: bool = False,
) -> Iterator[Tuple[np.ndarray, ...]]:
    
    if not arrays:
        raise ValueError("At least one array is required.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    arrs = [np.asarray(a) for a in arrays]
    check_consistent_length(*arrs)
    n = len(arrs[0])
    order = np.random.default_rng(seed).permutation(n) if shuffle else None
    for start in range(0, n, batch_size):  # loop over batches, not elements
        stop = min(start + batch_size, n)
        if drop_last and stop - start < batch_size:
            break
        sl = order[start:stop] if order is not None else slice(start, stop)
        out = tuple(a[sl] for a in arrs)
        yield out if len(out) > 1 else out[0]
