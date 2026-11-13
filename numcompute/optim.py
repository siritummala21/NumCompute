
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = ["grad", "jacobian", "check_grad", "line_search", "gradient_descent"]

_METHODS = ("central", "forward", "backward")


def _check_positive_finite(value, name: str) -> float:
    """Return ``value`` as float if it is a finite number > 0, else raise.

    Comparisons with NaN are always False, so ``h <= 0`` alone let ``h=nan``
    through and produced NaN gradients (Assignment 2.1 feedback).
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__} {value!r}.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}.")
    return float(value)


def _steps(x: np.ndarray, h: float, relative: bool) -> np.ndarray:
    h = _check_positive_finite(h, "h")
    return h * np.maximum(1.0, np.abs(x)) if relative else np.full(x.shape, float(h))


def _perturbations(x: np.ndarray, hs: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    n = x.size
    E = np.diag(hs)                              
    base = np.broadcast_to(x, (n, n))
    if method == "central":
        return base + E, base - E, 2.0 * hs
    if method == "forward":
        return base + E, None, hs
    return None, base - E, hs                    


def _eval_rows(f: Callable, X: np.ndarray, vectorized: bool, shape) -> np.ndarray:
    if vectorized:
        return np.asarray(f(X), dtype=float)
    return np.array([np.asarray(f(row.reshape(shape)), dtype=float) for row in X])


def grad(
    f: Callable, x, h: float = 1e-5, method: str = "central",
    relative: bool = True, vectorized: bool = False,
) -> np.ndarray:
    
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    x_arr = np.asarray(x, dtype=float)
    shape = x_arr.shape
    xf = x_arr.ravel()
    if xf.size == 0:
        raise ValueError("x must contain at least one element.")
    hs = _steps(xf, h, relative)
    Xp, Xm, den = _perturbations(xf, hs, method)
    f0 = None
    if method != "central":
        f0 = np.asarray(f(x_arr.copy()), dtype=float)
        if f0.size != 1:
            raise ValueError(f"f must return a scalar, got shape {f0.shape}. Use jacobian() for vector outputs.")
        f0 = float(f0)
    fp = _eval_rows(f, Xp, vectorized, shape) if Xp is not None else f0
    fm = _eval_rows(f, Xm, vectorized, shape) if Xm is not None else f0
    g = (np.asarray(fp) - np.asarray(fm))
    if g.ndim > 1 or (np.ndim(g) == 1 and g.size != xf.size):
        raise ValueError("f must return a scalar. Use jacobian() for vector-valued functions.")
    g = g / den
    return float(g[0]) if shape == () else g.reshape(shape)


def jacobian(
    F: Callable, x, h: float = 1e-5, method: str = "central",
    relative: bool = True, vectorized: bool = False,
) -> np.ndarray:
    
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    x_arr = np.asarray(x, dtype=float)
    shape = x_arr.shape
    xf = x_arr.ravel()
    if xf.size == 0:
        raise ValueError("x must contain at least one element.")
    hs = _steps(xf, h, relative)
    Xp, Xm, den = _perturbations(xf, hs, method)

    def rows(X):
        R = _eval_rows(F, X, vectorized, shape)
        return R.reshape(xf.size, -1)            

    f0 = None
    if method != "central":
        f0 = np.asarray(F(x_arr.copy()), dtype=float).ravel()[None, :]
    fp = rows(Xp) if Xp is not None else f0
    fm = rows(Xm) if Xm is not None else f0
    J = (fp - fm) / den[:, None]                 
    return J.T


def check_grad(f: Callable, grad_f: Callable, x, h: float = 1e-6) -> float:
    
    g_fd = np.atleast_1d(grad(f, x, h=h))
    g = np.atleast_1d(np.asarray(grad_f(np.asarray(x, dtype=float)), dtype=float))
    return float(np.linalg.norm(g_fd - g) / max(1.0, np.linalg.norm(g_fd) + np.linalg.norm(g)))


def line_search(
    f: Callable, x, direction, grad_x=None, alpha0: float = 1.0, rho: float = 0.5,
    c: float = 1e-4, max_iter: int = 50,
) -> float:
    
    if not 0 < rho < 1 or not 0 < c < 1:
        raise ValueError("rho and c must lie in (0, 1).")
    alpha0 = _check_positive_finite(alpha0, "alpha0")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be an int >= 1, got {max_iter!r}.")
    x = np.asarray(x, dtype=float)
    d = np.asarray(direction, dtype=float)
    if d.shape != x.shape:
        raise ValueError(f"direction shape {d.shape} does not match x shape {x.shape}.")
    g = grad(f, x) if grad_x is None else np.asarray(grad_x, dtype=float)
    slope = float(np.sum(g * d))
    if slope >= 0:
        raise ValueError("direction is not a descent direction (grad . direction >= 0).")
    fx = float(f(x))
    alpha = alpha0
    for _ in range(max_iter):
        if float(f(x + alpha * d)) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


def gradient_descent(
    f: Callable, x0, grad_f: Optional[Callable] = None, tol: float = 1e-6,
    max_iter: int = 1000, alpha0: float = 1.0,
) -> Dict[str, object]:
    
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be a finite number >= 0, got {tol!r}.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError(f"max_iter must be an int >= 0, got {max_iter!r}.")
    _check_positive_finite(alpha0, "alpha0")
    x = np.asarray(x0, dtype=float).copy()
    history = [float(f(x))]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g = grad(f, x) if grad_f is None else np.asarray(grad_f(x), dtype=float)
        if np.linalg.norm(g) < tol:
            converged = True
            it -= 1
            break
        step = line_search(f, x, -g, grad_x=g, alpha0=alpha0)
        x = x - step * g
        history.append(float(f(x)))
    return {"x": x, "fun": history[-1], "n_iter": it, "converged": converged, "history": np.array(history)}
