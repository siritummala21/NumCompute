
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "BaseEstimator",
    "TransformerMixin",
    "Pipeline",
    "make_pipeline",
    "FeatureUnion",
    "ColumnTransformer",
    "FunctionTransformer",
    "MeanRegressor",
    "MajorityClassifier",
    "LeastSquaresRegressor",
    "LogisticRegressionGD",
]



class BaseEstimator:
    

    @classmethod
    def _param_names(cls) -> List[str]:
        sig = inspect.signature(cls.__init__)
        return [p.name for p in sig.parameters.values()
                if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]

    def get_params(self) -> Dict[str, Any]:
        
        return {k: getattr(self, k) for k in self._param_names()}

    def set_params(self, **params) -> "BaseEstimator":
        
        valid = set(self._param_names())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Invalid parameter {k!r} for {type(self).__name__}. Valid: {sorted(valid)}")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class TransformerMixin:
    

    def fit_transform(self, X, y=None) -> np.ndarray:
        _fit(self, X, y)            # works for subclasses whose fit takes X only
        return self.transform(X)


def _accepts_y(method) -> bool:
    """Whether a bound ``fit``-like method takes ``y`` as a second positional argument."""
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return True
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return any(p.kind == p.VAR_POSITIONAL for p in params) or len(positional) >= 2


def _fit(est, X, y=None):
    """Call ``est.fit`` with or without ``y``, as its signature allows.

    External transformers often define ``fit(self, X)`` only, and may return
    ``None`` instead of ``self``; neither should break a pipeline
    (Assignment 2.1 feedback).
    """
    return est.fit(X, y) if _accepts_y(est.fit) else est.fit(X)


def _fit_transform(est, X, y=None):
    """Fit then transform ``X`` with any transformer; never chains on fit's return value."""
    if hasattr(est, "fit_transform"):
        method = est.fit_transform
        return method(X, y) if _accepts_y(method) else method(X)
    _fit(est, X, y)
    return est.transform(X)


def _is_transformer(obj) -> bool:
    return hasattr(obj, "fit") and hasattr(obj, "transform")


def _is_passthrough(obj) -> bool:
    return obj is None or (isinstance(obj, str) and obj == "passthrough")



class Pipeline(BaseEstimator):
    
    def __init__(self, steps: Sequence[Tuple[str, Any]]):
        self.steps = list(steps)
        self._validate()

    def _validate(self) -> None:
        if not self.steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [n for n, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique, got {names}.")
        for name, est in self.steps[:-1]:
            if not (_is_passthrough(est) or _is_transformer(est)):
                raise TypeError(
                    f"All intermediate steps must implement fit and transform; "
                    f"step {name!r} ({type(est).__name__}) does not."
                )
        name, last = self.steps[-1]
        if not (_is_passthrough(last) or hasattr(last, "fit")):
            raise TypeError(f"Final step {name!r} must implement fit.")

    @property
    def named_steps(self) -> Dict[str, Any]:
        return dict(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Pipeline(self.steps[idx])
        if isinstance(idx, str):
            return self.named_steps[idx]
        return self.steps[idx][1]

    def get_params(self) -> Dict[str, Any]:
        return {"steps": self.steps}

    def __repr__(self) -> str:
        inner = ",\n  ".join(f"({n!r}, {e!r})" for n, e in self.steps)
        return f"Pipeline([\n  {inner}\n])"

    def _fit_transform_head(self, X, y):
        Xt = X
        for _, est in self.steps[:-1]:
            if _is_passthrough(est):
                continue
            Xt = _fit_transform(est, Xt, y)
        return Xt

    def _transform_head(self, X):
        Xt = X
        for _, est in self.steps[:-1]:
            if not _is_passthrough(est):
                Xt = est.transform(Xt)
        return Xt

    def fit(self, X, y=None) -> "Pipeline":
        
        Xt = self._fit_transform_head(X, y)
        last = self.steps[-1][1]
        if not _is_passthrough(last):
            _fit(last, Xt, y)
        return self

    def fit_transform(self, X, y=None) -> np.ndarray:
        Xt = self._fit_transform_head(X, y)
        last = self.steps[-1][1]
        if _is_passthrough(last):
            return Xt
        if not hasattr(last, "transform"):
            raise AttributeError(f"Final step {type(last).__name__} has no transform method.")
        return _fit_transform(last, Xt, y)

    def transform(self, X) -> np.ndarray:
        Xt = self._transform_head(X)
        last = self.steps[-1][1]
        if _is_passthrough(last):
            return Xt
        if not hasattr(last, "transform"):
            raise AttributeError(f"Final step {type(last).__name__} has no transform method.")
        return last.transform(Xt)

    def predict(self, X) -> np.ndarray:
        last = self.steps[-1][1]
        if not hasattr(last, "predict"):
            raise AttributeError(f"Final step {type(last).__name__} has no predict method.")
        return last.predict(self._transform_head(X))

    def predict_proba(self, X) -> np.ndarray:
        last = self.steps[-1][1]
        if not hasattr(last, "predict_proba"):
            raise AttributeError(f"Final step {type(last).__name__} has no predict_proba method.")
        return last.predict_proba(self._transform_head(X))


def make_pipeline(*steps) -> Pipeline:
    
    names: List[str] = []
    for s in steps:
        base = "passthrough" if _is_passthrough(s) else type(s).__name__.lower()
        name, i = base, 1
        while name in names:
            i += 1
            name = f"{base}-{i}"
        names.append(name)
    return Pipeline(list(zip(names, steps)))



def _as_2d(X) -> np.ndarray:
    X = np.asarray(X)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class FeatureUnion(BaseEstimator, TransformerMixin):
    

    def __init__(self, transformer_list: Sequence[Tuple[str, Any]]):
        self.transformer_list = list(transformer_list)
        for name, t in self.transformer_list:
            if not _is_transformer(t):
                raise TypeError(f"{name!r} is not a transformer (needs fit and transform).")

    def fit(self, X, y=None) -> "FeatureUnion":
        for _, t in self.transformer_list:
            _fit(t, X, y)
        return self

    def transform(self, X) -> np.ndarray:
        return np.hstack([_as_2d(t.transform(X)) for _, t in self.transformer_list])


class ColumnTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, transformers: Sequence[Tuple[str, Any, Sequence[int]]], remainder: str = "drop"):
        self.transformers = list(transformers)
        self.remainder = remainder
        if remainder not in ("drop", "passthrough"):
            raise ValueError("remainder must be 'drop' or 'passthrough'.")

    def _columns(self, cols, n_features) -> np.ndarray:
        cols = np.asarray(cols)
        if cols.dtype == bool:
            if cols.size != n_features:
                raise ValueError("Boolean column mask length does not match number of features.")
            return np.flatnonzero(cols)
        cols = cols.astype(int)
        if cols.size and (cols.max() >= n_features or cols.min() < -n_features):
            raise IndexError(f"Column index out of range for X with {n_features} features.")
        return cols % n_features

    def fit(self, X, y=None) -> "ColumnTransformer":
        self.fit_transform(X, y)
        return self

    def _apply(self, X, fit: bool, y=None) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"ColumnTransformer expects a 2D array, got shape {X.shape}.")
        n = X.shape[1]
        if fit:
            self.n_features_in_ = n
        elif n != self.n_features_in_:
            raise ValueError(f"X has {n} features, expected {self.n_features_in_}.")
        used = np.zeros(n, dtype=bool)
        outs = []
        for name, t, cols in self.transformers:
            idx = self._columns(cols, n)
            used[idx] = True
            sub = X[:, idx]
            if isinstance(t, str) and t == "drop":
                continue
            if _is_passthrough(t):
                outs.append(sub)
            elif fit:
                outs.append(_as_2d(_fit_transform(t, sub, y)))
            else:
                outs.append(_as_2d(t.transform(sub)))
        if self.remainder == "passthrough" and (~used).any():
            outs.append(X[:, ~used])
        if not outs:
            return np.empty((X.shape[0], 0))
        try:
            return np.hstack([o.astype(float) for o in outs])
        except (TypeError, ValueError):
            return np.hstack(outs)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self._apply(X, fit=True, y=y)

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("ColumnTransformer is not fitted yet. Call 'fit' first.")
        return self._apply(X, fit=False)


class FunctionTransformer(BaseEstimator, TransformerMixin):
    

    def __init__(self, func: Optional[Callable] = None, inverse_func: Optional[Callable] = None):
        self.func = func
        self.inverse_func = inverse_func

    def fit(self, X, y=None) -> "FunctionTransformer":
        return self

    def transform(self, X):
        return X if self.func is None else self.func(X)

    def inverse_transform(self, X):
        return X if self.inverse_func is None else self.inverse_func(X)



def _check_Xy(X, y=None):
    from .utils import check_array, check_consistent_length
    X = check_array(X, allow_nan=False)
    if y is None:
        return X, None
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}.")
    check_consistent_length(X, y)
    return X, y


class MeanRegressor(BaseEstimator):
    

    def fit(self, X, y) -> "MeanRegressor":
        _, y = _check_Xy(X, y)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X) -> np.ndarray:
        X, _ = _check_Xy(X)
        return np.full(X.shape[0], self.mean_)


class MajorityClassifier(BaseEstimator):
    

    def fit(self, X, y) -> "MajorityClassifier":
        _, y = _check_Xy(X, y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.majority_ = self.classes_[np.argmax(counts)]
        self.class_prior_ = counts / counts.sum()
        return self

    def predict(self, X) -> np.ndarray:
        X, _ = _check_Xy(X)
        return np.full(X.shape[0], self.majority_)

    def predict_proba(self, X) -> np.ndarray:
        X, _ = _check_Xy(X)
        return np.tile(self.class_prior_, (X.shape[0], 1))


class LeastSquaresRegressor(BaseEstimator):
    

    def __init__(self, fit_intercept: bool = True, alpha: float = 0.0):
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def fit(self, X, y) -> "LeastSquaresRegressor":
        X, y = _check_Xy(X, y)
        y = y.astype(float)
        if self.fit_intercept:
            xm, ym = X.mean(0), y.mean()
            Xc, yc = X - xm, y - ym
        else:
            xm, ym, Xc, yc = np.zeros(X.shape[1]), 0.0, X, y
        if self.alpha > 0:
            d = X.shape[1]
            Xc = np.vstack([Xc, np.sqrt(self.alpha) * np.eye(d)])
            yc = np.concatenate([yc, np.zeros(d)])
        self.coef_ = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        self.intercept_ = float(ym - xm @ self.coef_)
        return self

    def predict(self, X) -> np.ndarray:
        X, _ = _check_Xy(X)
        return X @ self.coef_ + self.intercept_


class LogisticRegressionGD(BaseEstimator):
    
    def __init__(self, lr: float = 0.1, n_iter: int = 500, alpha: float = 0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.alpha = alpha

    def fit(self, X, y) -> "LogisticRegressionGD":
        from .utils import sigmoid
        X, y = _check_Xy(X, y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(f"LogisticRegressionGD is binary; got {self.classes_.size} classes.")
        t = (y == self.classes_[1]).astype(float)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        for _ in range(self.n_iter):  
            p = sigmoid(X @ w + b)
            err = p - t
            w -= self.lr * (X.T @ err / n + self.alpha * w)
            b -= self.lr * err.mean()
        self.coef_, self.intercept_ = w, b
        return self

    def decision_function(self, X) -> np.ndarray:
        X, _ = _check_Xy(X)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        from .utils import sigmoid
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]
