
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .pipeline import BaseEstimator, TransformerMixin
from .utils import check_array

__all__ = ["StandardScaler", "MinMaxScaler", "SimpleImputer", "OneHotEncoder"]


class _FittedMixin:
    _fitted_attr = ""

    def _check_is_fitted(self) -> None:
        if not hasattr(self, self._fitted_attr):
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet. Call 'fit' first."
            )

    def _check_n_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was fitted with "
                f"{self.n_features_in_} features."
            )



class StandardScaler(TransformerMixin, BaseEstimator, _FittedMixin):
    
    _fitted_attr = "mean_"

    def __init__(self, with_mean: bool = True, with_std: bool = True, ddof: int = 0):
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof

    def fit(self, X, y=None) -> "StandardScaler":
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Work in units of a per-column power of two (exact) so that tiny
            # (1e-170) and huge (1e308) values neither underflow nor overflow.
            absmax = np.nanmax(np.abs(X), axis=0)
            exp = np.clip(np.frexp(np.where(np.isfinite(absmax), absmax, 0.0))[1], -1021, 1023)
            unit = np.ldexp(1.0, exp)
            Xs = X / unit
            mean = np.nanmean(Xs, axis=0) * unit
            std = np.nanstd(Xs, axis=0, ddof=self.ddof) * unit
            constant = np.nanmax(X, axis=0) == np.nanmin(X, axis=0)
        self.mean_ = np.where(np.isnan(mean), 0.0, mean) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            # A column is constant only if max == min (exact). The old absolute
            # threshold std < 10*eps wrongly treated [1e-16, 2e-16, 3e-16] as
            # constant (Assignment 2.1 feedback).
            std = np.where(np.isnan(std) | constant | (std == 0), 1.0, std)
            self.scale_ = std
        else:
            self.scale_ = np.ones(X.shape[1])
        with np.errstate(over="ignore"):
            self.var_ = self.scale_ ** 2      # inf if std > ~1.3e154 (not representable)
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X = check_array(X)
        self._check_n_features(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X = check_array(X)
        self._check_n_features(X)
        return X * self.scale_ + self.mean_



class MinMaxScaler(TransformerMixin, BaseEstimator, _FittedMixin):
    

    _fitted_attr = "scale_"

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), clip: bool = False):
        self.feature_range = feature_range
        self.clip = clip

    def fit(self, X, y=None) -> "MinMaxScaler":
        lo, hi = self.feature_range
        if lo >= hi:
            raise ValueError(f"feature_range minimum must be smaller than maximum, got {self.feature_range}.")
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dmin = np.nanmin(X, axis=0)
            dmax = np.nanmax(X, axis=0)
        dmin = np.where(np.isnan(dmin), 0.0, dmin)
        dmax = np.where(np.isnan(dmax), 0.0, dmax)
        with np.errstate(over="ignore"):
            rng = dmax - dmin
        # halves are exact and avoid overflow when dmax - dmin exceeds the float range
        half_rng = dmax * 0.5 - dmin * 0.5
        safe = np.where(dmax == dmin, 1.0, half_rng)
        self.data_min_, self.data_max_, self.data_range_ = dmin, dmax, rng
        self.scale_ = ((hi - lo) * 0.5) / safe
        self.min_ = lo - dmin * self.scale_
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X = check_array(X)
        self._check_n_features(X)
        out = X * self.scale_ + self.min_
        if self.clip:
            np.clip(out, self.feature_range[0], self.feature_range[1], out=out)
        return out

    def inverse_transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X = check_array(X)
        self._check_n_features(X)
        return (X - self.min_) / self.scale_



class SimpleImputer(TransformerMixin, BaseEstimator, _FittedMixin):
    
    _fitted_attr = "statistics_"
    _strategies = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy: str = "mean", fill_value: Optional[float] = None):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None) -> "SimpleImputer":
        if self.strategy not in self._strategies:
            raise ValueError(f"strategy must be one of {self._strategies}, got {self.strategy!r}.")
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        fallback = 0.0 if self.fill_value is None else float(self.fill_value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.strategy == "mean":
                stats = np.nanmean(X, axis=0)
            elif self.strategy == "median":
                stats = np.nanmedian(X, axis=0)
            elif self.strategy == "constant":
                stats = np.full(X.shape[1], fallback)
            else:
                stats = np.empty(X.shape[1])
                for j in range(X.shape[1]):  
                    col = X[:, j][~np.isnan(X[:, j])]
                    if col.size == 0:
                        stats[j] = np.nan
                        continue
                    vals, counts = np.unique(col, return_counts=True)
                    stats[j] = vals[np.argmax(counts)]  
        self.statistics_ = np.where(np.isnan(stats), fallback, stats)
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X = check_array(X)
        self._check_n_features(X)
        return np.where(np.isnan(X), self.statistics_, X)



class OneHotEncoder(TransformerMixin, BaseEstimator, _FittedMixin):
    
    _fitted_attr = "categories_"

    def __init__(self, categories: Union[str, Sequence] = "auto", handle_unknown: str = "error",
                 drop_first: bool = False, dtype=float):
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first
        self.dtype = dtype

    @staticmethod
    def _prepare(X) -> np.ndarray:
        X = check_array(X, dtype=None)
        if X.dtype == object:
            X = X.astype(str)
        return X

    def fit(self, X, y=None) -> "OneHotEncoder":
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")
        X = self._prepare(X)
        self.n_features_in_ = X.shape[1]
        if isinstance(self.categories, str):
            if self.categories != "auto":
                raise ValueError("categories must be 'auto' or a list of arrays.")
            cats = [np.unique(X[:, j]) for j in range(X.shape[1])]
        else:
            if len(self.categories) != X.shape[1]:
                raise ValueError(
                    f"categories has {len(self.categories)} entries but X has {X.shape[1]} features."
                )
            cats = [np.unique(np.asarray(c, dtype=X.dtype)) for c in self.categories]
        self.categories_ = cats
        return self

    def _encode_column(self, col: np.ndarray, cats: np.ndarray, j: int) -> np.ndarray:
        idx = np.searchsorted(cats, col)
        idx_c = np.clip(idx, 0, len(cats) - 1)
        found = cats[idx_c] == col
        if np.issubdtype(cats.dtype, np.floating):
            found |= np.isnan(cats[idx_c]) & np.isnan(col)
        if not found.all() and self.handle_unknown == "error":
            unknown = np.unique(col[~found])
            raise ValueError(f"Found unknown categories {unknown.tolist()} in column {j} during transform.")
        out = np.zeros((col.shape[0], len(cats)), dtype=self.dtype)
        rows = np.flatnonzero(found)
        out[rows, idx_c[found]] = 1
        return out[:, 1:] if self.drop_first else out

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X = self._prepare(X)
        self._check_n_features(X)
        blocks = [self._encode_column(X[:, j], c, j) for j, c in enumerate(self.categories_)]
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0), dtype=self.dtype)

    def inverse_transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        if self.drop_first:
            raise NotImplementedError("inverse_transform is not supported with drop_first=True.")
        X = check_array(X)
        sizes = [len(c) for c in self.categories_]
        if X.shape[1] != sum(sizes):
            raise ValueError(f"Expected {sum(sizes)} columns, got {X.shape[1]}.")
        splits = np.split(X, np.cumsum(sizes)[:-1], axis=1)
        cols = [c[np.argmax(block, axis=1)] for c, block in zip(self.categories_, splits)]
        return np.column_stack(cols)

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        self._check_is_fitted()
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        names = []
        for f, cats in zip(input_features, self.categories_):
            cs = cats[1:] if self.drop_first else cats
            names.extend(f"{f}={c}" for c in cs)
        return names
