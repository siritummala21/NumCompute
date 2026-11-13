"""Reading CSV files as a stream of chunks (no pandas).

* ``read_csv`` loads a whole file into a ``Dataset``.
* ``iter_csv_chunks`` reads the file lazily and yields one ``Dataset`` of
  ``chunk_size`` rows at a time, so only one chunk is in memory.
* ``chunk_stream`` splits arrays that are already in memory into chunks.

Missing values use the group's tokens (``numcompute.io.DEFAULT_MISSING``,
compared case-insensitively). They become NaN in numeric columns and None in
categorical columns.
"""

import csv
import os
from dataclasses import dataclass, field

import numpy as np

from numcompute import io as _group_io

from ._validation import check_random_state, check_scalar

DEFAULT_MISSING = tuple(dict.fromkeys(t.lower() for t in _group_io.DEFAULT_MISSING))


@dataclass
class Dataset:
    """One file or one chunk: numeric features, categorical features, labels."""

    X: np.ndarray                         # (n, n_numeric) float, NaN = missing
    X_cat: np.ndarray                     # (n, n_categorical) object, None = missing
    y: np.ndarray = None                  # (n,) int if all labels are integers, else str
    numeric_names: list = field(default_factory=list)
    categorical_names: list = field(default_factory=list)
    target_name: str = None

    def __len__(self):
        return self.X.shape[0]


def _to_float(column, missing, name):
    """Convert a column of strings to float, with missing tokens as NaN."""
    column = np.char.strip(column)
    is_missing = np.isin(np.char.lower(column), missing)
    try:
        return np.where(is_missing, "nan", column).astype(float)
    except ValueError:
        raise ValueError(f"Column {name!r} has a value that is not a number. If it is "
                         f"categorical, list it in categorical=.") from None


def _to_labels(column):
    """Integer labels when every label is a whole number, otherwise strings."""
    column = np.char.strip(column)
    try:
        values = column.astype(float)
    except ValueError:
        return column.astype(object)
    if np.all(values == np.round(values)):
        return values.astype(np.int64)
    return values


def _make_dataset(header, rows, target, categorical, missing):
    """Turn rows of strings into a Dataset, converting one column at a time."""
    for name in list(categorical) + ([target] if target else []):
        if name not in header:
            raise KeyError(f"Column {name!r} is not in the header {header}.")
    table = np.array(rows, dtype=str).reshape(len(rows), len(header))
    col = {name: j for j, name in enumerate(header)}
    num_names = [c for c in header if c not in categorical and c != target]
    cat_names = [c for c in header if c in categorical and c != target]

    X = np.empty((len(rows), len(num_names)))
    for j, name in enumerate(num_names):
        X[:, j] = _to_float(table[:, col[name]], missing, name)
    X_cat = np.empty((len(rows), len(cat_names)), dtype=object)
    for j, name in enumerate(cat_names):
        values = np.char.strip(table[:, col[name]])
        X_cat[:, j] = np.where(np.isin(np.char.lower(values), missing), None,
                               values.astype(object))
    y = None
    if target is not None:
        if np.isin(np.char.lower(np.char.strip(table[:, col[target]])), missing).any():
            raise ValueError(f"The target column {target!r} has missing values.")
        y = _to_labels(table[:, col[target]])
    return Dataset(X, X_cat, y, num_names, cat_names, target)


def _rows(path, delimiter):
    """Yield (header, None) once, then (line_number, row) for every data row."""
    with open(path, newline="", encoding="utf-8-sig") as fh:   # -sig handles an Excel BOM
        reader = csv.reader(fh, delimiter=delimiter)
        header = [h.strip() for h in next(reader, [])]
        if not header:
            raise ValueError(f"{path!r} is empty.")
        yield header, None
        for line, row in enumerate(reader, start=2):
            if not row:                                        # blank line
                continue
            if len(row) != len(header):
                raise ValueError(f"Line {line} has {len(row)} fields, the header has "
                                 f"{len(header)}.")
            yield line, row


def read_csv(path, target=None, categorical=(), delimiter=",",
             missing_values=DEFAULT_MISSING):
    """Load the whole CSV file as one Dataset."""
    missing = tuple(m.lower() for m in missing_values)
    reader = _rows(path, delimiter)
    header, _ = next(reader)
    rows = [row for _, row in reader]
    if not rows:
        raise ValueError(f"{path!r} has a header but no data rows.")
    return _make_dataset(header, rows, target, categorical, missing)


def iter_csv_chunks(path, chunk_size, target=None, categorical=(), delimiter=",",
                    missing_values=DEFAULT_MISSING):
    """Yield the file as Datasets of ``chunk_size`` rows (the last may be smaller).

    Complexity: O(rows * columns) time overall; O(chunk_size * columns) memory.
    """
    chunk_size = check_scalar(chunk_size, "chunk_size", kind="int", min_val=1)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    missing = tuple(m.lower() for m in missing_values)

    def generate():
        """Read rows and yield a Dataset every chunk_size rows."""
        reader = _rows(path, delimiter)
        header, _ = next(reader)
        rows = []
        for _, row in reader:
            rows.append(row)
            if len(rows) == chunk_size:
                yield _make_dataset(header, rows, target, categorical, missing)
                rows = []
        if rows:
            yield _make_dataset(header, rows, target, categorical, missing)

    # the checks above run now, not when the first chunk is requested
    return generate()


def chunk_stream(X, y=None, chunk_size=100, shuffle=False, random_state=None):
    """Split arrays into consecutive chunks to simulate a stream.

    Yields ``X_chunk``, or ``(X_chunk, y_chunk)`` when ``y`` is given.
    """
    X = np.asarray(X)
    chunk_size = check_scalar(chunk_size, "chunk_size", kind="int", min_val=1)
    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
    order = np.arange(X.shape[0])
    if shuffle:
        check_random_state(random_state).shuffle(order)

    def generate():
        """Yield the chunks in order."""
        for start in range(0, X.shape[0], chunk_size):
            idx = order[start:start + chunk_size]
            yield X[idx] if y is None else (X[idx], y[idx])

    return generate()


def write_csv(path, X, header, y=None, target_name="target", delimiter=","):
    """Write features (and labels) to CSV; NaN and None become empty fields."""
    X = np.asarray(X, dtype=object)
    if X.ndim != 2 or X.shape[1] != len(header):
        raise ValueError("X must be 2-D with one column per header name.")

    def text(v):
        """Text for one cell."""
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(list(header) + ([target_name] if y is not None else []))
        for i in range(X.shape[0]):
            row = [text(v) for v in X[i]]
            writer.writerow(row + [text(y[i])] if y is not None else row)


__all__ = ["Dataset", "read_csv", "iter_csv_chunks", "chunk_stream", "write_csv",
           "DEFAULT_MISSING"]
