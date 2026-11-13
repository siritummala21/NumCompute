"""NumCompute-Stream: streaming learning on top of the group's NumCompute.

Modules
-------
io             read a CSV file whole or chunk by chunk
stats          running mean/variance, quantiles and histograms (update_stats)
preprocessing  StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
"""

from ._validation import NotFittedError
from .io import Dataset, chunk_stream, iter_csv_chunks, read_csv, write_csv
from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import RunningStats, StreamingHistogram, StreamingQuantile, StreamStats, update_stats

__version__ = "0.1.0"

__all__ = [
    "NotFittedError",
    "Dataset",
    "read_csv",
    "iter_csv_chunks",
    "chunk_stream",
    "write_csv",
    "RunningStats",
    "StreamingQuantile",
    "StreamingHistogram",
    "StreamStats",
    "update_stats",
    "StandardScaler",
    "MinMaxScaler",
    "SimpleImputer",
    "Imputer",
    "OneHotEncoder",
]
