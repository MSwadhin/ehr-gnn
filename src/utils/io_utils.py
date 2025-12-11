import os
from typing import Any

import pandas as pd


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists. If 'path' is a file path,
    ensure its parent directory exists.
    """
    directory = path
    if os.path.splitext(path)[1]:
        directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Wrapper around pandas.read_csv with sensible defaults for MIMIC-sized CSVs.
    """
    default_kwargs = {
        "low_memory": False
    }
    default_kwargs.update(kwargs)
    return pd.read_csv(path, **default_kwargs)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Write pandas DataFrame to Parquet, ensuring directory exists.
    We force timestamps to millisecond precision so that Spark 4
    can read them (Spark doesn't like TIMESTAMP(NANOS)).
    """
    ensure_dir(path)
    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        coerce_timestamps="ms",
        allow_truncated_timestamps=True,
    )



def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
