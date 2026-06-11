from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from pandas.errors import EmptyDataError
except Exception:  # pragma: no cover - fallback for older pandas versions
    EmptyDataError = ValueError


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except EmptyDataError:
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)