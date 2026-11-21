from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agy_trader.core.models import ensure_utc

REQUIRED_BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
CANONICAL_BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]


class CorporateActionMismatchError(ValueError):
    pass


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(pd.DatetimeIndex([], tz="UTC"), name="timestamp")

    if series.isna().any():
        raise ValueError("Timestamps must not contain NaN/NaT")

    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")

    if pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("Timestamps must be timezone-aware (received naive datetime64)")

    converted: list[pd.Timestamp] = []
    for raw in series.tolist():
        if pd.isna(raw):
            raise ValueError("Timestamps must not contain NaN/NaT")
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware")
        converted.append(ts.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(converted, tz="UTC"), index=series.index, name="timestamp")


def _empty_bars_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "vwap": pd.Series(dtype="float64"),
        }
    )


class ParquetBarStore:
    """Atomic Parquet OHLCV bar storage partitioned by timeframe and symbol."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, timeframe: str = "1d") -> Path:
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            raise ValueError("Symbol must not be empty")
        return self.base_dir / f"bars_{timeframe}" / f"{clean_symbol}.parquet"

    def save_bars(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = "1d",
        allow_overwrite: bool = False,
    ) -> int:
        missing_cols = [col for col in REQUIRED_BAR_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required bar columns: {missing_cols}")

        if df.empty:
            return 0

        work = df.copy()
        work["timestamp"] = _normalize_timestamp_series(work["timestamp"])
        if "vwap" not in work.columns:
            work["vwap"] = float("nan")
        work = work[CANONICAL_BAR_COLUMNS]

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            work[col] = pd.to_numeric(work[col], errors="raise").astype("float64")
        work["vwap"] = pd.to_numeric(work["vwap"], errors="coerce").astype("float64")

        if not np.isfinite(work[numeric_cols].to_numpy()).all():
            raise ValueError("OHLCV columns must contain only finite values")

        non_null_vwap = work["vwap"].dropna()
        if not non_null_vwap.empty and (
            not np.isfinite(non_null_vwap.to_numpy()).all() or (non_null_vwap <= 0.0).any()
        ):
            raise ValueError("Non-null vwap values must be finite and strictly positive")

        price_cols = ["open", "high", "low", "close"]
        if (work[price_cols] <= 0.0).any().any() or work[price_cols].isna().any().any():
            raise ValueError("OHLC prices must be strictly positive and non-null")
        if (work["volume"] < 0.0).any() or work["volume"].isna().any():
            raise ValueError("Volume must be non-negative and non-null")
        if (work["high"] < work["low"]).any():
            raise ValueError("Bar high must be >= low")
        if ((work["high"] < work["open"]) | (work["high"] < work["close"])).any():
            raise ValueError("Bar high must be >= open and close")
        if ((work["low"] > work["open"]) | (work["low"] > work["close"])).any():
            raise ValueError("Bar low must be <= open and close")

        target_path = self.path_for(symbol, timeframe=timeframe)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and not allow_overwrite:
            existing = pd.read_parquet(target_path)
            if not existing.empty:
                existing = existing.copy()
                existing["timestamp"] = _normalize_timestamp_series(existing["timestamp"])
                if "vwap" not in existing.columns:
                    existing["vwap"] = float("nan")
                existing = existing[CANONICAL_BAR_COLUMNS]

                existing_dedup = existing.drop_duplicates(subset=["timestamp"], keep="last")
                work_dedup = work.drop_duplicates(subset=["timestamp"], keep="last")
                overlap = existing_dedup[["timestamp", "close"]].merge(
                    work_dedup[["timestamp", "close"]],
                    on="timestamp",
                    how="inner",
                    suffixes=("_old", "_new"),
                )
                if not overlap.empty:
                    rel_diff = (overlap["close_new"] - overlap["close_old"]).abs() / overlap[
                        "close_old"
                    ]
                    if (rel_diff > 1e-4).any():
                        max_diff = float(rel_diff.max())
                        raise CorporateActionMismatchError(
                            f"Overlapping close mismatch for {symbol.strip().upper()} "
                            f"({timeframe}): max relative diff {max_diff:.6%} > 1e-4; "
                            "pass allow_overwrite=True to replace existing bars."
                        )

                combined = pd.concat([existing, work], ignore_index=True)
            else:
                combined = work
        else:
            combined = work

        combined = (
            combined.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp", ascending=True)
            .reset_index(drop=True)
        )

        fd, tmp_name = tempfile.mkstemp(
            dir=str(target_path.parent),
            prefix=f".{target_path.stem}_",
            suffix=".parquet.tmp",
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            combined.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, target_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return len(combined)

    def load_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        target_path = self.path_for(symbol, timeframe=timeframe)
        if not target_path.exists():
            return _empty_bars_frame()

        df = pd.read_parquet(target_path)
        if df.empty:
            return _empty_bars_frame()

        df = df.copy()
        df["timestamp"] = _normalize_timestamp_series(df["timestamp"])
        if "vwap" not in df.columns:
            df["vwap"] = float("nan")
        df = df[CANONICAL_BAR_COLUMNS]

        if start is not None:
            start_utc = pd.Timestamp(ensure_utc(start))
            df = df[df["timestamp"] >= start_utc]
        if end is not None:
            end_utc = pd.Timestamp(ensure_utc(end))
            df = df[df["timestamp"] <= end_utc]

        return df.sort_values("timestamp", ascending=True).reset_index(drop=True)

    def list_symbols(self, timeframe: str = "1d") -> list[str]:
        tf_dir = self.base_dir / f"bars_{timeframe}"
        if not tf_dir.exists():
            return []
        return sorted(p.stem.upper() for p in tf_dir.glob("*.parquet") if p.is_file())

    def load_universe_bars(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        return {
            sym.strip().upper(): self.load_bars(
                sym, timeframe=timeframe, start=start, end=end
            )
            for sym in symbols
            if sym.strip()
        }
