"""Convert daily stock prices into 24 monthly bars and technical indicators.

Only Pandas and Python's standard library are used. Calculations never write
files; write_symbol_files is the separate output boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import pandas as pd


EXPECTED_TICKERS = (
    "AAPL", "AMD", "AMZN", "AVGO", "CSCO",
    "MSFT", "NFLX", "PEP", "TMUS", "TSLA",
)
INPUT_COLUMNS = [
    "date", "volume", "open", "high", "low", "close", "adjclose", "ticker",
]
PRICE_COLUMNS = ["open", "high", "low", "close", "adjclose"]
NUMERIC_COLUMNS = ["volume", *PRICE_COLUMNS]


@dataclass(frozen=True)
class IndicatorConfig:
    """Periods are months because indicators are applied after resampling."""

    donchian_window: int = 20
    bollinger_window: int = 20
    bollinger_k: float = 2.0
    zscore_window: int = 20

    def __post_init__(self) -> None:
        for name in ("donchian_window", "bollinger_window", "zscore_window"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2.")
        if not isfinite(self.bollinger_k) or self.bollinger_k <= 0:
            raise ValueError("bollinger_k must be finite and positive.")


def load_daily_prices(input_path: str | Path) -> pd.DataFrame:
    """Read, validate, and chronologically sort the daily CSV.

    Reject incomplete prices instead of letting first()/last() silently skip
    them. Reject duplicate keys instead of guessing which record is correct.
    """
    daily = pd.read_csv(input_path, dtype={"ticker": "string"})
    missing = sorted(set(INPUT_COLUMNS) - set(daily.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if daily.empty:
        raise ValueError("The input CSV is empty.")

    daily = daily.loc[:, INPUT_COLUMNS].copy()
    daily["date"] = pd.to_datetime(
        daily["date"], format="%Y-%m-%d", errors="raise"
    )
    daily["ticker"] = daily["ticker"].str.strip().str.upper()
    if daily[["date", "ticker"]].isna().any().any():
        raise ValueError("date and ticker must not be missing.")

    symbols = set(daily["ticker"])
    if symbols != set(EXPECTED_TICKERS):
        missing_symbols = sorted(set(EXPECTED_TICKERS) - symbols)
        extra_symbols = sorted(symbols - set(EXPECTED_TICKERS))
        raise ValueError(
            f"Unexpected ticker set; missing={missing_symbols}, extra={extra_symbols}"
        )

    for column in NUMERIC_COLUMNS:
        daily[column] = pd.to_numeric(daily[column], errors="raise")
    numeric = daily[NUMERIC_COLUMNS]
    if numeric.isna().any().any() or numeric.isin(
        [float("inf"), -float("inf")]
    ).any().any():
        raise ValueError("Numeric input values must be finite and non-null.")
    if daily[PRICE_COLUMNS].le(0).any().any():
        raise ValueError("Prices must be positive.")
    if daily["volume"].lt(0).any() or daily["volume"].mod(1).ne(0).any():
        raise ValueError("Volume must be a nonnegative integer.")
    daily["volume"] = daily["volume"].astype("int64")

    invalid_ohlc = (
        daily["low"].gt(daily[["open", "close"]].min(axis=1))
        | daily["high"].lt(daily[["open", "close"]].max(axis=1))
        | daily["low"].gt(daily["high"])
    )
    if invalid_ohlc.any():
        raise ValueError("Daily OHLC values have inconsistent high/low bounds.")
    if daily.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate (ticker, date) records are not allowed.")

    return daily.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_daily_calendar(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Require the same 24 consecutive observed calendar months per ticker.

    This checks monthly coverage, not completeness against an exchange holiday
    calendar. No missing month is fabricated or filled with a made-up price.
    """
    month = daily["date"].dt.to_period("M")
    periods = pd.period_range(month.min(), month.max(), freq="M")
    if len(periods) != 24:
        raise ValueError(f"Expected 24 consecutive months; found {len(periods)}.")

    counts = daily.assign(month=month).groupby(["ticker", "month"]).size()
    expected = pd.MultiIndex.from_product(
        [EXPECTED_TICKERS, periods], names=["ticker", "month"]
    )
    if counts.reindex(expected, fill_value=0).eq(0).any():
        raise ValueError("Every ticker must have observations in all 24 months.")
    return periods.to_timestamp("M").rename("date")


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute OHLC snapshots/extrema, total volume, and last adjusted close."""
    ordered = daily.sort_values(["ticker", "date"]).set_index("date")
    monthly = (
        ordered.groupby("ticker")
        .resample("ME")
        .agg({
            "volume": "sum",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "adjclose": "last",
        })
        .reset_index()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return monthly.loc[:, INPUT_COLUMNS]


def sma_seeded_ema(close: pd.Series, period: int) -> pd.Series:
    """Seed EMA at month N with SMA(N), then apply alpha=2/(N+1).

    Plain ewm(adjust=False, min_periods=N) still seeds at the first close.
    Replacing the first N observations with N-1 NaNs and one SMA supplies
    the required seed without a Python loop over monthly records.
    """
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError("EMA period must be a positive integer.")
    if close.isna().any():
        raise ValueError("EMA input must not contain missing closes.")
    seeded = close.astype("float64").copy()
    if len(seeded) < period:
        return pd.Series(float("nan"), index=close.index, name=close.name)
    seeded.iloc[:period - 1] = float("nan")
    seeded.iloc[period - 1] = close.iloc[:period].mean()
    return seeded.ewm(span=period, adjust=False, min_periods=1).mean()


def add_indicators(
    monthly: pd.DataFrame, config: IndicatorConfig = IndicatorConfig()
) -> pd.DataFrame:
    """Add trailing indicators independently within each ticker.

    Windows include the current completed month. Sample standard deviation
    (ddof=1) follows the assignment's unqualified rolling(n).std() formula.
    """
    result = monthly.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    groups = result.groupby("ticker", sort=False)
    close = groups["close"]
    for period in (10, 20):
        result[f"sma_{period}"] = close.transform(
            lambda s: s.rolling(period, min_periods=period).mean()
        )
        result[f"ema_{period}"] = close.transform(
            lambda s: sma_seeded_ema(s, period)
        )

    n = config.donchian_window
    upper = groups["high"].transform(
        lambda s: s.rolling(n, min_periods=n).max()
    )
    lower = groups["low"].transform(
        lambda s: s.rolling(n, min_periods=n).min()
    )
    result[f"donchian_upper_{n}"] = upper
    result[f"donchian_lower_{n}"] = lower
    result[f"donchian_middle_{n}"] = (upper + lower) / 2

    n = config.bollinger_window
    middle = close.transform(lambda s: s.rolling(n, min_periods=n).mean())
    sd = close.transform(lambda s: s.rolling(n, min_periods=n).std(ddof=1))
    result[f"bollinger_middle_{n}"] = middle
    result[f"bollinger_upper_{n}"] = middle + config.bollinger_k * sd
    result[f"bollinger_lower_{n}"] = middle - config.bollinger_k * sd

    n = config.zscore_window
    mean = close.transform(lambda s: s.rolling(n, min_periods=n).mean())
    sd = close.transform(lambda s: s.rolling(n, min_periods=n).std(ddof=1))
    result[f"zscore_{n}"] = (result["close"] - mean) / sd.where(sd.ne(0))
    return result


def validate_monthly_output(
    monthly: pd.DataFrame, expected_dates: pd.DatetimeIndex
) -> None:
    """Validate the entire result before any output CSV is written."""
    if set(monthly["ticker"]) != set(EXPECTED_TICKERS):
        raise ValueError("Output must contain exactly the 10 expected tickers.")
    if monthly.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate monthly output keys.")
    if monthly[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("Monthly prices and volume must not be missing.")
    for ticker, frame in monthly.groupby("ticker", sort=True):
        actual = pd.DatetimeIndex(frame["date"])
        if len(frame) != 24 or not actual.equals(expected_dates):
            raise ValueError(f"{ticker} must have exactly the same 24 ordered months.")
    if monthly.select_dtypes(include="number").isin(
        [float("inf"), -float("inf")]
    ).any().any():
        raise ValueError("Output must not contain infinite numeric values.")


def write_symbol_files(monthly: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Write a validated monthly result; keep legitimate indicator NaNs blank."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ticker, frame in monthly.groupby("ticker", sort=True):
        path = output_dir / f"result_{ticker}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d", na_rep="")
        paths.append(path)
    return paths


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    config: IndicatorConfig = IndicatorConfig(),
) -> pd.DataFrame:
    """Coordinate reading, pure calculations, validation, and separate writing."""
    daily = load_daily_prices(input_path)
    expected_dates = validate_daily_calendar(daily)
    monthly = aggregate_monthly(daily)
    result = add_indicators(monthly, config)
    validate_monthly_output(result, expected_dates)
    write_symbol_files(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", default="data/output_file.csv")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--donchian-window", type=int, default=20)
    parser.add_argument("--bollinger-window", type=int, default=20)
    parser.add_argument("--bollinger-k", type=float, default=2.0)
    parser.add_argument("--zscore-window", type=int, default=20)
    args = parser.parse_args()
    try:
        config = IndicatorConfig(
            donchian_window=args.donchian_window,
            bollinger_window=args.bollinger_window,
            bollinger_k=args.bollinger_k,
            zscore_window=args.zscore_window,
        )
        result = run_pipeline(args.input_csv, args.output_dir, config)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(
        f"Wrote {result['ticker'].nunique()} CSV files to {Path(args.output_dir)}; "
        f"24 rows per file; "
        f"{result['date'].min():%Y-%m} through {result['date'].max():%Y-%m}."
    )


if __name__ == "__main__":
    main()
