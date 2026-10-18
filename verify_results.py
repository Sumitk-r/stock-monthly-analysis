"""Independently audit the default output using Python's standard library.

This intentionally does not import the production pipeline or reuse its
Pandas aggregation/window code. It checks every numeric output cell against
sorted source rows, direct arithmetic, statistics.stdev, and an EMA recurrence.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev


TICKERS = ("AAPL", "AMD", "AMZN", "AVGO", "CSCO", "MSFT", "NFLX", "PEP", "TMUS", "TSLA")
BASE_COLUMNS = ["date", "volume", "open", "high", "low", "close", "adjclose", "ticker"]
INDICATOR_COLUMNS = [
    "sma_10", "ema_10", "sma_20", "ema_20",
    "donchian_upper_20", "donchian_lower_20", "donchian_middle_20",
    "bollinger_middle_20", "bollinger_upper_20", "bollinger_lower_20", "zscore_20",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return reader.fieldnames, rows


def month_end(month):
    import calendar
    year, number = map(int, month.split("-"))
    return f"{month}-{calendar.monthrange(year, number)[1]:02d}"


def audit(source_path, output_dir):
    _, daily = read_csv(source_path)
    daily.sort(key=lambda row: (row["ticker"], row["date"]))
    require({row["ticker"] for row in daily} == set(TICKERS), "Source ticker set differs.")
    months = sorted({row["date"][:7] for row in daily})
    ids = [int(month[:4]) * 12 + int(month[5:]) - 1 for month in months]
    require(len(ids) == 24 and ids == list(range(ids[0], ids[0] + 24)), "Wrong month range.")

    buckets = {}
    for row in daily:
        buckets.setdefault((row["ticker"], row["date"][:7]), []).append(row)
    paths = sorted(output_dir.glob("result_*.csv"))
    require(
        [path.name for path in paths] == [f"result_{ticker}.csv" for ticker in TICKERS],
        "Expected exactly the ten named result files.",
    )

    cells_checked = 0
    unavailable = 0
    for ticker in TICKERS:
        bars = []
        for month in months:
            rows = buckets.get((ticker, month), [])
            require(bool(rows), f"Missing month: {ticker}, {month}")
            bars.append({
                "date": month_end(month), "ticker": ticker,
                "volume": sum(int(row["volume"]) for row in rows),
                "open": float(rows[0]["open"]),
                "high": max(float(row["high"]) for row in rows),
                "low": min(float(row["low"]) for row in rows),
                "close": float(rows[-1]["close"]),
                "adjclose": float(rows[-1]["adjclose"]),
            })

        headers, output = read_csv(output_dir / f"result_{ticker}.csv")
        require(headers == BASE_COLUMNS + INDICATOR_COLUMNS, f"Wrong columns: {ticker}")
        require(len(output) == 24, f"Wrong row count: {ticker}")
        closes = [bar["close"] for bar in bars]
        ema = {10: None, 20: None}
        for i, (bar, actual) in enumerate(zip(bars, output)):
            expected = dict(bar)
            for period in (10, 20):
                mean = fmean(closes[i - period + 1:i + 1]) if i >= period - 1 else None
                expected[f"sma_{period}"] = mean
                if i == period - 1:
                    ema[period] = mean
                elif i >= period:
                    alpha = 2 / (period + 1)
                    ema[period] = (closes[i] - ema[period]) * alpha + ema[period]
                expected[f"ema_{period}"] = ema[period]

            if i >= 19:
                sample = bars[i - 19:i + 1]
                c = closes[i - 19:i + 1]
                mean = fmean(c)
                sd = stdev(c)
                upper = max(row["high"] for row in sample)
                lower = min(row["low"] for row in sample)
                expected.update({
                    "donchian_upper_20": upper, "donchian_lower_20": lower,
                    "donchian_middle_20": (upper + lower) / 2,
                    "bollinger_middle_20": mean, "bollinger_upper_20": mean + 2 * sd,
                    "bollinger_lower_20": mean - 2 * sd,
                    "zscore_20": (closes[i] - mean) / sd if sd != 0 else None,
                })
            else:
                expected.update({column: None for column in INDICATOR_COLUMNS[4:]})

            for column in BASE_COLUMNS + INDICATOR_COLUMNS:
                context = f"{ticker} {bar['date']} {column}"
                value = expected[column]
                if column in ("date", "ticker"):
                    require(actual[column] == value, f"Incorrect {context}")
                    continue
                cells_checked += 1
                if value is None:
                    require(actual[column] == "", f"Expected blank: {context}")
                    unavailable += 1
                else:
                    observed = float(actual[column])
                    require(math.isfinite(observed), f"Non-finite: {context}")
                    require(
                        math.isclose(observed, value, rel_tol=1e-10, abs_tol=1e-10),
                        f"Incorrect {context}: observed={observed}, expected={value}",
                    )

    return {
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Independent standard-library audit of every output cell",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_rows": len(daily),
        "source_first_date": min(row["date"] for row in daily),
        "source_last_date": max(row["date"] for row in daily),
        "source_rows_per_ticker": {
            ticker: sum(row["ticker"] == ticker for row in daily) for ticker in TICKERS
        },
        "output_files": len(paths),
        "rows_per_file": 24,
        "total_output_rows": len(paths) * 24,
        "columns_per_file": len(BASE_COLUMNS + INDICATOR_COLUMNS),
        "numeric_cells_checked_including_blanks": cells_checked,
        "unavailable_indicator_cells_checked": unavailable,
        "float_comparison_relative_tolerance": 1e-10,
        "float_comparison_absolute_tolerance": 1e-10,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("data/output_file.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--report", type=Path, default=Path("verification_report.json"))
    args = parser.parse_args()
    report = audit(args.input_csv, args.output_dir)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
