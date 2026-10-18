# Monthly Stock Aggregation and Technical Indicators

A Python/Pandas pipeline that converts daily stock prices into monthly OHLC bars, calculates technical indicators, and writes one CSV per ticker.

## Setup and execution

Tested with **Python 3.12.14** and **Pandas 2.2.3**. Python 3.12 is recommended to reproduce the tested environment. Pandas is the only directly required third-party package.

Extract the project and open a terminal in the folder containing `README.md`, `stock_pipeline.py`, and `requirements.txt`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe stock_pipeline.py data/output_file.csv --output-dir output
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_results.py
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python stock_pipeline.py data/output_file.csv --output-dir output
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python verify_results.py
```

These commands use the virtual environment directly; activation is unnecessary. The dataset is included, so the pipeline requires no network access after dependencies are installed.

Expected pipeline result:

```text
Wrote 10 CSV files to output; 24 rows per file; 2018-01 through 2019-12.
```

The verifier checks the default indicator configuration and writes `verification_report.json`. A successful check reports `"status": "PASS"`.

## Dataset and output

The input is the supplied [assignment dataset](https://github.com/sandeep-tt/tt-intern-dataset), stored unchanged at `data/output_file.csv`.

| Property | Value |
| --- | --- |
| Daily records | 5,030 |
| Records per ticker | 503 |
| Observed dates | 2018-01-02 through 2019-12-31 |
| Tickers | AAPL, AMD, AMZN, AVGO, CSCO, MSFT, NFLX, PEP, TMUS, TSLA |
| Monthly coverage per ticker | January 2018 through December 2019 |
| Missing input values / duplicate ticker-date keys | 0 / 0 |

Input rows are shuffled. The pipeline sorts by ticker and date before aggregation.

Outputs are written as `output/result_{SYMBOL}.csv`. Each of the ten files contains **24 data rows plus a header**, with **19 columns**. Dates are chronological calendar month-end labels, from `2018-01-31` through `2019-12-31`.

Source checksums, the original commit, and license details are recorded in [data/SOURCE.md](data/SOURCE.md).

## Monthly aggregation

Aggregation runs independently for each ticker using month-end resampling (`ME`).

| Column | Aggregation |
| --- | --- |
| `open` | First observed trading day's open |
| `close` | Last observed trading day's close |
| `high` | Maximum daily high |
| `low` | Minimum daily low |
| `volume` | Sum of daily volumes |
| `adjclose` | Last observed trading day's adjusted close |

The month-end label can fall on a non-trading day. Open and close are still taken from the month's first and last observed trading records.

## Assumptions

The assignment leaves some indicator parameters unspecified. The following choices are used consistently and can be reproduced from the code:

| Item | Choice |
| --- | --- |
| Indicator timeframe | Calculate indicators after monthly aggregation; periods represent months. |
| Price field | Monthly `close` for SMA, EMA, Bollinger Bands, and Z-score. |
| Donchian input | Monthly `high` and `low`, following the explicit formulas in the assignment. |
| SMA / EMA periods | 10 and 20 months. |
| EMA seed | At month N, use the SMA of the first N monthly closes. |
| Donchian lookback | 20 months. |
| Bollinger lookback / multiplier | 20 months / k = 2. |
| Z-score | Trailing 20-month mean and standard deviation. |
| Standard deviation | Sample standard deviation (`ddof=1`), matching the supplied Pandas `.std()` formula. |
| Window boundaries | Trailing windows include the current completed month. |
| Precision | No rounding of intermediate calculations or forced two-decimal output. |

Donchian Channels use highs and lows because their specific formula requires these fields; this resolves the assignment's broader note about monthly closing prices.

Full windows are required. SMA 10 and EMA 10 have nine initial blank values and 15 valid values per ticker. The 20-month indicators have 19 initial blanks and five valid values on this dataset. All 24 monthly rows are retained. A Z-score also remains blank when its rolling standard deviation is zero; Bollinger Bands then collapse to their middle band.

Invalid or missing required values, nonfinite numbers, inconsistent OHLC bounds, duplicate keys, unexpected tickers, and missing months are rejected. Validation checks monthly coverage using the supplied observations; it does not certify completeness against an external exchange calendar.

## Indicator formulas

Let C[t], H[t], and L[t] be monthly close, high, and low. Month numbers below are one-based, and each window ends at the current month t.

```text
SMA_N[t] = mean(C[t-N+1], ..., C[t])

alpha = 2 / (N + 1)
EMA_N[N] = mean(C[1], ..., C[N])
EMA_N[t] = EMA_N[t-1] + alpha * (C[t] - EMA_N[t-1]), for t > N

Donchian upper[t]  = max(H[t-N+1], ..., H[t])
Donchian lower[t]  = min(L[t-N+1], ..., L[t])
Donchian middle[t] = (upper[t] + lower[t]) / 2

Bollinger middle[t] = mean(C[t-N+1], ..., C[t])
sd[t] = sample standard deviation of the same N closes
Bollinger upper/lower[t] = middle[t] +/- k * sd[t]

Z[t] = (C[t] - mean(C[t-N+1], ..., C[t])) / sd[t]
```

The EMA helper explicitly supplies the initial SMA before applying `ewm(adjust=False)`. Setting `min_periods=N` alone would hide early values while retaining Pandas' first-close seed.

With the default configuration, the Bollinger middle band equals `sma_20`. The Z-score measures standardized price deviation; it is the indicator requested under the assignment's volatility label.

## Output columns

```text
date,volume,open,high,low,close,adjclose,ticker,
sma_10,ema_10,sma_20,ema_20,
donchian_upper_20,donchian_lower_20,donchian_middle_20,
bollinger_middle_20,bollinger_upper_20,bollinger_lower_20,zscore_20
```

Each channel or band component is a separate column. Unavailable indicator values are blank CSV fields. Files are written with `index=False`.

Optional CLI flags are `--donchian-window`, `--bollinger-window`, `--bollinger-k`, and `--zscore-window`. Window-based column suffixes reflect the selected lookback. SMA/EMA periods remain 10 and 20. Use a separate output directory for custom configurations; `verify_results.py` audits the defaults.

## Implementation structure

| Function | Responsibility |
| --- | --- |
| `load_daily_prices` | Read, validate, and sort input. |
| `validate_daily_calendar` | Require 24 shared consecutive months. |
| `aggregate_monthly` | Calculate monthly OHLC, volume, and adjusted close. |
| `sma_seeded_ema` / `add_indicators` | Calculate indicators independently per ticker. |
| `validate_monthly_output` | Check ticker coverage, dates, row counts, and nonfinite output. |
| `write_symbol_files` | Write the ten CSV files. |
| `run_pipeline` | Coordinate the stages. |

Production calculations use Pandas aggregation, `groupby.transform`, rolling windows, and `ewm`. There are no Python loops over price records and no third-party technical-analysis libraries. Small loops over periods, validation fields, and output partitions are used where appropriate.

## Verification

All **14 unit/integration tests pass**. They cover shuffled input, hand-calculated formulas, EMA initialization, ticker isolation, warm-up periods, zero variance, invalid input, future-data independence, and repeatable output generation.

The independent verifier checks every output cell against the source data using standard-library CSV reading, direct arithmetic, `statistics.stdev`, and a separate EMA recurrence. For the included dataset, it checks **4,080 numeric cells**, including **1,890 unavailable indicator cells**, across all 240 monthly records. Numeric comparisons use relative and absolute tolerances of `1e-10`.

For a reference check, AAPL's December 2019 row has SMA 10 = **221.901999**, EMA 10 = **232.603642**, SMA 20 = **206.471500**, EMA 20 = **212.646963**, and Z-score 20 = **2.519774**, displayed to six decimals. CSVs retain greater precision.

## References

- [Pandas 2.2: resample](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.DataFrame.resample.html)
- [Pandas 2.2: rolling windows](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.Series.rolling.html)
- [Pandas 2.2: exponentially weighted calculations](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.DataFrame.ewm.html)
- [Pandas 2.2: rolling standard deviation](https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.core.window.rolling.Rolling.std.html)
- [StockCharts: simple and exponential moving averages](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
