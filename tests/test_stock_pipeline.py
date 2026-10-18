"""Behavioral tests with hand-derived expectations; no third-party test runner."""

import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from stock_pipeline import (
    EXPECTED_TICKERS,
    IndicatorConfig,
    add_indicators,
    aggregate_monthly,
    load_daily_prices,
    run_pipeline,
    sma_seeded_ema,
    validate_daily_calendar,
)


ROOT = Path(__file__).resolve().parents[1]


def simple_monthly(closes, ticker="AAPL"):
    c = pd.Series(closes, dtype="float64")
    return pd.DataFrame({
        "date": pd.date_range("2018-01-31", periods=len(c), freq="ME"),
        "volume": 100,
        "open": c,
        "high": c + 10,
        "low": c - 5,
        "close": c,
        "adjclose": c - 1,
        "ticker": ticker,
    })


class MonthlyLogicTests(unittest.TestCase):
    def test_unsorted_daily_rows_use_first_open_and_last_close(self):
        daily = pd.DataFrame({
            "date": pd.to_datetime([
                "2018-01-31", "2018-02-28", "2018-01-02", "2018-02-01"
            ]),
            "volume": [200, 400, 100, 300],
            "open": [150.0, 180.0, 100.0, 160.0],
            "high": [170.0, 200.0, 130.0, 190.0],
            "low": [140.0, 175.0, 90.0, 155.0],
            "close": [165.0, 195.0, 120.0, 185.0],
            "adjclose": [164.0, 194.0, 119.0, 184.0],
            "ticker": "AAPL",
        })
        result = aggregate_monthly(daily)
        january = result.iloc[0]
        self.assertEqual(str(january["date"].date()), "2018-01-31")
        self.assertEqual(january["open"], 100)
        self.assertEqual(january["close"], 165)
        self.assertEqual(january["high"], 170)
        self.assertEqual(january["low"], 90)
        self.assertEqual(january["volume"], 300)
        self.assertEqual(january["adjclose"], 164)
        self.assertEqual(result.iloc[1]["open"], 160)
        self.assertEqual(result.iloc[1]["close"], 195)

    def test_ema_uses_sma_seed_and_recursive_multiplier(self):
        # N=3 => seed=(10+20+30)/3=20, alpha=1/2, then 30, then 40.
        close = pd.Series([10, 20, 30, 40, 50], dtype="float64")
        actual = sma_seeded_ema(close, 3)
        expected = pd.Series([math.nan, math.nan, 20, 30, 40])
        assert_series_equal(actual, expected)
        # min_periods alone hides early values; it does not change the seed.
        first_close_seeded = close.ewm(span=3, adjust=False, min_periods=3).mean()
        self.assertEqual(first_close_seeded.iloc[2], 22.5)

    def test_ema_short_history_remains_missing(self):
        self.assertTrue(sma_seeded_ema(pd.Series([10.0, 20.0]), 3).isna().all())

    def test_hand_calculated_channels_bands_and_zscore(self):
        monthly = simple_monthly([10, 20, 30, 40])
        config = IndicatorConfig(3, 3, 2.0, 3)
        result = add_indicators(monthly, config)
        # At month 3: closes 10,20,30 => mean=20, sample SD=10, Z=1.
        third = result.iloc[2]
        self.assertEqual(third["donchian_upper_3"], 40)
        self.assertEqual(third["donchian_lower_3"], 5)
        self.assertEqual(third["donchian_middle_3"], 22.5)
        self.assertAlmostEqual(third["bollinger_middle_3"], 20)
        self.assertAlmostEqual(third["bollinger_upper_3"], 40)
        self.assertAlmostEqual(third["bollinger_lower_3"], 0)
        self.assertAlmostEqual(third["zscore_3"], 1)
        self.assertTrue(result["zscore_3"].iloc[:2].isna().all())
        self.assertEqual(result.iloc[3]["donchian_lower_3"], 15)

    def test_full_window_warmup_keeps_all_24_rows(self):
        result = add_indicators(simple_monthly(range(10, 34)))
        self.assertEqual(len(result), 24)
        for column in ("sma_10", "ema_10"):
            self.assertTrue(result[column].iloc[:9].isna().all())
            self.assertEqual(result[column].notna().sum(), 15)
        for column in (
            "sma_20", "ema_20", "donchian_upper_20", "donchian_lower_20",
            "bollinger_middle_20", "bollinger_upper_20", "bollinger_lower_20",
            "zscore_20",
        ):
            self.assertTrue(result[column].iloc[:19].isna().all())
            self.assertEqual(result[column].notna().sum(), 5)
        self.assertEqual(result.iloc[9]["sma_10"], 14.5)
        self.assertEqual(result.iloc[19]["sma_20"], 19.5)

    def test_constant_closes_give_undefined_zscore_and_collapsed_bands(self):
        result = add_indicators(simple_monthly([100] * 24))
        self.assertTrue(result["zscore_20"].isna().all())
        for column in ("bollinger_middle_20", "bollinger_upper_20", "bollinger_lower_20"):
            self.assertTrue(result[column].iloc[19:].eq(100).all())

    def test_tickers_are_isolated_and_input_order_does_not_matter(self):
        aapl = simple_monthly(range(10, 34), "AAPL")
        amd = simple_monthly(range(1000, 1024), "AMD")
        expected = pd.concat(
            [add_indicators(aapl), add_indicators(amd)], ignore_index=True
        )
        shuffled = pd.concat([aapl, amd]).sample(frac=1, random_state=11)
        actual = add_indicators(shuffled)
        assert_frame_equal(actual, expected)

    def test_future_prices_do_not_change_past_indicators(self):
        monthly = simple_monthly(range(10, 34))
        before = add_indicators(monthly)
        changed = monthly.copy()
        changed.loc[23, ["open", "high", "low", "close"]] += 1000
        after = add_indicators(changed)
        assert_frame_equal(before.iloc[:23], after.iloc[:23])

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (
            {"donchian_window": 0}, {"bollinger_window": 1},
            {"zscore_window": 2.5}, {"bollinger_k": float("inf")},
            {"bollinger_k": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IndicatorConfig(**kwargs)


class SourceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = ROOT / "data" / "output_file.csv"
        cls.daily = load_daily_prices(cls.source_path)

    def test_actual_aapl_january_2018_bar(self):
        result = aggregate_monthly(self.daily)
        bar = result.loc[
            result.ticker.eq("AAPL") & result.date.eq(pd.Timestamp("2018-01-31"))
        ].iloc[0]
        # Fixed values independently inspected from the source's Jan 2/Jan 31 rows.
        self.assertAlmostEqual(bar["open"], 170.16000366210938)
        self.assertAlmostEqual(bar["close"], 167.42999267578125)
        self.assertAlmostEqual(bar["high"], 180.10000610351562)
        self.assertAlmostEqual(bar["low"], 164.6999969482422)
        self.assertEqual(bar["volume"], 659679400)

    def test_missing_month_is_rejected(self):
        daily = self.daily.loc[~(
            self.daily.ticker.eq("AMD")
            & self.daily.date.dt.to_period("M").eq(pd.Period("2018-06"))
        )]
        with self.assertRaisesRegex(ValueError, "all 24 months"):
            validate_daily_calendar(daily)

    def test_wrong_time_span_is_rejected(self):
        daily = self.daily.loc[self.daily.date.lt("2019-12-01")]
        with self.assertRaisesRegex(ValueError, "24 consecutive months"):
            validate_daily_calendar(daily)

    def test_bad_input_is_rejected(self):
        for case in ("duplicate", "null_price", "bad_date", "missing_ticker", "invalid_ohlc"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                bad = self.daily.copy()
                if case == "duplicate":
                    bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)
                elif case == "null_price":
                    bad.loc[0, "open"] = float("nan")
                elif case == "bad_date":
                    bad["date"] = bad["date"].astype("string")
                    bad.loc[0, "date"] = "not-a-date"
                elif case == "missing_ticker":
                    bad = bad.loc[bad.ticker.ne("TSLA")]
                elif case == "invalid_ohlc":
                    bad.loc[0, "high"] = 1.0
                path = Path(directory) / "bad.csv"
                bad.to_csv(path, index=False, date_format="%Y-%m-%d")
                with self.assertRaises(ValueError):
                    load_daily_prices(path)

    def test_pipeline_writes_exactly_ten_24_row_files_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_pipeline(self.source_path, directory)
            files = sorted(Path(directory).glob("result_*.csv"))
            self.assertEqual(len(result), 240)
            self.assertEqual(
                [path.name for path in files],
                [f"result_{ticker}.csv" for ticker in EXPECTED_TICKERS],
            )
            first_run_bytes = {path.name: path.read_bytes() for path in files}
            for path in files:
                frame = pd.read_csv(path)
                self.assertEqual(frame.shape, (24, 19))
                self.assertEqual(frame.ticker.nunique(), 1)
                self.assertEqual(frame.date.iloc[0], "2018-01-31")
                self.assertEqual(frame.date.iloc[-1], "2019-12-31")
                self.assertEqual(frame.sma_20.notna().sum(), 5)
                self.assertNotIn("Unnamed: 0", frame.columns)
            run_pipeline(self.source_path, directory)
            for path in files:
                self.assertEqual(path.read_bytes(), first_run_bytes[path.name])


if __name__ == "__main__":
    unittest.main()
