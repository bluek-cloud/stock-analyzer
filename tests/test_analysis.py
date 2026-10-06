import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

import backtest_engine as bt
import accumulation_scanner as scanner
from setup_signals import build_setup_signals, current_setups, bullish_divergence_at, prepare_setup_data


def prices(n=250):
    return pd.DataFrame({"Open": 100., "High": 101., "Low": 99.,
                         "Close": 100., "Volume": 1000.},
                        index=pd.bdate_range("2023-01-02", periods=n))


def breakout():
    d = prices()
    d.loc[d.index[-1], ["Open", "High", "Low", "Close", "Volume"]] = [100., 111., 99., 110., 2000.]
    return d


class SetupTests(unittest.TestCase):
    def test_regime_or_pattern_text_does_not_prove_a_setup(self):
        for regime in ("強勢", "강세 추세", "횡보 박스", "변동성 폭발", "에너지 응축 (스퀴즈)"):
            self.assertEqual(bt.match_current_setup({"regime": regime, "bullish_div": True}, ["상승 장악형"]), (None, None))

    def test_raw_ohlcv_backtest_has_no_missing_loss_error(self):
        self.assertEqual(bt.run_stock_backtest(prices())["total_trades"], 0)

    def test_breakouts_need_volume(self):
        d = breakout()
        keys = ("BOX_BREAKOUT", "BOLLINGER_SQUEEZE_BREAKOUT")
        self.assertTrue(all(k in current_setups(d) for k in keys))
        d.loc[d.index[-1], "Volume"] = 1000.
        self.assertTrue(all(k not in current_setups(d) for k in keys))

    def test_squeeze_requires_history(self):
        d = breakout().iloc[-100:]
        self.assertNotIn("BOLLINGER_SQUEEZE_BREAKOUT", current_setups(d))

    def test_engulfing_and_hammer_positive_examples(self):
        d = prices()
        d.loc[d.index[-2], ["Open", "High", "Low", "Close"]] = [101., 102., 98., 99.]
        d.loc[d.index[-1], ["Open", "High", "Low", "Close"]] = [98.5, 102., 98., 101.5]
        self.assertIn("BULLISH_ENGULFING", current_setups(d))
        d = prices()
        d.loc[d.index[-1], ["Open", "High", "Low", "Close"]] = [99.9, 100.1, 97., 100.]
        self.assertIn("HAMMER_BOTTOM", current_setups(d))

    def test_ma200_requires_proximity_and_ma5_recovery(self):
        d = prices()
        d.loc[d.index[-1], ["Open", "Close"]] = [100., 100.5]
        self.assertIn("MA200_PULLBACK", current_setups(d))
        self.assertNotIn("MA200_PULLBACK", current_setups(d.iloc[-100:]))
        d.loc[d.index[-1], ["Open", "Close"]] = [99., 99.5]
        self.assertNotIn("MA200_PULLBACK", current_setups(d))

    def test_zero_volume_cannot_be_a_tradable_signal(self):
        d = breakout()
        d.loc[d.index[-1], "Volume"] = 0.
        self.assertEqual(current_setups(d), [])

    def test_box_requires_narrow_prior_range(self):
        d = breakout()
        d.loc[d.index[-15], "Low"] = 80.
        self.assertNotIn("BOX_BREAKOUT", current_setups(d))

    def test_downside_burst_is_not_upside_breakout(self):
        d = prices()
        d.loc[d.index[-1], ["Open", "High", "Low", "Close", "Volume"]] = [100., 101., 94., 95., 3000.]
        self.assertNotIn("BOLLINGER_SQUEEZE_BREAKOUT", current_setups(d))

    def test_divergence_requires_price_and_oscillator_lows(self):
        d = prepare_setup_data(prices(40))
        d["RSI"], d["OBV"] = 30., 100.
        d.loc[d.index[20], ["Low", "RSI"]] = [95., 20.]
        d.loc[d.index[38], ["Low", "RSI"]] = [94., 30.]
        self.assertTrue(bullish_divergence_at(d, 39))
        d.loc[d.index[38], "Low"] = 96.
        self.assertFalse(bullish_divergence_at(d, 39))
        d.loc[d.index[38], ["Low", "RSI"]] = [94., 20.]
        self.assertFalse(bullish_divergence_at(d, 39))

    def test_monotonic_rise_is_not_divergence(self):
        d = prices(100)
        c = np.arange(100., 200.)
        d["Close"], d["Open"], d["Low"], d["High"] = c, c-.5, c-1, c+1
        d["RSI"] = 30.  # stale supplied indicators must not change the definition
        self.assertEqual(bt.run_stock_backtest(d, "BULLISH_DIVERGENCE")["total_trades"], 0)

    def test_historical_signal_matches_current_and_ignores_future(self):
        d = breakout()
        future = prices(25)
        future.index = pd.bdate_range(d.index[-1]+pd.Timedelta(days=1), periods=25)
        future[["Open", "High", "Low", "Close"]] *= 1.2
        extended = pd.concat([d, future])
        historical = build_setup_signals(extended)
        pd.testing.assert_frame_equal(build_setup_signals(d), historical.iloc[:len(d)])
        res = bt.run_stock_backtest(extended, "BOX_BREAKOUT")
        self.assertEqual(res["total_trades"], 1)
        self.assertEqual(res["recent_trades"][0]["entry_date"], str(d.index[-1].date()))
        self.assertAlmostEqual(res["avg_return"], 9.09, places=2)
        self.assertEqual(bt.match_current_setup({}, df=d)[0], current_setups(d)[0])

    def test_invalid_prices_and_horizon_return_errors(self):
        d = prices()
        d.loc[d.index[30], "Close"] = np.nan
        self.assertIn("error", bt.run_stock_backtest(d))
        self.assertIn("error", bt.run_stock_backtest(prices(), hold_days=0))
        self.assertIn("error", bt.run_stock_backtest(prices(), setup_type="TYPO"))

    def test_unverified_stats_never_reach_report(self):
        for key, stats in bt.load_backtest_stats().items():
            self.assertFalse(bt.verified_stats(stats))
            self.assertEqual(bt.match_current_setup({"matched_setups": [key]}), (key, None))
            self.assertEqual(bt.format_stats_for_report(stats), "")
            self.assertNotIn(str(stats["win_rate_20d"]), bt.format_stats_for_llm(stats))

    def test_weekly_signal_does_not_use_daily_stats(self):
        stats = dict(verification_status="verified", signal_version="2", timeframe="daily",
                     provenance={k: "fixture" for k in ("source", "period", "universe", "trade_log", "generator")})
        with patch.object(bt, "load_backtest_stats", return_value={"BOX_BREAKOUT": stats}):
            self.assertEqual(bt.match_current_setup({"matched_setups": ["BOX_BREAKOUT"], "is_short_term": False}), ("BOX_BREAKOUT", None))


class ScannerTests(unittest.TestCase):
    def test_unknown_small_caps_are_excluded(self):
        d = pd.DataFrame({"Code": ["000010", "000020", "000030", "000040"],
                          "Name": ["Example"]*4, "Marcap": [0, np.nan, 9e10, 1e11], "Dept": [np.nan]*4})
        self.assertEqual([r["Code"] for r in scanner.filter_universe_candidates(d)], ["000040"])

    def test_investor_failure_is_unknown_not_zero(self):
        with patch.object(scanner.requests, "get", return_value=Mock(status_code=503)):
            self.assertIsNone(scanner.get_investor_net_buys("000010"))

    def test_investor_values_accept_numbers_and_commas(self):
        rows = [{"foreignerPureBuyQuant": "1,000", "organPureBuyQuant": -500} for _ in range(20)]
        with patch.object(scanner.requests, "get", return_value=Mock(status_code=200, json=lambda: rows)):
            self.assertEqual(scanner.get_investor_net_buys("000010"), (20000, -10000, 10000))

    def test_incomplete_or_missing_investor_data_is_unknown(self):
        for rows in ([{}]*20, [{"foreignerPureBuyQuant": 0, "organPureBuyQuant": 0}]*5):
            with patch.object(scanner.requests, "get", return_value=Mock(status_code=200, json=lambda: rows)):
                self.assertIsNone(scanner.get_investor_net_buys("000010"))

    def test_missing_investor_data_keeps_unknown_badge(self):
        d = prices(120)
        d["Volume"] = 20_000_000.
        d.loc[d.index[-5], "Volume"] = 80_000_000.
        with patch.object(scanner, "get_investor_net_buys", return_value=None):
            result = scanner.evaluate_stock_accumulation_df(d, code="000010", min_accum_candles=1)
        self.assertIsNotNone(result)
        self.assertIsNone(result["total_smart_money"])
        self.assertFalse(result["investor_data_available"])
        self.assertEqual(result["badge"], "⚪ 수급 미확인")


if __name__ == "__main__":
    unittest.main()
