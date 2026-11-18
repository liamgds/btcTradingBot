# Unit tests for single parts of the bot. Run: python tests/test_units.py

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import strategy
from config import Config
from performance_analytics import PerformanceAnalytics


class TestPositionSizing(unittest.TestCase):
    def size(self, equity=10_000.0, entry=100.0, sl=50.0, conf=0.5,
             returns=None, losses=0, drawdown=0.0, daily_pnl=0.0):
        return strategy.compute_size(equity, entry, sl, conf, returns or [],
                                     losses, drawdown, daily_pnl, dict(strategy.DEFAULT_CFG))

    def test_normal_trade_risks_one_and_a_half_percent(self):
        qty, info = self.size()
        self.assertAlmostEqual(qty, 3.0)
        self.assertEqual(info["dollar_risk"], 150.0)

    def test_position_is_capped_at_ten_percent_of_account(self):
        qty, _ = self.size(entry=100.0, sl=99.0)
        self.assertAlmostEqual(qty * 100.0, 1_000.0)

    def test_stop_at_entry_price_is_blocked(self):
        qty, info = self.size(entry=100.0, sl=100.0)
        self.assertEqual(qty, 0.0)
        self.assertEqual(info["blocked"], "invalid_sl")

    def test_stop_above_entry_price_is_blocked(self):
        qty, info = self.size(entry=100.0, sl=105.0)
        self.assertEqual(qty, 0.0)
        self.assertEqual(info["blocked"], "invalid_sl")

    def test_size_shrinks_as_drawdown_grows(self):
        self.assertAlmostEqual(self.size(drawdown=4.9)[0], 3.0)
        self.assertAlmostEqual(self.size(drawdown=5.0)[0], 2.25)
        self.assertAlmostEqual(self.size(drawdown=8.0)[0], 1.5)
        self.assertAlmostEqual(self.size(drawdown=10.0)[0], 0.75)
        self.assertAlmostEqual(self.size(drawdown=11.99)[0], 0.75)

    def test_trading_stops_at_drawdown_limit(self):
        qty, info = self.size(drawdown=12.0)
        self.assertEqual(qty, 0.0)
        self.assertIn("dd_halt", info["blocked"])

    def test_size_shrinks_after_losing_streak(self):
        self.assertAlmostEqual(self.size(losses=1)[0], 3.0)
        self.assertAlmostEqual(self.size(losses=2)[0], 2.25)
        self.assertAlmostEqual(self.size(losses=3)[0], 1.5)
        self.assertAlmostEqual(self.size(losses=10)[0], 1.5)

    def test_trading_stops_when_recent_win_rate_is_too_low(self):
        qty, info = self.size(returns=[0.01] * 7 + [-0.01] * 13)
        self.assertEqual(qty, 0.0)
        self.assertIn("perf_halt", info["blocked"])

    def test_win_rate_exactly_on_the_limits(self):
        _, at_halt_limit = self.size(returns=[0.01] * 8 + [-0.01] * 12)
        self.assertEqual(at_halt_limit["l6_mon"], 0.5)
        _, at_warn_limit = self.size(returns=[0.01] * 9 + [-0.01] * 11)
        self.assertEqual(at_warn_limit["l6_mon"], 1.0)

    def test_size_drops_after_big_daily_profit(self):
        self.assertAlmostEqual(self.size(daily_pnl=300.0)[0], 3.0)
        self.assertAlmostEqual(self.size(daily_pnl=300.01)[0], 1.8)

    def test_confidence_outside_range_is_clipped(self):
        self.assertAlmostEqual(self.size(conf=5.0)[0], 3.9)
        self.assertAlmostEqual(self.size(conf=-5.0)[0], 2.1)

    def test_volatile_results_reduce_size(self):
        qty, info = self.size(returns=[0.5, -0.5] * 3)
        self.assertEqual(info["l2_vol"], 0.3)
        self.assertAlmostEqual(qty, 0.9)

    def test_identical_results_do_not_divide_by_zero(self):
        qty, info = self.size(returns=[0.01] * 6)
        self.assertEqual(info["l2_vol"], 1.0)
        self.assertAlmostEqual(qty, 3.0)

    def test_too_few_results_are_ignored(self):
        _, info = self.size(returns=[0.5, -0.5, 0.5, -0.5])
        self.assertEqual(info["l2_vol"], 1.0)

    def test_empty_account_gives_zero_size(self):
        qty, _ = self.size(equity=0.0)
        self.assertEqual(qty, 0.0)


class TestPerformanceStats(unittest.TestCase):
    def stats(self, pnls):
        analytics = PerformanceAnalytics(initial_capital=10_000.0)
        for pnl in pnls:
            analytics.record_trade({"pnl": pnl})
        return analytics

    def test_no_trades_gives_zero_not_a_crash(self):
        analytics = self.stats([])
        self.assertEqual(analytics.get_win_rate(), 0.0)
        self.assertEqual(analytics.get_profit_factor(), 0.0)
        self.assertEqual(analytics.get_comprehensive_stats()["total_trades"], 0)

    def test_all_winning_trades(self):
        analytics = self.stats([50.0, 20.0, 5.0])
        self.assertEqual(analytics.get_win_rate(), 100.0)
        self.assertEqual(analytics.get_profit_factor(), float("inf"))

    def test_all_losing_trades(self):
        analytics = self.stats([-50.0, -20.0])
        self.assertEqual(analytics.get_win_rate(), 0.0)
        self.assertEqual(analytics.get_profit_factor(), 0.0)

    def test_mixed_trades(self):
        analytics = self.stats([100.0, -50.0, 60.0, 40.0])
        self.assertEqual(analytics.get_win_rate(), 75.0)
        self.assertAlmostEqual(analytics.get_profit_factor(), 4.0)
        self.assertAlmostEqual(analytics.current_capital, 10_150.0)

    def test_breakeven_trade_is_not_a_win(self):
        analytics = self.stats([0.0, 10.0])
        self.assertEqual(analytics.get_win_rate(), 50.0)

    def test_trade_with_missing_profit_is_not_a_win(self):
        analytics = PerformanceAnalytics(initial_capital=10_000.0)
        analytics.record_trade({"symbol": "BTCUSDT"})
        self.assertEqual(analytics.get_win_rate(), 0.0)
        self.assertEqual(analytics.current_capital, 10_000.0)

    def test_streaks_are_counted(self):
        analytics = self.stats([10.0, 10.0, -5.0, -5.0, -5.0, 10.0])
        self.assertEqual(analytics.max_winning_streak, 2)
        self.assertEqual(analytics.max_losing_streak, 3)
        self.assertEqual(analytics.current_streak, 1)

    def test_drawdown_keeps_its_worst_value(self):
        analytics = self.stats([-1_000.0, 500.0])
        self.assertAlmostEqual(analytics.max_drawdown, 10.0)
        self.assertAlmostEqual(analytics.current_drawdown, 5.0)

    def test_new_high_resets_current_drawdown(self):
        analytics = self.stats([-1_000.0, 1_500.0])
        self.assertEqual(analytics.current_drawdown, 0.0)
        self.assertAlmostEqual(analytics.max_drawdown, 10.0)
        self.assertAlmostEqual(analytics.peak_capital, 10_500.0)


class TestSettingsCheck(unittest.TestCase):
    FIELDS = ("TRADING_MODE", "BINANCE_API_KEY", "BINANCE_API_SECRET", "SYMBOL",
              "TIMEFRAME", "CHECK_INTERVAL_SECONDS", "BACKTEST_INITIAL_CAPITAL", "LOG_LEVEL")

    def setUp(self):
        self.saved = {name: getattr(Config, name) for name in self.FIELDS}
        Config.TRADING_MODE = "paper"
        Config.BINANCE_API_KEY = ""
        Config.BINANCE_API_SECRET = ""
        Config.SYMBOL = "BTCUSDT"
        Config.TIMEFRAME = "15m"
        Config.CHECK_INTERVAL_SECONDS = 60
        Config.BACKTEST_INITIAL_CAPITAL = 10_000.0
        Config.LOG_LEVEL = "INFO"

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(Config, name, value)

    def error(self):
        with self.assertRaises(ValueError) as caught:
            Config.validate()
        return str(caught.exception)

    def test_paper_mode_needs_no_keys(self):
        self.assertTrue(Config.validate())

    def test_live_mode_is_refused(self):
        Config.TRADING_MODE = "live"
        Config.BINANCE_API_KEY = "key"
        Config.BINANCE_API_SECRET = "secret"
        self.assertIn("disabled", self.error())

    def test_unknown_or_empty_mode_is_refused(self):
        for mode in ("real", "", "PAPER"):
            Config.TRADING_MODE = mode
            self.assertIn("TRADING_MODE", self.error())

    def test_demo_mode_without_keys_is_refused(self):
        Config.TRADING_MODE = "demo"
        message = self.error()
        self.assertIn("BINANCE_API_KEY", message)
        self.assertIn("BINANCE_API_SECRET", message)

    def test_demo_mode_with_only_one_key_is_refused(self):
        Config.TRADING_MODE = "demo"
        Config.BINANCE_API_KEY = "key"
        message = self.error()
        self.assertNotIn("BINANCE_API_KEY", message)
        self.assertIn("BINANCE_API_SECRET", message)

    def test_demo_and_testnet_pass_with_both_keys(self):
        Config.BINANCE_API_KEY = "key"
        Config.BINANCE_API_SECRET = "secret"
        for mode in ("demo", "testnet"):
            Config.TRADING_MODE = mode
            self.assertTrue(Config.validate())

    def test_bad_timeframe_is_refused(self):
        for timeframe in ("7m", "15", ""):
            Config.TIMEFRAME = timeframe
            self.assertIn("TIMEFRAME", self.error())

    def test_symbol_is_cleaned_up(self):
        Config.SYMBOL = "btc/usdt "
        Config.validate()
        self.assertEqual(Config.SYMBOL, "BTCUSDT")

    def test_symbol_with_nothing_usable_is_refused(self):
        Config.SYMBOL = "/ !"
        self.assertIn("SYMBOL", self.error())

    def test_check_interval_limit(self):
        Config.CHECK_INTERVAL_SECONDS = 4
        self.assertIn("CHECK_INTERVAL_SECONDS", self.error())
        Config.CHECK_INTERVAL_SECONDS = 5
        self.assertTrue(Config.validate())

    def test_starting_capital_must_be_positive(self):
        for capital in (0, -100):
            Config.BACKTEST_INITIAL_CAPITAL = capital
            self.assertIn("BACKTEST_INITIAL_CAPITAL", self.error())

    def test_every_problem_is_reported_at_once(self):
        Config.TRADING_MODE = "real"
        Config.TIMEFRAME = "7m"
        Config.LOG_LEVEL = "LOUD"
        message = self.error()
        for name in ("TRADING_MODE", "TIMEFRAME", "LOG_LEVEL"):
            self.assertIn(name, message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
