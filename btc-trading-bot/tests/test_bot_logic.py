import sys
import os
from unittest.mock import patch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import pandas as pd

import strategy
import exchange as exchange_module
from exchange import OrderTooSmall
from binance.client import Client as RealBinanceClient
from bot import TradingBot
from test_orchestration import FakeExchange, SAMPLE_CSV, use_clean_workdir

BTC_RULES = {"filters": [
    {"filterType": "LOT_SIZE", "stepSize": "0.00001000", "minQty": "0.00001000"},
    {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
    {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
]}


def paper_exchange(price):
    ex = exchange_module.BinanceExchange("", "", mode="paper")
    ex.market_price = price
    ex.get_current_price = lambda symbol: ex.market_price
    return ex


def new_bot(fill_limit_orders=True):
    use_clean_workdir()
    fake = FakeExchange(SAMPLE_CSV, strategy.DEFAULT_CFG)
    fake.fill_limit_orders = fill_limit_orders
    return TradingBot(exchange=fake), fake


def signal_once():
    calls = {"n": 0}

    def fake_check_entry(row, cfg, vov_halted):
        calls["n"] += 1
        return (True, "ok") if calls["n"] == 1 else (False, "no_cross")

    return patch.object(strategy, "check_entry", fake_check_entry), calls


def test_order_sizes_follow_exchange_rules():
    ex = paper_exchange(100371.92)
    order = ex.place_limit_order("BTCUSDT", "BUY", 0.009963789, 100371.92048038)
    assert order["origQty"] == 0.00996, order["origQty"]
    assert order["price"] == 100371.92, order["price"]
    print("PASS: order quantity and price are rounded to what Binance accepts")


def test_tiny_orders_are_refused():
    ex = paper_exchange(100000.0)
    try:
        ex.place_market_order("BTCUSDT", "SELL", 0.00001, 100000.0)
        raise AssertionError("a $1 order should have been refused")
    except OrderTooSmall:
        pass
    print("PASS: orders below the exchange minimum are refused before being sent")


def test_paper_limit_order_fills_only_when_price_reaches_it():
    ex = paper_exchange(101000.0)
    order = ex.place_limit_order("BTCUSDT", "BUY", 0.01, 100000.0)
    assert ex.get_order_status("BTCUSDT", order["orderId"])["status"] == "NEW"
    ex.market_price = 99990.0
    status = ex.get_order_status("BTCUSDT", order["orderId"])
    assert status["status"] == "FILLED" and status["avg_fill_price"] == 100000.0
    print("PASS: a paper limit order only fills once the price reaches it")


def test_forming_candle_is_ignored():
    bot, _ = new_bot()
    current = pd.Timestamp.now(tz="UTC").floor("15min")
    times = [current - pd.Timedelta(minutes=30), current - pd.Timedelta(minutes=15), current]
    df = pd.DataFrame({"timestamp": times, "close": [1.0, 2.0, 3.0]})
    closed = bot._closed_candles(df)
    assert list(closed["close"]) == [1.0, 2.0]
    print("PASS: the candle that is still forming is left out of every decision")


def test_each_candle_is_processed_once():
    bot, fake = new_bot()
    with patch.object(strategy, "check_entry", wraps=strategy.check_entry) as check:
        bot.run_once()
        bot.run_once()
        bot.run_once()
        assert check.call_count == 1, check.call_count
        fake.advance()
        bot.run_once()
        assert check.call_count == 2, check.call_count
    print("PASS: a candle is only acted on once, however often the bot polls")


def test_restart_does_not_repeat_a_candle():
    bot, fake = new_bot()
    bot.run_once()
    second_fake = FakeExchange(SAMPLE_CSV, strategy.DEFAULT_CFG)
    second_bot = TradingBot(exchange=second_fake)
    with patch.object(strategy, "check_entry", wraps=strategy.check_entry) as check:
        second_bot.run_once()
        assert check.call_count == 0
    print("PASS: after a restart the bot does not act on the same candle twice")


def test_unfilled_limit_order_is_cancelled():
    bot, fake = new_bot(fill_limit_orders=False)
    signal, _ = signal_once()
    with signal:
        bot.run_once()
        assert bot.risk_manager.pending_entry is not None
        assert not bot.risk_manager.has_open_position()
        order_id = bot.risk_manager.pending_entry["order_id"]

        fake.advance()
        bot.run_once()
        assert bot.risk_manager.pending_entry is not None, "order should still be waiting after 1 candle"

        fake.advance()
        bot.run_once()
    assert bot.risk_manager.pending_entry is None
    assert not bot.risk_manager.has_open_position()
    assert fake.cancelled == [order_id]
    print("PASS: a limit order that never fills is cancelled after 2 candles")


def test_partly_filled_order_becomes_a_position():
    bot, fake = new_bot(fill_limit_orders=False)
    signal, _ = signal_once()
    with signal:
        bot.run_once()
        order_id = bot.risk_manager.pending_entry["order_id"]
        fake.limit_orders[order_id]["filled"] = 0.004
        for _ in range(2):
            fake.advance()
            bot.run_once()
    assert bot.risk_manager.pending_entry is None
    assert bot.risk_manager.has_open_position()
    assert bot.risk_manager.position["qty"] == 0.004
    print("PASS: a partly filled order is tracked with the amount that actually filled")


def test_failed_exit_is_retried():
    bot, fake = new_bot()
    signal, _ = signal_once()
    with signal:
        bot.run_once()
        assert bot.risk_manager.has_open_position()

        def forced_exit(row, position, cfg, fee_maker):
            return position, [], {"price": float(row["close"]), "reason": "StructuralStop"}

        fake.fail_sells = True
        fake.advance()
        with patch.object(strategy, "evaluate_position_candle", forced_exit):
            bot.run_once()
        assert bot.risk_manager.has_open_position(), "position must stay open while the sell has not gone through"
        assert bot.risk_manager.position["exit_pending"]["reason"] == "StructuralStop"
        assert len(bot.risk_manager.trades) == 0

        fake.fail_sells = False
        bot.run_once()
    assert not bot.risk_manager.has_open_position()
    assert len(bot.risk_manager.trades) == 1
    assert fake.market_orders[-1][0] == "SELL"
    print("PASS: a failed exit order is retried instead of being recorded as done")


def test_state_is_kept_separate_per_mode():
    bot, _ = new_bot()
    path = bot.risk_manager.state_path
    assert "paper" in path and "BTCUSDT" in path, path
    print("PASS: saved positions are kept in a separate file for each mode")


def main():
    tests = [
        test_order_sizes_follow_exchange_rules,
        test_tiny_orders_are_refused,
        test_paper_limit_order_fills_only_when_price_reaches_it,
        test_forming_candle_is_ignored,
        test_each_candle_is_processed_once,
        test_restart_does_not_repeat_a_candle,
        test_unfilled_limit_order_is_cancelled,
        test_partly_filled_order_becomes_a_position,
        test_failed_exit_is_retried,
        test_state_is_kept_separate_per_mode,
    ]
    with patch.object(RealBinanceClient, "ping", lambda self: None), \
            patch.object(RealBinanceClient, "get_symbol_info", lambda self, symbol: BTC_RULES):
        for t in tests:
            t()
    print(f"\nALL {len(tests)} BOT LOGIC CHECKS PASSED")


if __name__ == "__main__":
    main()
