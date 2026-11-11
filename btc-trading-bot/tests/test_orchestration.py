import sys
import os
import glob
import random

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

import strategy
from bot import TradingBot

RELAXED = {
    "adx_min": 12.0, "di_spread_min": 1.0, "sus_vol_threshold": 0.9,
    "min_session_score": 0, "rsi_floor": 35.0, "rsi_ceil": 70.0,
}
SAMPLE_CSV = os.path.join(ROOT, "sample_data", "sample_btcusdt_15m.csv")


class FakeExchange:
    def __init__(self, csv_path: str, cfg: dict):
        self.df = pd.read_csv(csv_path)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.cursor = max(cfg["ema_trend"], cfg["atr_regime_period"]) * 2 + 100
        self.limit_orders = {}
        self.cancelled = []
        self.market_orders = []
        self.fill_limit_orders = True
        self.fail_sells = False

    def prepare(self):
        pass

    def get_account_balance(self, asset="USDT"):
        return 10_000.0

    def get_current_price(self, symbol):
        return float(self.df.iloc[self.cursor]["close"])

    def get_historical_klines(self, symbol, interval, start_time, end_time=None, limit=1000):
        lo = max(0, self.cursor - limit)
        return self.df.iloc[lo:self.cursor + 1].reset_index(drop=True)

    def place_market_order(self, symbol, side, quantity, expected_price, max_slippage_percent=None):
        if side == "SELL" and self.fail_sells:
            raise Exception("simulated network failure")
        self.market_orders.append((side, quantity))
        return {"orderId": f"FAKE_{random.randint(0, 10**9)}", "symbol": symbol, "side": side,
                "origQty": quantity, "executedQty": quantity, "price": expected_price,
                "avg_fill_price": expected_price, "status": "FILLED", "fills": []}

    def place_limit_order(self, symbol, side, quantity, price):
        oid = f"FAKE_LIMIT_{random.randint(0, 10**9)}"
        self.limit_orders[oid] = {"qty": quantity, "price": price, "filled": 0.0, "status": "NEW"}
        return {"orderId": oid, "symbol": symbol, "side": side, "origQty": quantity,
                "executedQty": 0.0, "price": price, "status": "NEW"}

    def get_order_status(self, symbol, order_id):
        order = self.limit_orders[order_id]
        if self.fill_limit_orders and order["status"] == "NEW":
            order["status"], order["filled"] = "FILLED", order["qty"]
        return {"status": order["status"], "executedQty": order["filled"],
                "price": order["price"], "avg_fill_price": order["price"]}

    def cancel_order(self, symbol, order_id):
        self.cancelled.append(order_id)
        if self.limit_orders[order_id]["status"] == "NEW":
            self.limit_orders[order_id]["status"] = "CANCELED"
        return {"orderId": order_id, "status": "CANCELED"}

    def advance(self):
        self.cursor += 1
        return self.cursor < len(self.df) - 1


def use_clean_workdir():
    test_tmp = os.path.abspath(os.path.join(ROOT, "tests", ".tmp_test_run"))
    os.makedirs(test_tmp, exist_ok=True)
    os.chdir(test_tmp)
    for stale in glob.glob("logs/risk_state*.json"):
        os.remove(stale)


def main():
    if not os.path.exists(SAMPLE_CSV):
        print(f"Sample data not found at {SAMPLE_CSV}.")
        print("Run: python tests/generate_sample_data.py")
        sys.exit(1)

    use_clean_workdir()
    orig_default_cfg = dict(strategy.DEFAULT_CFG)
    strategy.DEFAULT_CFG.update(RELAXED)

    fake_exchange = FakeExchange(SAMPLE_CSV, strategy.DEFAULT_CFG)
    first_candle = fake_exchange.df["timestamp"].iloc[fake_exchange.cursor + 1]
    bot = TradingBot(exchange=fake_exchange)

    cycles = 0
    entries_seen = 0
    exits_seen = 0
    max_cycles = len(fake_exchange.df) - fake_exchange.cursor - 2

    while fake_exchange.advance() and cycles < max_cycles:
        had_position_before = bot.risk_manager.has_open_position()
        bot.run_once()
        has_position_after = bot.risk_manager.has_open_position()
        if not had_position_before and has_position_after:
            entries_seen += 1
        if had_position_before and not has_position_after:
            exits_seen += 1
        cycles += 1

    live_trades = bot.risk_manager.trades
    print(f"\nCycles run: {cycles}")
    print(f"Entries opened: {entries_seen}")
    print(f"Full exits: {exits_seen}")
    print(f"Trades recorded: {len(live_trades)}")
    print(f"Final equity: {bot.risk_manager.equity:.2f}")

    assert entries_seen > 0, "No entries fired, orchestration or filters broken"
    assert len(live_trades) == exits_seen, "Trade count mismatch"
    for t in live_trades:
        assert t["exit_time"] >= t["entry_time"], "Exit before entry"
        assert t["qty"] > 0
        assert np.isfinite(t["pnl"]) and np.isfinite(t["pnl_pct"]), "Non-finite P&L"

    backtest = strategy.run(fake_exchange.df, capital=10_000.0, cfg=strategy.DEFAULT_CFG)
    backtest_trades = [t for t in backtest["trades"]
                       if t["entry_time"] >= first_candle and t["exit_reason"] != "EndOfData"]
    assert len(backtest_trades) == len(live_trades), (
        f"Backtest made {len(backtest_trades)} trades but the live loop made {len(live_trades)}"
    )
    for bt, lv in zip(backtest_trades, live_trades):
        assert bt["entry_time"] == lv["entry_time"], "Live loop and backtest entered at different times"
        assert bt["exit_reason"] == lv["exit_reason"], "Live loop and backtest exited for different reasons"
        assert abs(bt["pnl"] - lv["pnl"]) < 0.01, "Live loop and backtest disagree on profit"
    print(f"Live loop matches the backtest trade for trade ({len(live_trades)} trades)")

    strategy.DEFAULT_CFG.clear()
    strategy.DEFAULT_CFG.update(orig_default_cfg)

    print("\nALL ORCHESTRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
