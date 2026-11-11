import time
import signal
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from logger import get_logger
from exchange import BinanceExchange, OrderTooSmall
from risk_manager import RiskManager
import strategy


class TradingBot:
    FINISHED_ORDER_STATES = ("FILLED", "CANCELED", "REJECTED", "EXPIRED", "EXPIRED_IN_MATCH", "UNKNOWN")

    def __init__(self, exchange: Optional[BinanceExchange] = None):
        Config.validate()

        self.logger = get_logger(Config.LOG_LEVEL, Config.LOG_FILE,
                                 Config.LOG_MAX_BYTES, Config.LOG_BACKUP_COUNT)
        self.logger.info(Config.mode_banner())
        self.logger.info("Initializing Trading Bot...")

        self.cfg = dict(strategy.DEFAULT_CFG)
        self.exchange = exchange or BinanceExchange(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_API_SECRET,
            mode=Config.TRADING_MODE,
        )
        self.exchange.prepare()

        try:
            initial_balance = self.exchange.get_account_balance("USDT")
        except Exception as e:
            self.logger.warning(f"Could not get account balance, using default: {e}")
            initial_balance = Config.BACKTEST_INITIAL_CAPITAL

        state_path = f"logs/risk_state_{Config.TRADING_MODE}_{Config.SYMBOL}.json"
        self.risk_manager = RiskManager(initial_capital=initial_balance, cfg=self.cfg,
                                        state_path=state_path)
        self._vov_halted: Optional[bool] = None

        self.running = False
        self.logger.info(Config.display())

        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        raise KeyboardInterrupt

    def start(self):
        self.logger.info("=" * 50)
        self.logger.info(f"Starting {Config.TRADING_MODE.upper()} Trading Bot")
        self.logger.info(f"Symbol: {Config.SYMBOL} | Timeframe: {Config.TIMEFRAME}")
        self.logger.info("=" * 50)
        self.running = True

        try:
            while self.running:
                try:
                    self.run_once()
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}")
                time.sleep(Config.CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        finally:
            self.stop()

    def run_once(self):
        df_raw = self._closed_candles(self._get_market_data())
        d = strategy.build_indicators(df_raw, self.cfg)
        if d.empty:
            self.logger.warning("Not enough candle history yet to calculate the indicators")
            return
        row = d.iloc[-1]
        today = row["timestamp"].date()

        if self.risk_manager.pending_entry is not None:
            self._check_pending_entry()

        position = self.risk_manager.position
        if position is not None and position.get("exit_pending"):
            self._finish_exit(row, today)
            return

        candle_id = str(row["timestamp"])
        if candle_id == self.risk_manager.last_candle:
            self.logger.info(f"Waiting for the next {Config.TIMEFRAME} candle | "
                             f"{Config.SYMBOL} last close ${float(row['close']):.2f}")
            return

        self._update_vov(d)

        if self.risk_manager.has_open_position():
            self._manage_open_position(row, today)
        elif self.risk_manager.pending_entry is not None:
            self._age_pending_entry()

        if not self.risk_manager.has_open_position() and self.risk_manager.pending_entry is None:
            self._look_for_entry(row, d, today)

        self.risk_manager.mark_candle(candle_id)

    def _closed_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        candle_length = timedelta(minutes=self._timeframe_to_minutes(Config.TIMEFRAME))
        now = pd.Timestamp.now(tz="UTC")
        return df[df["timestamp"] + candle_length <= now].reset_index(drop=True)

    def _update_vov(self, d: pd.DataFrame):
        values = d["vov"] if self._vov_halted is None else d["vov"].iloc[-1:]
        halted = bool(self._vov_halted)
        for v in values:
            if v > self.cfg["vov_halt_threshold"]:
                halted = True
            elif v < self.cfg["vov_resume_threshold"]:
                halted = False
        self._vov_halted = halted

    def _look_for_entry(self, row: pd.Series, d: pd.DataFrame, today):
        can_open, why = self.risk_manager.can_open_new_position(today)
        if not can_open:
            self.logger.info(f"Not opening new position: {why}")
            return

        ok, reason = strategy.check_entry(row, self.cfg, self._vov_halted)
        indicators = {
            "Price": float(row["close"]), "ADX": float(row["adx"]),
            "RSI": float(row["rsi"]), "VoV": round(float(row["vov"]), 3),
            "VoV_Halt": self._vov_halted,
        }
        indicators["Reason" if ok else "Blocked"] = reason
        self.logger.log_signal("BUY" if ok else "HOLD", Config.SYMBOL, float(row["close"]), indicators)
        if ok:
            self._execute_entry(row, d, today)

    def _manage_open_position(self, row: pd.Series, today):
        position = self.risk_manager.position
        fee_maker = self.cfg["fee_maker"] / 100

        position, partial_exits, full_exit = strategy.evaluate_position_candle(
            row, position, self.cfg, fee_maker
        )

        for pe in partial_exits:
            try:
                pe["price"] = self._sell(pe["qty"], pe["price"])
            except Exception as e:
                self.logger.warning(f"Tranche {pe['label']} sell skipped, keeping that part open: {e}")
                position["qty"] += pe["qty"]
                continue
            pnl = self.risk_manager.record_partial_exit(pe, today)
            self.logger.log_trade("SELL", Config.SYMBOL, pe["qty"], pe["price"],
                                  pe["qty"] * pe["price"], reason=f"Tranche {pe['label']}", pnl=pnl)

        if full_exit is not None:
            position["exit_pending"] = {"price": float(full_exit["price"]), "reason": full_exit["reason"]}
            self.risk_manager.update_position_state(position)
            self._finish_exit(row, today)
        else:
            self.risk_manager.update_position_state(position)

    def _sell(self, qty: float, expected_price: float) -> float:
        order = self.exchange.place_market_order(Config.SYMBOL, "SELL", qty, expected_price)
        price = float(order.get("avg_fill_price") or order.get("price") or 0.0)
        return price if price > 0 else float(expected_price)

    def _finish_exit(self, row: pd.Series, today):
        position = self.risk_manager.position
        pending = position["exit_pending"]
        qty = position["qty"]

        try:
            price = self._sell(qty, pending["price"])
        except OrderTooSmall:
            self.logger.warning(f"Remaining {qty:.8f} {Config.SYMBOL} is too small to sell, closing the position anyway")
            price = pending["price"]
        except Exception as e:
            self.logger.error(f"Exit order failed ({pending['reason']}), will try again next cycle: {e}")
            return

        trade = self.risk_manager.record_full_exit(
            price, pending["reason"], row["timestamp"], today,
            self.cfg["fee_maker"] / 100, self.cfg["slippage_normal"] / 100,
        )
        self.logger.log_trade("SELL", Config.SYMBOL, qty, price, qty * price,
                              reason=pending["reason"], pnl=trade["pnl"], pnl_percent=trade["pnl_pct"])
        self.logger.info(self.risk_manager.display_statistics())

    def _execute_entry(self, row: pd.Series, d: pd.DataFrame, today):
        entry = float(row["close"])
        idx = len(d) - 1
        sl = float(strategy.compute_stop_loss(d, idx, entry, self.cfg))
        conf = strategy.confidence_score(row, self.cfg)

        equity, recent_returns, consec_losses, dd_pct, day_pnl = self.risk_manager.sizing_inputs(today)
        qty, sizing = strategy.compute_size(
            equity, entry, sl, conf, recent_returns, consec_losses, dd_pct, day_pnl, self.cfg
        )

        if qty <= 0 or "blocked" in sizing:
            self.logger.warning(f"Entry signal fired but sizing blocked it: {sizing}")
            return

        setup = {
            "sl": sl, "atr": float(row["atr"]), "vov": float(row["vov"]),
            "confidence": conf, "signal_time": row["timestamp"],
            "limit_price": entry, "candles_waited": 0,
        }
        self.logger.info(f"Attempting BUY {qty:.8f} {Config.SYMBOL} @ ${entry:.2f} | sizing={sizing}")

        try:
            if self.cfg["use_limit_orders"]:
                order = self.exchange.place_limit_order(Config.SYMBOL, "BUY", qty, entry)
                setup["order_id"] = order["orderId"]
                self.risk_manager.set_pending_entry(setup)
                self._check_pending_entry()
            else:
                order = self.exchange.place_market_order(
                    Config.SYMBOL, "BUY", qty, entry, Config.MAX_SLIPPAGE_PERCENT
                )
                fill_price = float(order.get("avg_fill_price") or order.get("price") or 0.0)
                self._open_position(setup, float(order["executedQty"]), fill_price)
        except OrderTooSmall as e:
            self.logger.warning(f"Entry skipped: {e}")
        except Exception as e:
            self.logger.error(f"Buy order failed: {e}")

    def _check_pending_entry(self, expire: bool = False):
        pending = self.risk_manager.pending_entry
        order_id = pending["order_id"]

        if expire:
            try:
                self.exchange.cancel_order(Config.SYMBOL, order_id)
            except Exception as e:
                self.logger.warning(f"Could not cancel order {order_id}: {e}")

        try:
            status = self.exchange.get_order_status(Config.SYMBOL, order_id)
        except Exception as e:
            self.logger.warning(f"Could not check order {order_id}: {e}")
            return

        if status.get("status") not in self.FINISHED_ORDER_STATES:
            return

        self.risk_manager.set_pending_entry(None)
        filled = float(status.get("executedQty") or 0.0)
        if filled > 0:
            fill_price = float(status.get("avg_fill_price") or status.get("price") or 0.0)
            self._open_position(pending, filled, fill_price)
        else:
            self.logger.info(f"Limit BUY order {order_id} was not filled and is no longer open")

    def _age_pending_entry(self):
        pending = self.risk_manager.pending_entry
        pending["candles_waited"] += 1
        if pending["candles_waited"] >= self.cfg["limit_order_ttl"]:
            self._check_pending_entry(expire=True)
        else:
            self.risk_manager.set_pending_entry(pending)

    def _open_position(self, setup: dict, qty: float, fill_price: float):
        if fill_price <= 0:
            self.logger.warning(f"Exchange returned no fill price, using the order price ${setup['limit_price']:.2f}")
            fill_price = setup["limit_price"]

        atr_v = setup["atr"]
        position = {
            "entry": fill_price, "qty": qty, "qty_original": qty,
            "sl": setup["sl"], "tp_t1": fill_price + self.cfg["tp_t1_atr_mult"] * atr_v,
            "trail_stop": fill_price - 2.0 * atr_v, "atr_entry": atr_v,
            "in_profit": False, "t1_done": False, "t2_done": False,
            "candles_held": 0, "partial_pnl": 0.0,
            "entry_time": setup["signal_time"],
            "vov_at_entry": setup["vov"],
            "confidence": setup["confidence"],
        }
        self.risk_manager.open_position(position)
        self.logger.log_trade("BUY", Config.SYMBOL, qty, fill_price, qty * fill_price,
                              reason="Entry: all filters passed")
        self.logger.info(f"Position opened: SL=${setup['sl']:.2f}, T1=${position['tp_t1']:.2f}")

    def _get_market_data(self) -> pd.DataFrame:
        lookback_candles = max(self.cfg["ema_trend"], self.cfg["atr_regime_period"]) * 2 + 100
        tf_minutes = self._timeframe_to_minutes(Config.TIMEFRAME)
        start_time = datetime.now(timezone.utc) - timedelta(minutes=tf_minutes * lookback_candles)
        return self.exchange.get_historical_klines(
            symbol=Config.SYMBOL, interval=Config.TIMEFRAME,
            start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            limit=1000,
        )

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return {"m": value, "h": value * 60, "d": value * 1440,
                "w": value * 10080, "M": value * 43200}.get(unit, 15)

    def stop(self):
        self.logger.info("Stopping trading bot...")
        self.running = False
        if self.risk_manager.pending_entry is not None:
            self._check_pending_entry(expire=True)
        if self.risk_manager.has_open_position():
            self.logger.info("A position is still open. It is saved and will be managed when the bot starts again.")
        self.logger.info(self.risk_manager.display_statistics())
        self.logger.info("Bot stopped successfully")

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "mode": Config.TRADING_MODE,
            "symbol": Config.SYMBOL,
            "has_position": self.risk_manager.has_open_position(),
            "equity": self.risk_manager.equity,
            "trades": len(self.risk_manager.trades),
        }
