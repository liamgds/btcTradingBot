import json
import math
import os
from datetime import date as date_type
from typing import Dict, List, Optional, Tuple


class RiskManager:
    def __init__(self, initial_capital: float, cfg: Dict,
                state_path: str = "logs/risk_state.json"):
        self.cfg = cfg
        self.state_path = state_path

        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.position: Optional[Dict] = None
        self.daily_pnl: Dict[str, float] = {}
        self.recent_returns: List[float] = []
        self.consec_losses = 0
        self.consec_wins = 0
        self.trades: List[Dict] = []
        self.pending_entry: Optional[Dict] = None
        self.last_candle: Optional[str] = None

        self._load_state()

    @property
    def dd_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity * 100

    def today_pnl(self, today: date_type) -> float:
        return self.daily_pnl.get(str(today), 0.0)

    def sizing_inputs(self, today: date_type) -> Tuple[float, List[float], int, float, float]:
        return (self.equity, self.recent_returns, self.consec_losses,
                self.dd_pct, self.today_pnl(today))

    def can_open_new_position(self, today: date_type) -> Tuple[bool, str]:
        if self.position is not None:
            return False, "position_already_open"
        if self.today_pnl(today) < -(self.equity * self.cfg["max_daily_loss_pct"] / 100):
            return False, "daily_loss_limit_hit"
        return True, "ok"

    def has_open_position(self) -> bool:
        return self.position is not None

    def open_position(self, position: Dict):
        self.position = position
        self._save_state()

    def record_partial_exit(self, pe: Dict, today: date_type) -> float:
        entry = self.position["entry"]
        p_gross = (pe["price"] - entry) * pe["qty"]
        p_fee = (entry + pe["price"]) * pe["qty"] * pe["fee_rate"]
        p_pnl = p_gross - p_fee
        self.equity += p_pnl
        self.daily_pnl[str(today)] = self.today_pnl(today) + p_pnl
        self.position["partial_pnl"] = self.position.get("partial_pnl", 0) + p_pnl
        self._save_state()
        return p_pnl

    def record_full_exit(self, exit_price: float, reason: str, exit_time,
                         today: date_type, fee_maker: float, fee_slip: float) -> Dict:
        entry = self.position["entry"]
        gross = (exit_price - entry) * self.position["qty"]
        cost = (entry + exit_price) * self.position["qty"] * (fee_maker + fee_slip)
        pnl = gross - cost + self.position.get("partial_pnl", 0)

        self.equity += (gross - cost)
        self.daily_pnl[str(today)] = self.today_pnl(today) + (gross - cost)
        self.peak_equity = max(self.peak_equity, self.equity)

        pnl_pct = (pnl / (self.position["qty_original"] * entry) * 100
                  if entry > 0 and self.position["qty_original"] > 0 else 0.0)
        if pnl > 0:
            self.consec_wins += 1
            self.consec_losses = 0
        else:
            self.consec_losses += 1
            self.consec_wins = 0

        self.recent_returns.append(pnl_pct / 100)
        if len(self.recent_returns) > self.cfg["vol_lookback_trades"] * 2:
            self.recent_returns.pop(0)

        trade = {
            "entry_time": self.position["entry_time"],
            "exit_time": exit_time,
            "entry_price": entry,
            "exit_price": exit_price,
            "qty": self.position["qty_original"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
        }
        self.trades.append(trade)
        self.position = None
        self._save_state()
        return trade

    def update_position_state(self, position: Dict):
        self.position = position
        self._save_state()

    def set_pending_entry(self, pending: Optional[Dict]):
        self.pending_entry = pending
        self._save_state()

    def mark_candle(self, candle_id: str):
        self.last_candle = candle_id
        self._save_state()

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        state = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "position": self.position,
            "daily_pnl": self.daily_pnl,
            "recent_returns": self.recent_returns,
            "consec_losses": self.consec_losses,
            "consec_wins": self.consec_wins,
            "pending_entry": self.pending_entry,
            "last_candle": self.last_candle,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)

            equity = state.get("equity", self.equity)
            peak_equity = state.get("peak_equity", self.peak_equity)
            self.equity = equity if math.isfinite(equity) else self.equity
            self.peak_equity = peak_equity if math.isfinite(peak_equity) else self.peak_equity
            self.position = state.get("position")
            self.daily_pnl = {k: v for k, v in state.get("daily_pnl", {}).items() if math.isfinite(v)}
            self.recent_returns = [r for r in state.get("recent_returns", []) if math.isfinite(r)]
            self.consec_losses = state.get("consec_losses", 0)
            self.consec_wins = state.get("consec_wins", 0)
            self.pending_entry = state.get("pending_entry")
            self.last_candle = state.get("last_candle")
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    def display_statistics(self) -> str:
        wins = [t for t in self.trades if t["pnl"] > 0]
        wr = (len(wins) / len(self.trades) * 100) if self.trades else 0.0
        return (f"Equity: ${self.equity:,.2f} | Trades: {len(self.trades)} | "
               f"Win rate: {wr:.1f}% | Drawdown: {self.dd_pct:.2f}% | "
               f"Consecutive losses: {self.consec_losses}")
