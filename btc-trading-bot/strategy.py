import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(hi: pd.Series, lo: pd.Series, cl: pd.Series, p: int = 14) -> pd.Series:
    prev = cl.shift(1)
    tr   = pd.concat([hi-lo, (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()


def _adx_full(hi: pd.Series, lo: pd.Series, cl: pd.Series,
               p: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    prev_h = hi.shift(1); prev_l = lo.shift(1); prev_c = cl.shift(1)
    tr     = pd.concat([hi-lo, (hi-prev_c).abs(), (lo-prev_c).abs()], axis=1).max(axis=1)
    up     = hi - prev_h; dn = prev_l - lo
    pdm    = pd.Series(np.where((up>dn)&(up>0), up, 0.0), index=hi.index)
    ndm    = pd.Series(np.where((dn>up)&(dn>0), dn, 0.0), index=hi.index)
    tr_s   = tr.ewm(alpha=1/p, adjust=False).mean()
    pdi    = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / tr_s.replace(0, np.nan)
    ndi    = 100 * ndm.ewm(alpha=1/p, adjust=False).mean() / tr_s.replace(0, np.nan)
    dx     = 100 * (pdi-ndi).abs() / (pdi+ndi).replace(0, np.nan)
    adx_s  = dx.ewm(alpha=1/p, adjust=False).mean()
    return adx_s, pdi, ndi

DEFAULT_CFG: Dict = {
    "ema_fast":             9,
    "ema_slow":            21,
    "ema_trend":          200,
    "rsi_period":          14,
    "rsi_floor":           45.0,
    "rsi_ceil":            62.0,
    "rsi_extreme":         75.0,
    "adx_period":          14,
    "adx_min":             20.0,
    "adx_slope_lookback":   5,
    "di_spread_min":        3.0,
    "vol_ma_period":       20,
    "sus_vol_lookback":     4,
    "sus_vol_threshold":    1.05,
    "atr_period":          14,
    "atr_regime_period":   50,
    "atr_regime_low":       0.60,
    "atr_regime_high":      2.00,
    "min_session_score":    1,
    "vov_fast_period":      5,
    "vov_slow_period":     14,
    "vov_halt_threshold":   1.80,
    "vov_resume_threshold": 1.40,
    "sl_swing_lookback":    5,
    "sl_atr_buffer_mult":   1.00,
    "sl_hard_floor_mult":   3.00,
    "tp_t1_atr_mult":       1.50,
    "tp_t2_adx_map": {
        20: 2.0, 25: 2.5, 30: 3.0, 35: 3.5, 40: 4.0, 999: 5.0
    },
    "trail_mult_map": {
        20: 1.0, 25: 1.2, 35: 1.5, 999: 2.0
    },
    "max_hold_candles":    32,
    "adx_decay_floor":     15.0,
    "risk_pct":             1.50,
    "max_position_pct":    10.00,
    "max_total_exposure":  20.00,
    "vol_target_pct":       1.20,
    "vol_lookback_trades":  20,
    "vol_scale_min":        0.30,
    "vol_scale_max":        1.50,
    "conf_scale_min":       0.70,
    "conf_scale_max":       1.30,
    "dd_caution":           5.0,
    "dd_defensive":         8.0,
    "dd_survival":         10.0,
    "dd_halt":             12.0,
    "am_loss_1":            2,
    "am_loss_2":            3,
    "am_scale_1":           0.75,
    "am_scale_2":           0.50,
    "monitor_window":      20,
    "monitor_warn_wr":      0.45,
    "monitor_halt_wr":      0.40,
    "use_limit_orders":    True,
    "limit_order_ttl":      2,
    "max_daily_loss_pct":   4.0,
    "daily_profit_lock_pct": 3.0,
    "daily_profit_scale":   0.60,
    "fee_maker":            0.10,
    "fee_taker":            0.10,
    "slippage_normal":      0.03,
    "slippage_stressed":    0.15,
}


def build_indicators(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    d = df.copy()

    d["ema_fast"]   = _ema(d["close"], cfg["ema_fast"])
    d["ema_slow"]   = _ema(d["close"], cfg["ema_slow"])
    d["ema_trend"]  = _ema(d["close"], cfg["ema_trend"])
    d["rsi"]        = _rsi(d["close"], cfg["rsi_period"])
    d["atr"]        = _atr(d["high"], d["low"], d["close"], cfg["atr_period"])
    d["atr_fast"]   = _atr(d["high"], d["low"], d["close"], cfg["vov_fast_period"])

    d["adx"], d["plus_di"], d["minus_di"] = _adx_full(
        d["high"], d["low"], d["close"], cfg["adx_period"]
    )

    atr_base       = d["atr"].rolling(cfg["atr_regime_period"]).mean()
    d["atr_ratio"] = d["atr"] / atr_base.replace(0, np.nan)

    d["vov"]       = d["atr_fast"] / d["atr"].replace(0, np.nan)

    vol_avg        = d["volume"].rolling(cfg["vol_ma_period"]).mean()
    rel_vol        = d["volume"] / vol_avg.replace(0, np.nan)
    d["sus_vol"]   = rel_vol.rolling(cfg["sus_vol_lookback"]).mean()

    d["di_spread"] = d["plus_di"] - d["minus_di"]
    d["adx_rising"] = d["adx"] > d["adx"].shift(cfg["adx_slope_lookback"])

    hour          = d["timestamp"].dt.hour
    conditions    = [(hour < 8), (hour < 10), (hour < 13), (hour < 21)]
    scores        = [0, 1, 2, 3]
    d["session"]  = np.select(conditions, scores, default=2)

    d["ema_above"]   = (d["ema_fast"] > d["ema_slow"]).astype(int)
    d["bull_cross"]  = (d["ema_above"] == 1) & (d["ema_above"].shift(1) == 0)
    d["bear_cross"]  = (d["ema_above"] == 0) & (d["ema_above"].shift(1) == 1)

    d["atr_cap"]  = d["atr"].rolling(50).quantile(0.75).fillna(d["atr"])

    return d.dropna().reset_index(drop=True)


def confidence_score(row: pd.Series, cfg: Dict) -> float:
    adx_s  = float(np.clip((row["adx"] - 20) / 30, 0, 1))

    centre = (cfg["rsi_floor"] + cfg["rsi_ceil"]) / 2
    half   = (cfg["rsi_ceil"] - cfg["rsi_floor"]) / 2
    rsi_s  = float(np.clip(1 - abs(row["rsi"] - centre) / half, 0, 1))

    vol_s  = float(np.clip((row["sus_vol"] - 1.0) / 2.0, 0, 1))

    di_s   = float(np.clip((row["di_spread"] - 3) / 17, 0, 1))

    sess_s = float(np.clip(0.5 + 0.5 * row["session"] / 3, 0.5, 1.0))

    score  = 0.30*adx_s + 0.20*rsi_s + 0.20*vol_s + 0.20*di_s + 0.10*sess_s
    return float(np.clip(score, 0, 1))


def adx_to_tp(adx: float, cfg: Dict) -> float:
    mapping = cfg["tp_t2_adx_map"]
    for threshold in sorted(mapping.keys()):
        if adx < threshold:
            return mapping[threshold]
    return mapping[max(mapping.keys())]


def adx_to_trail(adx: float, cfg: Dict) -> float:
    mapping = cfg["trail_mult_map"]
    for threshold in sorted(mapping.keys()):
        if adx < threshold:
            return mapping[threshold]
    return mapping[max(mapping.keys())]


def check_entry(row: pd.Series, cfg: Dict,
                vov_halted: bool) -> Tuple[bool, str]:
    if vov_halted:
        return False, "vov_crash_filter"
    if not row["bull_cross"]:
        return False, "no_cross"
    if row["close"] <= row["ema_trend"]:
        return False, "below_trend"
    if row["adx"] < cfg["adx_min"]:
        return False, "adx_low"
    if not row["adx_rising"]:
        return False, "adx_falling"
    if row["di_spread"] < cfg["di_spread_min"]:
        return False, "di_spread"
    if row["sus_vol"] < cfg["sus_vol_threshold"]:
        return False, "sus_vol"
    if not (cfg["rsi_floor"] <= row["rsi"] < cfg["rsi_ceil"]):
        return False, "rsi_range"
    if not (cfg["atr_regime_low"] <= row["atr_ratio"] <= cfg["atr_regime_high"]):
        return False, "atr_regime"
    if row["session"] < cfg["min_session_score"]:
        return False, "session"
    return True, "ok"


def compute_stop_loss(df: pd.DataFrame, idx: int,
                      entry: float, cfg: Dict) -> float:
    lookback   = cfg["sl_swing_lookback"]
    start      = max(0, idx - lookback)
    swing_low  = df["low"].iloc[start:idx+1].min()
    atr_val    = df["atr"].iloc[idx]

    struct_sl  = swing_low - cfg["sl_atr_buffer_mult"] * atr_val

    hard_floor = entry - cfg["sl_hard_floor_mult"] * min(atr_val, df["atr_cap"].iloc[idx])

    return max(struct_sl, hard_floor)


def compute_size(equity: float, entry: float, sl: float,
                 conf: float, recent_returns: List[float],
                 consec_losses: int, dd_pct: float,
                 daily_pnl: float, cfg: Dict) -> Tuple[float, Dict]:
    sl_dist = entry - sl
    if sl_dist <= 0:
        return 0.0, {"blocked": "invalid_sl"}

    l1 = cfg["risk_pct"] / 100

    if len(recent_returns) >= 5:
        rv = np.std(recent_returns[-cfg["vol_lookback_trades"]:])
        l2 = float(np.clip(
            (cfg["vol_target_pct"]/100) / rv if rv > 0 else 1.0,
            cfg["vol_scale_min"], cfg["vol_scale_max"]
        ))
    else:
        l2 = 1.0

    l3 = cfg["conf_scale_min"] + conf * (cfg["conf_scale_max"] - cfg["conf_scale_min"])
    l3 = float(np.clip(l3, cfg["conf_scale_min"], cfg["conf_scale_max"]))

    if   dd_pct >= cfg["dd_halt"]:      return 0.0, {"blocked": f"dd_halt_{dd_pct:.1f}%"}
    elif dd_pct >= cfg["dd_survival"]:  l4 = 0.25
    elif dd_pct >= cfg["dd_defensive"]: l4 = 0.50
    elif dd_pct >= cfg["dd_caution"]:   l4 = 0.75
    else:                               l4 = 1.00

    if   consec_losses >= cfg["am_loss_2"]: l5 = cfg["am_scale_2"]
    elif consec_losses >= cfg["am_loss_1"]: l5 = cfg["am_scale_1"]
    else:                                    l5 = 1.00

    l6 = 1.0
    if len(recent_returns) >= cfg["monitor_window"]:
        recent_wr = (np.array(recent_returns[-cfg["monitor_window"]:]) > 0).mean()
        if recent_wr < cfg["monitor_halt_wr"]:
            return 0.0, {"blocked": f"perf_halt_wr={recent_wr:.2f}"}
        elif recent_wr < cfg["monitor_warn_wr"]:
            l6 = 0.50

    l7 = cfg["daily_profit_scale"] if daily_pnl > equity * cfg["daily_profit_lock_pct"] / 100 else 1.0

    adj_risk_pct = l1 * l2 * l3 * l4 * l5 * l6 * l7
    dollar_risk  = equity * adj_risk_pct
    qty          = dollar_risk / sl_dist
    qty          = min(qty, equity * cfg["max_position_pct"] / 100 / entry)

    return qty, {
        "l1_base": round(l1*100, 3), "l2_vol": round(l2, 3),
        "l3_conf": round(l3, 3),     "l4_dd":  round(l4, 3),
        "l5_am":   round(l5, 3),     "l6_mon": round(l6, 3),
        "l7_dprot":round(l7, 3),     "adj_risk_pct": round(adj_risk_pct*100, 3),
        "dollar_risk": round(dollar_risk, 2), "qty": round(qty, 8),
    }


def evaluate_position_candle(row: pd.Series, position: Dict, cfg: Dict,
                             fee_maker: float) -> Tuple[Dict, List[Dict], Optional[Dict]]:
    entry = position["entry"]
    atr_e = position["atr_entry"]
    adx_c = float(row["adx"])

    tp_mult = adx_to_tp(adx_c, cfg)
    tp_t2   = entry + tp_mult * atr_e

    trail_mult = adx_to_trail(adx_c, cfg)
    trail_mult = max(trail_mult, cfg["trail_mult_map"][min(cfg["trail_mult_map"])])
    new_trail  = row["close"] - trail_mult * float(row["atr"])
    if not position.get("in_profit"):
        if row["close"] >= entry + 0.5 * atr_e:
            position["in_profit"] = True
    if position.get("in_profit") and new_trail > position["trail_stop"]:
        position["trail_stop"] = new_trail

    sl = position["sl"]
    partial_exits: List[Dict] = []
    full_exit: Optional[Dict] = None

    if row["low"] <= sl:
        full_exit = {"price": sl, "reason": "StructuralStop"}

    elif not position.get("t1_done") and row["high"] >= position["tp_t1"]:
        p_qty   = position["qty"] * 0.333
        p_price = position["tp_t1"]
        partial_exits.append({"qty": p_qty, "price": p_price, "fee_rate": fee_maker, "label": "T1"})
        position["qty"]    -= p_qty
        position["t1_done"] = True
        position["sl"]      = entry

    elif position.get("t1_done") and not position.get("t2_done") and row["high"] >= tp_t2:
        p_qty   = position["qty"] * 0.500
        p_price = tp_t2
        partial_exits.append({"qty": p_qty, "price": p_price, "fee_rate": fee_maker, "label": "T2"})
        position["qty"]    -= p_qty
        position["t2_done"] = True
        position["trail_stop"] = max(
            position["trail_stop"], row["close"] - 2.0 * float(row["atr"])
        )

    elif position.get("t1_done") and position.get("in_profit"):
        if row["low"] <= position["trail_stop"]:
            full_exit = {"price": position["trail_stop"], "reason": "AdaptiveTrail"}

    elif float(row["rsi"]) > cfg["rsi_extreme"]:
        full_exit = {"price": row["close"], "reason": "RSI_Extreme"}

    elif adx_c < cfg["adx_decay_floor"]:
        full_exit = {"price": row["close"], "reason": "ADX_Decay"}

    elif position.get("candles_held", 0) >= cfg["max_hold_candles"]:
        in_strong_profit = row["close"] >= entry + 1.5 * atr_e
        if not in_strong_profit:
            full_exit = {"price": row["close"], "reason": "TimeStop"}

    if full_exit is None:
        position["candles_held"] = position.get("candles_held", 0) + 1

    return position, partial_exits, full_exit


def run(df_raw: pd.DataFrame, capital: float = 10_000.0,
        cfg: Optional[Dict] = None) -> Dict:
    if cfg is None:
        cfg = DEFAULT_CFG

    df = build_indicators(df_raw, cfg)

    equity          = capital
    peak_equity     = capital
    position        = None
    trades: List    = []
    equity_curve    = []
    daily_pnl       = {}
    recent_returns: List[float] = []
    consec_losses   = 0
    vov_halted      = False

    fee_maker  = cfg["fee_maker"]  / 100
    fee_slip   = cfg["slippage_normal"] / 100

    for i in range(len(df)):
        row      = df.iloc[i]
        date_key = row["timestamp"].date()
        day_pnl  = daily_pnl.get(date_key, 0.0)

        vov = float(row["vov"]) if not np.isnan(row["vov"]) else 1.0
        if vov > cfg["vov_halt_threshold"]:
            vov_halted = True
        elif vov < cfg["vov_resume_threshold"]:
            vov_halted = False

        if day_pnl < -(equity * cfg["max_daily_loss_pct"] / 100) and position is None:
            equity_curve.append({"timestamp": row["timestamp"], "equity": equity})
            continue

        peak_equity = max(peak_equity, equity)
        dd_pct      = (peak_equity - equity) / peak_equity * 100

        if position is not None:
            entry = position["entry"]

            position, partial_exits, full_exit = evaluate_position_candle(
                row, position, cfg, fee_maker
            )

            for pe in partial_exits:
                p_gross = (pe["price"] - entry) * pe["qty"]
                p_fee   = (entry + pe["price"]) * pe["qty"] * pe["fee_rate"]
                p_pnl   = p_gross - p_fee
                equity += p_pnl
                day_pnl += p_pnl
                daily_pnl[date_key] = day_pnl
                position["partial_pnl"] = position.get("partial_pnl", 0) + p_pnl

            if full_exit is not None:
                exit_p = full_exit["price"]; reason = full_exit["reason"]
                gross = (exit_p - entry) * position["qty"]
                cost  = (entry + exit_p) * position["qty"] * (fee_maker + fee_slip)
                pnl   = gross - cost + position.get("partial_pnl", 0)
                equity   += gross - cost
                day_pnl  += gross - cost
                daily_pnl[date_key] = day_pnl

                pnl_pct = pnl / (position["qty_original"] * entry) * 100

                if pnl > 0:
                    consec_losses = 0
                else:
                    consec_losses += 1

                recent_returns.append(pnl_pct / 100)
                if len(recent_returns) > cfg["vol_lookback_trades"] * 2:
                    recent_returns.pop(0)

                trades.append({
                    "entry_time":  position["entry_time"],
                    "exit_time":   row["timestamp"],
                    "entry_price": entry,
                    "exit_price":  exit_p,
                    "pnl":         pnl,
                    "pnl_pct":     pnl_pct,
                    "exit_reason": reason,
                    "vov_at_entry": position.get("vov_at_entry", 1.0),
                })
                position = None

        if position is None:
            ok, block = check_entry(row, cfg, vov_halted)
            if ok:
                entry = row["close"]
                sl    = compute_stop_loss(df, i, entry, cfg)
                conf  = confidence_score(row, cfg)

                qty, sizing = compute_size(
                    equity, entry, sl, conf, recent_returns,
                    consec_losses, dd_pct, day_pnl, cfg
                )

                if qty > 0 and "blocked" not in sizing:
                    atr_v = float(row["atr"])
                    tp_t1 = entry + cfg["tp_t1_atr_mult"] * atr_v

                    position = {
                        "entry":        entry,
                        "qty":          qty,
                        "qty_original": qty,
                        "sl":           sl,
                        "tp_t1":        tp_t1,
                        "trail_stop":   entry - 2.0 * atr_v,
                        "atr_entry":    atr_v,
                        "in_profit":    False,
                        "t1_done":      False,
                        "t2_done":      False,
                        "candles_held": 0,
                        "partial_pnl":  0.0,
                        "entry_time":   row["timestamp"],
                        "vov_at_entry": vov,
                        "confidence":   conf,
                    }

        equity_curve.append({"timestamp": row["timestamp"], "equity": equity})

    if position is not None:
        last  = df.iloc[-1]
        gross = (last["close"] - position["entry"]) * position["qty"]
        cost  = (position["entry"] + last["close"]) * position["qty"] * (fee_maker + fee_slip)
        pnl   = gross - cost + position.get("partial_pnl", 0)
        equity += gross - cost
        pnl_pct = pnl / (position["qty_original"] * position["entry"]) * 100
        trades.append({
            "entry_time":  position["entry_time"],
            "exit_time":   last["timestamp"],
            "entry_price": position["entry"],
            "exit_price":  last["close"],
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "exit_reason": "EndOfData",
        })

    return {"trades": trades, "equity_curve": equity_curve, "final_equity": equity}
