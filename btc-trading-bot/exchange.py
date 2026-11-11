import time
import hashlib
import random
from decimal import Decimal
from typing import Optional, Dict, List, Tuple

import pandas as pd
from retry import retry


class OrderTooSmall(Exception):
    pass


class CircuitBreaker:
    def __init__(self, threshold: int = 5, reset_timeout: int = 300):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"

    def can_proceed(self) -> Tuple[bool, str]:
        if self.state == "CLOSED":
            return True, "OK"
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.reset_timeout:
                self.state = "HALF_OPEN"
                self.failure_count = 0
                return True, "Attempting recovery"
            return False, f"Circuit breaker OPEN ({self.failure_count} failures)"
        return True, "Testing connection"


class BinanceExchange:
    _EXPECTED_ENDPOINT_FRAGMENT = {
        "demo": "demo-api.binance.com",
        "testnet": "testnet.binance.vision",
    }
    _DEFAULT_FILTERS = {"step": "0.00001", "tick": "0.01", "min_qty": "0.00001", "min_notional": "5"}

    def __init__(self, api_key: str, api_secret: str, mode: str = "paper"):
        if mode == "live":
            raise ValueError(
                "BinanceExchange refuses to construct in 'live' mode. This "
                "build is demo-only (paper/demo/testnet)."
            )
        if mode not in ("paper", "demo", "testnet"):
            raise ValueError(f"Unknown trading mode '{mode}'. Must be 'paper', 'demo', or 'testnet'.")

        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode
        self.paper_mode = (mode == "paper") or not api_key or not api_secret

        from binance.client import Client
        if self.paper_mode:
            self.client = None
            self.public_client = Client()
        else:
            self.client = Client(
                api_key, api_secret,
                demo=(mode == "demo"),
                testnet=(mode == "testnet"),
            )
            self.public_client = self.client
            self._verify_endpoint()

        self.recent_orders: Dict[str, float] = {}
        self.order_cache_duration = 30
        self.circuit_breaker = CircuitBreaker(threshold=5, reset_timeout=300)
        self._open_limit_orders: Dict[str, Dict] = {}
        self._filters: Dict[str, Dict[str, Decimal]] = {}

    def prepare(self):
        if self.paper_mode:
            return
        try:
            server_time = self.client.get_server_time()["serverTime"]
            self.client.timestamp_offset = server_time - int(time.time() * 1000)
        except Exception:
            pass

    def symbol_filters(self, symbol: str) -> Dict[str, Decimal]:
        if symbol in self._filters:
            return self._filters[symbol]

        found = dict(self._DEFAULT_FILTERS)
        loaded = False
        try:
            info = self.public_client.get_symbol_info(symbol)
            for f in (info or {}).get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    found["step"], found["min_qty"] = f["stepSize"], f["minQty"]
                elif f["filterType"] == "PRICE_FILTER":
                    found["tick"] = f["tickSize"]
                elif f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
                    found["min_notional"] = f.get("minNotional", found["min_notional"])
            loaded = info is not None
        except Exception:
            pass

        filters = {k: Decimal(str(v)) for k, v in found.items()}
        if loaded:
            self._filters[symbol] = filters
        return filters

    def _order_quantity(self, symbol: str, quantity: float, price: float) -> Decimal:
        f = self.symbol_filters(symbol)
        qty = (Decimal(str(float(quantity))) // f["step"]) * f["step"]
        if qty <= 0 or qty < f["min_qty"] or qty * Decimal(str(float(price))) < f["min_notional"]:
            raise OrderTooSmall(f"{float(quantity):.8f} {symbol} is below the minimum order size")
        return qty

    def _order_price(self, symbol: str, price: float) -> Decimal:
        tick = self.symbol_filters(symbol)["tick"]
        return (Decimal(str(float(price))) // tick) * tick

    def _verify_endpoint(self):
        if self.paper_mode:
            if self.client is not None:
                raise RuntimeError("Safety check failed: paper mode has an authenticated client, it must not.")
            return

        expected_fragment = self._EXPECTED_ENDPOINT_FRAGMENT.get(self.mode)
        if expected_fragment is None:
            raise RuntimeError(f"Safety check failed: no known-good endpoint defined for mode '{self.mode}'.")

        flags_ok = {
            "demo": getattr(self.client, "demo", False) and not getattr(self.client, "testnet", False),
            "testnet": getattr(self.client, "testnet", False) and not getattr(self.client, "demo", False),
        }[self.mode]
        if not flags_ok:
            raise RuntimeError(
                f"Safety check failed: mode='{self.mode}' but client.demo="
                f"{getattr(self.client, 'demo', None)}, client.testnet="
                f"{getattr(self.client, 'testnet', None)}. Refusing to proceed."
            )

        try:
            resolved_url = self.client._create_api_uri("account")
        except Exception as e:
            raise RuntimeError(f"Safety check failed: could not resolve API URL to verify ({e}).")

        if expected_fragment not in resolved_url:
            raise RuntimeError(
                f"Safety check failed: mode='{self.mode}' resolved to '{resolved_url}', "
                f"which does not contain the expected '{expected_fragment}'. Refusing to proceed."
            )

    def _check_circuit_breaker(self):
        can_proceed, reason = self.circuit_breaker.can_proceed()
        if not can_proceed:
            raise Exception(f"Circuit breaker preventing operation: {reason}")

    def _order_hash(self, symbol: str, side: str, quantity: float) -> str:
        data = f"{symbol}{side}{quantity:.8f}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _is_recent_order(self, order_hash: str) -> bool:
        now = time.time()
        self.recent_orders = {h: t for h, t in self.recent_orders.items()
                               if now - t < self.order_cache_duration}
        return order_hash in self.recent_orders

    def _mark_order_placed(self, order_hash: str):
        self.recent_orders[order_hash] = time.time()

    @retry(tries=3, delay=2, backoff=2)
    def get_account_balance(self, asset: str = "USDT") -> float:
        self._check_circuit_breaker()
        if self.paper_mode:
            return 10_000.0
        try:
            info = self.client.get_account()
            for bal in info["balances"]:
                if bal["asset"] == asset:
                    self.circuit_breaker.record_success()
                    return float(bal["free"])
            self.circuit_breaker.record_success()
            return 0.0
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get balance: {e}")

    @retry(tries=3, delay=2, backoff=2)
    def get_current_price(self, symbol: str) -> float:
        self._check_circuit_breaker()
        try:
            ticker = self.public_client.get_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])
            self.circuit_breaker.record_success()
            return price
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get price for {symbol}: {e}")

    @retry(tries=3, delay=2, backoff=2)
    def get_historical_klines(self, symbol: str, interval: str,
                              start_time: str, end_time: Optional[str] = None,
                              limit: int = 1000) -> pd.DataFrame:
        self._check_circuit_breaker()
        try:
            all_klines: List = []
            current_start = start_time
            while True:
                klines = self.public_client.get_historical_klines(
                    symbol=symbol, interval=interval,
                    start_str=current_start, end_str=end_time, limit=limit,
                )
                if not klines:
                    break
                all_klines.extend(klines)
                if len(klines) < limit:
                    break
                current_start = str(klines[-1][0] + 1)
                if len(all_klines) >= 20_000:
                    break

            if not all_klines:
                raise Exception("No data returned from exchange")

            df = pd.DataFrame(all_klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)

            if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
                raise Exception("Data contains NaN values")

            self.circuit_breaker.record_success()
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get historical data: {e}")

    def place_market_order(self, symbol: str, side: str, quantity: float,
                           expected_price: float, max_slippage_percent: Optional[float] = None) -> Dict:
        self._verify_endpoint()
        self._check_circuit_breaker()
        qty = self._order_quantity(symbol, quantity, expected_price)

        order_hash = self._order_hash(symbol, side, float(qty))
        if self._is_recent_order(order_hash):
            raise Exception(f"Duplicate order detected (same order within {self.order_cache_duration}s)")

        if self.paper_mode:
            current_price = self.get_current_price(symbol)
            fill_price = current_price * (1 + random.uniform(-0.001, 0.001))
            slippage = abs((fill_price - expected_price) / expected_price * 100)
            if max_slippage_percent is not None and slippage > max_slippage_percent:
                raise Exception(f"Simulated slippage {slippage:.2f}% exceeds maximum {max_slippage_percent}%")
            self._mark_order_placed(order_hash)
            self.circuit_breaker.record_success()
            return {
                "orderId": f"PAPER_{int(time.time() * 1000)}",
                "symbol": symbol, "side": side, "type": "MARKET",
                "origQty": float(qty), "executedQty": float(qty),
                "price": fill_price, "avg_fill_price": fill_price,
                "actual_slippage_percent": slippage,
                "status": "FILLED", "transactTime": int(time.time() * 1000),
                "paper_trade": True,
            }

        try:
            if side == "BUY":
                order = self.client.order_market_buy(symbol=symbol, quantity=format(qty, "f"))
            else:
                order = self.client.order_market_sell(symbol=symbol, quantity=format(qty, "f"))

            if order["status"] not in ("FILLED", "PARTIALLY_FILLED"):
                raise Exception(f"Order not filled: {order['status']}")

            executed_qty = float(order["executedQty"])
            if executed_qty == 0:
                raise Exception("Order executed but quantity is zero")

            total_cost, total_qty = 0.0, 0.0
            for fill in order.get("fills", []):
                fp, fq = float(fill["price"]), float(fill["qty"])
                total_cost += fp * fq
                total_qty += fq
            if total_qty > 0:
                avg_price = total_cost / total_qty
            elif float(order.get("cummulativeQuoteQty") or 0) > 0:
                avg_price = float(order["cummulativeQuoteQty"]) / executed_qty
            else:
                avg_price = expected_price

            slippage = abs((avg_price - expected_price) / expected_price * 100)
            if max_slippage_percent is not None and slippage > max_slippage_percent:
                print(f"WARNING: Slippage {slippage:.2f}% exceeds maximum {max_slippage_percent}%")

            order["avg_fill_price"] = avg_price
            order["actual_slippage_percent"] = slippage
            order["executedQty"] = executed_qty
            self._mark_order_placed(order_hash)
            self.circuit_breaker.record_success()
            return order
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Order failed: {e}")

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        self._verify_endpoint()
        self._check_circuit_breaker()
        limit_price = self._order_price(symbol, price)
        qty = self._order_quantity(symbol, quantity, float(limit_price))

        if self.paper_mode:
            order_id = f"PAPER_LIMIT_{int(time.time() * 1000)}"
            self._open_limit_orders[order_id] = {
                "symbol": symbol, "side": side, "qty": float(qty), "price": float(limit_price),
            }
            self.circuit_breaker.record_success()
            return {
                "orderId": order_id, "symbol": symbol, "side": side, "type": "LIMIT",
                "origQty": float(qty), "executedQty": 0.0, "price": float(limit_price),
                "status": "NEW", "transactTime": int(time.time() * 1000),
                "paper_trade": True,
            }

        try:
            place = self.client.order_limit_buy if side == "BUY" else self.client.order_limit_sell
            order = place(symbol=symbol, quantity=format(qty, "f"), price=format(limit_price, "f"))
            self.circuit_breaker.record_success()
            return order
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Limit order failed: {e}")

    def get_order_status(self, symbol: str, order_id: str) -> Dict:
        if self.paper_mode:
            info = self._open_limit_orders.get(str(order_id))
            if info is None:
                return {"status": "UNKNOWN", "executedQty": 0.0}
            price = self.get_current_price(symbol)
            crossed = price <= info["price"] if info["side"] == "BUY" else price >= info["price"]
            if crossed:
                self._open_limit_orders.pop(str(order_id), None)
                return {"status": "FILLED", "executedQty": info["qty"],
                        "price": info["price"], "avg_fill_price": info["price"]}
            return {"status": "NEW", "executedQty": 0.0, "price": info["price"]}

        try:
            result = self.client.get_order(symbol=symbol, orderId=order_id)
            executed = float(result.get("executedQty") or 0)
            quote = float(result.get("cummulativeQuoteQty") or 0)
            result["executedQty"] = executed
            if executed > 0 and quote > 0:
                result["avg_fill_price"] = quote / executed
            else:
                result["avg_fill_price"] = float(result.get("price") or 0)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Failed to get order status: {e}")

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        self._check_circuit_breaker()
        if self.paper_mode:
            self._open_limit_orders.pop(str(order_id), None)
            return {"orderId": order_id, "status": "CANCELED", "paper_trade": True}
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            self.circuit_breaker.record_success()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise Exception(f"Cancel order failed: {e}")

    def get_circuit_breaker_status(self) -> Dict:
        return {
            "state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "threshold": self.circuit_breaker.threshold,
        }
