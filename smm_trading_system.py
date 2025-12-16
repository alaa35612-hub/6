"""
Experimental automated trader for Simple Market Metrics v4.1 signals.
- Obeys compute_smm() outputs (buy_signal / sell_signal / profit_target_hit).
- Supports Binance USDT-M Futures testnet or paper trading; no live trading allowed.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Strategy wiring ------------------------------------------------------------
TRADING_MODE = "PAPER"  # Options: "TESTNET" or "PAPER"
CLOSE_ON_OPPOSITE_SIGNAL = True
STOP_LOSS_ENABLED = False  # Optional only; no explicit SL in strategy by default
STOP_LOSS_PCT = 0.02  # 2% default if enabled (disabled by default)
LEVERAGE = 20
RISK_FRACTION = 0.20

OHLCV_TIMEFRAME = "1m"
OHLCV_LIMIT = 600
DROP_LAST_BAR = True


# Import compute_smm from the strategy file (space in filename)
_strategy_path = Path(__file__).resolve().parent / "Simple Market Metrics.py"
_spec = None
if _strategy_path.exists():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("smm_strategy", _strategy_path)
    smm_strategy = importlib.util.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(smm_strategy)
    compute_smm = smm_strategy.compute_smm  # type: ignore
else:
    raise FileNotFoundError("Simple Market Metrics.py not found")


# Data models ----------------------------------------------------------------
@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    qty: float
    entry_time: pd.Timestamp


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_usdt: float
    pnl_percent: float
    fees: float
    close_reason: str


@dataclass
class PerformanceReport:
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    profit_factor: float
    max_drawdown: float


# Utilities ------------------------------------------------------------------
def _resolve_ccxt():
    import ccxt  # type: ignore

    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "future",
            "defaultMarket": "future",
        },
    })
    if TRADING_MODE.upper() == "TESTNET":
        exchange.set_sandbox_mode(True)
    return exchange


def _normalize_ohlcv(ohlcv: List[List[float]]) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    arr = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(arr["timestamp"].astype(int), unit="ms", utc=True),
            "open": arr["open"],
            "high": arr["high"],
            "low": arr["low"],
            "close": arr["close"],
            "volume": arr["volume"],
        }
    )
    return df


def _round_to_step(amount: float, market: Dict[str, Any]) -> float:
    step_size = None
    precision = market.get("precision", {}).get("amount")
    if "limits" in market and "amount" in market["limits"]:
        step_size = market["limits"]["amount"].get("min")
    if step_size and step_size > 0:
        amount = math.floor(amount / step_size) * step_size
    if precision is not None:
        amount = float(f"{amount:.{int(precision)}f}")
    return amount


# Signal Engine --------------------------------------------------------------
class SignalEngine:
    def __init__(self, exchange, symbol: str):
        self.exchange = exchange
        self.symbol = symbol

    def fetch_ohlcv(self) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=OHLCV_TIMEFRAME, limit=OHLCV_LIMIT)
        df = _normalize_ohlcv(ohlcv)
        if DROP_LAST_BAR:
            df = df.iloc[:-1]
        return df

    def run_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
        params = {
            "use_heikin_ashi": True,
            "require_heikin_ashi": True,
            "enable_signals": True,
            "enable_chop_filter": True,
            "enable_profit_targets": True,
        }
        return compute_smm(df, params)


# Execution Engine -----------------------------------------------------------
class ExecutionEngine:
    def __init__(self, exchange):
        self.exchange = exchange
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []

    def get_free_balance(self) -> float:
        if TRADING_MODE.upper() == "PAPER":
            # Placeholder balance for paper mode; replace with persisted balance as needed.
            return 1000.0
        balance = self.exchange.fetch_balance()
        return float(balance.get("free", {}).get("USDT", 0.0))

    def get_market(self, symbol: str) -> Dict[str, Any]:
        markets = self.exchange.load_markets()
        return markets[symbol]

    def compute_order_size(self, symbol: str, price: float) -> float:
        free_balance = self.get_free_balance()
        notional = free_balance * RISK_FRACTION * LEVERAGE
        raw_qty = notional / price
        market = self.get_market(symbol)
        qty = _round_to_step(raw_qty, market)
        min_qty = market.get("limits", {}).get("amount", {}).get("min")
        if min_qty and qty < min_qty:
            raise ValueError(f"Quantity {qty} below minQty {min_qty}")
        return qty

    def open_position(self, symbol: str, side: str, price: float, timestamp: pd.Timestamp):
        if symbol in self.positions:
            return  # One position per symbol for simplicity
        qty = self.compute_order_size(symbol, price)
        position = Position(symbol=symbol, side=side, entry_price=price, qty=qty, entry_time=timestamp)
        self.positions[symbol] = position
        if TRADING_MODE.upper() == "TESTNET":
            self.exchange.set_leverage(LEVERAGE, symbol)
            self.exchange.create_order(symbol, "market", "buy" if side == "long" else "sell", qty)
        print(f"Opened {side} {symbol}: qty={qty} price={price} mode={TRADING_MODE}")

    def close_position(self, symbol: str, price: float, timestamp: pd.Timestamp, reason: str):
        if symbol not in self.positions:
            return
        pos = self.positions.pop(symbol)
        if TRADING_MODE.upper() == "TESTNET":
            close_side = "sell" if pos.side == "long" else "buy"
            self.exchange.create_order(symbol, "market", close_side, pos.qty)
        pnl = (price - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - price) * pos.qty
        pnl_pct = pnl / (pos.entry_price * pos.qty) * 100 if pos.entry_price > 0 else 0.0
        trade = TradeRecord(
            symbol=symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            qty=pos.qty,
            exit_time=timestamp,
            exit_price=price,
            pnl_usdt=pnl,
            pnl_percent=pnl_pct,
            fees=0.0 if TRADING_MODE.upper() == "PAPER" else math.nan,
            close_reason=reason,
        )
        self.trades.append(trade)
        print(f"Closed {symbol} {pos.side} qty={pos.qty} at {price} due to {reason} | PnL={pnl}")


# Analytics ------------------------------------------------------------------
class Analytics:
    def __init__(self, trades: Iterable[TradeRecord]):
        self.trades = list(trades)

    def equity_curve(self) -> List[float]:
        equity = 0.0
        curve = []
        for t in self.trades:
            equity += t.pnl_usdt
            curve.append(equity)
        return curve

    def max_drawdown(self) -> float:
        curve = self.equity_curve()
        peak = -math.inf
        max_dd = 0.0
        for value in curve:
            peak = max(peak, value)
            drawdown = peak - value
            max_dd = max(max_dd, drawdown)
        return max_dd

    def summary(self) -> PerformanceReport:
        total = len(self.trades)
        wins = [t for t in self.trades if t.pnl_usdt > 0]
        losses = [t for t in self.trades if t.pnl_usdt < 0]
        total_pnl = sum(t.pnl_usdt for t in self.trades)
        avg_win = sum(t.pnl_usdt for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl_usdt for t in losses) / len(losses) if losses else 0.0
        profit_factor = (sum(t.pnl_usdt for t in wins) / abs(sum(t.pnl_usdt for t in losses))) if losses else math.inf
        return PerformanceReport(
            total_trades=total,
            win_rate=len(wins) / total * 100 if total else 0.0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_pnl=total_pnl,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown(),
        )


# Orchestration --------------------------------------------------------------
def evaluate_signals(symbol: str, df_out: pd.DataFrame, executor: ExecutionEngine):
    last = df_out.iloc[-1]
    now = last["datetime"] if isinstance(last["datetime"], pd.Timestamp) else pd.to_datetime(last["datetime"])
    price = float(last.get("real_close_used", last.get("close")))

    # Close logic
    if symbol in executor.positions:
        close_reason = None
        if bool(last.get("profit_target_hit", False)):
            close_reason = "profit_target"
        pos = executor.positions[symbol]
        if CLOSE_ON_OPPOSITE_SIGNAL:
            if pos.side == "long" and bool(last.get("sell_signal", False)):
                close_reason = close_reason or "opposite_signal"
            elif pos.side == "short" and bool(last.get("buy_signal", False)):
                close_reason = close_reason or "opposite_signal"
        if STOP_LOSS_ENABLED:
            if pos.side == "long" and price <= pos.entry_price * (1 - STOP_LOSS_PCT):
                close_reason = close_reason or "stop_loss"
            elif pos.side == "short" and price >= pos.entry_price * (1 + STOP_LOSS_PCT):
                close_reason = close_reason or "stop_loss"
        if close_reason:
            executor.close_position(symbol, price, now, close_reason)
            return

    # Entry logic
    if bool(last.get("buy_signal", False)):
        executor.open_position(symbol, "long", price, now)
    elif bool(last.get("sell_signal", False)):
        executor.open_position(symbol, "short", price, now)


def run_once(symbols: List[str]):
    exchange = _resolve_ccxt()
    executor = ExecutionEngine(exchange)
    for symbol in symbols:
        try:
            signal_engine = SignalEngine(exchange, symbol)
            df = signal_engine.fetch_ohlcv()
            if df.empty:
                continue
            df_out, _ = signal_engine.run_signals(df)
            evaluate_signals(symbol, df_out, executor)
        except Exception as exc:
            print(f"Error on {symbol}: {exc}")
    analytics = Analytics(executor.trades)
    summary = analytics.summary()
    print("Summary:", summary)


if __name__ == "__main__":
    # Example usage
    symbols_to_trade = ["BTC/USDT"]
    run_once(symbols_to_trade)
