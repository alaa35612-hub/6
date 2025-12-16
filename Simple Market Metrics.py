# smm_v41.py
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Scanner settings (CCXT)
# ----------------------------
SCAN_SETTINGS = {
    # Exchange: Binance USDT-M Futures
    "exchange_id": "binanceusdm",

    # Timeframe for OHLCV: "1m", "2m", "5m", "15m", "1h", ...
    "timeframe": "1m",

    # How many historical bars to fetch per symbol
    "ohlcv_limit": 600,

    # Drop last bar to avoid using a potentially-incomplete candle
    "drop_last_bar": True,

    # Signal age filter (in bars): show signals within last N closed bars
    "signal_max_age_bars": 5,

    # Market universe filters
    "only_active": True,
    "quote": "USDT",

    # Optional cap for faster scans (None = scan all)
    "max_symbols": None,

    # Rate-limit / throttling
    "enable_rate_limit": True,
    "request_timeout_ms": 20000,
}


# ----------------------------
# Pine-compat helpers
# ----------------------------
def nz(x: float, repl: float = 0.0) -> float:
    """Pine nz(): replace na with repl."""
    return repl if (x is None or (isinstance(x, float) and np.isnan(x))) else x


def is_na(x: float) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def tv_ema_step(prev_ema: float, x: float, length: int) -> float:
    """TradingView-like EMA step (standard EMA)."""
    alpha = 2.0 / (length + 1.0)
    if is_na(prev_ema):
        return x
    return alpha * x + (1.0 - alpha) * prev_ema


def ema_tv(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    prev = np.nan
    for i in range(len(x)):
        xi = x[i]
        if np.isnan(xi):
            out[i] = prev
            continue
        if np.isnan(prev):
            prev = xi  # First valid value seeds the EMA
        else:
            prev = tv_ema_step(prev, float(xi), length)
        out[i] = prev
    return out


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """TradingView ta.atr() uses Wilder's RMA of TR."""
    n = len(close)
    tr = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

    atr = np.full(n, np.nan, dtype=float)
    if n < length:
        return atr

    # First value is SMA of TR
    seed = float(np.nanmean(tr[:length]))
    atr[length - 1] = seed
    prev = seed
    for i in range(length, n):
        # RMA formula: (prev * (len - 1) + tr) / len
        prev = (prev * (length - 1) + tr[i]) / length
        atr[i] = prev
    return atr


def mfi_tv(hlc3: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
    """
    Standard Money Flow Index (ta.mfi).
    Matches TradingView logic:
    1. Raw Money Flow = HLC3 * Volume
    2. Positive Flow / Negative Flow based on HLC3 change.
    3. Sum over length.
    4. MFI = 100 - (100 / (1 + Ratio))
    """
    n = len(hlc3)
    out = np.full(n, np.nan, dtype=float)

    pos_flow = np.zeros(n, dtype=float)
    neg_flow = np.zeros(n, dtype=float)

    # Calculate flows
    for i in range(1, n):
        raw = hlc3[i] * volume[i]
        if hlc3[i] > hlc3[i - 1]:
            pos_flow[i] = raw
        elif hlc3[i] < hlc3[i - 1]:
            neg_flow[i] = raw

    if n < length + 1:
        return out

    # Initial window sum
    pos_roll = np.sum(pos_flow[1 : length + 1])
    neg_roll = np.sum(neg_flow[1 : length + 1])

    # Calculate first point (at index `length`)
    # Logic note: Pine index `length` means we have looking back `length` bars (1 to length)
    
    for i in range(length, n):
        if i > length:
            # Add new, remove old
            pos_roll += pos_flow[i] - pos_flow[i - length]
            neg_roll += neg_flow[i] - neg_flow[i - length]

        if neg_roll == 0.0 and pos_roll == 0.0:
            out[i] = 50.0  # Flat
        elif neg_roll == 0.0:
            out[i] = 100.0
        else:
            ratio = pos_roll / neg_roll
            out[i] = 100.0 - (100.0 / (1.0 + ratio))
    return out


def to_heikin_ashi(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    TradingView HA definition:
      ha_close = (o+h+l+c)/4
      ha_open  = (prev_ha_open + prev_ha_close)/2, seed: (o0+c0)/2
      ha_high  = max(high, ha_open, ha_close)
      ha_low   = min(low,  ha_open, ha_close)
    Returns (df_ha, real_close_default=original close).
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.full_like(ha_close, np.nan, dtype=float)
    ha_high = np.full_like(ha_close, np.nan, dtype=float)
    ha_low = np.full_like(ha_close, np.nan, dtype=float)

    if len(df) > 0:
        ha_open[0] = (o[0] + c[0]) / 2.0
        ha_high[0] = max(h[0], ha_open[0], ha_close[0])
        ha_low[0] = min(l[0], ha_open[0], ha_close[0])

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close[i])

    out = df.copy()
    out["open"] = ha_open
    out["high"] = ha_high
    out["low"] = ha_low
    out["close"] = ha_close

    real_close_default = df["close"].copy()
    return out, real_close_default


def pivot_high_tv(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    ta.pivothigh
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=float)
    w = left + right + 1
    if n < w:
        return out

    for i in range(w - 1, n):
        pivot_idx = i - right
        start = pivot_idx - left
        end = pivot_idx + right
        window = high[start : end + 1]
        pv = high[pivot_idx]
        mx = np.max(window)
        # Strict check: pv must be the unique max or strictly max
        # Pine typically returns the one with largest index if multiple match, 
        # but here we follow strict pivot definition for safety.
        if pv == mx and np.sum(window == mx) == 1:
            out[i] = pv
    return out


def pivot_low_tv(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan, dtype=float)
    w = left + right + 1
    if n < w:
        return out

    for i in range(w - 1, n):
        pivot_idx = i - right
        start = pivot_idx - left
        end = pivot_idx + right
        window = low[start : end + 1]
        pv = low[pivot_idx]
        mn = np.min(window)
        if pv == mn and np.sum(window == mn) == 1:
            out[i] = pv
    return out


def _round_to_tick(x: float, tick: float) -> float:
    if is_na(x) or tick <= 0:
        return x
    return round(x / tick) * tick


# ----------------------------
# Main compute
# ----------------------------
def compute_smm(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    """
    Python port of Simple Market Metrics v4.1
    """
    p = {
        # HA control
        "use_heikin_ashi": bool(params.get("use_heikin_ashi", False)),
        "data_is_heikin_ashi": bool(params.get("data_is_heikin_ashi", False)),
        "require_heikin_ashi": bool(params.get("require_heikin_ashi", False)),
        # real close
        "real_close_series": params.get("real_close_series", None),
        # core toggles
        "enable_signals": bool(params.get("enable_signals", True)),
        "enable_chop_filter": bool(params.get("enable_chop_filter", True)),
        "enable_profit_targets": bool(params.get("enable_profit_targets", True)),
        "profit_target_value": int(params.get("profit_target_value", 20)),
        "max_profit_target_lines": int(params.get("max_profit_target_lines", 10)),
        "enable_profit_wave": bool(params.get("enable_profit_wave", True)),
        "enable_sr": bool(params.get("enable_sr", True)),
        "sr_extend_until": str(params.get("sr_extend_until", "Close")),  # Close/Touch
        "sr_style": str(params.get("sr_style", "Dotted")),
        "sr_width": int(params.get("sr_width", 2)),
        # constants
        "pivot_len": int(params.get("pivot_len", 20)),
        "sr_max_lines": int(params.get("sr_max_lines", 50)),
        "sr_creation_lookback": int(params.get("sr_creation_lookback", 500)),
        # price precision
        "mintick": float(params.get("mintick", 0.25)),
        "round_to_tick": bool(params.get("round_to_tick", False)),
    }

    required_cols = ["open", "high", "low", "close", "volume"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"df is missing required column: {c}")

    df_in = df.copy()

    if "datetime" in df_in.columns:
        dt = pd.to_datetime(df_in["datetime"])
    else:
        dt = pd.to_datetime(df_in.index)

    # 1. HA Conversion
    ha_enabled = False
    real_close_used = None

    if p["use_heikin_ashi"]:
        df_src, real_close_default = to_heikin_ashi(df_in)
        ha_enabled = True
        if p["real_close_series"] is None:
            real_close_used = real_close_default.to_numpy(dtype=float)
        else:
            rc = p["real_close_series"]
            real_close_used = (rc.to_numpy(dtype=float) if isinstance(rc, pd.Series) else np.asarray(rc, dtype=float))
    else:
        df_src = df_in
        ha_enabled = p["data_is_heikin_ashi"]
        if p["real_close_series"] is None:
            real_close_used = df_src["close"].to_numpy(dtype=float)
        else:
            rc = p["real_close_series"]
            real_close_used = (rc.to_numpy(dtype=float) if isinstance(rc, pd.Series) else np.asarray(rc, dtype=float))

    # Mimic Pine HA gating: if user requires HA but we aren't using it, disable signals
    components_enabled = True
    if p["require_heikin_ashi"] and not ha_enabled:
        p["enable_signals"] = False
        p["enable_profit_targets"] = False
        p["enable_profit_wave"] = False
        p["enable_sr"] = False
        components_enabled = False

    # Source Data (HA if enabled)
    srcOpen = df_src["open"].to_numpy(dtype=float)
    srcHigh = df_src["high"].to_numpy(dtype=float)
    srcLow = df_src["low"].to_numpy(dtype=float)
    srcClose = df_src["close"].to_numpy(dtype=float)
    srcHlc3 = (srcHigh + srcLow + srcClose) / 3.0
    vol = df_src["volume"].to_numpy(dtype=float)

    n = len(df_src)

    # 2. Indicators
    profitWaveEmaFast = ema_tv(srcClose, 8)
    profitWaveEmaMedium = ema_tv(srcClose, 13)
    profitWaveEmaSlow = ema_tv(srcClose, 21)

    atr_8 = atr_wilder(srcHigh, srcLow, srcClose, 8)
    mfi_10 = mfi_tv(srcHlc3, vol, 10)

    # 3. Chop components (Recursive)
    u8t = 1.0
    r7c = np.full(n, np.nan, dtype=float)
    p5y = np.full(n, np.nan, dtype=float)
    d4h = np.full(n, np.nan, dtype=float)
    v5g = np.full(n, np.nan, dtype=float)
    p1b = np.full(n, np.nan, dtype=float)
    b6t = np.full(n, np.nan, dtype=float)
    x9n = np.full(n, np.nan, dtype=float)
    d1v = np.full(n, np.nan, dtype=float)

    for i in range(n):
        prev_close = srcClose[i - 1] if i > 0 else np.nan
        prev_high = srcHigh[i - 1] if i > 0 else np.nan
        prev_low = srcLow[i - 1] if i > 0 else np.nan

        # r7c: max(Range, |High-PrevClose|, |Low-PrevClose|)
        r7c[i] = max(
            srcHigh[i] - srcLow[i],
            abs(srcHigh[i] - nz(prev_close)),
            abs(srcLow[i] - nz(prev_close)),
        )
        
        # p5y / d4h logic
        if i == 0:
            p5y[i] = 0.0
            d4h[i] = 0.0
        else:
            cond_up = (srcHigh[i] - nz(prev_high)) > (nz(prev_low) - srcLow[i])
            p5y[i] = max(srcHigh[i] - nz(prev_high), 0.0) if cond_up else 0.0
            
            cond_down = (nz(prev_low) - srcLow[i]) > (srcHigh[i] - nz(prev_high))
            d4h[i] = max(nz(prev_low) - srcLow[i], 0.0) if cond_down else 0.0

        prev_v5g = nz(v5g[i - 1]) if i > 0 else 0.0
        prev_p1b = nz(p1b[i - 1]) if i > 0 else 0.0
        prev_b6t = nz(b6t[i - 1]) if i > 0 else 0.0

        v5g[i] = prev_v5g - (prev_v5g / u8t) + r7c[i]
        p1b[i] = prev_p1b - (prev_p1b / u8t) + p5y[i]
        b6t[i] = prev_b6t - (prev_b6t / u8t) + d4h[i]

        if v5g[i] == 0:
            x9n[i] = 0.0 # avoid nan in further math
            d1v[i] = 0.0
        else:
            x9n[i] = p1b[i] / v5g[i] * 100.0
            d1v[i] = b6t[i] / v5g[i] * 100.0

    # 4. TrendSwitch logic (Stateful)
    t5c = ((srcHigh + srcLow) / 2.0) - (1.3 * atr_8)
    j6r = ((srcHigh + srcLow) / 2.0) + (1.3 * atr_8)

    r8d = np.full(n, np.nan, dtype=float)
    t9f = np.full(n, np.nan, dtype=float)
    x2m = np.full(n, np.nan, dtype=float)  # 1.0 (bull) or -1.0 (bear)

    for i in range(n):
        if i == 0:
            r8d[i] = t5c[i]
            t9f[i] = j6r[i]
            x2m[i] = 1.0
            continue

        r8d_prev = r8d[i - 1]
        t9f_prev = t9f[i - 1]
        prev_close = srcClose[i - 1]

        # r8d update
        if (not is_na(r8d_prev)) and (prev_close > r8d_prev):
            r8d[i] = max(t5c[i], r8d_prev)
        else:
            r8d[i] = t5c[i]

        # t9f update
        if (not is_na(t9f_prev)) and (prev_close < t9f_prev):
            t9f[i] = min(j6r[i], t9f_prev)
        else:
            t9f[i] = j6r[i]

        # x2m state switch
        if srcClose[i] > t9f_prev:
            x2m[i] = 1.0
        elif srcClose[i] < r8d_prev:
            x2m[i] = -1.0
        else:
            x2m[i] = nz(x2m[i - 1], 1.0)

    trendSwitch_state = np.array(["none"] * n, dtype=object)
    for i in range(n):
        if x2m[i] == 1.0:
            trendSwitch_state[i] = "bull"
        elif x2m[i] == -1.0:
            trendSwitch_state[i] = "bear"

    events: Dict[str, List[Dict[str, Any]]] = {
        "signals": [],
        "profit_targets": [],
        "support_lines": [],
        "resistance_lines": [],
    }

    # 5. Signals & Event Logic
    
    # S/R Pivots
    left = right = p["pivot_len"]
    ph1 = pivot_high_tv(srcHigh, left, right)
    lp8 = pivot_low_tv(srcLow, left, right)

    active_res: List[int] = []
    active_sup: List[int] = []
    displayed_target_indices: List[int] = []

    y7c = "none" # State: "none", "buy", "sell"
    prev_bg_state = "none"
    u0k: Optional[int] = None # bar index of signal
    x0l: float = np.nan       # profit target price

    profit_target_hit = np.zeros(n, dtype=bool)
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    signal_state = np.array(["none"] * n, dtype=object)
    profit_target_price = np.full(n, np.nan, dtype=float)

    for i in range(n):
        bg_state = trendSwitch_state[i]
        if i == 0:
            prev_bg_state = bg_state

        # Reset signal state if trend changes (x5y != x5y[1] in Pine)
        if i > 0 and bg_state != prev_bg_state:
            y7c = "none"
        prev_bg_state = bg_state

        # Conditions
        o1z = (x2m[i] == 1.0) # Trend Bull
        k8s = (x2m[i] == -1.0) # Trend Bear
        y2g = o1z
        w1r = k8s

        rc = float(real_close_used[i]) # Real close for crossovers

        # HA Candle Logic
        # a4h (Bull Candle): Close > Open AND Open == Low AND RealClose > FastEMA AND RealClose > SlowEMA
        a4h = (srcClose[i] > srcOpen[i]) and (srcOpen[i] == srcLow[i]) and (rc > profitWaveEmaFast[i]) and (rc > profitWaveEmaSlow[i])
        
        # p4r (Bear Candle): Close < Open AND Open == High AND RealClose < FastEMA AND RealClose < SlowEMA
        p4r = (srcClose[i] < srcOpen[i]) and (srcOpen[i] == srcHigh[i]) and (rc < profitWaveEmaFast[i]) and (rc < profitWaveEmaSlow[i])

        # Chop Filter Logic
        if p["enable_chop_filter"]:
            # e1p: x9n > d1v and x9n >= 45
            e1p = (math.floor(nz(x9n[i])) > math.floor(nz(d1v[i]))) and (math.floor(nz(x9n[i])) >= 45)
            # t2y: d1v > x9n and d1v >= 45
            t2y = (math.floor(nz(d1v[i])) > math.floor(nz(x9n[i]))) and (math.floor(nz(d1v[i])) >= 45)
            # MFI checks
            w2f = (not is_na(mfi_10[i])) and (mfi_10[i] > 52)
            b7m = (not is_na(mfi_10[i])) and (mfi_10[i] < 48)
        else:
            e1p = True
            t2y = True
            w2f = True
            b7m = True

        # Signal Triggers
        # o4k (Buy): TrendBull AND SignalBull AND CandleBull AND NotAlreadyBuy AND MFI_Ok AND Chop_Ok
        o4k = o1z and y2g and a4h and (y7c != "buy") and w2f and e1p
        # o5t (Sell): TrendBear AND SignalBear AND CandleBear AND NotAlreadySell AND MFI_Ok AND Chop_Ok
        o5t = k8s and w1r and p4r and (y7c != "sell") and b7m and t2y

        # Process Buy
        if o4k and p["enable_signals"]:
            y7c = "buy"
            buy_signal[i] = True
            u0k = i
            events["signals"].append({
                "datetime": dt.iloc[i], "type": "buy", "price": rc, "bar": i
            })

        # Process Sell
        if o5t and p["enable_signals"]:
            y7c = "sell"
            sell_signal[i] = True
            u0k = i
            events["signals"].append({
                "datetime": dt.iloc[i], "type": "sell", "price": rc, "bar": i
            })

        # Profit Target Logic Creation
        if (o4k or o5t) and p["enable_signals"] and p["enable_profit_targets"]:
            # Limit lines
            if len(displayed_target_indices) >= p["max_profit_target_lines"]:
                oldest_idx = displayed_target_indices.pop(0)
                events["profit_targets"][oldest_idx]["removed_bar"] = i
                events["profit_targets"][oldest_idx]["removed_datetime"] = dt.iloc[i]

            direction = "buy" if o4k else "sell"
            # Calculate target
            shift = (p["profit_target_value"] * p["mintick"])
            if o4k:
                x0l = rc + shift
            else:
                x0l = rc - shift

            if p["round_to_tick"]:
                x0l = _round_to_tick(x0l, p["mintick"])

            events["profit_targets"].append({
                "datetime": dt.iloc[i],
                "direction": direction,
                "target_price": float(x0l),
                "created_bar": i,
                "hit_bar": None,
                "hit_datetime": None,
                "removed_bar": None,
                "removed_datetime": None,
            })
            displayed_target_indices.append(len(events["profit_targets"]) - 1)

        # Reset Logic (Signal State Reset)
        # If Buy state, but RealClose < SlowEMA and HA_Close < SlowEMA -> Reset
        if y7c == "buy" and (rc < profitWaveEmaSlow[i]) and (srcClose[i] < profitWaveEmaSlow[i]):
            y7c = "none"
            u0k = None

        # If Sell state, but RealClose > SlowEMA and HA_Close > SlowEMA -> Reset
        if y7c == "sell" and (rc > profitWaveEmaSlow[i]) and (srcClose[i] > profitWaveEmaSlow[i]):
            y7c = "none"
            u0k = None

        # Check Profit Target Hit
        o7g = False
        if u0k is not None and i > u0k and (not is_na(x0l)):
            if y7c == "buy":
                o7g = srcHigh[i] >= x0l
            elif y7c == "sell":
                o7g = srcLow[i] <= x0l

        if o7g:
            profit_target_hit[i] = True
            # Mark the line as hit
            if len(events["profit_targets"]) > 0:
                # Find the most recent active target matching direction
                for idx in range(len(events["profit_targets"]) - 1, -1, -1):
                    ev = events["profit_targets"][idx]
                    if ev["hit_bar"] is None and ev["direction"] == y7c and ev["removed_bar"] is None:
                        ev["hit_bar"] = i
                        ev["hit_datetime"] = dt.iloc[i]
                        break
            x0l = np.nan
            u0k = None

        signal_state[i] = y7c
        profit_target_price[i] = x0l

        # Support / Resistance Logic
        if p["enable_sr"]:
            # Resistance
            if not np.isnan(ph1[i]):
                pivot_bar = i - right
                if pivot_bar >= i - p["sr_creation_lookback"]:
                    events["resistance_lines"].append({
                        "datetime": dt.iloc[i],
                        "level": float(srcHigh[pivot_bar]),
                        "start_bar": pivot_bar,
                        "end_bar": i,
                        "stop_reason": None,
                        "active": True
                    })
                    active_res.append(len(events["resistance_lines"]) - 1)

            # Support
            if not np.isnan(lp8[i]):
                pivot_bar = i - right
                if pivot_bar >= i - p["sr_creation_lookback"]:
                    events["support_lines"].append({
                        "datetime": dt.iloc[i],
                        "level": float(srcLow[pivot_bar]),
                        "start_bar": pivot_bar,
                        "end_bar": i,
                        "stop_reason": None,
                        "active": True
                    })
                    active_sup.append(len(events["support_lines"]) - 1)

            # Check for breaks
            z8v = srcClose[i] if p["sr_extend_until"] == "Close" else srcHigh[i]
            for idx in list(active_res):
                ln = events["resistance_lines"][idx]
                if not ln["active"]: continue
                ln["end_bar"] = i
                if z8v >= ln["level"]:
                    ln["stop_reason"] = "break"
                    ln["active"] = False
                    active_res.remove(idx)

            t0w = srcClose[i] if p["sr_extend_until"] == "Close" else srcLow[i]
            for idx in list(active_sup):
                ln = events["support_lines"][idx]
                if not ln["active"]: continue
                ln["end_bar"] = i
                if t0w <= ln["level"]:
                    ln["stop_reason"] = "break"
                    ln["active"] = False
                    active_sup.remove(idx)

    # Output DataFrame
    df_out = pd.DataFrame(
        {
            "datetime": dt.values,
            "profitWaveEmaFast": profitWaveEmaFast,
            "profitWaveEmaMedium": profitWaveEmaMedium,
            "profitWaveEmaSlow": profitWaveEmaSlow,
            "mfi_10": mfi_10,
            "atr_8": atr_8,
            "trendSwitch_state": trendSwitch_state,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "signal_state": signal_state,
            "profit_target_price": profit_target_price,
            "profit_target_hit": profit_target_hit,
            "real_close_used": real_close_used,
            "ha_enabled": np.array([ha_enabled] * n, dtype=bool),
        }
    )
    df_out.index = df_in.index
    return df_out, events


# ----------------------------
# CCXT Scanner (Binance USDT-M)
# ----------------------------
def _safe_import_ccxt():
    try:
        import ccxt  # type: ignore
        return ccxt
    except ImportError as e:
        return None

def _normalize_ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    arr = np.asarray(ohlcv, dtype=float)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True),
            "open": arr[:, 1], "high": arr[:, 2], "low": arr[:, 3], "close": arr[:, 4], "volume": arr[:, 5],
        }
    )
    return df

def scan_binance_usdtm_smm(smm_params: Dict[str, Any], scan_params: Optional[Dict[str, Any]] = None):
    cfg = dict(SCAN_SETTINGS)
    if scan_params:
        cfg.update(scan_params)

    ccxt = _safe_import_ccxt()
    if ccxt is None:
        print("Warning: CCXT not installed. Skipping live scan.")
        return pd.DataFrame(), {}

    try:
        exchange_cls = getattr(ccxt, cfg["exchange_id"])
        ex = exchange_cls({"enableRateLimit": bool(cfg["enable_rate_limit"]), "timeout": int(cfg["request_timeout_ms"])})
        markets = ex.load_markets()
    except Exception as e:
        print(f"Exchange connection failed: {e}")
        return pd.DataFrame(), {}

    symbols = [m["symbol"] for _, m in markets.items() if m.get("swap") and m.get("quote") == cfg["quote"] and m.get("active")]
    if cfg.get("max_symbols") is not None:
        symbols = symbols[: int(cfg["max_symbols"])]

    rows, scan_events = [], {}
    print(f"Scanning {len(symbols)} symbols...")

    for sym in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(sym, timeframe=cfg["timeframe"], limit=int(cfg["ohlcv_limit"]))
            df = _normalize_ohlcv_to_df(ohlcv)
            if df.empty or len(df) < 50: continue
            if bool(cfg["drop_last_bar"]): df = df.iloc[:-1]

            df_out, events = compute_smm(df, smm_params)
            
            # Check last N bars
            tail = df_out.iloc[-int(cfg["signal_max_age_bars"]):]
            if tail["buy_signal"].any() or tail["sell_signal"].any():
                last_row = df_out.iloc[-1]
                sig_type = "buy" if tail["buy_signal"].any() else "sell" # Simplified
                if tail["buy_signal"].iloc[-1]: sig_type = "buy"
                elif tail["sell_signal"].iloc[-1]: sig_type = "sell"

                rows.append({
                    "symbol": sym, "timeframe": cfg["timeframe"], "signal": sig_type,
                    "price": last_row["real_close_used"], "trend": last_row["trendSwitch_state"]
                })
                scan_events[sym] = events
        except Exception:
            continue

    return pd.DataFrame(rows), scan_events

# ----------------------------
# Automatic Demo Logic
# ----------------------------
def generate_mock_data(length=500):
    """Generates synthetic price data to demo the scanner without CCXT."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq="1min")
    x = np.linspace(0, 4*np.pi, length)
    price = 100 + 10*np.sin(x) + np.random.normal(0, 0.5, length)
    
    # Create trending behavior
    for i in range(1, length):
        price[i] = price[i-1] + np.random.normal(0, 0.2)
        if i > length // 2: price[i] += 0.05 # slight uptrend

    df = pd.DataFrame({
        "datetime": dates,
        "open": price, "high": price + 0.5, "low": price - 0.5, "close": price + np.random.normal(0, 0.1, length),
        "volume": np.random.randint(100, 1000, length)
    })
    # Ensure High/Low consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df

def run_auto_demo():
    print("--- SMM v4.1 Python Auto-Runner ---")
    
    # 1. Test basic mock scan
    print("Generating synthetic data for logic verification...")
    df = generate_mock_data(600)
    
    params = {
        "use_heikin_ashi": True,
        "require_heikin_ashi": True,
        "mintick": 0.01,
        "enable_signals": True,
        "enable_chop_filter": True, # Test the complex chop logic
        "enable_profit_targets": True,
    }
    
    start_t = time.time()
    df_out, events = compute_smm(df, params)
    dt = time.time() - start_t
    
    print(f"Computation finished in {dt:.4f}s")
    print(f"Data Points: {len(df_out)}")
    
    sig_buys = df_out["buy_signal"].sum()
    sig_sells = df_out["sell_signal"].sum()
    targets = len(events["profit_targets"])
    
    print(f"\n[Results on Synthetic Data]")
    print(f"Buy Signals: {sig_buys}")
    print(f"Sell Signals: {sig_sells}")
    print(f"Profit Targets Generated: {targets}")
    
    if sig_buys > 0 or sig_sells > 0:
        print("\nLast 5 Signals:")
        sigs = pd.concat([df_out[df_out["buy_signal"]==True], df_out[df_out["sell_signal"]==True]]).sort_index().tail(5)
        print(sigs[["datetime", "trendSwitch_state", "buy_signal", "sell_signal", "real_close_used"]])
    else:
        print("\nNo signals in synthetic set (this is possible depending on random seed). Logic executed successfully.")

    # 2. Try Live Scan if CCXT is there
    print("\nAttempting Live Binance Scan (requires 'ccxt')...")
    try:
        res, _ = scan_binance_usdtm_smm(params, {"max_symbols": 50})
        if not res.empty:
            print(res.to_string())
        else:
            print("No signals found in live scan or CCXT not available/configured.")
    except Exception as e:
        print(f"Scan skipped: {e}")

if __name__ == "__main__":
    run_auto_demo()
