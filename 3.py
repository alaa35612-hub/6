"""
Python implementation of FT Concept v2.2 indicator logic.
Converted from Pine Script v5 to Python for CCXT-based scanning on Binance USDT-M futures.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import ccxt
import pandas as pd
import numpy as np

# =============================== CONFIG ===============================
# Exchange / scanner configuration
EXCHANGE_ID = "binanceusdm"
USE_ALL_SYMBOLS = True
SYMBOLS_WHITELIST = ["BTC/USDT", "ETH/USDT"]
SCAN_TIMEFRAMES = ["5m", "15m", "1h"]
# HTF FVG timeframes are configured separately from the scan timeframes
HTF_FVG_TIMEFRAMES = ["5m", "15m", "1h"]
MAX_HISTORY_BARS = 500
ALERT_LOOKBACK_BARS = 3
ALERT_LOOKBACK_MINUTES = 60
SCAN_INTERVAL_SECONDS = 0  # 0 to run single pass

# Pine inputs translated to Python defaults
showLine = True
biasUpAlert = True
biasDownAlert = True
showLabelBias = True

# HTF FVG settings
settings_ltf_hide = True
settings_border_show = True
settings_mitigated_show = True
settings_mitigated_type = "Wick filled"
settings_CE_show = True
settings_CE_color = (0, 0, 0, 60)
settings_CE_style = "····"
settings_label_show = True
settings_label_color = (0, 0, 0, 10)
settings_label_bgcolor = (255, 255, 255, 100)
settings_label_size = "small"
settings_padding = 4
settings_buffer = 6

HTF_SETTINGS_DEFAULTS = [
    {"show": True, "htf": tf, "color_bull": (0, 255, 0, 90), "color_bear": (0, 0, 255, 90), "max_count": 20}
    for tf in HTF_FVG_TIMEFRAMES
]

# LuxAlgo Imbalance settings
show_fvg = True
bull_fvg_css = (33, 87, 243, 255)
bear_fvg_css = (255, 17, 0, 255)
fvg_usewidth = False
fvg_gapwidth = 0.0
fvg_method = "Points"  # Points, %, ATR
fvg_extend = 0

# SB CISD settings
Tracking_Method = "Classic"
length = 10
Minimum_Sequence_Length = 0
Maximum_Sequence_Length = 100
textSize = "tiny"
lineWidth = 3
cBull_sb = (8, 153, 129)
cBear_sb = (242, 54, 69)
cSweepH = (120, 123, 134, 132)
cSweepL = (120, 123, 134, 132)

# iFVG settings
alert_tf = "All"

# Alert toggles
alert_fvg_bull = True
alert_fvg_bear = True
alert_cisd_bull = True
alert_cisd_bear = True
alert_ifvg_bull = True
alert_ifvg_bear = True

# =============================== DATA CLASSES ===============================

@dataclass
class Settings:
    CE_show: bool = True
    CE_style: str = settings_CE_style
    CE_color: tuple = settings_CE_color
    Border_show: bool = settings_border_show
    mitigated_show: bool = settings_mitigated_show
    mitigated_type: str = settings_mitigated_type
    mitigated_color_bull: tuple = (128, 128, 128, 95)
    mitigated_color_bear: tuple = (128, 128, 128, 95)
    ltf_hide: bool = settings_ltf_hide
    label_show: bool = settings_label_show
    label_color: tuple = settings_label_color
    label_bgcolor: tuple = settings_label_bgcolor
    label_size: str = settings_label_size
    padding: int = settings_padding
    buffer: int = settings_buffer


@dataclass
class ImbalanceSettings:
    show: bool = True
    htf: str = "5"
    color_bull: tuple = (0, 255, 0, 90)
    color_bear: tuple = (0, 0, 255, 90)
    max_count: int = 20


@dataclass
class Imbalance:
    open_time: int = 0
    close_time: int = 0
    open: float = 0.0
    middle: float = 0.0
    close: float = 0.0
    mitigated: bool = False
    mitigated_time: Optional[int] = None


@dataclass
class ImbalanceStructure:
    imbalance: List[Imbalance] = field(default_factory=list)
    settings: ImbalanceSettings = field(default_factory=ImbalanceSettings)

    def add_imbalance(self, o: float, c: float, o_time: int):
        imb = Imbalance(open_time=o_time, open=o, middle=(o + c) / 2, close=c)
        self.imbalance.insert(0, imb)
        if len(self.imbalance) > 100:
            self.imbalance.pop()

    def check_mitigated(self, o: float, h: float, l: float, c: float, now_time: int, settings: Settings):
        removal = []
        for idx, imb in enumerate(self.imbalance):
            if imb.mitigated:
                continue
            mt = settings.mitigated_type
            if mt == "None":
                imb.mitigated = False
            elif mt == "Wick filled":
                imb.mitigated = (imb.open <= imb.close and l <= imb.open) or (imb.open > imb.close and h >= imb.open)
            elif mt == "Body filled":
                body_low, body_high = min(o, c), max(o, c)
                imb.mitigated = (imb.open < imb.close and body_low <= imb.open) or (imb.open > imb.close and body_high >= imb.open)
            elif mt == "Wick filled half":
                imb.mitigated = (imb.open <= imb.close and l <= imb.middle) or (imb.open > imb.close and h >= imb.middle)
            elif mt == "Body filled half":
                body_low, body_high = min(o, c), max(o, c)
                imb.mitigated = (imb.open <= imb.close and body_low <= imb.middle) or (imb.open > imb.close and body_high >= imb.middle)
            if imb.mitigated:
                imb.mitigated_time = now_time
                if not settings.mitigated_show:
                    removal.append(idx)
        for i in reversed(removal):
            self.imbalance.pop(i)

    def find_imbalance(self, o, h, l, c, t, o1, h1, l1, c1, t1, o2, h2, l2, c2, t2):
        if self.settings.show and (h < l2 or l > h2):
            o_val = l2 if h < l2 else h2
            c_val = h if h < l2 else l
            if not self.imbalance or self.imbalance[0].open_time < t2:
                self.add_imbalance(o_val, c_val, t2)

    def process(self, series, settings: Settings):
        visible = 0
        if self.settings.show:
            if not settings.ltf_hide:
                visible = 1
            # series contain tuples for o,h,l,c,t etc
            for row in series:
                (o, h, l, c, t), (o1, h1, l1, c1, t1), (o2, h2, l2, c2, t2) = row
                self.find_imbalance(o, h, l, c, t, o1, h1, l1, c1, t1, o2, h2, l2, c2, t2)
                self.check_mitigated(o, h, l, c, t, settings)
                visible = 1
        return visible


@dataclass
class CISDBin:
    active: bool
    x1: int
    y1: float
    broken: bool
    direction: int


@dataclass
class CISDState:
    trend: int = 0
    bull_bin: Optional[CISDBin] = None
    bear_bin: Optional[CISDBin] = None
    swingsH: List[Tuple[int, float]] = field(default_factory=list)
    swingsL: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class IFVGState:
    active: bool = False
    start_bar: Optional[int] = None
    detected_bar: Optional[int] = None
    c1_high: Optional[float] = None
    c1_low: Optional[float] = None
    c3_high: Optional[float] = None
    c3_low: Optional[float] = None
    direction: int = 0
    validation_end: Optional[int] = None


# =============================== HELPERS ===============================

def init_exchange():
    exchange = getattr(ccxt, EXCHANGE_ID)()
    exchange.load_markets()
    return exchange


def _normalize_symbol(exchange: ccxt.Exchange, symbol: str) -> Optional[str]:
    """Return the CCXT market symbol for a user-provided base/quote pair.

    Binance USDT-M futures symbols are typically formatted like "BTC/USDT:USDT".
    This helper resolves simple pairs (e.g., "BTC/USDT") to the futures market
    if available so users do not have to include the contract suffix manually.
    """

    markets = exchange.markets
    if symbol in markets:
        return symbol
    # Try to find a contract with the same base/quote
    for m in markets.values():
        if m.get("base") and m.get("quote") and f"{m['base']}/{m['quote']}" == symbol:
            return m["symbol"]
    return None


def get_symbols(exchange: ccxt.Exchange) -> List[str]:
    """Return the set of Binance USDT-M swap symbols to scan."""

    markets = exchange.markets if exchange.markets else exchange.load_markets()

    if USE_ALL_SYMBOLS:
        symbols = [
            s
            for s, m in markets.items()
            if m.get("quote") == "USDT"
            and m.get("swap")
            and m.get("contract")
            and (m.get("linear") or m.get("future"))
        ]
        # Binance lists both with and without ":USDT" suffix; deduplicate to avoid double scans
        return sorted(list(dict.fromkeys(symbols)))

    resolved = []
    for sym in SYMBOLS_WHITELIST:
        resolved_sym = _normalize_symbol(exchange, sym)
        if resolved_sym:
            resolved.append(resolved_sym)
        else:
            logging.warning("Symbol %s not found in exchange markets; skipping", sym)
    return resolved


def fetch_ohlcv_dataframe(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


def fetch_htf_context(exchange: ccxt.Exchange, symbol: str) -> Dict[str, pd.DataFrame]:
    """Fetch HTF FVG data independently from the scan timeframe selection."""

    context = {}
    for tf in HTF_FVG_TIMEFRAMES:
        try:
            context[tf] = fetch_ohlcv_dataframe(exchange, symbol, tf, MAX_HISTORY_BARS)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to fetch HTF data for %s %s: %s", symbol, tf, exc)
    return context


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def pivot_high(values: pd.Series, left: int, right: int = 1):
    res = []
    for i in range(len(values)):
        if i < left or i + right >= len(values):
            res.append(np.nan)
            continue
        window = values[i - left:i + right + 1]
        if values[i] == window.max():
            res.append(values[i])
        else:
            res.append(np.nan)
    return pd.Series(res, index=values.index)


def pivot_low(values: pd.Series, left: int, right: int = 1):
    res = []
    for i in range(len(values)):
        if i < left or i + right >= len(values):
            res.append(np.nan)
            continue
        window = values[i - left:i + right + 1]
        if values[i] == window.min():
            res.append(values[i])
        else:
            res.append(np.nan)
    return pd.Series(res, index=values.index)


# =============================== CORE LOGIC ===============================

def process_bias_system(df: pd.DataFrame) -> Tuple[List[bool], List[bool]]:
    __arrLow: List[float] = []
    __arrHigh: List[float] = []
    _continueFlag = False
    _bias = None
    bullBias = False
    bearBias = False
    prev_bias = None
    countBull = 0
    countBear = 0
    bull_signals = []
    bear_signals = []

    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values

    for i in range(len(df)):
        isUpCandle = closes[i] > opens[i]
        isDownCandle = closes[i] < opens[i]
        prev_close = closes[i - 1] if i >= 1 else np.nan
        prev_open = opens[i - 1] if i >= 1 else np.nan
        _pair_updown = (prev_close > prev_open and closes[i] < opens[i]) if i >= 1 else False
        _pair_downup = (prev_close < prev_open and closes[i] > opens[i]) if i >= 1 else False

        if i == 0:
            bull_signals.append(False)
            bear_signals.append(False)
            continue

        if not _continueFlag:
            if _pair_downup:
                _barLower = lows[i] if lows[i] < lows[i - 1] else lows[i - 1]
                if __arrLow:
                    __arrLow[0] = _barLower
                else:
                    __arrLow.append(_barLower)
        if len(__arrLow) > 0:
            _valLow = __arrLow[0]
            if lows[i] < _valLow and closes[i] - _valLow > 0 and closes[i] != lows[i] and len(__arrHigh) == 0:
                _barLower = lows[i]
                if __arrLow:
                    __arrLow[0] = _barLower
                else:
                    __arrLow.append(_barLower)
            _latestLowVal = __arrLow[0]
            if _latestLowVal - closes[i] > 0 and isDownCandle:
                lookup = max(highs[i - 3:i + 1]) if i >= 3 else max(highs[: i + 1])
                if __arrHigh:
                    __arrHigh[0] = lookup
                else:
                    __arrHigh.append(lookup)
                _continueFlag = True
                _bias = -1
        if _continueFlag:
            if _pair_updown:
                _barHigher = highs[i] if highs[i] > highs[i - 1] else highs[i - 1]
                if __arrHigh:
                    __arrHigh[0] = _barHigher
                else:
                    __arrHigh.append(_barHigher)
        if len(__arrHigh) > 0:
            _valHigh = __arrHigh[0]
            if highs[i] > _valHigh and closes[i] < _valHigh and highs[i] != opens[i]:
                _barHigher = highs[i]
                if __arrHigh:
                    __arrHigh[0] = _valHigh
                else:
                    __arrHigh.append(_valHigh)
            _latestHighVal = __arrHigh[0]
            if closes[i] - _latestHighVal > 0 and isUpCandle:
                lookup = min(lows[i - 3:i + 1]) if i >= 3 else min(lows[: i + 1])
                if __arrLow:
                    __arrLow[0] = lookup
                else:
                    __arrLow.append(lookup)
                __arrHigh.clear()
                _continueFlag = False
                _bias = 1

        __isFvgUp = (i >= 2) and (lows[i] > highs[i - 2]) and (closes[i - 1] > highs[i - 2])
        __isFvgDown = (i >= 2) and (highs[i] < lows[i - 2]) and (closes[i - 1] < lows[i - 2])

        if prev_bias is not None and prev_bias != _bias:
            countBull = 0
            countBear = 0

        if __isFvgUp and _bias == 1:
            if countBull == 0:
                bullBias = True
                countBull += 1
        if __isFvgDown and _bias == -1:
            if countBear == 0:
                bearBias = True
                countBear += 1

        bull_signals.append(bullBias and biasUpAlert)
        bear_signals.append(bearBias and biasDownAlert)
        if bullBias and biasUpAlert:
            bullBias = False
        if bearBias and biasDownAlert:
            bearBias = False

        prev_bias = _bias

    return bull_signals, bear_signals


def imbalance_detection_lux(show, use_width, method, width, top, btm, condition, atr=None, count=0):
    if not atr:
        atr = 0
    dist = abs(top - btm)
    is_width = True
    if use_width:
        if method == "Points":
            is_width = dist > width
        elif method == "%":
            is_width = (dist / btm * 100) > width if btm else False
        elif method == "ATR":
            is_width = dist > atr * width
    is_true = show and condition and is_width
    count += 1 if is_true else 0
    return is_true, count


def bull_filled_lux(condition, btm, lows):
    btms = []
    count = 0
    for cond, lv in zip(condition, btm):
        if cond:
            btms.insert(0, lv)
        for value in list(btms):
            if lv < value:
                btms.remove(value)
                count += 1
    return count


def bear_filled_lux(condition, top, highs):
    tops = []
    count = 0
    for cond, hv in zip(condition, top):
        if cond:
            tops.insert(0, hv)
        for value in list(tops):
            if hv > value:
                tops.remove(value)
                count += 1
    return count


def detect_cisd(df: pd.DataFrame) -> Tuple[List[bool], List[bool]]:
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values
    bullish = [False] * len(df)
    bearish = [False] * len(df)

    ph = pivot_high(df["high"], length)
    pl = pivot_low(df["low"], length)

    oBull = None
    oBear = None
    trend = 0

    for i in range(1, len(df)):
        bull_sb = closes[i] > opens[i]
        bear_sb = closes[i] < opens[i]
        bull_prev = closes[i - 1] > opens[i - 1]
        bear_prev = closes[i - 1] < opens[i - 1]

        if not bull_prev and bear_prev and bull_sb:
            oBull = {"x1": i, "y": opens[i], "broken": False, "active": True}
        if not bear_prev and bull_prev and bear_sb:
            oBear = {"x1": i, "y": opens[i], "broken": False, "active": True}

        if oBull and oBull["active"]:
            if i - oBull["x1"] <= Maximum_Sequence_Length:
                if closes[i] < oBull["y"]:
                    if i - oBull["x1"] >= Minimum_Sequence_Length:
                        trend = -1
                        bearish[i] = True
                        oBull["active"] = False
                    else:
                        oBull = None
            else:
                oBull = None

        if oBear and oBear["active"]:
            if i - oBear["x1"] <= Maximum_Sequence_Length:
                if closes[i] > oBear["y"]:
                    if i - oBear["x1"] >= Minimum_Sequence_Length:
                        trend = 1
                        bullish[i] = True
                        oBear["active"] = False
                    else:
                        oBear = None
            else:
                oBear = None

    return bullish, bearish


def detect_ifvg(df: pd.DataFrame) -> Tuple[List[bool], List[bool]]:
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    bullish = [False] * len(df)
    bearish = [False] * len(df)
    state = IFVGState()
    for i in range(2, len(df)):
        bullishFVG = highs[i - 2] < lows[i]
        bearishFVG = lows[i - 2] > highs[i]
        if bullishFVG:
            state.active = True
            state.direction = 1
            state.start_bar = i - 2
            state.detected_bar = i
            state.c1_high = highs[i - 2]
            state.c1_low = lows[i - 2]
            state.c3_high = highs[i]
            state.c3_low = lows[i]
            state.validation_end = i + 4
        if bearishFVG:
            state.active = True
            state.direction = -1
            state.start_bar = i - 2
            state.detected_bar = i
            state.c1_high = highs[i - 2]
            state.c1_low = lows[i - 2]
            state.c3_high = highs[i]
            state.c3_low = lows[i]
            state.validation_end = i + 4

        validated = False
        if state.active and i <= (state.validation_end or i):
            if state.direction == 1 and closes[i] < (state.c1_high or closes[i]):
                validated = True
            if state.direction == -1 and closes[i] > (state.c1_low or closes[i]):
                validated = True
            if validated:
                if state.direction == 1:
                    bearish[i] = True
                else:
                    bullish[i] = True
                state.active = False
    return bullish, bearish


# =============================== SIGNAL ENGINE ===============================

def process_htf_fvgs(htf_context: Dict[str, pd.DataFrame]):
    """Process HTF FVG structures independently of the scan timeframes."""

    settings = Settings()
    structures = {cfg["htf"]: ImbalanceStructure(settings=ImbalanceSettings(**cfg)) for cfg in HTF_SETTINGS_DEFAULTS}
    for tf, structure in structures.items():
        if tf not in htf_context:
            continue
        df_htf = htf_context[tf]
        series = []
        for i in range(2, len(df_htf)):
            row = (
                (df_htf.loc[df_htf.index[i], "open"], df_htf.loc[df_htf.index[i], "high"], df_htf.loc[df_htf.index[i], "low"], df_htf.loc[df_htf.index[i], "close"], df_htf.loc[df_htf.index[i], "timestamp"]),
                (df_htf.loc[df_htf.index[i - 1], "open"], df_htf.loc[df_htf.index[i - 1], "high"], df_htf.loc[df_htf.index[i - 1], "low"], df_htf.loc[df_htf.index[i - 1], "close"], df_htf.loc[df_htf.index[i - 1], "timestamp"]),
                (df_htf.loc[df_htf.index[i - 2], "open"], df_htf.loc[df_htf.index[i - 2], "high"], df_htf.loc[df_htf.index[i - 2], "low"], df_htf.loc[df_htf.index[i - 2], "close"], df_htf.loc[df_htf.index[i - 2], "timestamp"]),
            )
            series.append(row)
        structure.process(series, settings)
    return structures


def process_symbol(symbol: str, timeframe: str, df: pd.DataFrame, htf_context: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
    signals = []
    if htf_context is not None:
        process_htf_fvgs(htf_context)
    bull_bias, bear_bias = process_bias_system(df)

    atr = compute_atr(df, 14)
    bull_fvg_series = []
    bear_fvg_series = []
    for i in range(len(df)):
        if i < 2:
            bull_fvg_series.append(False)
            bear_fvg_series.append(False)
            continue
        bull_cond, _ = imbalance_detection_lux(show_fvg, fvg_usewidth, fvg_method, fvg_gapwidth, df.loc[df.index[i], "low"], df
                                               .loc[df.index[i - 2], "high"], df.loc[df.index[i], "low"] > df.loc[df.index[i - 2], "high"] and df.loc[df.index[i - 1], "close"] > df.loc[df.index[i - 2], "high"], atr.iloc[i], 0)
        bear_cond, _ = imbalance_detection_lux(show_fvg, fvg_usewidth, fvg_method, fvg_gapwidth, df.loc[df.index[i - 2], "low"], df.loc[df.index[i], "high"], df.loc[df.index[i], "high"] < df.loc[df.index[i - 2], "low"] and df.loc[df.index[i - 1], "close"] < df.loc[df.index[i - 2], "low"], atr.iloc[i], 0)
        bull_fvg_series.append(bull_cond)
        bear_fvg_series.append(bear_cond)

    cisd_bull, cisd_bear = detect_cisd(df)
    ifvg_bull, ifvg_bear = detect_ifvg(df)

    latest_index = len(df) - 1
    cutoff_index = max(0, latest_index - ALERT_LOOKBACK_BARS)
    cutoff_time = df.loc[df.index[latest_index], "timestamp"] - ALERT_LOOKBACK_MINUTES * 60 * 1000 if ALERT_LOOKBACK_MINUTES else 0

    for i in range(cutoff_index, len(df)):
        ts = df.loc[df.index[i], "timestamp"]
        if ALERT_LOOKBACK_MINUTES and ts < cutoff_time:
            continue
        price = df.loc[df.index[i], "close"]
        t_readable = pd.to_datetime(ts, unit="ms")
        if bull_bias[i]:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bias Up", "price": price, "time": t_readable})
        if bear_bias[i]:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bias Down", "price": price, "time": t_readable})
        if bull_fvg_series[i] and alert_fvg_bull:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bullish FVG", "price": price, "time": t_readable})
        if bear_fvg_series[i] and alert_fvg_bear:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bearish FVG", "price": price, "time": t_readable})
        if cisd_bull[i] and alert_cisd_bull:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bullish CISD", "price": price, "time": t_readable})
        if cisd_bear[i] and alert_cisd_bear:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bearish CISD", "price": price, "time": t_readable})
        if ifvg_bull[i] and alert_ifvg_bull:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bullish iFVG", "price": price, "time": t_readable})
        if ifvg_bear[i] and alert_ifvg_bear:
            signals.append({"symbol": symbol, "timeframe": timeframe, "signal": "Bearish iFVG", "price": price, "time": t_readable})
    return signals


# =============================== MAIN ===============================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exchange = init_exchange()
    symbols = get_symbols(exchange)
    logging.info("Scanning %d symbols", len(symbols))
    if not symbols:
        logging.error("No symbols available for scanning. Check exchange markets or whitelist settings.")
        return

    def run_once():
        for symbol in symbols:
            htf_context = fetch_htf_context(exchange, symbol)
            for tf in SCAN_TIMEFRAMES:
                try:
                    df = fetch_ohlcv_dataframe(exchange, symbol, tf, MAX_HISTORY_BARS)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Fetch failed for %s %s: %s", symbol, tf, exc)
                    continue
                signals = process_symbol(symbol, tf, df, htf_context)
                for sig in signals:
                    logging.info("%s, %s, %s, %.4f, %s", sig["symbol"], sig["timeframe"], sig["signal"], sig["price"], sig["time"])

    run_once()
    if SCAN_INTERVAL_SECONDS > 0:
        while True:
            time.sleep(SCAN_INTERVAL_SECONDS)
            run_once()


if __name__ == "__main__":
    main()
