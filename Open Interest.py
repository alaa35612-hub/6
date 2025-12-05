import ccxt
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import pstdev
from tabulate import tabulate
from typing import Dict, List, Optional, Tuple

# ==========================================
# 1. إعدادات الاستراتيجية (Config)
# ==========================================

@dataclass
class Thresholds:
    """القيم الأساسية قبل أي ضبط ديناميكي."""

    # ستظل هذه القيم أساساً، لكن سيتم تعديلها لاحقاً إحصائياً بحسب تذبذب كل أصل.
    bearish_oi_increase: float = 3.0
    bearish_price_max_drop: float = -0.5
    bearish_price_limit_drop: float = -2.5
    bullish_price_drop: float = -3.5
    bullish_oi_drop: float = -3.5
    exhaustion_oi_drop: float = -1.5
    min_volatility: float = 0.4
    max_volatility: float = 2.5
    funding_high: float = 0.01
    funding_extreme_high: float = 0.07
    funding_extreme_low: float = -0.05
    basis_extreme_pos: float = 1.5
    basis_extreme_neg: float = -1.5
    oi_liquidity_hot: float = 5.0
    top_ratio_high: float = 1.5
    top_ratio_low: float = 0.8
    top_ratio_extreme_high: float = 2.5
    top_ratio_extreme_low: float = 0.5


@dataclass
class DynamicTuning:
    """عوامل تضخيم/تهدئة ديناميكية مشتقة من التوزيع التاريخي."""

    price_sigma_mult: float = 1.25
    oi_sigma_mult: float = 1.15
    vol_sensitivity: float = 0.25
    min_samples: int = 20
    flash_sigma_mult: float = 3.0
    momentum_floor: float = 0.05
    price_trend_lookback: int = 10
    outlier_sigma: float = 4.0
    multi_timeframes: Tuple[str, ...] = ("5m", "1h", "4h")
    funding_history: int = 50
    liquidation_lookback: int = 50


@dataclass
class Config:
    timeframe: str = "15m"
    limit_coins: int = 200
    lookback: int = 50
    thresholds: Thresholds = Thresholds()
    dynamic: DynamicTuning = DynamicTuning()
    throttle_delay: float = 0.15
    long_short_period: str = "5m"
    orderbook_limit: int = 50
    retry_attempts: int = 3
    retry_backoff: float = 0.75
    cache_ttl: float = 30.0


CONFIG = Config()

# ------------------------------------------
# قاموس تحويل المصطلحات (Mapping Dictionary)
# ------------------------------------------
TERM_MAPPING: Dict[str, Tuple[str, str]] = {
    "Sucker Pattern": ("Adverse Positioning", "High_OI_Accumulation + Negative_Price_Drift"),
    "Trapped Traders": ("Adverse Positioning", "High_OI_Accumulation + Negative_Price_Drift"),
    "Price Creeping": ("Consolidation", "Low_Volatility_Range"),
    "Flat": ("Consolidation", "Low_Volatility_Range"),
    "Price Tanking": ("Liquidation Cascade", "Sharp_Price_Drop + OI_Drop"),
    "Killing everyone": ("Liquidation Cascade", "Sharp_Price_Drop + OI_Drop"),
    "Fuel for the fire": ("Short Squeeze", "Rising_Price + Decreasing_OI"),
    "Faith in trend": ("Trend Momentum", "Correlation(Price, OI)"),
}

# ==========================================
# 2. تهيئة الاتصال بالمنصة
# ==========================================
print("🔄 جاري الاتصال بمنصة Binance Futures...")
exchange = ccxt.binanceusdm(
    {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
)

# تحميل الأسواق مرة واحدة لتصفية عقود USDT-M فقط.
exchange.load_markets()
FUTURES_USDT = {
    symbol
    for symbol, meta in exchange.markets.items()
    if meta.get("linear") and meta.get("quote") == "USDT" and meta.get("active", True)
}

# مخزن مؤقت للنتائج الباهظة زمنياً
_TICKERS_CACHE: Dict[str, Tuple[float, Dict]] = {}

# ==========================================
# 3. الدوال المساعدة (Helper Functions)
# ==========================================


def request_with_retry(method, *args, **kwargs):
    """تنفيذ طلب مع آلية إعادة المحاولة والتدرج البسيط."""

    attempts = CONFIG.retry_attempts
    delay = CONFIG.retry_backoff
    for attempt in range(1, attempts + 1):
        try:
            return method(*args, **kwargs)
        except ccxt.NotSupported as exc:
            # لا جدوى من إعادة المحاولة إذا كان المنفذ غير مدعوم أصلاً.
            print(f"⚠️ المنصة لا تدعم الطلب: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            sleep_for = delay * attempt
            print(f"↻ إعادة المحاولة ({attempt}/{attempts}) بعد خطأ: {exc} -> الانتظار {sleep_for:.1f}s")
            time.sleep(sleep_for)


def _cache_valid(cache_key: str) -> bool:
    ts = _TICKERS_CACHE.get(cache_key, (0,))[0]
    return (time.time() - ts) < CONFIG.cache_ttl


def filter_outliers(series: List[float], sigma: float) -> List[float]:
    """إزالة القيم المتطرفة لتقليل الضوضاء في الأسواق منخفضة السيولة."""

    if not series:
        return series
    mean_val = sum(series) / len(series)
    std_val = pstdev(series) if len(series) > 1 else 0
    if std_val == 0:
        return series
    filtered = [x for x in series if abs(x - mean_val) <= sigma * std_val]
    return filtered if len(filtered) >= max(3, len(series) // 2) else series


def align_by_timestamp(ohlcv: List[List[float]], oi_history: List[Dict]) -> Tuple[List[List[float]], List[Dict]]:
    """مواءمة بيانات السعر وOI زمنياً لتقليل الانحراف."""

    if not ohlcv or not oi_history:
        return ohlcv, oi_history

    timeframe_ms = exchange.parse_timeframe(CONFIG.timeframe) * 1000
    oi_map = {int(item.get("timestamp") // timeframe_ms): item for item in oi_history}
    aligned_ohlcv: List[List[float]] = []
    aligned_oi: List[Dict] = []
    for candle in ohlcv:
        bucket = int(candle[0] // timeframe_ms)
        if bucket in oi_map:
            aligned_ohlcv.append(candle)
            aligned_oi.append(oi_map[bucket])
    # إذا قلَّت العينات نرجع الأصل بدون تعديل
    if len(aligned_ohlcv) < CONFIG.dynamic.min_samples:
        return ohlcv, oi_history
    return aligned_ohlcv, aligned_oi


def get_top_symbols(limit: int) -> List[str]:
    """جلب أعلى عملات العقود الدائمة USDT-M من حيث حجم التداول."""

    cache_key = "tickers"
    try:
        if not _cache_valid(cache_key):
            _TICKERS_CACHE[cache_key] = (time.time(), request_with_retry(exchange.fetch_tickers))
        tickers = _TICKERS_CACHE[cache_key][1]
        sorted_tickers = sorted(
            tickers.items(),
            key=lambda item: item[1].get("quoteVolume", 0),
            reverse=True,
        )

        symbols = [symbol for symbol, data in sorted_tickers if symbol in FUTURES_USDT]
        return symbols[:limit]
    except Exception as exc:  # noqa: BLE001 - نعرض الخطأ للمستخدم
        print(f"⚠️ خطأ في جلب الرموز: {exc}")
        return []


def fetch_ohlcv_and_oi(symbol: str) -> Optional[Tuple[List[List[float]], List[Dict]]]:
    """جلب OHLCV والـ OI التاريخي للرمز."""

    try:
        ohlcv = request_with_retry(exchange.fetch_ohlcv, symbol, CONFIG.timeframe, limit=CONFIG.lookback + 1)
        oi_history = request_with_retry(
            exchange.fetch_open_interest_history,
            symbol,
            CONFIG.timeframe,
            limit=CONFIG.lookback + 1,
        )
        if not isinstance(ohlcv, list) or not all(isinstance(c, (list, tuple)) for c in ohlcv):
            print(f"⚠️ صيغة OHLCV غير متوقعة لـ {symbol} - تم التجاوز")
            return None
        if not isinstance(oi_history, list) or not all(isinstance(item, dict) for item in oi_history):
            print(f"⚠️ صيغة OI غير متوقعة لـ {symbol} - تم التجاوز")
            return None
        ohlcv, oi_history = align_by_timestamp(ohlcv, oi_history)
        if len(ohlcv) <= CONFIG.dynamic.min_samples or len(oi_history) <= CONFIG.dynamic.min_samples:
            print(f"⚠️ بيانات غير كافية لـ {symbol} - تم التجاوز")
            return None
        return ohlcv, oi_history
    except Exception as exc:  # noqa: BLE001 - نعرض الخطأ للمستخدم
        print(f"⚠️ تعذر جلب البيانات لـ {symbol}: {exc}")
        return None


def fetch_risk_metrics(symbol: str) -> Optional[Dict]:
    """جلب بيانات إضافية: سعر العقد، المؤشر، الأساس، التمويل، أحجام الشراء/البيع وغيرها."""

    try:
        ticker = request_with_retry(exchange.fetch_ticker, symbol)

        # أسعار رئيسية
        futures_price = float(ticker.get("last") or ticker.get("close"))
        mark_price = float(ticker.get("info", {}).get("markPrice", futures_price))
        index_price = float(ticker.get("info", {}).get("indexPrice", futures_price))

        # الأساس = الفرق بين سعر العقود وسعر المؤشر
        basis = futures_price - index_price
        basis_pct = (basis / index_price) * 100 if index_price else 0.0

        # تمويل
        funding_rate = None
        funding_history = []
        try:
            funding = request_with_retry(exchange.fetch_funding_rate, symbol)
            funding_rate = float(funding.get("fundingRate")) if funding else None
            funding_history_resp = request_with_retry(
                exchange.fetch_funding_rate_history,
                symbol,
                None,
                None,
                CONFIG.dynamic.funding_history,
            )
            if funding_history_resp:
                funding_history = [float(row.get("fundingRate") or 0) for row in funding_history_resp]
        except Exception:
            funding_rate = None

        # نسب المتداولين الكبار (إذا توفرت من واجهة بيانات بينانس)
        top_ratio = None
        try:
            endpoint = getattr(exchange, "fapiPublicGetTopLongShortAccountRatio", None)
            if endpoint:
                resp = request_with_retry(
                    endpoint,
                    {"symbol": symbol.replace("/", ""), "period": CONFIG.long_short_period, "limit": 1},
                )
                if resp:
                    top_ratio = float(resp[0].get("longShortRatio"))
        except Exception:
            top_ratio = None

        # أحجام التكر و نسبة الشراء/البيع
        quote_volume = float(ticker.get("quoteVolume") or 0)
        taker_buy_quote = float(ticker.get("takerBuyQuoteVolume") or 0)
        taker_sell_quote = max(quote_volume - taker_buy_quote, 0)
        buy_sell_ratio = (taker_buy_quote / taker_sell_quote) if taker_sell_quote else None

        # نسبة الفائدة المفتوحة للقيمة السوقية (نستخدم حجم التداول كبديل للسيولة)
        oi_value = float(ticker.get("info", {}).get("openInterestValue", 0))
        oi_to_liquidity = (oi_value / quote_volume) if quote_volume else None

        # عمق دفتر الأوامر وفارق السبريد كمقياس للسيولة اللحظية
        orderbook = request_with_retry(exchange.fetch_order_book, symbol, CONFIG.orderbook_limit)
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        top_bid = bids[0][0] if bids else None
        top_ask = asks[0][0] if asks else None
        spread_pct = ((top_ask - top_bid) / top_bid * 100) if top_bid and top_ask else None

        def _depth(side: List[List[float]], levels: int = 10) -> float:
            return sum([price * size for price, size in side[:levels]]) if side else 0.0

        bid_depth = _depth(bids)
        ask_depth = _depth(asks)
        depth_ratio = (bid_depth / ask_depth) if ask_depth else None

        # تدفقات التصفيات (إن وُجدت)
        liquidations: List[Dict] = []
        try:
            if exchange.has.get("fetchLiquidations") and getattr(exchange, "fetch_liquidations", None):
                liq_resp = request_with_retry(
                    exchange.fetch_liquidations,
                    symbol,
                    None,
                    CONFIG.dynamic.liquidation_lookback,
                    {"limit": CONFIG.dynamic.liquidation_lookback},
                )
                if isinstance(liq_resp, list):
                    liquidations = [item for item in liq_resp if isinstance(item, dict)]
                elif isinstance(liq_resp, dict) and isinstance(liq_resp.get("data"), list):
                    liquidations = [item for item in liq_resp.get("data") if isinstance(item, dict)]
        except Exception:
            liquidations = []

        long_liq = sum(float(item.get("base", 0)) for item in liquidations if item.get("side") == "long")
        short_liq = sum(float(item.get("base", 0)) for item in liquidations if item.get("side") == "short")

        buy_pressure = bid_depth + taker_buy_quote
        sell_pressure = ask_depth + taker_sell_quote
        liquidity_score = (buy_pressure / sell_pressure) if sell_pressure else None

        return {
            "futures_price": futures_price,
            "mark_price": mark_price,
            "index_price": index_price,
            "basis": basis,
            "basis_pct": basis_pct,
            "funding_rate": funding_rate,
            "funding_history": funding_history,
            "top_long_short_ratio": top_ratio,
            "taker_buy_quote": taker_buy_quote,
            "taker_sell_quote": taker_sell_quote,
            "buy_sell_ratio": buy_sell_ratio,
            "oi_to_liquidity": oi_to_liquidity,
            "oi_value": oi_value,
            "spread_pct": spread_pct,
            "depth_ratio": depth_ratio,
            "liquidity_score": liquidity_score,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "long_liquidations": long_liq,
            "short_liquidations": short_liq,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ تعذر جلب مقاييس المخاطر لـ {symbol}: {exc}")
        return None


def compute_changes(
    ohlcv: List[List[float]], oi_history: List[Dict]
) -> Tuple[
    float,
    float,
    float,
    List[float],
    List[float],
]:
    """يحسب التغيرات بالنسبة المئوية والتذبذب البسيط + سلاسل تاريخية."""

    closes_raw = [candle[4] for candle in ohlcv[-CONFIG.lookback :]]
    oi_series_raw = [float(point["openInterestAmount"]) for point in oi_history[-CONFIG.lookback :]]

    closes = filter_outliers(closes_raw, CONFIG.dynamic.outlier_sigma)
    oi_series = filter_outliers(oi_series_raw, CONFIG.dynamic.outlier_sigma)

    price_returns = [((closes[i] - closes[i - 1]) / closes[i - 1]) * 100 for i in range(1, len(closes))]
    oi_returns = [((oi_series[i] - oi_series[i - 1]) / oi_series[i - 1]) * 100 for i in range(1, len(oi_series))]

    if not price_returns or not oi_returns:
        return 0.0, 0.0, 0.0, price_returns, oi_returns

    price_change_pct = price_returns[-1]
    oi_change_pct = oi_returns[-1]
    volatility = (pstdev(closes) / closes[-1] * 100) if len(closes) > 1 else 0.0

    return (
        round(price_change_pct, 2),
        round(oi_change_pct, 2),
        round(volatility, 2),
        price_returns,
        oi_returns,
    )


def compute_trend(series: List[float], lookback: int) -> int:
    """ترند بسيط: مقارنة المتوسط القصير بالمتوسط الطويل لتقدير الاتجاه العام."""

    if len(series) < lookback + 5:
        return 0

    short_avg = sum(series[-lookback:]) / lookback
    long_avg = sum(series) / len(series)
    if short_avg > long_avg * 1.002:
        return 1
    if short_avg < long_avg * 0.998:
        return -1
    return 0


def multi_timeframe_trend(symbol: str) -> Dict[str, int]:
    """حساب الترند عبر عدة أطر زمنية لإشارة أكثر موثوقية."""

    trends: Dict[str, int] = {}
    for tf in CONFIG.dynamic.multi_timeframes:
        try:
            ohlcv = request_with_retry(exchange.fetch_ohlcv, symbol, tf, limit=CONFIG.dynamic.price_trend_lookback + 20)
            closes = [c[4] for c in ohlcv]
            trends[tf] = compute_trend(closes, min(CONFIG.dynamic.price_trend_lookback, len(closes) - 1))
        except Exception:
            trends[tf] = 0
    return trends


def score_confidence(long_score: int, short_score: int, coverage: float) -> float:
    """
    تحويل الفارق بين الدرجات إلى نسبة مئوية مبسطة لقياس دقة/ثقة الإشارة
    مع وزن إضافي بناءً على جودة البيانات المتاحة.
    """

    total = max(long_score + short_score, 1)
    diff = abs(long_score - short_score)
    raw = min(100.0, (diff / total) * 100)
    # تغطية البيانات (0-100) تقلل الثقة إذا كانت المدخلات ناقصة.
    weighted = raw * max(0.25, coverage / 100)
    return round(weighted, 1)


def classify_momentum(price_chg: float, oi_chg: float) -> str:
    """تصنيف الزخم اللحظي وفق حالات السعر/الفائدة المفتوحة."""

    floor = CONFIG.dynamic.momentum_floor
    price_up = price_chg > floor
    price_down = price_chg < -floor
    oi_up = oi_chg > floor
    oi_down = oi_chg < -floor

    if price_up and oi_up:
        return "زخم صعودي حقيقي (Price↑ + OI↑)"
    if price_up and oi_down:
        return "Short Squeeze محتمل (Price↑ + OI↓)"
    if price_down and oi_up:
        return "زخم هبوطي حقيقي (Price↓ + OI↑)"
    if price_down and oi_down:
        return "Long Squeeze محتمل (Price↓ + OI↓)"
    return "زخم جانبي/ضعيف"


def detect_flash_event(
    price_chg: float,
    oi_chg: float,
    price_returns: List[float],
    oi_returns: List[float],
) -> Optional[str]:
    """رصد أحداث الفلاش عبر انحرافات سعرية/‏OI حادة عن التوزيع التاريخي."""

    if len(price_returns) < 5 or len(oi_returns) < 5:
        return None

    price_sigma = pstdev(price_returns)
    oi_sigma = pstdev(oi_returns)
    p_thr = CONFIG.dynamic.flash_sigma_mult * price_sigma
    oi_thr = CONFIG.dynamic.flash_sigma_mult * oi_sigma

    if price_chg > p_thr and oi_chg < -oi_thr:
        return "Flash Short Squeeze (قفزة + تفريغ OI)"
    if price_chg < -p_thr and oi_chg < -oi_thr:
        return "Flash Long Squeeze (انهيار + تفريغ OI)"
    return None


# ==========================================
# 4. المنطق الاستراتيجي
# ==========================================


def adjust_thresholds_dynamic(
    volatility: float, price_returns: List[float], oi_returns: List[float]
) -> Thresholds:
    """تعديل ديناميكي للعتبات بناءً على التذبذب وتوزيع التغيرات التاريخية."""

    base = CONFIG.thresholds
    tuning = CONFIG.dynamic

    price_mu = sum(price_returns) / len(price_returns)
    oi_mu = sum(oi_returns) / len(oi_returns)

    price_sigma = pstdev(price_returns)
    oi_sigma = pstdev(oi_returns)

    vol_scale = 1 + tuning.vol_sensitivity * max(0, (volatility - base.min_volatility))
    price_band = tuning.price_sigma_mult * price_sigma
    oi_band = tuning.oi_sigma_mult * oi_sigma

    return Thresholds(
        bearish_oi_increase=max(base.bearish_oi_increase, oi_mu + oi_band) * vol_scale,
        bearish_price_max_drop=min(base.bearish_price_max_drop, price_mu + price_band) * vol_scale,
        bearish_price_limit_drop=min(base.bearish_price_limit_drop, price_mu - price_band) * vol_scale,
        bullish_price_drop=min(base.bullish_price_drop, price_mu - price_band * 1.1) * vol_scale,
        bullish_oi_drop=min(base.bullish_oi_drop, oi_mu - oi_band * 1.1) * vol_scale,
        exhaustion_oi_drop=min(base.exhaustion_oi_drop, oi_mu - oi_band) * vol_scale,
        min_volatility=base.min_volatility,
        max_volatility=base.max_volatility,
    )


def evaluate_signal(
    price_chg: float,
    oi_chg: float,
    volatility: float,
    price_returns: List[float],
    oi_returns: List[float],
    metrics: Dict,
) -> Tuple[str, str, int, int, float]:
    """تطبيق قواعد الاستراتيجية وإرجاع الإشارة مع المبرر ونقاط الصعود/الهبوط ونسبة ثقة مبسطة."""

    t = adjust_thresholds_dynamic(volatility, price_returns, oi_returns)

    # إشارات تأكيد/إلغاء بناءً على الأساس والتمويل ونسبة المتداولين الكبار
    basis_raw = metrics.get("basis_pct")
    basis_pct = basis_raw or 0.0
    funding = metrics.get("funding_rate")
    top_ratio = metrics.get("top_long_short_ratio")
    buy_sell_ratio = metrics.get("buy_sell_ratio")
    oi_to_liquidity = metrics.get("oi_to_liquidity")
    spread_pct = metrics.get("spread_pct")
    depth_ratio = metrics.get("depth_ratio")
    liquidity_score = metrics.get("liquidity_score")
    long_liq = metrics.get("long_liquidations")
    short_liq = metrics.get("short_liquidations")
    funding_history = metrics.get("funding_history", [])

    momentum = classify_momentum(price_chg, oi_chg)
    flash_event = detect_flash_event(price_chg, oi_chg, price_returns, oi_returns)
    # سلسلة الإغلاقات جاهزة بالفعل في metrics["ohlcv_closes"], لذا نمررها مباشرة لتفادي فهرسة غير صحيحة
    price_trend = compute_trend(metrics.get("ohlcv_closes", []) or [0], CONFIG.dynamic.price_trend_lookback)
    oi_trend = compute_trend(metrics.get("oi_series", []), CONFIG.dynamic.price_trend_lookback)
    mtf_trends = metrics.get("mtf_trends", {})

    long_score = 0
    short_score = 0
    notes: List[str] = []

    # تغطية البيانات: كلما زادت المقاييس المتاحة ارتفعت الثقة في القرار
    coverage_checks = [
        basis_raw is not None,
        funding is not None,
        top_ratio is not None,
        buy_sell_ratio is not None,
        oi_to_liquidity is not None,
        spread_pct is not None,
        depth_ratio is not None,
        price_trend != 0,
        oi_trend != 0,
        momentum not in {"زخم جانبي/ضعيف"},
        any(val != 0 for val in mtf_trends.values()) if mtf_trends else False,
    ]
    coverage_pct = (sum(coverage_checks) / len(coverage_checks)) * 100
    missing_metrics = [
        name
        for flag, name in zip(
            coverage_checks,
            [
                "basis",
                "funding",
                "top accounts",
                "buy/sell ratio",
                "oi/liquidity",
                "spread",
                "depth",
                "price trend",
                "oi trend",
                "momentum",
                "multi timeframe",
            ],
        )
        if not flag
    ]

    # ترجيح التمويل والأساس كعوامل تشبع/حذر
    if funding is not None:
        if funding >= t.funding_extreme_high:
            notes.append("تمويل موجب متطرف = تشبع شرائي")
            short_score += 2
        elif funding >= t.funding_high:
            notes.append("تمويل موجب مرتفع")
            short_score += 1
        elif funding <= t.funding_extreme_low:
            notes.append("تمويل سلبي متطرف = تشبع بيعي")
            long_score += 2
    if funding_history:
        avg_funding = sum(funding_history) / len(funding_history)
        if avg_funding > t.funding_high:
            notes.append("متوسط تمويل مرتفع تاريخياً -> تشبع شرائي مزمن")
            short_score += 1
        if avg_funding < t.funding_extreme_low:
            notes.append("متوسط تمويل سلبي حاد -> ضغط قصير محتمل")
            long_score += 1
    if basis_pct >= t.basis_extreme_pos:
        notes.append("أساس موجب مرتفع (كونتانجو مبالغ)")
        short_score += 1
    if basis_pct <= t.basis_extreme_neg:
        notes.append("أساس سالب كبير (باكوارد)")
        long_score += 1
    if oi_to_liquidity and oi_to_liquidity >= t.oi_liquidity_hot:
        notes.append("رافعة مرتفعة: OI/السيولة في خطر")
        short_score += 1
    if spread_pct and spread_pct > 0.25:
        notes.append("سبريد عريض -> سيولة ضعيفة")
        short_score += 1
    if depth_ratio and depth_ratio > 1.5:
        notes.append("عمق شراء يضغط للأعلى")
        long_score += 1
    elif depth_ratio and depth_ratio < 0.67:
        notes.append("عمق بيع يضغط للأسفل")
        short_score += 1
    if liquidity_score and liquidity_score < 0.9:
        notes.append("توازن السيولة ضعيف لصالح البائعين")
        short_score += 1
    elif liquidity_score and liquidity_score > 1.1:
        notes.append("توازن السيولة لصالح المشترين")
        long_score += 1

    if long_liq or short_liq:
        if long_liq > short_liq * 2:
            notes.append("تصفية لونغات مرتفعة -> احتمالية ارتداد صعودي")
            long_score += 1
        elif short_liq > long_liq * 2:
            notes.append("تصفية شورتات مرتفعة -> احتمالية تهدئة صعودية")
            short_score += 1

    # تأثير نسبة كبار المتداولين مع القراءة المعاكسة عند التطرف
    if top_ratio is not None:
        if top_ratio >= t.top_ratio_extreme_high:
            notes.append("حيتان لونغ بشكل مفرط (إشارة معاكسة محتملة)")
            short_score += 2
        elif top_ratio >= t.top_ratio_high:
            notes.append("حيتان منحازة لونغ")
            long_score += 1
        elif top_ratio <= t.top_ratio_extreme_low:
            notes.append("حيتان شورت بشكل مفرط (إشارة معاكسة صعودية)")
            long_score += 2
        elif top_ratio <= t.top_ratio_low:
            notes.append("حيتان منحازة شورت")
            short_score += 1

    # الزخم اللحظي
    if "صعودي" in momentum and "حقيقي" in momentum:
        long_score += 2
    if "هبوطي" in momentum and "حقيقي" in momentum:
        short_score += 2
    if "Short Squeeze" in momentum:
        long_score += 1
        notes.append("سوق يصعد بتفريغ شورتات")
    if "Long Squeeze" in momentum:
        short_score += 1
        notes.append("سوق يهبط بتفريغ لونغات")

    # ترند عام للسعر ولـ OI يضيف أولوية إضافية لاتجاه الترند السائد
    if price_trend == 1:
        long_score += 1
        notes.append("ترند سعري عام صاعد يدعم اللونغ")
    elif price_trend == -1:
        short_score += 1
        notes.append("ترند سعري عام هابط يدعم الشورت")

    if oi_trend == 1:
        long_score += 1
        notes.append("ترند OI صاعد = دخول سيولة جديدة")
    elif oi_trend == -1:
        short_score += 1
        notes.append("ترند OI هابط = تفريغ مراكز")

    if mtf_trends:
        bull_count = sum(1 for v in mtf_trends.values() if v == 1)
        bear_count = sum(1 for v in mtf_trends.values() if v == -1)
        if bull_count >= 2:
            notes.append("توافق أطر زمنية صعودي")
            long_score += 2
        elif bear_count >= 2:
            notes.append("توافق أطر زمنية هبوطي")
            short_score += 2

    if buy_sell_ratio:
        if buy_sell_ratio >= 1.2:
            notes.append("تفضيل شراء من التيكرز")
            long_score += 1
        elif buy_sell_ratio <= 0.8:
            notes.append("تفضيل بيع من التيكرز")
            short_score += 1

    # إشارات أساسية موسعة + القواعد النصية
    if t.bearish_price_limit_drop < price_chg < t.bearish_price_max_drop and oi_chg > t.bearish_oi_increase:
        short_score += 2
        notes.append("مصيدة لونغ: سعر مسطح/OI يقفز")

    if price_chg < t.bullish_price_drop and oi_chg < t.bullish_oi_drop:
        long_score += 2
        notes.append("استسلام/Capitulation: سعر وOI ينهاران")

    if price_chg > 0 and oi_chg < t.exhaustion_oi_drop:
        notes.append("إنهاك صعودي: سعر ↑ مقابل OI ↓")
        short_score += 1

    if price_chg < t.bearish_price_limit_drop and oi_chg > 0:
        notes.append("كسر دعم بدون تفريغ OI -> مقاومة محتملة")
        short_score += 1

    if price_chg > 1.0 and -1.5 <= oi_chg <= 0:
        notes.append("وقود Short Squeeze: سعر يرتفع مع تفريغ OI")
        long_score += 1

    # Long Rule 1: ترند صاعد + OI↑ + تمويل ≤0 + حيتان شورت + أساس ≤0
    if price_trend == 1 and oi_trend == 1 and (funding or 0) <= 0 and (top_ratio is None or top_ratio < t.top_ratio_low) and basis_pct <= 0:
        notes.append("لونغ 1: زخم صعودي مع تشبع بيعي (تمويل ≤0 وحيتان شورت)")
        long_score += 3

    # Long Rule 2: اختراق مدعوم بـ OI↑ وتمويل غير متطرف وأساس طبيعي
    if price_chg > abs(t.bearish_price_max_drop) and oi_chg > max(0, t.bearish_oi_increase / 2) and (funding is None or funding < t.funding_high) and abs(basis_pct) < abs(t.basis_extreme_pos):
        notes.append("لونغ 2: اختراق مدعوم بتدفق OI وتمويل غير متطرف")
        long_score += 2

    # Long Rule 3: Short Trap (نزول بطيء + OI↑ + تمويل سلبي + حيتان تتحول لونغ)
    if price_chg < 0 and oi_chg > t.bearish_oi_increase and (funding or 0) < 0 and (top_ratio is None or top_ratio >= 1.0):
        notes.append("لونغ 3: تراكم شورتات مع تمويل سالب -> احتمال Short Squeeze")
        long_score += 2

    # Short Rule 1: تشبع شرائي واضح (ترند صاعد + تمويل/أساس مرتفع + OI/Liq حار + حيتان لونغ)
    if price_trend == 1 and (funding or 0) >= t.funding_extreme_high and basis_pct >= t.basis_extreme_pos and (oi_to_liquidity or 0) >= t.oi_liquidity_hot and (top_ratio or 0) >= t.top_ratio_high:
        notes.append("شورت 1: تشبع شرائي (تمويل/أساس/رافعة مرتفعة والحيتان لونغ)")
        short_score += 3

    # Short Rule 2: اختراق كاذب/Short Squeeze (سعر↑ قوي + OI↓ + تمويل يقفز)
    if price_chg > abs(t.bearish_price_max_drop) and oi_chg < t.exhaustion_oi_drop and (funding or 0) >= t.funding_high:
        notes.append("شورت 2: اختراق كاذب/Short Squeeze غير مستدام")
        short_score += 2

    # Short Rule 3: Long Trap (صعود بطيء + OI↑ قوي + تمويل يرتفع + حيتان تخفف شراء)
    if price_chg > 0 and oi_chg > t.bearish_oi_increase and (funding or 0) > 0 and (top_ratio is not None and top_ratio < t.top_ratio_high):
        notes.append("شورت 3: تراكم لونغات برافعة مع خروج الحيتان")
        short_score += 2

    # أحداث الفلاش تعطل الدخول اللحظي وتوجه للخروج/جني أرباح
    if flash_event:
        if "Short Squeeze" in flash_event:
            notes.append("فلاش صعودي: جني أرباح/انتظار قبل أي لونغ جديد")
            short_score += 1
        elif "Long Squeeze" in flash_event:
            notes.append("فلاش هبوطي: تغطية شورت/انتظار قبل بيع جديد")
            long_score += 1
        joined = " | ".join(notes)
        return "⚪️ NEUTRAL/WAIT", joined, long_score, short_score, score_confidence(long_score, short_score, coverage_pct)

    if missing_metrics:
        notes.append("بيانات ناقصة: " + ", ".join(missing_metrics))

    # ترجيح نهائي مع حماية من التشبع المفرط
    if long_score > short_score + 1:
        return (
            "🟢 LONG",
            " | ".join(notes) or momentum,
            long_score,
            short_score,
            score_confidence(long_score, short_score, coverage_pct),
        )
    if short_score > long_score + 1:
        return (
            "🔴 SHORT",
            " | ".join(notes) or momentum,
            long_score,
            short_score,
            score_confidence(long_score, short_score, coverage_pct),
        )
    if long_score == short_score and long_score > 0:
        return (
            "⚪️ NEUTRAL/WAIT",
            "إشارات متعارضة: " + (" | ".join(notes) or momentum),
            long_score,
            short_score,
            score_confidence(long_score, short_score, coverage_pct),
        )

    return "NEUTRAL", "-", long_score, short_score, score_confidence(long_score, short_score, coverage_pct)


# ==========================================
# 5. تحليل السوق بالكامل
# ==========================================


def analyze_market() -> Tuple[List[List[str]], List[List[str]]]:
    print(f"🔎 جاري فحص أفضل {CONFIG.limit_coins} عملة رقمية... (قد يستغرق وقتاً)")
    symbols = get_top_symbols(CONFIG.limit_coins)

    longs: List[List[str]] = []
    shorts: List[List[str]] = []
    scanned = 0

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{CONFIG.limit_coins}] فحص {symbol}...", end="\r")
        payload = fetch_ohlcv_and_oi(symbol)
        if not payload:
            continue

        scanned += 1
        ohlcv, oi_history = payload
        price_chg, oi_chg, volatility, price_returns, oi_returns = compute_changes(ohlcv, oi_history)
        metrics = fetch_risk_metrics(symbol) or {}
        metrics["ohlcv_closes"] = [candle[4] for candle in ohlcv[-CONFIG.lookback :]]
        metrics["oi_series"] = [float(point["openInterestAmount"]) for point in oi_history[-CONFIG.lookback :]]
        metrics["mtf_trends"] = multi_timeframe_trend(symbol)
        signal, rationale, long_score, short_score, confidence = evaluate_signal(
            price_chg,
            oi_chg,
            volatility,
            price_returns,
            oi_returns,
            metrics,
        )
        momentum = classify_momentum(price_chg, oi_chg)
        flash = detect_flash_event(price_chg, oi_chg, price_returns, oi_returns)

        futures_price = metrics.get("futures_price")
        basis_pct = metrics.get("basis_pct")
        funding_rate = metrics.get("funding_rate")
        top_ratio = metrics.get("top_long_short_ratio")
        oi_to_liquidity = metrics.get("oi_to_liquidity")
        spread_pct = metrics.get("spread_pct")
        depth_ratio = metrics.get("depth_ratio")
        liquidity_score = metrics.get("liquidity_score")

        if signal != "NEUTRAL":
            row = [
                symbol,
                f"{price_chg}%",
                f"{oi_chg}%",
                f"{volatility}%",
                f"{futures_price}" if futures_price is not None else "-",
                f"{basis_pct:.2f}%" if basis_pct is not None else "-",
                f"{funding_rate:.4f}" if funding_rate is not None else "-",
                f"{top_ratio:.2f}" if top_ratio is not None else "-",
                f"{oi_to_liquidity:.2f}" if oi_to_liquidity is not None else "-",
                f"{spread_pct:.3f}%" if spread_pct is not None else "-",
                f"{depth_ratio:.2f}" if depth_ratio is not None else "-",
                f"{liquidity_score:.2f}" if liquidity_score is not None else "-",
                str(long_score),
                str(short_score),
                f"{confidence}%",
                momentum,
                flash or "-",
                signal,
                rationale,
            ]
            if "LONG" in signal:
                longs.append(row)
            elif "SHORT" in signal:
                shorts.append(row)

        time.sleep(CONFIG.throttle_delay)

    # ترتيب المخرجات تنازلياً حسب قوة/ثقة الإشارة
    longs.sort(key=lambda r: float(r[11].replace("%", "")), reverse=True)
    shorts.sort(key=lambda r: float(r[11].replace("%", "")), reverse=True)

    print(f"\n✅ تم فحص {scanned} أزواج بعينات كافية من أصل {len(symbols)}")
    return longs, shorts


# ==========================================
# 6. مخرجات التقرير
# ==========================================


def render_report(longs: List[List[str]], shorts: List[List[str]]) -> None:
    print("\n" + "=" * 70)
    print(f"📊 تقرير التحليل - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    headers = [
        "Symbol",
        "Price %",
        "OI %",
        "Vol %",
        "Fut Px",
        "Basis %",
        "Funding",
        "Top L/S",
        "OI/Liq",
        "Spread %",
        "DepthR",
        "LiqScore",
        "LScore",
        "SScore",
        "Conf %",
        "Momentum",
        "Flash",
        "Signal",
        "Action",
        "Reason",
    ]

    def annotate(rows: List[List[str]], bias: str) -> List[List[str]]:
        """إضافة توصية دخول واضحة لكل صف لتسهيل القراءة بعد التحليل."""

        action = "ادخل شراء" if bias == "LONG" else "ادخل بيع"
        enriched: List[List[str]] = []
        for row in rows:
            # row schema before: [symbol, price%, oi%, vol%, fut, basis, funding, top, oi/liquidity, spread, depth, liqScore, L, S, Conf, momentum, flash, signal, reason]
            enriched.append(row[:-1] + [action, row[-1]])
        return enriched

    if longs:
        print("\n🟢 فرص شراء محتملة (Long Candidates):")
        print(tabulate(annotate(longs, "LONG"), headers=headers, tablefmt="grid"))
    else:
        print("\n🟢 لا توجد فرص Long مطابقة حالياً.")

    if shorts:
        print("\n🔴 فرص بيع محتملة (Short Candidates):")
        print(tabulate(annotate(shorts, "SHORT"), headers=headers, tablefmt="grid"))
    else:
        print("\n🔴 لا توجد فرص Short مطابقة حالياً.")

    print("\n🔁 القاعدة الذهبية (المحدَّثة):")
    print(
        "تداول مع الاتجاه السائد فقط عند تأكيده بتدفق أموال جديدة (OI) مع غياب اختلال"
        " تمويلي/أساسي مفرط؛ إذا كان هناك تشبع (تمويل أو أساس أو OI/سيولة مرتفع جدًا)"
        " فضِّل الانتظار أو التداول عكسيًا بعد انتهاء الفلاش. راقب دائمًا اختلاف الحيتان"
        " عن الجمهور وتمييز حركة الزخم الحقيقي (Price/OI معًا) من الحركة القائمة على"
        " تصفيات فقط."
    )
    print("- السعر ينخفض + OI يرتفع = هبوط مؤكد/مصيدة لونغ محتملة")
    print("- السعر ينخفض بشدة + OI ينخفض بشدة = استسلام/احتمال انعكاس صعودي")
    print("- السعر يرتفع + OI ينخفض = شورت سكويز/ضعف استدامة الصعود")
    print("- تمويل/أساس موجب حاد + OI/سيولة مرتفع = تشبع شراء وخطر انعكاس هابط")
    print("- تمويل/أساس سالب بحدة + تفريغ OI = تشبع بيع وفرصة ارتداد")

    if longs or shorts:
        print("\n📌 قرار الدخول المقترح بعد التحليل:")
        for row in annotate(longs, "LONG"):
            symbol, lscore, sscore, conf, momentum, flash, signal, action, reason = (
                row[0],
                row[9],
                row[10],
                row[11],
                row[12],
                row[13],
                row[14],
                row[15],
                row[16],
            )
            print(
                f"✅ {symbol}: {action} | {signal} | ثقة {conf} | L:{lscore} / S:{sscore} | {momentum} | {flash} | {reason}"
            )
        for row in annotate(shorts, "SHORT"):
            symbol, lscore, sscore, conf, momentum, flash, signal, action, reason = (
                row[0],
                row[9],
                row[10],
                row[11],
                row[12],
                row[13],
                row[14],
                row[15],
                row[16],
            )
            print(
                f"⚠️ {symbol}: {action} | {signal} | ثقة {conf} | L:{lscore} / S:{sscore} | {momentum} | {flash} | {reason}"
            )


# ==========================================
# 7. نقطة الدخول الرئيسية
# ==========================================


if __name__ == "__main__":
    try:
        long_signals, short_signals = analyze_market()
        render_report(long_signals, short_signals)
    except KeyboardInterrupt:
        print("\nتم إيقاف البرنامج.")
