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


@dataclass
class DynamicTuning:
    """عوامل تضخيم/تهدئة ديناميكية مشتقة من التوزيع التاريخي."""

    price_sigma_mult: float = 1.25
    oi_sigma_mult: float = 1.15
    vol_sensitivity: float = 0.25
    min_samples: int = 20
    flash_sigma_mult: float = 3.0
    momentum_floor: float = 0.05


@dataclass
class Config:
    timeframe: str = "15m"
    limit_coins: int = 200
    lookback: int = 50
    thresholds: Thresholds = Thresholds()
    dynamic: DynamicTuning = DynamicTuning()
    throttle_delay: float = 0.15
    long_short_period: str = "5m"


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

# ==========================================
# 3. الدوال المساعدة (Helper Functions)
# ==========================================


def get_top_symbols(limit: int) -> List[str]:
    """جلب أعلى عملات العقود الدائمة USDT-M من حيث حجم التداول."""

    try:
        tickers = exchange.fetch_tickers()
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
        ohlcv = exchange.fetch_ohlcv(symbol, CONFIG.timeframe, limit=CONFIG.lookback + 1)
        oi_history = exchange.fetch_open_interest_history(
            symbol,
            CONFIG.timeframe,
            limit=CONFIG.lookback + 1,
        )
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
        ticker = exchange.fetch_ticker(symbol)

        # أسعار رئيسية
        futures_price = float(ticker.get("last") or ticker.get("close"))
        mark_price = float(ticker.get("info", {}).get("markPrice", futures_price))
        index_price = float(ticker.get("info", {}).get("indexPrice", futures_price))

        # الأساس = الفرق بين سعر العقود وسعر المؤشر
        basis = futures_price - index_price
        basis_pct = (basis / index_price) * 100 if index_price else 0.0

        # تمويل
        funding_rate = None
        try:
            funding = exchange.fetch_funding_rate(symbol)
            funding_rate = float(funding.get("fundingRate")) if funding else None
        except Exception:
            funding_rate = None

        # نسب المتداولين الكبار (إذا توفرت من واجهة بيانات بينانس)
        top_ratio = None
        try:
            endpoint = getattr(exchange, "fapiPublicGetTopLongShortAccountRatio", None)
            if endpoint:
                resp = endpoint({"symbol": symbol.replace("/", ""), "period": CONFIG.long_short_period, "limit": 1})
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

        return {
            "futures_price": futures_price,
            "mark_price": mark_price,
            "index_price": index_price,
            "basis": basis,
            "basis_pct": basis_pct,
            "funding_rate": funding_rate,
            "top_long_short_ratio": top_ratio,
            "taker_buy_quote": taker_buy_quote,
            "taker_sell_quote": taker_sell_quote,
            "buy_sell_ratio": buy_sell_ratio,
            "oi_to_liquidity": oi_to_liquidity,
            "oi_value": oi_value,
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

    closes = [candle[4] for candle in ohlcv[-CONFIG.lookback :]]
    price_returns = [((closes[i] - closes[i - 1]) / closes[i - 1]) * 100 for i in range(1, len(closes))]

    oi_series = [float(point["openInterestAmount"]) for point in oi_history[-CONFIG.lookback :]]
    oi_returns = [((oi_series[i] - oi_series[i - 1]) / oi_series[i - 1]) * 100 for i in range(1, len(oi_series))]

    price_change_pct = price_returns[-1]
    oi_change_pct = oi_returns[-1]
    volatility = pstdev(closes) / closes[-1] * 100

    return (
        round(price_change_pct, 2),
        round(oi_change_pct, 2),
        round(volatility, 2),
        price_returns,
        oi_returns,
    )


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
) -> Tuple[str, str]:
    """تطبيق قواعد الاستراتيجية وإرجاع الإشارة مع المبرر."""

    t = adjust_thresholds_dynamic(volatility, price_returns, oi_returns)

    # إشارات تأكيد/إلغاء بناءً على الأساس والتمويل ونسبة المتداولين الكبار
    basis_pct = metrics.get("basis_pct") or 0.0
    funding = metrics.get("funding_rate")
    top_ratio = metrics.get("top_long_short_ratio")
    buy_sell_ratio = metrics.get("buy_sell_ratio")
    oi_to_liquidity = metrics.get("oi_to_liquidity")

    momentum = classify_momentum(price_chg, oi_chg)
    flash_event = detect_flash_event(price_chg, oi_chg, price_returns, oi_returns)

    long_score = 0
    short_score = 0
    notes: List[str] = []

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
    if basis_pct >= t.basis_extreme_pos:
        notes.append("أساس موجب مرتفع (كونتانجو مبالغ)")
        short_score += 1
    if basis_pct <= t.basis_extreme_neg:
        notes.append("أساس سالب كبير (باكوارد)")
        long_score += 1
    if oi_to_liquidity and oi_to_liquidity >= t.oi_liquidity_hot:
        notes.append("رافعة مرتفعة: OI/السيولة في خطر")
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

    if buy_sell_ratio:
        if buy_sell_ratio >= 1.2:
            notes.append("تفضيل شراء من التيكرز")
            long_score += 1
        elif buy_sell_ratio <= 0.8:
            notes.append("تفضيل بيع من التيكرز")
            short_score += 1

    # إشارات أساسية موسعة
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

    # تأثير نسبة كبار المتداولين
    if top_ratio is not None:
        if top_ratio >= t.top_ratio_high:
            notes.append("حيتان منحازة لونغ بشكل مرتفع")
            long_score += 1
        elif top_ratio <= t.top_ratio_low:
            notes.append("حيتان منحازة شورت بقوة")
            short_score += 1

    # أحداث الفلاش تعطل الدخول اللحظي وتوجه للخروج/جني أرباح
    if flash_event:
        if "Short Squeeze" in flash_event:
            notes.append("فلاش صعودي: فكر في جني أرباح اللونغ/تحوط")
            short_score += 1
        elif "Long Squeeze" in flash_event:
            notes.append("فلاش هبوطي: فكر في تغطية الشورت/شراء عكسي")
            long_score += 1

    # ترجيح نهائي مع حماية من التشبع المفرط
    if long_score > short_score + 1:
        return "🟢 LONG", " | ".join(notes) or momentum
    if short_score > long_score + 1:
        return "🔴 SHORT", " | ".join(notes) or momentum
    if long_score == short_score and long_score > 0:
        return "⚪️ NEUTRAL/WAIT", "إشارات متعارضة: " + (" | ".join(notes) or momentum)

    return "NEUTRAL", "-"


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
        signal, rationale = evaluate_signal(price_chg, oi_chg, volatility, price_returns, oi_returns, metrics)
        momentum = classify_momentum(price_chg, oi_chg)
        flash = detect_flash_event(price_chg, oi_chg, price_returns, oi_returns)

        futures_price = metrics.get("futures_price")
        basis_pct = metrics.get("basis_pct")
        funding_rate = metrics.get("funding_rate")
        top_ratio = metrics.get("top_long_short_ratio")
        oi_to_liquidity = metrics.get("oi_to_liquidity")

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
            # row schema before: [symbol, price%, oi%, vol%, fut, basis, funding, top, oi/liquidity, momentum, flash, signal, reason]
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

    print("\n🔁 القاعدة الذهبية السريعة:")
    print("- السعر ينخفض + OI يرتفع = إشارة هبوطية قوية")
    print("- السعر ينخفض بشدة + OI ينخفض بشدة = احتمال انعكاس صعودي")
    print("- السعر يرتفع + OI ينخفض = ضعف في الاتجاه الصاعد")
    print("- Basis موجب + تمويل مرتفع + OI مرتفع = ضغط بيع محتمل")
    print("- Basis سالب + تمويل سلبي + تفريغ OI = احتمالية ارتداد صعودي")

    if longs or shorts:
        print("\n📌 قرار الدخول المقترح بعد التحليل:")
        for row in annotate(longs, "LONG"):
            print(f"✅ {row[0]}: {row[9]} | {row[10]}")
        for row in annotate(shorts, "SHORT"):
            print(f"⚠️ {row[0]}: {row[9]} | {row[10]}")


# ==========================================
# 7. نقطة الدخول الرئيسية
# ==========================================


if __name__ == "__main__":
    try:
        long_signals, short_signals = analyze_market()
        render_report(long_signals, short_signals)
    except KeyboardInterrupt:
        print("\nتم إيقاف البرنامج.")
