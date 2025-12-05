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
    bearish_oi_increase: float = 3.0
    bearish_price_max_drop: float = -0.5
    bearish_price_limit_drop: float = -2.5
    bullish_price_drop: float = -3.5
    bullish_oi_drop: float = -3.5
    exhaustion_oi_drop: float = -1.5
    min_volatility: float = 0.4
    max_volatility: float = 2.5


@dataclass
class Config:
    timeframe: str = "15m"
    limit_coins: int = 200
    lookback: int = 3
    thresholds: Thresholds = Thresholds()
    throttle_delay: float = 0.15


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
exchange = ccxt.binanceusdm({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})

# ==========================================
# 3. الدوال المساعدة (Helper Functions)
# ==========================================


def get_top_symbols(limit: int) -> List[str]:
    """جلب أعلى العملات من حيث حجم التداول (Quote Volume)."""
    try:
        tickers = exchange.fetch_tickers()
        sorted_tickers = sorted(
            tickers.items(),
            key=lambda item: item[1].get("quoteVolume", 0),
            reverse=True,
        )
        symbols = [symbol for symbol, data in sorted_tickers if symbol.endswith("/USDT")]
        return symbols[:limit]
    except Exception as exc:  # noqa: BLE001 - نعرض الخطأ للمستخدم
        print(f"⚠️ خطأ في جلب الرموز: {exc}")
        return []


def fetch_ohlcv_and_oi(symbol: str) -> Optional[Tuple[List[List[float]], List[Dict]]]:
    """جلب OHLCV والـ OI التاريخي للرمز."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, CONFIG.timeframe, limit=CONFIG.lookback + 1)
        oi_history = exchange.fetch_open_interest_history(
            symbol, CONFIG.timeframe, limit=CONFIG.lookback + 1
        )
        if len(ohlcv) < 2 or len(oi_history) < 2:
            return None
        return ohlcv, oi_history
    except Exception:
        return None


def compute_changes(ohlcv: List[List[float]], oi_history: List[Dict]) -> Tuple[float, float, float]:
    """يحسب التغيرات بالنسبة المئوية والتذبذب البسيط."""
    close_prices = [candle[4] for candle in ohlcv[-CONFIG.lookback:]]
    price_change_pct = ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100

    current_oi = float(oi_history[-1]["openInterestAmount"])
    prev_oi = float(oi_history[-2]["openInterestAmount"])
    oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100

    volatility = pstdev(close_prices) / close_prices[-1] * 100
    return round(price_change_pct, 2), round(oi_change_pct, 2), round(volatility, 2)


# ==========================================
# 4. المنطق الاستراتيجي
# ==========================================


def adjust_thresholds_by_volatility(volatility: float) -> Thresholds:
    """تعديل ديناميكي للعتبات بناءً على التذبذب الحالي."""
    scale = 1.0
    if volatility < CONFIG.thresholds.min_volatility:
        scale = 0.7
    elif volatility > CONFIG.thresholds.max_volatility:
        scale = 1.3

    base = CONFIG.thresholds
    return Thresholds(
        bearish_oi_increase=base.bearish_oi_increase * scale,
        bearish_price_max_drop=base.bearish_price_max_drop * scale,
        bearish_price_limit_drop=base.bearish_price_limit_drop * scale,
        bullish_price_drop=base.bullish_price_drop * scale,
        bullish_oi_drop=base.bullish_oi_drop * scale,
        exhaustion_oi_drop=base.exhaustion_oi_drop * scale,
        min_volatility=base.min_volatility,
        max_volatility=base.max_volatility,
    )


def evaluate_signal(price_chg: float, oi_chg: float, volatility: float) -> Tuple[str, str]:
    """تطبيق قواعد الاستراتيجية وإرجاع الإشارة مع المبرر."""
    t = adjust_thresholds_by_volatility(volatility)

    # 1) المصيدة الهبوطية (Trapped Longs)
    if t.bearish_price_limit_drop < price_chg < t.bearish_price_max_drop and oi_chg > t.bearish_oi_increase:
        return "🔴 SHORT", "Sucker Pattern: Price flat/down + OI spiking"

    # 2) الانعكاس الصعودي (Capitulation)
    if price_chg < t.bullish_price_drop and oi_chg < t.bullish_oi_drop:
        return "🟢 LONG", "Capitulation: Price & OI collapse"

    # 3) إنهاك الاتجاه الصاعد
    if price_chg > 0 and oi_chg < t.exhaustion_oi_drop:
        return "⚪️ EXIT/CAUTIOUS LONG", "Trend Exhaustion: Price up with falling OI"

    # 4) تأكيد المقاومة بالعالقين (Breakdown بدون خروج)
    if price_chg < t.bearish_price_limit_drop and oi_chg > 0:
        return "🔴 SHORT", "Trapped Resistance: Breakdown without OI flush"

    # 5) ضغط شراء (Short squeeze محتمل)
    if price_chg > 1.0 and -1.5 <= oi_chg <= 0:
        return "🟢 LONG", "Short squeeze fuel: Price rising while OI unwinds"

    return "NEUTRAL", "-"


# ==========================================
# 5. تحليل السوق بالكامل
# ==========================================


def analyze_market() -> Tuple[List[List[str]], List[List[str]]]:
    print(f"🔎 جاري فحص أفضل {CONFIG.limit_coins} عملة رقمية... (قد يستغرق وقتاً)")
    symbols = get_top_symbols(CONFIG.limit_coins)

    longs: List[List[str]] = []
    shorts: List[List[str]] = []

    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{CONFIG.limit_coins}] فحص {symbol}...", end="\r")
        payload = fetch_ohlcv_and_oi(symbol)
        if not payload:
            continue

        ohlcv, oi_history = payload
        price_chg, oi_chg, volatility = compute_changes(ohlcv, oi_history)
        signal, rationale = evaluate_signal(price_chg, oi_chg, volatility)

        if signal != "NEUTRAL":
            row = [
                symbol,
                f"{price_chg}%",
                f"{oi_chg}%",
                f"{volatility}%",
                signal,
                rationale,
            ]
            if "LONG" in signal:
                longs.append(row)
            elif "SHORT" in signal:
                shorts.append(row)

        time.sleep(CONFIG.throttle_delay)

    return longs, shorts


# ==========================================
# 6. مخرجات التقرير
# ==========================================


def render_report(longs: List[List[str]], shorts: List[List[str]]) -> None:
    print("\n" + "=" * 70)
    print(f"📊 تقرير التحليل - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    headers = ["Symbol", "Price %", "OI %", "Vol %", "Signal", "Reason"]

    if longs:
        print("\n🟢 فرص شراء محتملة (Long Candidates):")
        print(tabulate(longs, headers=headers, tablefmt="grid"))
    else:
        print("\n🟢 لا توجد فرص Long مطابقة حالياً.")

    if shorts:
        print("\n🔴 فرص بيع محتملة (Short Candidates):")
        print(tabulate(shorts, headers=headers, tablefmt="grid"))
    else:
        print("\n🔴 لا توجد فرص Short مطابقة حالياً.")

    print("\n🔁 القاعدة الذهبية السريعة:")
    print("- السعر ينخفض + OI يرتفع = إشارة هبوطية قوية")
    print("- السعر ينخفض بشدة + OI ينخفض بشدة = احتمال انعكاس صعودي")
    print("- السعر يرتفع + OI ينخفض = ضعف في الاتجاه الصاعد")


# ==========================================
# 7. نقطة الدخول الرئيسية
# ==========================================


if __name__ == "__main__":
    try:
        long_signals, short_signals = analyze_market()
        render_report(long_signals, short_signals)
    except KeyboardInterrupt:
        print("\nتم إيقاف البرنامج.")
