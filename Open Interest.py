import ccxt
import pandas as pd
import time
from datetime import datetime
from tabulate import tabulate # لتنسيق الجدول بشكل جميل

# ==========================================
# 1. إعدادات الاستراتيجية (Config)
# ==========================================
TIMEFRAME = '15m'       # الإطار الزمني (ساعة واحدة)
LIMIT_COINS = 500       # عدد العملات التي سيتم فحصها (الأعلى سيولة)
Looking_Back = 2       # عدد الشموع للعودة للوراء لحساب التغير

# عتبات الاستراتيجية (Thresholds)
# لتحديد الـ Sucker Pattern (تراكم المراكز الخاسرة)
BEARISH_OI_INCREASE = 3.0    # زيادة في OI أكثر من 3%
BEARISH_PRICE_DROP = -0.5    # انخفاض طفيف أو تماسك (أقل من 0.5% هبوط)
BEARISH_PRICE_MAX_DROP = -2.5 # ألا يكون انهياراً كاملاً بعد

# لتحديد الـ Liquidation Cascade (فرصة شراء ارتداد)
BULLISH_PRICE_DROP = -3.5    # انخفاض حاد في السعر أكثر من 3.5%
BULLISH_OI_DROP = -3.5       # انخفاض حاد في OI أكثر من 3.5%

# ==========================================
# 2. تهيئة الاتصال بالمنصة
# ==========================================
print("🔄 جاري الاتصال بمنصة Binance Futures...")
exchange = ccxt.binanceusdm({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# ==========================================
# 3. الدوال المساعدة (Helper Functions)
# ==========================================

def get_top_symbols(limit):
    """جلب أعلى العملات من حيث حجم التداول"""
    try:
        tickers = exchange.fetch_tickers()
        # ترتيب العملات حسب حجم التداول (Quote Volume)
        sorted_tickers = sorted(tickers.items(), key=lambda item: item[1]['quoteVolume'], reverse=True)
        # تصفية الرموز لتشمل فقط USDT
        symbols = [symbol for symbol, data in sorted_tickers if '/USDT' in symbol]
        return symbols[:limit]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def get_data(symbol):
    """جلب بيانات السعر و OI التاريخية"""
    try:
        # جلب الشموع (OHLCV)
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=Looking_Back+1)
        if len(ohlcv) < 2: return None
        
        # جلب تاريخ Open Interest
        # ملاحظة: هذه الدالة قد تختلف استجابتها حسب قيود المنصة
        oi_history = exchange.fetch_open_interest_history(symbol, TIMEFRAME, limit=Looking_Back+1)
        if len(oi_history) < 2: return None

        # استخراج القيم الحالية والسابقة
        current_price = ohlcv[-1][4] # Close price
        prev_price = ohlcv[-2][4]
        
        current_oi = float(oi_history[-1]['openInterestAmount'])
        prev_oi = float(oi_history[-2]['openInterestAmount'])

        # حساب نسبة التغير
        price_change_pct = ((current_price - prev_price) / prev_price) * 100
        oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100

        return {
            'symbol': symbol,
            'price': current_price,
            'price_chg': round(price_change_pct, 2),
            'oi_chg': round(oi_change_pct, 2)
        }
    except Exception as e:
        # بعض العملات قد لا توفر بيانات OI تاريخية بسهولة
        return None

def analyze_market():
    print(f"🔎 جاري فحص أفضل {LIMIT_COINS} عملة رقمية... (قد يستغرق وقتاً)")
    symbols = get_top_symbols(LIMIT_COINS)
    
    opportunities = []

    for i, symbol in enumerate(symbols):
        # طباعة مؤشر تقدم بسيط
        print(f"[{i+1}/{LIMIT_COINS}] فحص {symbol}...", end="\r")
        
        data = get_data(symbol)
        if not data: continue

        signal = "NEUTRAL"
        rationale = "-"
        
        p_chg = data['price_chg']
        oi_chg = data['oi_chg']

        # ---------------------------------------------------------
        # تطبيق القواعد الاستراتيجية (Logic Application)
        # ---------------------------------------------------------

        # 1. استراتيجية المصيدة الهبوطية (Bearish Trap)
        # السعر يتماسك أو يهبط ببطء + زيادة كبيرة في OI
        if (BEARISH_PRICE_MAX_DROP < p_chg < 0.5) and (oi_chg > BEARISH_OI_INCREASE):
            signal = "🔴 SHORT (Trapped Longs)"
            rationale = "Price Flat/Down + OI Spiking"

        # 2. استراتيجية الانعكاس الصعودي (Bullish Capitulation)
        # انهيار سعري حاد + خروج جماعي (انخفاض OI)
        elif (p_chg < BULLISH_PRICE_DROP) and (oi_chg < BULLISH_OI_DROP):
            signal = "🟢 LONG (Reversal)"
            rationale = "Capitulation: Price & OI Dump"

        # إضافة العملة للقائمة إذا كان هناك إشارة
        if signal != "NEUTRAL":
            opportunities.append([
                symbol, 
                data['price'], 
                f"{p_chg}%", 
                f"{oi_chg}%", 
                signal,
                rationale
            ])
            
        # تأخير بسيط لتجنب الحظر (Rate Limit)
        time.sleep(0.1)

    print("\n" + "="*60)
    print(f"📊 تقرير التحليل - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    if opportunities:
        headers = ["Symbol", "Price", "Price %", "OI %", "Signal", "Reason"]
        print(tabulate(opportunities, headers=headers, tablefmt="grid"))
    else:
        print("⚠️ لم يتم العثور على فرص تطابق الشروط بدقة حالياً.")
        print("جرب توسيع نطاق البحث أو تغيير الإطار الزمني.")

# ==========================================
# تشغيل البرنامج
# ==========================================
if __name__ == "__main__":
    try:
        analyze_market()
    except KeyboardInterrupt:
        print("\nتم إيقاف البرنامج.")
