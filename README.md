# Simple Market Metrics Trading System (Experimental)

## (أ) قائمة متطلبات تنفيذية مختصرة
- التشغيل التجريبي فقط عبر `TRADING_MODE` (`TESTNET` مع `ccxt` Sandbox أو `PAPER` للمحاكاة الداخلية). لا يسمح بصفقات حقيقية.
- الاستراتيجية تعتمد فقط على مخرجات `compute_smm()` الحالية: `buy_signal` للدخول Long و`sell_signal` للدخول Short.
- الرافعة ثابتة 20x وحجم الصفقة = 20% من الرصيد المتاح مضروبًا في الرافعة، مع مراعاة دقة وحدود السوق من بيانات `ccxt`.
- الخروج عند `profit_target_hit` أو إشارة معاكسة إذا كان `CLOSE_ON_OPPOSITE_SIGNAL=True`. لا يوجد إيقاف خسارة صريح؛ خيار `STOP_LOSS_ENABLED` متاح لكنه معطل افتراضيًا.
- تسجيل كل صفقة (حتى في المحاكاة) مع تقرير أداء شامل (win rate، profit factor، max drawdown، ...).

## (ب) تصميم معماري (Modules + Data Flow)
- **Signal Engine** (`SignalEngine`): يجلب OHLCV من `binanceusdm`، يشغّل `compute_smm()` لإنتاج الإشارات والأحداث.
- **Execution Engine** (`ExecutionEngine`): يحسب الكمية باستخدام الرافعة/الرصيد وحدود السوق، يفتح/يغلق المراكز في وضع `TESTNET` أو يحاكي أوامر `PAPER`، ويحافظ على سجل الصفقات.
- **Analytics** (`Analytics`): يبني منحنى رأس المال من سجل الصفقات ويحسب المقاييس (win rate، avg_win/loss، profit factor، max drawdown، ...).
- **Orchestrator** (`run_once`): يمر على قائمة الرموز، يجلب البيانات، يشغّل الإشارات، ثم ينفّذ قرارات الدخول/الخروج بالاعتماد على `evaluate_signals`.

## (ج) Function Signatures الأساسية
- `SignalEngine.fetch_ohlcv() -> pd.DataFrame`
- `SignalEngine.run_signals(df) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]`
- `ExecutionEngine.compute_order_size(symbol, price) -> float`
- `ExecutionEngine.open_position(symbol, side, price, timestamp)`
- `ExecutionEngine.close_position(symbol, price, timestamp, reason)`
- `Analytics.summary() -> PerformanceReport`
- `evaluate_signals(symbol, df_out, executor) -> None`
- `run_once(symbols: List[str]) -> None`

## (د) كود Python skeleton قابل للتوسعة
يوجد في `smm_trading_system.py`:
- إعداد CCXT لـ `binanceusdm` مع `set_sandbox_mode(True)` في وضع `TESTNET`، أو مسار محاكاة كامل في `PAPER`.
- جلب OHLCV (1m، 600 شمعة، مع إسقاط آخر شمعة غير مكتملة إذا لزم).
- تشغيل `compute_smm()` مباشرة من ملف الاستراتيجية الحالي (دون تعديل المنطق).
- اتخاذ القرار: دخول Long عند `buy_signal`، دخول Short عند `sell_signal`؛ الإغلاق عند `profit_target_hit` أو إشارة معاكسة (اختياري) أو إيقاف خسارة اختياري.
- تنفيذ تجريبي: إرسال أوامر سوق إلى Testnet أو تسجيل أوامر افتراضية في `PAPER` مع حساب PnL.
- تسجيل الصفقات وبناء تقرير الأداء عبر `Analytics`.

## (هـ) Assumptions & Limitations
- لا يوجد Stop Loss صريح في منطق الاستراتيجية؛ تم ترك خيار `STOP_LOSS_ENABLED` معطلًا افتراضيًا حتى لا يغير السلوك الأصلي.
- النظام أحادي الدورة `run_once` لأغراض العرض؛ يمكن تغليفه في حلقة مجدولة للإنتاج التجريبي.
- الترصيد في وضع `PAPER` افتراضي إلى 1000 USDT لأغراض المثال؛ يوصى بربطه بمصدر رصيد محفوظ للمحاكاة الدقيقة.
- أي فشل شبكي/قيود rate-limit من `ccxt` يجب التعامل معه عند التشغيل الفعلي (السجل يلتقط الاستثناءات الحالية فقط).
