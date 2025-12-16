أنت مهندس برمجيات تداول خوارزمي. لدي ملف Python لاستراتيجية تداول (Simple Market Metrics v4.1) تقوم بتوليد إشارات شراء/بيع من داخل الكود عبر compute_smm() وتحديدًا مخرجات مثل buy_signal و sell_signal مع فلاتر/شروط الاستراتيجية الموجودة (مثل Chop/MFI وHeikin Ashi وProfit Targets). المطلوب: بناء نظام تداول آلي تجريبي فقط مرتبط بعقود Binance USDT-M Futures باستخدام CCXT، مع الالتزام التام بمنطق الإشارات الموجود في الكود دون تعديل أو اختراع شروط جديدة.

1) شرط التشغيل التجريبي (إلزامي)

النظام يجب أن يعمل في وضع تجريبي فقط:

إما عبر Binance Futures Testnet إن أمكن،

أو عبر Paper Trading (محاكاة داخلية) إذا كان testnet غير متاح.


ممنوع تنفيذ صفقات حقيقية على الحساب الحقيقي. اجعل هناك متغير إعداد واضح مثل TRADING_MODE = "TESTNET" أو "PAPER"، ووضّح كيف يتم تفعيله.

في وضع PAPER: نفّذ الصفقات افتراضيًا كسجل محاكاة (Virtual Orders) مع احتساب PnL بناءً على الأسعار (الـ OHLCV/آخر سعر)، دون إرسال أوامر حقيقية للبورصة.


2) مصدر الإشارات (إلزامي)

اعتمد فقط على إشارات الاستراتيجية الناتجة من الكود الحالي (مخرجات compute_smm())، خصوصًا:

دخول Long عند buy_signal == True

دخول Short عند sell_signal == True


لا تغيّر شروط الإشارة أو تسمياتها أو منطقها.


3) إدارة المخاطر وإعدادات الصفقة (حسب المطلوب)

الرافعة المالية ثابتة: 20x

مبلغ الدخول لكل صفقة: 20% من الرصيد المتاح (Free Balance) بالـ USDT

طبّق إدارة المخاطر الموجودة في الكود كما هي.

إذا لم يوجد Stop Loss صريح في الكود، اذكر ذلك بوضوح ولا تضف Stop Loss افتراضيًا.

يمكن إضافة Stop Loss كخيار “اختياري” فقط (disabled by default) مع توضيح ذلك.



4) الخروج من الصفقة

الإغلاق يتم وفق ما هو موجود في الاستراتيجية:

عند تحقق profit_target_hit (إن كان جزءًا من مخرجات الكود)

أو عند ظهور إشارة معاكسة (إن كان ذلك مطلوبًا ضمن النظام) — اجعل هذه القاعدة قابلة للتفعيل عبر إعداد: CLOSE_ON_OPPOSITE_SIGNAL = True/False


اشرح بوضوح سبب الإغلاق في سجل الصفقات.


5) الربط بـ Binance Futures عبر CCXT (تنفيذ تجريبي)

استخدم CCXT مع binanceusdm.

افصل النظام إلى طبقات واضحة:

1. Signal Engine: جلب بيانات OHLCV + تشغيل compute_smm() + إنتاج إشارات.


2. Execution Engine: (Testnet/Paper) إدارة أوامر الدخول/الخروج + الرافعة + حساب الكمية.


3. Analytics: سجل الصفقات + إحصائيات الأداء.



راعِ rate limits والأخطاء (Network / Timeout / Insufficient margin).


6) حساب حجم الصفقة (Position Sizing)

اكتب دالة واضحة لحساب الكمية qty:

leverage=20

risk_fraction=0.20

تعتمد على free_balance وسعر الدخول الحالي

تراعي precision/stepSize/minQty من exchange.market(symbol) أو ما يعادله في CCXT


امنع رفض الأوامر بسبب القيود.


7) سجل الصفقات + الأداء (إلزامي)

سجّل كل صفقة منفّذة (حتى في PAPER) في Trade Log يحتوي على:

symbol, side, entry_time, entry_price, qty

exit_time, exit_price

pnl_usdt, pnl_percent

fees (إن أمكن تقديرها أو اجعلها 0 في paper مع خيار إدخالها)

close_reason: (profit_target / opposite_signal / manual / error)


ثم احسب تقرير شامل:

total_trades

win_rate

avg_win, avg_loss

total_pnl

profit_factor

max_drawdown (من equity curve)


8) مخرجاتك المطلوبة (تنسيق إلزامي)

أخرج النتيجة بهذا الترتيب: (أ) قائمة متطلبات تنفيذية مختصرة
(ب) تصميم معماري (Modules + Data Flow)
(ج) Function Signatures الأساسية
(د) كود Python skeleton قابل للتوسعة يوضح:

إعداد CCXT (testnet/paper)

جلب OHLCV

تشغيل compute_smm()

اتخاذ القرار

تنفيذ تجريبي (testnet أو paper)

تسجيل الصفقات والتقرير (هـ) Assumptions & Limitations
