import pandas as pd

# قراءة البيانات
df = pd.read_csv('arab_markets_complete.csv')

# تنظيف البيانات
df['RSI'] = pd.to_numeric(df['RSI'], errors='coerce')
df['change'] = pd.to_numeric(df['change'], errors='coerce')
df['price_earnings_ttm'] = pd.to_numeric(df['price_earnings_ttm'], errors='coerce')
df['dividend_yield_recent'] = pd.to_numeric(df['dividend_yield_recent'], errors='coerce')
df['market_cap_basic'] = pd.to_numeric(df['market_cap_basic'], errors='coerce')

print("=" * 80)
print("📊 DASHBOARD الأسواق العربية (شامل السعودية)")
print("=" * 80)

# 1. ملخص الأسواق
print("\n1️⃣ ملخص كل سوق:")
market_summary = df.groupby('market').agg({
    'name': 'count',
    'close': 'mean',
    'change': 'mean',
    'RSI': 'mean',
    'market_cap_basic': 'sum'
}).round(2)

market_summary.columns = ['عدد الأسهم', 'متوسط السعر', 'متوسط التغير%', 'متوسط RSI', 'القيمة السوقية (B)']
market_summary['القيمة السوقية (B)'] = market_summary['القيمة السوقية (B)'] / 1e9
market_summary = market_summary.sort_values('القيمة السوقية (B)', ascending=False)
print(market_summary.to_string())

# 2. أفضل 15 فرصة استثمارية
print("\n" + "=" * 80)
print("2️⃣ أفضل 15 فرصة استثمارية (RSI<40, P/E<15, عائد>3%):")
print("=" * 80)

opportunities = df[
    (df['RSI'] < 40) & 
    (df['price_earnings_ttm'] < 15) & 
    (df['price_earnings_ttm'] > 0) &
    (df['dividend_yield_recent'] > 3)
].nlargest(15, 'dividend_yield_recent')

if len(opportunities) > 0:
    for i, (idx, row) in enumerate(opportunities.iterrows(), 1):
        print(f"\n{i}. {row['name']} ({row['market']})")
        print(f"   سعر: {row['close']:.2f} | RSI: {row['RSI']:.1f} | P/E: {row['price_earnings_ttm']:.1f}")
        print(f"   عائد: {row['dividend_yield_recent']:.1f}% | قطاع: {row['sector']}")
else:
    print("   لا توجد فرص بالمعايير الحالية")

# 3. حالة السوق العامة
print("\n" + "=" * 80)
print("3️⃣ حالة السوق العامة:")
print("=" * 80)

total = len(df)
up = len(df[df['change'] > 0])
down = len(df[df['change'] < 0])
flat = total - up - down

print(f"📊 إجمالي الأسهم: {total}")
print(f"📈 صاعد: {up} ({up/total*100:.1f}%)")
print(f"📉 هابط: {down} ({down/total*100:.1f}%)")
print(f"➖ ثابت: {flat} ({flat/total*100:.1f}%)")

avg_rsi = df['RSI'].mean()
print(f"\n🎯 متوسط RSI: {avg_rsi:.1f}")
if avg_rsi < 30:
    print("   🔴 السوق في منطقة ذروة بيع - فرصة شراء")
elif avg_rsi > 70:
    print("   🟢 السوق في منطقة ذروة شراء - حذر")
else:
    print("   ⚪ السوق في منطقة محايدة")

# 4. أفضل الأسهم في كل سوق
print("\n" + "=" * 80)
print("4️⃣ أفضل سهم في كل سوق (حسب المعايير):")
print("=" * 80)

for market in df['market'].unique():
    market_df = df[df['market'] == market]
    
    # حساب درجة الجودة
    market_df = market_df.copy()
    market_df['score'] = (
        (70 - market_df['RSI'].fillna(50)) / 10 +  # RSI منخفض جيد
        (market_df['dividend_yield_recent'].fillna(0) / 5) +  # عوائد عالية جيد
        (15 / market_df['price_earnings_ttm'].fillna(15))  # P/E منخفض جيد
    )
    
    best = market_df.nlargest(1, 'score')
    if len(best) > 0:
        row = best.iloc[0]
        print(f"\n📍 {market}:")
        print(f"   🏆 {row['name']} | سعر {row['close']:.2f} | RSI {row['RSI']:.0f} | P/E {row['price_earnings_ttm']:.1f}")
        print(f"   عائد {row['dividend_yield_recent']:.1f}% | قطاع {row['sector']}")

# 5. إحصائيات سريعة
print("\n" + "=" * 80)
print("5️⃣ إحصائيات سريعة:")
print("=" * 80)

print(f"🏦 أكبر سوق: {market_summary.index[0]} ({market_summary.iloc[0]['القيمة السوقية (B)']:.0f} B AED)")
print(f"📈 أقوى سوق اليوم: {market_summary.sort_values('متوسط التغير%', ascending=False).index[0]} ({market_summary.sort_values('متوسط التغير%', ascending=False).iloc[0]['متوسط التغير%']:.2f}%)")
print(f"🎯 أكثر سوق هدوءًا: {market_summary.sort_values('متوسط RSI').index[0]} (RSI {market_summary.sort_values('متوسط RSI').iloc[0]['متوسط RSI']:.0f})")

print("\n" + "=" * 80)
print("✅ تم التحليل بنجاح!")
print("=" * 80)
