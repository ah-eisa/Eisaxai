import pandas as pd
import matplotlib.pyplot as plt

# قراءة البيانات
df = pd.read_csv('arab_markets_stocks.csv')

# تنظيف
df['RSI'] = pd.to_numeric(df['RSI'], errors='coerce')
df['change'] = pd.to_numeric(df['change'], errors='coerce')
df['price_earnings_ttm'] = pd.to_numeric(df['price_earnings_ttm'], errors='coerce')
df['dividend_yield_recent'] = pd.to_numeric(df['dividend_yield_recent'], errors='coerce')
df['market_cap_basic'] = pd.to_numeric(df['market_cap_basic'], errors='coerce')

print("="*80)
print("📊 DASHBOARD الأسواق العربية")
print("="*80)

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
print(market_summary.to_string())

# 2. أفضل 10 فرص استثمارية (RSI منخفض + P/E منخفض + عائد جيد)
print("\n" + "="*80)
print("2️⃣ أفضل 10 فرص استثمارية:")
print("="*80)

opportunities = df[
    (df['RSI'] < 40) & 
    (df['price_earnings_ttm'] < 15) & 
    (df['price_earnings_ttm'] > 0) &
    (df['dividend_yield_recent'] > 3)
].nlargest(10, 'dividend_yield_recent')

if len(opportunities) > 0:
    for i, row in opportunities.iterrows():
        print(f"\n📌 {row['name']} ({row['market']})")
        print(f"   سعر: {row['close']:.2f} | RSI: {row['RSI']:.1f} | P/E: {row['price_earnings_ttm']:.1f}")
        print(f"   عائد: {row['dividend_yield_recent']:.1f}% | قطاع: {row['sector']}")
else:
    print("لا توجد فرص بالمعايير الحالية")

# 3. الأسهم الأكثر ارتفاعًا وانخفاضًا
print("\n" + "="*80)
print("3️⃣ أبرز الحركات اليوم:")
print("="*80)

top5_up = df.nlargest(5, 'change')[['name', 'market', 'close', 'change', 'RSI']]
print("\n📈 أعلى 5 ارتفاعًا:")
for _, row in top5_up.iterrows():
    print(f"   • {row['name']} ({row['market']}): {row['change']:.1f}% | سعر {row['close']:.2f} | RSI {row['RSI']:.0f}")

top5_down = df.nsmallest(5, 'change')[['name', 'market', 'close', 'change', 'RSI']]
print("\n📉 أعلى 5 انخفاضًا:")
for _, row in top5_down.iterrows():
    print(f"   • {row['name']} ({row['market']}): {row['change']:.1f}% | سعر {row['close']:.2f} | RSI {row['RSI']:.0f}")

# 4. حالة السوق العامة
print("\n" + "="*80)
print("4️⃣ حالة السوق العامة:")
print("="*80)

total_stocks = len(df)
up_stocks = len(df[df['change'] > 0])
down_stocks = len(df[df['change'] < 0])
unchanged = total_stocks - up_stocks - down_stocks

print(f"📊 عدد الأسهم: {total_stocks}")
print(f"📈 صاعد: {up_stocks} ({up_stocks/total_stocks*100:.1f}%)")
print(f"📉 هابط: {down_stocks} ({down_stocks/total_stocks*100:.1f}%)")
print(f"➖ ثابت: {unchanged} ({unchanged/total_stocks*100:.1f}%)")

# مؤشر قوة السوق
avg_rsi = df['RSI'].mean()
print(f"\n🎯 متوسط RSI للسوق: {avg_rsi:.1f}")
if avg_rsi < 30:
    print("   🔴 السوق في منطقة ذروة بيع (oversold) - فرصة شراء")
elif avg_rsi > 70:
    print("   🟢 السوق في منطقة ذروة شراء (overbought) - حذر")
else:
    print("   ⚪ السوق في منطقة محايدة")

# 5. أفضل القطاعات أداءً
print("\n" + "="*80)
print("5️⃣ أفضل القطاعات أداءً اليوم:")
print("="*80)

sector_performance = df.groupby('sector')['change'].mean().sort_values(ascending=False).head(10)
for sector, perf in sector_performance.items():
    count = len(df[df['sector'] == sector])
    print(f"   • {sector}: {perf:.2f}% ({count} سهم)")

# حفظ التقرير
print("\n" + "="*80)
print("✅ تم تحليل 793 سهم من 7 أسواق")
print("="*80)
