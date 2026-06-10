import pandas as pd
import numpy as np

# قراءة البيانات
df = pd.read_csv('arab_markets_stocks.csv')

# تنظيف البيانات
df['RSI'] = pd.to_numeric(df['RSI'], errors='coerce')
df['change'] = pd.to_numeric(df['change'], errors='coerce')
df['price_earnings_ttm'] = pd.to_numeric(df['price_earnings_ttm'], errors='coerce')
df['dividend_yield_recent'] = pd.to_numeric(df['dividend_yield_recent'], errors='coerce')
df['market_cap_basic'] = pd.to_numeric(df['market_cap_basic'], errors='coerce')

print("=" * 80)
print("📊 تحليل الأسواق العربية")
print("=" * 80)

# 1. ملخص كل سوق
print("\n📈 1. أداء الأسواق اليوم:")
market_summary = df.groupby('market').agg({
    'name': 'count',
    'close': 'mean',
    'change': 'mean',
    'RSI': 'mean',
    'market_cap_basic': 'sum'
}).round(2)

market_summary.columns = ['عدد الأسهم', 'متوسط السعر', 'متوسط التغير%', 'متوسط RSI', 'إجمالي القيمة (B AED)']
market_summary['إجمالي القيمة (B AED)'] = market_summary['إجمالي القيمة (B AED)'] / 1e9
market_summary = market_summary.sort_values('إجمالي القيمة (B AED)', ascending=False)

print(market_summary.to_string())

# 2. أكثر 10 أسهم ارتفاعًا في كل الأسواق
print("\n📈 2. أكثر 10 أسهم ارتفاعًا اليوم:")
top_gainers = df.nlargest(10, 'change')[['name', 'market', 'close', 'change', 'RSI', 'sector']]
print(top_gainers.to_string(index=False))

# 3. أكثر 10 أسهم انخفاضًا
print("\n📉 3. أكثر 10 أسهم انخفاضًا اليوم:")
top_losers = df.nsmallest(10, 'change')[['name', 'market', 'close', 'change', 'RSI', 'sector']]
print(top_losers.to_string(index=False))

# 4. فرص استثمارية (RSI منخفض + P/E منخفض)
print("\n💰 4. فرص استثمارية (RSI < 35 و P/E < 15):")
opportunities = df[(df['RSI'] < 35) & (df['price_earnings_ttm'] < 15) & (df['price_earnings_ttm'] > 0)]
opportunities = opportunities.nlargest(15, 'market_cap_basic')[['name', 'market', 'close', 'RSI', 'price_earnings_ttm', 'dividend_yield_recent', 'sector']]

if len(opportunities) > 0:
    print(opportunities.to_string(index=False))
else:
    print("   لا توجد فرص بهذه المعايير حاليًا")

# 5. أسهم بعوائد توزيعات عالية
print("\n💸 5. أعلى 15 عائد توزيعات:")
top_div = df.nlargest(15, 'dividend_yield_recent')[['name', 'market', 'close', 'dividend_yield_recent', 'price_earnings_ttm', 'sector']]
print(top_div.to_string(index=False))

# 6. حالة كل سوق (RSI)
print("\n🎯 6. حالة الأسواق حسب RSI:")
def market_status(rsi):
    if rsi < 30:
        return "🔴 ذروة بيع (oversold)"
    elif rsi > 70:
        return "🟢 ذروة شراء (overbought)"
    elif rsi < 40:
        return "🟡 منخفض"
    elif rsi > 60:
        return "🟠 مرتفع"
    else:
        return "⚪ محايد"

for market in df['market'].unique():
    market_rsi = df[df['market'] == market]['RSI'].mean()
    print(f"   {market}: {market_rsi:.1f} → {market_status(market_rsi)}")

# 7. القطاعات الأكثر انتشارًا
print("\n🏭 7. أكبر 10 قطاعات من حيث عدد الشركات:")
top_sectors = df['sector'].value_counts().head(10)
for sector, count in top_sectors.items():
    pct = (count / len(df)) * 100
    print(f"   • {sector}: {count} سهم ({pct:.1f}%)")

# حفظ التحليل
df.to_csv('arab_markets_analyzed.csv', index=False)
print("\n" + "=" * 80)
print(f"✅ تم حفظ {len(df)} سهم في: arab_markets_analyzed.csv")
print("=" * 80)
