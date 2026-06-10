from tradingview_screener import Query
import pandas as pd
from datetime import datetime

# الحقول المشتركة
fields = [
    'name', 'close', 'change', 'volume', 'market_cap_basic',
    'price_earnings_ttm', 'dividend_yield_recent',
    'earnings_per_share_diluted_ttm', 'sector',
    'RSI', 'MACD.macd', 'MACD.signal', 'Stoch.K', 'Stoch.D', 'CCI20', 'AO'
]

# الأسواق مع الكود الصحيح
markets = {
    'uae': 'الإمارات',
    'ksa': 'السعودية',
    'egypt': 'مصر',
    'kuwait': 'الكويت',
    'qatar': 'قطر',
    'bahrain': 'البحرين',
    'morocco': 'المغرب',
    'tunisia': 'تونس',
}

print("=" * 70)
print("🌍 جاري جلب بيانات الأسواق العربية...")
print("=" * 70)

all_data = []

for market_code, market_name in markets.items():
    print(f"\n📊 جاري جلب {market_name}...")
    
    try:
        _, df = (Query()
            .set_markets(market_code)
            .select(*fields)
            .limit(500)
            .get_scanner_data())
        
        df['market'] = market_name
        df['market_code'] = market_code
        all_data.append(df)
        print(f"   ✅ {len(df)} سهم")
        
    except Exception as e:
        print(f"   ❌ خطأ: {e}")

# دمج كل البيانات
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # حفظ
    final_df.to_csv('arab_markets_complete.csv', index=False, encoding='utf-8')
    
    print("\n" + "=" * 70)
    print("📈 ملخص الأسواق:")
    print("=" * 70)
    
    summary = final_df.groupby('market').agg({
        'name': 'count',
        'close': 'mean',
        'change': 'mean',
        'RSI': 'mean',
        'market_cap_basic': 'sum'
    }).round(2)
    
    summary.columns = ['عدد الأسهم', 'متوسط السعر', 'متوسط التغير%', 'متوسط RSI', 'القيمة السوقية (B)']
    summary['القيمة السوقية (B)'] = summary['القيمة السوقية (B)'] / 1e9
    
    print(summary.to_string())
    
    print("\n" + "=" * 70)
    print(f"✅ تم حفظ {len(final_df)} سهم من {len(markets)} سوق")
    print("=" * 70)
    
else:
    print("❌ لم يتم جلب أي بيانات")
