import streamlit as st
import pandas as pd
import plotly.express as px

# إعداد الصفحة
st.set_page_config(page_title="تحليل الأسواق العربية", layout="wide", page_icon="📊")

# العنوان
st.title("📊 تحليل الأسواق العربية")
st.markdown("### بيانات لحظية من TradingView")

# قراءة البيانات
@st.cache_data
def load_data():
    df = pd.read_csv('arab_markets_complete.csv')
    df['RSI'] = pd.to_numeric(df['RSI'], errors='coerce')
    df['change'] = pd.to_numeric(df['change'], errors='coerce')
    df['price_earnings_ttm'] = pd.to_numeric(df['price_earnings_ttm'], errors='coerce')
    df['dividend_yield_recent'] = pd.to_numeric(df['dividend_yield_recent'], errors='coerce')
    df['market_cap_basic'] = pd.to_numeric(df['market_cap_basic'], errors='coerce')
    return df

df = load_data()

# ============================================================
# الشريط الجانبي (فلترة)
# ============================================================
st.sidebar.header("🔍 فلترة البيانات")

# فلتر السوق
markets = df['market'].unique().tolist()
selected_markets = st.sidebar.multiselect("اختر الأسواق", markets, default=markets[:3])

# فلتر RSI
rsi_range = st.sidebar.slider("نطاق RSI", 0, 100, (0, 100))

# فلتر P/E
pe_range = st.sidebar.slider("نطاق P/E", 0, 50, (0, 50))

# فلتر العائد
div_min = st.sidebar.slider("الحد الأدنى لعائد التوزيعات (%)", 0, 20, 0)

# تطبيق الفلاتر
filtered_df = df[
    (df['market'].isin(selected_markets)) &
    (df['RSI'].between(rsi_range[0], rsi_range[1])) &
    (df['price_earnings_ttm'].between(pe_range[0], pe_range[1])) &
    (df['dividend_yield_recent'] >= div_min)
]

# ============================================================
# مؤشرات عامة
# ============================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 إجمالي الأسهم", len(filtered_df))

with col2:
    up = len(filtered_df[filtered_df['change'] > 0])
    st.metric("📈 صاعد", f"{up} ({up/len(filtered_df)*100:.1f}%)" if len(filtered_df) > 0 else "0")

with col3:
    down = len(filtered_df[filtered_df['change'] < 0])
    st.metric("📉 هابط", f"{down} ({down/len(filtered_df)*100:.1f}%)" if len(filtered_df) > 0 else "0")

with col4:
    avg_rsi = filtered_df['RSI'].mean()
    st.metric("🎯 متوسط RSI", f"{avg_rsi:.1f}" if pd.notna(avg_rsi) else "N/A")

# ============================================================
# التبويبات
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 جدول الأسهم", "🎯 فرص استثمارية", "📊 تحليل الأسواق", "🏭 تحليل القطاعات"])

# تبويب 1: جدول الأسهم
with tab1:
    st.subheader("📈 قائمة الأسهم")
    
    display_cols = ['name', 'market', 'close', 'change', 'RSI', 'price_earnings_ttm', 'dividend_yield_recent', 'sector']
    display_df = filtered_df[display_cols].copy()
    
    # تنسيق الأرقام
    display_df['change'] = display_df['change'].round(2)
    display_df['RSI'] = display_df['RSI'].round(1)
    display_df['price_earnings_ttm'] = display_df['price_earnings_ttm'].round(1)
    display_df['dividend_yield_recent'] = display_df['dividend_yield_recent'].round(1)
    
    # إعادة تسمية الأعمدة
    display_df.columns = ['الاسم', 'السوق', 'السعر', 'التغير %', 'RSI', 'P/E', 'عائد %', 'القطاع']
    
    st.dataframe(display_df, use_container_width=True)
    
    # تحميل CSV
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 تحميل البيانات CSV", csv, "stocks_data.csv", "text/csv")

# تبويب 2: فرص استثمارية
with tab2:
    st.subheader("🎯 فرص استثمارية")
    
    # معايير الفرص: RSI منخفض + P/E منخفض + عائد جيد
    opportunities = filtered_df[
        (filtered_df['RSI'] < 40) & 
        (filtered_df['price_earnings_ttm'] < 15) & 
        (filtered_df['price_earnings_ttm'] > 0) &
        (filtered_df['dividend_yield_recent'] > 3)
    ].nlargest(20, 'dividend_yield_recent')
    
    if len(opportunities) > 0:
        for i, (idx, row) in enumerate(opportunities.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {row['name']}** ({row['market']})")
                    st.caption(f"قطاع: {row['sector']}")
                with col2:
                    st.markdown(f"📊 RSI: **{row['RSI']:.0f}**")
                    st.markdown(f"💰 عائد: **{row['dividend_yield_recent']:.1f}%**")
                st.markdown(f"📈 سعر: {row['close']:.2f} | P/E: {row['price_earnings_ttm']:.1f}")
                st.markdown("---")
    else:
        st.info("لا توجد فرص استثمارية بالمعايير الحالية. حاول تعديل الفلاتر.")

# تبويب 3: تحليل الأسواق
with tab3:
    st.subheader("📊 أداء الأسواق")
    
    # رسم بياني: متوسط التغير حسب السوق
    market_perf = filtered_df.groupby('market')['change'].mean().reset_index()
    fig1 = px.bar(market_perf, x='market', y='change', title='متوسط التغير اليومي حسب السوق',
                  color='change', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig1, use_container_width=True)
    
    # رسم بياني: متوسط RSI حسب السوق
    market_rsi = filtered_df.groupby('market')['RSI'].mean().reset_index()
    fig2 = px.bar(market_rsi, x='market', y='RSI', title='متوسط RSI حسب السوق',
                  color='RSI', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig2, use_container_width=True)
    
    # جدول ملخص الأسواق
    summary = filtered_df.groupby('market').agg({
        'name': 'count',
        'close': 'mean',
        'change': 'mean',
        'RSI': 'mean',
        'dividend_yield_recent': 'mean'
    }).round(2)
    summary.columns = ['عدد الأسهم', 'متوسط السعر', 'متوسط التغير%', 'متوسط RSI', 'متوسط العائد%']
    st.dataframe(summary, use_container_width=True)

# تبويب 4: تحليل القطاعات
with tab4:
    st.subheader("🏭 أداء القطاعات")
    
    # أكبر 10 قطاعات
    sector_counts = filtered_df['sector'].value_counts().head(10).reset_index()
    sector_counts.columns = ['القطاع', 'عدد الأسهم']
    
    fig3 = px.bar(sector_counts, x='القطاع', y='عدد الأسهم', title='أكبر 10 قطاعات من حيث عدد الشركات')
    st.plotly_chart(fig3, use_container_width=True)
    
    # أداء القطاعات
    sector_perf = filtered_df.groupby('sector')['change'].mean().sort_values(ascending=False).head(10).reset_index()
    sector_perf.columns = ['القطاع', 'متوسط التغير%']
    
    fig4 = px.bar(sector_perf, x='القطاع', y='متوسط التغير%', title='أفضل 10 قطاعات أداءً',
                  color='متوسط التغير%', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(f"📅 آخر تحديث: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 📊 عدد الأسهم: {len(filtered_df)}")
