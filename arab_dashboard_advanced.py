import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# إعداد الصفحة
st.set_page_config(
    page_title="تحليل الأسواق العربية",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للتحسين
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stock-card {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s;
    }
    .stock-card:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        background: #f8f9fa;
    }
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: white;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6b6b !important;
        color: white !important;
    }
    .stSlider > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# تحميل البيانات
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('arab_markets_complete.csv')
    df['RSI'] = pd.to_numeric(df['RSI'], errors='coerce')
    df['change'] = pd.to_numeric(df['change'], errors='coerce')
    df['price_earnings_ttm'] = pd.to_numeric(df['price_earnings_ttm'], errors='coerce')
    df['dividend_yield_recent'] = pd.to_numeric(df['dividend_yield_recent'], errors='coerce')
    df['market_cap_basic'] = pd.to_numeric(df['market_cap_basic'], errors='coerce')
    return df

df = load_data()

# العنوان الرئيسي
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📈 تحليل الأسواق العربية")
st.markdown("### 🚀 بيانات لحظية من TradingView | تحديث تلقائي كل 30 دقيقة")
st.markdown('</div>', unsafe_allow_html=True)

# الشريط الجانبي المتطور
with st.sidebar:
    st.markdown("## 🔍 فلتر البيانات")
    st.markdown("---")
    
    # الأسواق مع أيقونات
    markets = df['market'].unique().tolist()
    market_icons = {
        'السعودية': '🇸🇦', 'الإمارات': '🇦🇪', 'مصر': '🇪🇬', 
        'قطر': '🇶🇦', 'الكويت': '🇰🇼', 'المغرب': '🇲🇦',
        'تونس': '🇹🇳', 'البحرين': '🇧🇭'
    }
    
    market_labels = [f"{market_icons.get(m, '🏦')} {m}" for m in markets]
    selected_markets = st.multiselect("اختر الأسواق", markets, default=markets[:3], format_func=lambda x: f"{market_icons.get(x, '🏦')} {x}")
    
    st.markdown("---")
    
    # الفلاتر المتقدمة
    col1, col2 = st.columns(2)
    with col1:
        rsi_min = st.slider("RSI Min", 0, 100, 0)
    with col2:
        rsi_max = st.slider("RSI Max", 0, 100, 70)
    
    col1, col2 = st.columns(2)
    with col1:
        pe_min = st.slider("P/E Min", 0, 100, 0)
    with col2:
        pe_max = st.slider("P/E Max", 0, 100, 30)
    
    div_min = st.slider("الحد الأدنى لعائد التوزيعات (%)", 0.0, 20.0, 0.0, step=0.5)
    
    st.markdown("---")
    
    # فلتر RSI حالة
    st.markdown("### 🎯 حالة RSI")
    rsi_status = st.radio("اختر الحالة", ["الكل", "ذروة بيع (RSI < 30)", "ذروة شراء (RSI > 70)", "محايد"])
    
    st.markdown("---")
    st.caption(f"📊 إجمالي الأسهم المتاحة: {len(df)}")
    st.caption(f"🕐 آخر تحديث: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

# تطبيق الفلاتر
filtered_df = df[
    (df['market'].isin(selected_markets)) &
    (df['RSI'].between(rsi_min, rsi_max)) &
    (df['price_earnings_ttm'].between(pe_min, pe_max)) &
    (df['dividend_yield_recent'] >= div_min)
]

# فلتر RSI status
if rsi_status == "ذروة بيع (RSI < 30)":
    filtered_df = filtered_df[filtered_df['RSI'] < 30]
elif rsi_status == "ذروة شراء (RSI > 70)":
    filtered_df = filtered_df[filtered_df['RSI'] > 70]
elif rsi_status == "محايد":
    filtered_df = filtered_df[(filtered_df['RSI'] >= 30) & (filtered_df['RSI'] <= 70)]

# مؤشرات رئيسية متطورة
st.markdown("### 📊 نظرة عامة")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📊 إجمالي الأسهم", len(filtered_df))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    up = len(filtered_df[filtered_df['change'] > 0])
    up_pct = (up/len(filtered_df)*100) if len(filtered_df) > 0 else 0
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📈 صاعد", f"{up} ({up_pct:.1f}%)", delta=f"+{up_pct:.1f}%" if up_pct > 0 else None)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    down = len(filtered_df[filtered_df['change'] < 0])
    down_pct = (down/len(filtered_df)*100) if len(filtered_df) > 0 else 0
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("📉 هابط", f"{down} ({down_pct:.1f}%)", delta=f"-{down_pct:.1f}%" if down_pct > 0 else None)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    avg_rsi = filtered_df['RSI'].mean()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🎯 متوسط RSI", f"{avg_rsi:.1f}" if pd.notna(avg_rsi) else "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

# تبويبات متطورة
tab1, tab2, tab3, tab4 = st.tabs(["📈 جدول الأسهم المتقدم", "🎯 فرص استثمارية ذكية", "📊 تحليل الأسواق المتقدم", "🏭 تحليل القطاعات المتقدم"])

# تبويب 1: جدول الأسهم المتقدم
with tab1:
    st.markdown("### 📈 قائمة الأسهم")
    
    display_cols = ['name', 'market', 'close', 'change', 'RSI', 'price_earnings_ttm', 'dividend_yield_recent', 'sector', 'market_cap_basic']
    display_df = filtered_df[display_cols].copy()
    
    # إضافة أعمدة ملونة
    display_df['change_color'] = display_df['change'].apply(lambda x: f'<span style="color: {"green" if x > 0 else "red"}">{x:.2f}%</span>')
    display_df['RSI_color'] = display_df['RSI'].apply(lambda x: f'<span style="color: {"red" if x < 30 else "green" if x > 70 else "orange"}">{x:.1f}</span>')
    
    # تنسيق الأرقام
    display_df['close'] = display_df['close'].round(2)
    display_df['price_earnings_ttm'] = display_df['price_earnings_ttm'].round(1)
    display_df['dividend_yield_recent'] = display_df['dividend_yield_recent'].round(1)
    display_df['market_cap_basic'] = (display_df['market_cap_basic'] / 1e9).round(2)
    
    display_df.columns = ['الاسم', 'السوق', 'السعر', 'التغير %', 'RSI', 'P/E', 'عائد %', 'القطاع', 'القيمة السوقية (B)']
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # تنزيل CSV
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 تحميل البيانات CSV", csv, "stocks_data.csv", "text/csv")

# تبويب 2: فرص استثمارية ذكية
with tab2:
    st.markdown("### 🎯 الفرص الاستثمارية المميزة")
    
    # معايير متعددة
    opportunities_df = filtered_df.copy()
    opportunities_df['score'] = (
        (70 - opportunities_df['RSI'].fillna(50)) / 10 +
        (opportunities_df['dividend_yield_recent'].fillna(0) / 5) +
        (15 / opportunities_df['price_earnings_ttm'].fillna(15))
    )
    
    opportunities = opportunities_df.nlargest(10, 'score')
    
    if len(opportunities) > 0:
        for i, (idx, row) in enumerate(opportunities.iterrows(), 1):
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**{i}. {row['name']}**")
                    st.caption(f"📍 {row['market']} | 🏭 {row['sector']}")
                with col2:
                    rsi_color = "🟢" if row['RSI'] > 70 else "🔴" if row['RSI'] < 30 else "🟡"
                    st.markdown(f"{rsi_color} **RSI:** {row['RSI']:.0f}")
                    st.markdown(f"💰 **P/E:** {row['price_earnings_ttm']:.1f}")
                with col3:
                    st.markdown(f"**عائد:** {row['dividend_yield_recent']:.1f}%")
                    st.markdown(f"**سعر:** {row['close']:.2f}")
                st.progress(min(100, int(row['score'] * 10)), text=f"النتيجة: {row['score']:.1f}/10")
                st.markdown("---")
    else:
        st.info("لا توجد فرص استثمارية بالمعايير الحالية. جرب تعديل الفلاتر.")

# تبويب 3: تحليل الأسواق المتقدم
with tab3:
    st.markdown("### 📊 أداء الأسواق")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # رسم بياني للتغيرات
        market_perf = filtered_df.groupby('market')['change'].mean().reset_index()
        fig1 = px.bar(market_perf, x='market', y='change', 
                      title='متوسط التغير اليومي حسب السوق',
                      color='change', color_continuous_scale='RdYlGn',
                      text='change')
        fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # رسم بياني لـ RSI
        market_rsi = filtered_df.groupby('market')['RSI'].mean().reset_index()
        fig2 = px.bar(market_rsi, x='market', y='RSI', 
                      title='متوسط RSI حسب السوق',
                      color='RSI', color_continuous_scale='RdYlGn')
        fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="ذروة بيع")
        fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="ذروة شراء")
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Heatmap للمؤشرات
    st.markdown("### 🔥 Heatmap المؤشرات حسب السوق")
    heatmap_data = filtered_df.groupby('market')[['change', 'RSI', 'price_earnings_ttm', 'dividend_yield_recent']].mean().reset_index()
    fig3 = px.imshow(heatmap_data.set_index('market').T, 
                     text_auto=True, aspect="auto",
                     title="Heatmap المؤشرات حسب السوق",
                     color_continuous_scale='RdYlGn')
    st.plotly_chart(fig3, use_container_width=True)

# تبويب 4: تحليل القطاعات المتقدم
with tab4:
    st.markdown("### 🏭 تحليل القطاعات")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart للقطاعات
        sector_counts = filtered_df['sector'].value_counts().head(8).reset_index()
        sector_counts.columns = ['القطاع', 'عدد الأسهم']
        fig4 = px.pie(sector_counts, values='عدد الأسهم', names='القطاع', 
                      title='توزيع الأسهم حسب القطاع')
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        # أداء القطاعات
        sector_perf = filtered_df.groupby('sector')['change'].mean().sort_values(ascending=False).head(10).reset_index()
        sector_perf.columns = ['القطاع', 'متوسط التغير%']
        fig5 = px.bar(sector_perf, x='القطاع', y='متوسط التغير%',
                      title='أفضل القطاعات أداءً',
                      color='متوسط التغير%', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig5, use_container_width=True)
    
    # جدول القطاعات
    st.markdown("### 📋 تفاصيل القطاعات")
    sector_details = filtered_df.groupby('sector').agg({
        'name': 'count',
        'close': 'mean',
        'change': 'mean',
        'RSI': 'mean',
        'dividend_yield_recent': 'mean',
        'price_earnings_ttm': 'mean'
    }).round(2)
    sector_details.columns = ['عدد الأسهم', 'متوسط السعر', 'متوسط التغير%', 'متوسط RSI', 'متوسط العائد%', 'متوسط P/E']
    st.dataframe(sector_details, use_container_width=True)

# Footer متطور
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📅 آخر تحديث: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.caption(f"📊 عدد الأسهم المعروضة: {len(filtered_df)}")
with col3:
    st.caption("🚀 Powered by TradingView | Streamlit")
