"""
EisaX Arab Markets Dashboard  v5.1 - Premium Bilingual Edition
====================================
Data source: pipeline cache (parquet) — 15-min refresh via systemd service
Changes from v4.0:
  - Fixed missing translation keys ('min', 'max', all chart titles)
  - Fixed 🇸🇪 → 🌐 on language toggle
  - Fixed bare imports in portfolio tab (try/except guards)
  - Added AI Assistant tab (DeepSeek-powered NL queries)
  - Added dark mode toggle
  - Added Watchlist tab with persistent session storage
  - Added RSI vs P/E scatter (value screen chart)
  - Improved opportunity scoring (volume, SMA, volatility)
  - Cache health indicator in sidebar
  - RSI/P/E thresholds extracted as named constants
  - All chart titles now use t() translations
  - @st.cache_data on CSS builder (pure function)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import sqlite3 as _sl
from pathlib import Path
from zoneinfo import ZoneInfo
from config import DEEPSEEK_API_KEY, DEFAULT_MODEL

_WL_DB = Path('/home/ubuntu/investwise/data/watchlists.db')
_WL_DB.parent.mkdir(parents=True, exist_ok=True)


def _wl_init():
    with _sl.connect(str(_WL_DB)) as c:
        c.execute('CREATE TABLE IF NOT EXISTS watchlist (user_id TEXT, ticker TEXT, PRIMARY KEY(user_id, ticker))')


def _wl_load(user_id='default'):
    _wl_init()
    with _sl.connect(str(_WL_DB)) as c:
        return [r[0] for r in c.execute('SELECT ticker FROM watchlist WHERE user_id=? ORDER BY rowid', (user_id,)).fetchall()]


def _wl_add(ticker, user_id='default'):
    _wl_init()
    with _sl.connect(str(_WL_DB)) as c:
        c.execute('INSERT OR IGNORE INTO watchlist(user_id,ticker) VALUES(?,?)', (user_id, ticker))


def _wl_remove(ticker, user_id='default'):
    _wl_init()
    with _sl.connect(str(_WL_DB)) as c:
        c.execute('DELETE FROM watchlist WHERE user_id=? AND ticker=?', (user_id, ticker))

# ── Constants ────────────────────────────────────────────────────────────────
RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
CACHE_STALE_MINUTES = 30
DEEPSEEK_MODEL = DEFAULT_MODEL or "deepseek-chat"

PIE_COLORS       = ["#0f4c81","#0ea5a4","#3b82f6","#8b5cf6","#f59e0b","#10b981","#f97316","#ef4444"]
COLOR_SCALE_CHANGE = [(0.0,"#ef4444"),(0.5,"#f8fafc"),(1.0,"#10b981")]
COLOR_SCALE_RSI    = [(0.0,"#10b981"),(0.5,"#f59e0b"),(1.0,"#ef4444")]
DUBAI_TZ = ZoneInfo("Asia/Dubai")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EisaX | الأسواق العربية | Arab Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("language", "ar"),
    ("dark_mode", False),
    ("watchlist", _wl_load()),
    ("_wl_loaded", True),
    ("ai_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Translations ──────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "ar": {
        # Header
        "title": "EisaX | تحليل الأسواق العربية",
        "subtitle": "بيانات مباشرة | تحديث كل 15 دقيقة",
        "version": "الإصدار 5.1",
        # Sidebar
        "filters": "الفلاتر",
        "markets": "الأسواق",
        "select_markets": "اختر الأسواق",
        "rsi_range": "نطاق RSI",
        "pe_range": "نطاق P/E",
        "min_dividend": "الحد الأدنى للتوزيعات",
        "rsi_condition": "حالة RSI",
        "all": "الكل",
        "oversold": "ذروة بيع (RSI < 30)",
        "overbought": "ذروة شراء (RSI > 70)",
        "neutral": "محايد (30-70)",
        "refresh": "تحديث البيانات",
        "total_stocks": "إجمالي الأسهم",
        "last_update": "آخر تحديث",
        "min": "الحد الأدنى",
        "max": "الحد الأقصى",
        "cache_health": "صحة الكاش",
        "stale": "قديم",
        "fresh": "محدّث",
        # KPIs
        "overview": "نظرة عامة",
        "stocks_count": "الأسهم",
        "advancers": "صاعد",
        "decliners": "هابط",
        "avg_rsi": "متوسط RSI",
        "avg_change": "متوسط التغير",
        # Tabs
        "stocks_tab": "قائمة الأسهم",
        "opportunities_tab": "الفرص الاستثمارية",
        "analysis_tab": "تحليل الأسواق",
        "sectors_tab": "تحليل القطاعات",
        "portfolio_tab": "المحفظة",
        "watchlist_tab": "قائمة المراقبة",
        "ai_tab": "المساعد الذكي",
        "commodities_tab": "السلع",
        # Table columns
        "name": "الاسم",
        "market": "السوق",
        "price": "السعر",
        "change": "التغير",
        "sector": "القطاع",
        "pe": "P/E",
        "dividend": "عائد",
        "cap": "القيمة السوقية",
        "volume": "الحجم",
        # Opportunities
        "opp_title": "أفضل 10 فرص استثمارية",
        "opp_subtitle": "بناءً على تحليل RSI، عوائد التوزيعات، نسبة P/E، والزخم",
        "score": "النقاط",
        "trend_up": "اتجاه صاعد",
        "trend_down": "اتجاه هابط",
        "trend_mixed": "مختلط",
        # Charts
        "chart_avg_change": "متوسط التغير حسب السوق",
        "chart_avg_rsi": "متوسط RSI حسب السوق",
        "chart_top_gainers_losers": "أكثر الأسهم ارتفاعاً وانخفاضاً",
        "chart_sector_dist": "توزيع القطاعات",
        "chart_sector_perf": "أفضل القطاعات أداءً",
        "chart_rsi_heatmap": "خريطة حرارة RSI: القطاعات × الأسواق",
        "chart_rsi_pe_scatter": "مصفوفة القيمة: RSI مقابل P/E",
        # Portfolio
        "portfolio_title": "بناء وتحليل المحفظة",
        "portfolio_desc": "أدخل أسهم محفظتك لتحليل الأداء والمخاطر",
        "holdings": "المقتنيات",
        "format_example": "الصيغة: السوق:السهم الكمية سعر_الشراء",
        "example": "مثال: uae:EMAAR 1000 14.5",
        "analyze": "تحليل المحفظة",
        "total_value": "القيمة الإجمالية",
        "unrealized_pnl": "أرباح/خسائر غير محققة",
        "risk_score": "مستوى المخاطرة",
        "diversification": "التنويع",
        "low_risk": "منخفض",
        "medium_risk": "متوسط",
        "high_risk": "مرتفع",
        # Watchlist
        "watchlist_title": "قائمة المراقبة",
        "watchlist_empty": "قائمة المراقبة فارغة. أضف أسهماً من تبويب الأسهم.",
        "add_to_watchlist": "إضافة إلى المراقبة",
        "remove": "حذف",
        "watchlist_input": "أدخل رمز السهم (مثال: EMAAR)",
        "add": "إضافة",
        "commodities_unavailable": "بيانات السلع غير متاحة مؤقتاً",
        "one_day_change": "تغير يوم واحد",
        # AI
        "ai_title": "المساعد الذكي EisaX",
        "ai_desc": "اسأل أي سؤال عن السوق بالعربية أو الإنجليزية",
        "ai_placeholder": "مثال: ما هي أفضل الأسهم بعائد توزيعات فوق 5%؟",
        "ai_send": "إرسال",
        "ai_thinking": "جاري التحليل...",
        "ai_clear": "مسح المحادثة",
        "ai_context_label": "بيانات السوق المرفقة",
        # Common
        "loading": "جاري التحميل...",
        "error": "خطأ",
        "success": "تم بنجاح",
        "warning": "تنبيه",
        "no_data": "لا توجد بيانات",
        "download": "تحميل",
    },
    "en": {
        # Header
        "title": "EisaX | Arab Markets Analysis",
        "subtitle": "Live Data | Auto-refresh every 15 min",
        "version": "Version 5.1",
        # Sidebar
        "filters": "Filters",
        "markets": "Markets",
        "select_markets": "Select Markets",
        "rsi_range": "RSI Range",
        "pe_range": "P/E Range",
        "min_dividend": "Min Dividend Yield",
        "rsi_condition": "RSI Condition",
        "all": "All",
        "oversold": "Oversold (RSI < 30)",
        "overbought": "Overbought (RSI > 70)",
        "neutral": "Neutral (30-70)",
        "refresh": "Refresh Data",
        "total_stocks": "Total Stocks",
        "last_update": "Last Update",
        "min": "Min",
        "max": "Max",
        "cache_health": "Cache Health",
        "stale": "Stale",
        "fresh": "Fresh",
        # KPIs
        "overview": "Overview",
        "stocks_count": "Stocks",
        "advancers": "Advancers",
        "decliners": "Decliners",
        "avg_rsi": "Avg RSI",
        "avg_change": "Avg Change",
        # Tabs
        "stocks_tab": "Stocks List",
        "opportunities_tab": "Opportunities",
        "analysis_tab": "Market Analysis",
        "sectors_tab": "Sector Analysis",
        "portfolio_tab": "Portfolio",
        "watchlist_tab": "Watchlist",
        "ai_tab": "AI Assistant",
        "commodities_tab": "Commodities",
        # Table columns
        "name": "Name",
        "market": "Market",
        "price": "Price",
        "change": "Change",
        "sector": "Sector",
        "pe": "P/E",
        "dividend": "Div Yield",
        "cap": "Market Cap",
        "volume": "Volume",
        # Opportunities
        "opp_title": "Top 10 Investment Opportunities",
        "opp_subtitle": "Based on RSI, dividend yield, P/E ratio, and momentum analysis",
        "score": "Score",
        "trend_up": "Uptrend",
        "trend_down": "Downtrend",
        "trend_mixed": "Mixed",
        # Charts
        "chart_avg_change": "Average Change by Market",
        "chart_avg_rsi": "Average RSI by Market",
        "chart_top_gainers_losers": "Top Gainers & Losers",
        "chart_sector_dist": "Sector Distribution",
        "chart_sector_perf": "Top Performing Sectors",
        "chart_rsi_heatmap": "RSI Heatmap: Sectors × Markets",
        "chart_rsi_pe_scatter": "Value Matrix: RSI vs P/E",
        # Portfolio
        "portfolio_title": "Portfolio Builder & Analysis",
        "portfolio_desc": "Enter your holdings to analyze performance and risk",
        "holdings": "Holdings",
        "format_example": "Format: market:TICKER quantity cost_basis",
        "example": "Example: uae:EMAAR 1000 14.5",
        "analyze": "Analyze Portfolio",
        "total_value": "Total Value",
        "unrealized_pnl": "Unrealized P&L",
        "risk_score": "Risk Score",
        "diversification": "Diversification",
        "low_risk": "Low",
        "medium_risk": "Medium",
        "high_risk": "High",
        # Watchlist
        "watchlist_title": "Watchlist",
        "watchlist_empty": "Watchlist is empty. Add stocks from the Stocks tab.",
        "add_to_watchlist": "Add to Watchlist",
        "remove": "Remove",
        "watchlist_input": "Enter ticker (e.g. EMAAR)",
        "add": "Add",
        "commodities_unavailable": "Commodities data temporarily unavailable",
        "one_day_change": "1-Day Change",
        # AI
        "ai_title": "EisaX AI Assistant",
        "ai_desc": "Ask anything about the market in Arabic or English",
        "ai_placeholder": "E.g. Which stocks have dividend yield above 5%?",
        "ai_send": "Send",
        "ai_thinking": "Analysing...",
        "ai_clear": "Clear chat",
        "ai_context_label": "Market data attached",
        # Common
        "loading": "Loading...",
        "error": "Error",
        "success": "Success",
        "warning": "Warning",
        "no_data": "No data available",
        "download": "Download",
    },
}


def t(key: str) -> str:
    """Return translated string for the current language."""
    return TRANSLATIONS[st.session_state.language].get(key, key)


def now_dubai_str() -> str:
    return pd.Timestamp.now(tz=DUBAI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _tokenize_query(text: str) -> list[str]:
    """Simple tokenizer for relevance scoring (Arabic + English friendly)."""
    if not text:
        return []
    cleaned = (
        str(text).lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("|", " ")
        .replace("/", " ")
        .replace("\\", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    raw_tokens = [t.strip() for t in cleaned.split() if t.strip()]
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "what", "which",
        "how", "are", "is", "in", "on", "to", "of", "a", "an",
        "stock", "stocks", "market", "markets",
        "في", "من", "على", "الى", "إلى", "ما", "هو", "هي", "عن", "مع", "او", "أو", "كل",
        "سهم", "اسهم", "الاسهم", "السوق", "الأسواق",
    }
    return [t for t in raw_tokens if len(t) > 1 and t not in stop]


def build_ai_market_context(df: "pd.DataFrame", user_query: str, max_rows: int = 18) -> tuple[str, int]:
    """Build relevance-ranked context instead of sending arbitrary top rows."""
    if df is None or df.empty:
        return "No market data available.", 0

    cols = [
        "name", "market", "close", "change", "RSI",
        "price_earnings_ttm", "dividend_yield_recent", "sector", "SMA50", "SMA200"
    ]
    use_cols = [c for c in cols if c in df.columns]
    work = df[use_cols].copy()

    for c in ("close", "change", "RSI", "price_earnings_ttm", "dividend_yield_recent", "SMA50", "SMA200"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    q = (user_query or "").lower()
    tokens = _tokenize_query(user_query)

    wants_oversold = any(k in q for k in ("oversold", "ذروة بيع", "undervalued rsi"))
    wants_overbought = any(k in q for k in ("overbought", "ذروة شراء"))
    wants_dividend = any(k in q for k in ("dividend", "yield", "توزيع", "عائد"))
    wants_value = any(k in q for k in ("pe", "p/e", "valuation", "value", "قيمة", "تقييم"))
    wants_momentum = any(k in q for k in ("momentum", "trend", "زخم", "اتجاه"))
    wants_gainers = any(k in q for k in ("gainer", "top gain", "ارتفاع", "صاعد"))
    wants_losers = any(k in q for k in ("loser", "drop", "هبوط", "هابط"))

    score = pd.Series(0.0, index=work.index)
    if "change" in work.columns:
        score += work["change"].fillna(0).abs() * 0.08
    if "dividend_yield_recent" in work.columns:
        score += work["dividend_yield_recent"].fillna(0) * 0.05

    if wants_oversold and "RSI" in work.columns:
        score += ((35 - work["RSI"].fillna(50)).clip(lower=0, upper=25) / 8.0)
    if wants_overbought and "RSI" in work.columns:
        score += ((work["RSI"].fillna(50) - 65).clip(lower=0, upper=25) / 8.0)
    if wants_dividend and "dividend_yield_recent" in work.columns:
        score += (work["dividend_yield_recent"].fillna(0).clip(lower=0, upper=12) / 2.0)
    if wants_value and "price_earnings_ttm" in work.columns:
        pe = work["price_earnings_ttm"].replace(0, pd.NA)
        score += ((22 - pe.fillna(22)).clip(lower=0, upper=22) / 4.0)
    if wants_momentum and "change" in work.columns:
        score += (work["change"].fillna(0).abs().clip(lower=0, upper=8) / 1.5)
    if wants_gainers and "change" in work.columns:
        score += work["change"].fillna(0).clip(lower=0, upper=10) / 1.2
    if wants_losers and "change" in work.columns:
        score += ((-work["change"].fillna(0)).clip(lower=0, upper=10) / 1.2)

    if tokens:
        name_blob = (
            work.get("name", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
            + " "
            + work.get("sector", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
            + " "
            + work.get("market", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
        )
        for tk in tokens[:8]:
            score += name_blob.str.contains(tk, regex=False).astype(float) * 2.5

    work["_score"] = score
    ranked = work.sort_values("_score", ascending=False).head(max_rows).copy()
    ranked = ranked.drop(columns=["_score"], errors="ignore")

    preview = ranked.copy()
    for c in ("close", "change", "RSI", "price_earnings_ttm", "dividend_yield_recent"):
        if c in preview.columns:
            preview[c] = preview[c].round(2)

    csv_context = preview.to_csv(index=False)
    fact_lines = []
    for _, row in preview.head(10).iterrows():
        fact_lines.append(
            f"- {row.get('name','N/A')} ({row.get('market','N/A')}) | "
            f"Price {row.get('close','N/A')} | Change {row.get('change','N/A')}% | "
            f"RSI {row.get('RSI','N/A')} | P/E {row.get('price_earnings_ttm','N/A')} | "
            f"Div {row.get('dividend_yield_recent','N/A')}% | Sector {row.get('sector','N/A')}"
        )

    context_block = (
        f"Relevant market slice ({len(preview)} rows selected from {len(df)} filtered rows):\n"
        f"{csv_context}\n"
        f"Data Fact Cards:\n" + "\n".join(fact_lines)
    )
    return context_block, len(preview)


def ask_eisa_ai(messages, market_context: str, stock_count: int, language: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    language_name = "Arabic" if language == "ar" else "English"
    macro_ctx = ''
    try:
        from pipeline import scheduler
        cache_obj_ai = scheduler.cache
        df_com_ai, _ = cache_obj_ai.get_latest('commodities')
        if df_com_ai is not None and not df_com_ai.empty:
            for _, row in df_com_ai.iterrows():
                if row.get('name') in ['Gold', 'Crude Oil (WTI)', 'Silver']:
                    chg = row.get('change', 0) or 0
                    macro_ctx += f"{row['name']}: ${row['close']:.2f} ({chg:+.2f}%)  "
        df_crypto_ai, _ = cache_obj_ai.get_latest('crypto')
        if df_crypto_ai is not None and not df_crypto_ai.empty:
            btc = df_crypto_ai[df_crypto_ai['name'].str.upper().str.contains('BITCOIN|BTC', na=False)]
            if not btc.empty:
                b = btc.iloc[0]
                macro_ctx += f"BTC: ${b['close']:,.0f} ({b.get('change', 0):+.2f}%)"
    except Exception:
        pass

    macro_line = f"\nMacro context: {macro_ctx}" if macro_ctx else ""
    system_prompt = f"""You are Eisax, the official market intelligence assistant for EisaX.
You are powered by DeepSeek internally, but you must never introduce yourself as DeepSeek.
If asked who you are, reply that you are Eisax.
You are an expert financial analyst focused on Arab and GCC stock markets.
Always answer in {language_name}, matching the user's latest message.
Use ONLY the provided market data context for prices, RSI, daily change, sector, dividend yield, and valuation.
Every stock mention MUST include inline evidence in this style:
[Data: Price=..., Change=...%, RSI=..., P/E=..., Div=...%].
Never give direct buy or sell orders, never guarantee returns, and frame guidance as analysis only.
If the data is missing or insufficient, say that clearly instead of guessing.
Use a strict structured format:
1) Executive Summary
2) Top Opportunities (max 3, each with data evidence)
3) Risks / Watchouts
4) Monitoring Checklist (non-execution, what to track next)
Keep the answer concise and data-first.

Market data (filtered, {stock_count} stocks):
{market_context}
{macro_line}
"""

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": 0.2,
        "max_tokens": 900,
        "stream": False,
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ── CSS ───────────────────────────────────────────────────────────────────────
@st.cache_data
def _build_css(lang: str, dark: bool) -> str:
    direction   = "rtl" if lang == "ar" else "ltr"
    border_side = "left" if lang == "ar" else "right"
    card_border = "right" if lang == "ar" else "left"
    hover_tx    = "4px" if lang == "en" else "-4px"

    if dark:
        bg          = "#0f172a"
        surface     = "#1e293b"
        surface2    = "#334155"
        text        = "#e2e8f0"
        subtext     = "#94a3b8"
        border_col  = "#334155"
        header_bg   = "linear-gradient(135deg,#0f172a 0%,#1a1a4e 50%,#0f2a4a 100%)"
        card_bg     = "rgba(30,41,59,0.85)"
        tab_bg      = "rgba(30,41,59,0.9)"
        grid_col    = "#1e293b"
        app_bg      = "#0f172a"
        glass_bg    = "rgba(30,41,59,0.6)"
        glass_border = "rgba(100,116,139,0.2)"
        shadow_soft = "rgba(0,0,0,0.3)"
        accent_glow = "rgba(14,165,164,0.3)"
    else:
        bg          = "#f0f2f5"
        surface     = "white"
        surface2    = "#f8fafc"
        text        = "#0f172a"
        subtext     = "#64748b"
        border_col  = "#e2e8f0"
        header_bg   = "linear-gradient(135deg,#ffffff 0%,#e0ecff 50%,#e8f0fe 100%)"
        card_bg     = "rgba(255,255,255,0.85)"
        tab_bg      = "rgba(255,255,255,0.9)"
        grid_col    = "#f1f5f9"
        app_bg      = "linear-gradient(135deg,#f5f7fa 0%,#e8ecf4 50%,#f0f2f5 100%)"
        glass_bg    = "rgba(255,255,255,0.55)"
        glass_border = "rgba(15,76,129,0.12)"
        shadow_soft = "rgba(0,0,0,0.06)"
        accent_glow = "rgba(15,76,129,0.15)"

    return f"""
    <style>
        /* Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {{
            background: {app_bg};
            color: {text};
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        /* RTL scoped to content areas only — NOT .stApp root (breaks sidebar collapse) */
        .stMainBlockContainer, [data-testid="stSidebarContent"] {{
            direction: {direction};
        }}
        .main {{ padding: 1rem; }}

        /* ── Glassmorphism Header ──────────────────────────────────── */
        .main-header {{
            background: {header_bg};
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 2rem 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 32px {shadow_soft}, inset 0 1px 0 {glass_border};
            border: 1px solid {glass_border};
            position: relative;
            overflow: hidden;
        }}
        .main-header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 30% 30%, {accent_glow} 0%, transparent 60%);
            pointer-events: none;
        }}
        .main-header h1 {{
            color: #0f4c81;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: .4rem;
            letter-spacing: -0.02em;
            position: relative;
        }}
        .main-header p {{
            color: {subtext};
            font-size: .9rem;
            position: relative;
        }}

        /* ── Pulse indicator ──────────────────────────────────────── */
        @keyframes pulse-live {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(1.3); }}
        }}
        .live-dot {{
            display: inline-block;
            width: 8px; height: 8px;
            background: #10b981;
            border-radius: 50%;
            margin-inline-end: 6px;
            animation: pulse-live 2s ease-in-out infinite;
            vertical-align: middle;
        }}

        /* ── Sidebar ──────────────────────────────────────────────── */
        [data-testid="stSidebar"] {{
            background: {surface} !important;
            border-{border_side}: 1px solid {border_col};
        }}

        /* ── Metric card (glassmorphism) ──────────────────────────── */
        @keyframes countUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .metric-card {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 1.2rem 1rem;
            text-align: center;
            box-shadow: 0 4px 16px {shadow_soft};
            transition: all .35s cubic-bezier(.4,0,.2,1);
            border: 1px solid {glass_border};
            animation: countUp 0.6s ease-out;
        }}
        .metric-card:hover {{
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 12px 32px {shadow_soft}, 0 0 0 1px {accent_glow};
        }}
        .metric-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: #0f4c81;
            letter-spacing: -0.03em;
        }}
        .metric-label {{
            font-size: .82rem;
            color: {subtext};
            margin-top: .3rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        /* ── Sentiment Bar ────────────────────────────────────────── */
        .sentiment-bar {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid {glass_border};
            box-shadow: 0 2px 12px {shadow_soft};
        }}
        .sentiment-track {{
            height: 10px;
            border-radius: 8px;
            background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
            overflow: hidden;
            position: relative;
        }}
        .sentiment-fill {{
            height: 100%;
            border-radius: 8px;
            transition: width 1s ease-out;
        }}

        /* ── Stock card ───────────────────────────────────────────── */
        .stock-card {{
            background: {card_bg};
            backdrop-filter: blur(8px);
            border-radius: 16px;
            padding: 1.2rem;
            margin: .6rem 0;
            box-shadow: 0 2px 8px {shadow_soft};
            border-{card_border}: 4px solid #0f4c81;
            border-top: 1px solid {glass_border};
            transition: all .35s cubic-bezier(.4,0,.2,1);
        }}
        .stock-card:hover {{
            transform: translateX({hover_tx});
            box-shadow: 0 8px 24px {shadow_soft};
        }}

        /* ── Badges ───────────────────────────────────────────────── */
        .badge-success {{ background:#d1fae5; color:#065f46; padding:.25rem .75rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}
        .badge-warning {{ background:#fed7aa; color:#92400e; padding:.25rem .75rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}
        .badge-danger  {{ background:#fee2e2; color:#991b1b; padding:.25rem .75rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}
        .badge-info    {{ background:#dbeafe; color:#1e40af; padding:.25rem .75rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}

        /* ── Tabs (glow on active) ────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap:.5rem;
            background: {tab_bg};
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: .5rem;
            border: 1px solid {glass_border};
            margin-bottom: 1rem;
            flex-wrap: wrap;
            box-shadow: 0 2px 8px {shadow_soft};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 12px;
            padding: .5rem 1.2rem;
            font-weight: 600;
            color: {subtext};
            white-space: nowrap;
            transition: all .25s ease;
            font-family: 'Inter', sans-serif;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: {accent_glow};
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #0f4c81, #0ea5a4) !important;
            color: white !important;
            box-shadow: 0 4px 16px {accent_glow};
        }}

        /* ── Buttons ──────────────────────────────────────────────── */
        .stButton > button {{
            background: linear-gradient(135deg, #0f4c81, #0ea5a4);
            color: white;
            border: none;
            border-radius: 12px;
            padding: .55rem 1.1rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all .25s cubic-bezier(.4,0,.2,1);
            box-shadow: 0 2px 8px rgba(15,76,129,0.2);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(15,76,129,0.3);
        }}

        /* ── AI chat bubble ───────────────────────────────────────── */
        .chat-user {{
            background: linear-gradient(135deg, #0f4c81, #0ea5a4);
            color: white;
            border-radius: 20px 20px 4px 20px;
            padding: .85rem 1.2rem;
            margin: .5rem 0 .5rem auto;
            max-width: 75%;
            text-align: {("right" if lang == "ar" else "left")};
            box-shadow: 0 2px 8px rgba(15,76,129,0.2);
        }}
        .chat-ai {{
            background: {card_bg};
            backdrop-filter: blur(8px);
            color: {text};
            border-radius: 20px 20px 20px 4px;
            padding: .85rem 1.2rem;
            margin: .5rem auto .5rem 0;
            max-width: 85%;
            border: 1px solid {glass_border};
            box-shadow: 0 2px 8px {shadow_soft};
        }}

        /* ── Cache pill ───────────────────────────────────────────── */
        .cache-pill-fresh  {{ background:#d1fae5; color:#065f46; padding:.2rem .6rem; border-radius:20px; font-size:.7rem; }}
        .cache-pill-stale  {{ background:#fee2e2; color:#991b1b; padding:.2rem .6rem; border-radius:20px; font-size:.7rem; }}

        /* ── Positive / Negative ──────────────────────────────────── */
        .positive {{ color:#10b981; font-weight:700; }}
        .negative {{ color:#ef4444; font-weight:700; }}

        /* ── Premium divider ──────────────────────────────────────── */
        hr {{ margin:1.2rem 0; border:none; height:1px; background:linear-gradient(to right,transparent,{border_col},transparent); }}

        /* ── Footer ───────────────────────────────────────────────── */
        .eisax-footer {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: .8rem 1.5rem;
            margin-top: 1.5rem;
            border: 1px solid {glass_border};
            box-shadow: 0 -2px 12px {shadow_soft};
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: .5rem;
        }}
        .eisax-footer span {{
            font-size: .8rem;
            color: {subtext};
            font-weight: 500;
        }}
        .eisax-footer .brand {{
            font-weight: 700;
            background: linear-gradient(135deg, #0f4c81, #0ea5a4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ── Search box ───────────────────────────────────────────── */
        .stock-search {{
            background: {card_bg};
            border-radius: 12px;
            padding: .4rem .8rem;
            border: 1px solid {glass_border};
            margin-bottom: .8rem;
        }}

        /* ── Dataframe ────────────────────────────────────────────── */
        .stDataFrame {{ border-radius: 16px; overflow: hidden; }}

        /* ── Scrollbar ────────────────────────────────────────────── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: {surface2}; border-radius: 10px; }}
        ::-webkit-scrollbar-thumb {{ background: linear-gradient(135deg, #0f4c81, #0ea5a4); border-radius: 10px; }}

        /* ── Responsive ───────────────────────────────────────────── */
        @media (max-width:768px) {{
            .main-header h1 {{ font-size: 1.5rem; }}
            .metric-value    {{ font-size: 1.3rem; }}
            .eisax-footer {{ flex-direction: column; text-align: center; }}
        }}
    </style>
    """

st.markdown(
    _build_css(st.session_state.language, st.session_state.dark_mode),
    unsafe_allow_html=True,
)

# ── Market metadata ───────────────────────────────────────────────────────────
def _market_label(code: str) -> str:
    flags = {"uae":"🇦🇪","ksa":"🇸🇦","egypt":"🇪🇬","kuwait":"🇰🇼","qatar":"🇶🇦","bahrain":"🇧🇭","morocco":"🇲🇦","tunisia":"🇹🇳","america":"🇺🇸"}
    names_ar = {"uae":"الإمارات","ksa":"السعودية","egypt":"مصر","kuwait":"الكويت","qatar":"قطر","bahrain":"البحرين","morocco":"المغرب","tunisia":"تونس","america":"USA"}
    names_en = {"uae":"UAE","ksa":"Saudi Arabia","egypt":"Egypt","kuwait":"Kuwait","qatar":"Qatar","bahrain":"Bahrain","morocco":"Morocco","tunisia":"Tunisia","america":"USA"}
    names = names_ar if st.session_state.language == "ar" else names_en
    return f"{flags.get(code,'')} {names.get(code, code)}"

ARAB_MARKETS = ["uae","ksa","egypt","kuwait","qatar","bahrain","morocco","tunisia"]
GLOBAL_MARKETS = ARAB_MARKETS + ["america"]


def _format_price(value, market_code=None):
    if pd.isna(value):
        return "—"
    prefix = "$" if market_code == "america" else ""
    return f"{prefix}{value:,.2f}"


def _commodity_row(df_in: pd.DataFrame, name: str):
    if df_in is None or df_in.empty or "name" not in df_in.columns:
        return None
    match = df_in[df_in["name"] == name]
    return None if match.empty else match.iloc[0]

# ── Pipeline helpers ──────────────────────────────────────────────────────────
@st.cache_resource
def _get_pipeline():
    try:
        from pipeline import cache, fetcher
        from query_engine import QueryEngine
        return cache, fetcher, QueryEngine(cache, fetcher)
    except Exception:
        return None, None, None


def _load_from_cache(markets: list) -> tuple:
    cache_obj, _, _ = _get_pipeline()
    if cache_obj is None:
        return pd.DataFrame(), {}
    frames, status = [], {}
    for m in markets:
        df, ts = cache_obj.get_latest(m)
        if df is not None and not df.empty:
            df = df.copy()
            df["_market_code"] = m
            df["market"] = _market_label(m)
            frames.append(df)
            age = cache_obj.cache_age_minutes(m)
            status[m] = {
                "ts": ts,
                "age": round(age, 1) if age else None,
                "count": len(df),
                "stale": (age or 999) > CACHE_STALE_MINUTES,
            }
    non_empty = [f for f in frames if not f.empty and not f.isna().all(axis=None)]
    return (pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()), status


@st.cache_data(ttl=900)
def load_data_cached():
    df, status = _load_from_cache(GLOBAL_MARKETS)
    if df.empty:
        try:
            df = pd.read_csv("arab_markets_complete.csv")
            status = {"_source": "csv_fallback"}
        except Exception:
            return pd.DataFrame(), {}
    for col in ["RSI","change","price_earnings_ttm","dividend_yield_recent","market_cap_basic","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, status


# ── Load data ─────────────────────────────────────────────────────────────────
df, cache_status = load_data_cached()
_, fetcher_obj, qe = _get_pipeline()

# ── Chart helpers ─────────────────────────────────────────────────────────────
def style_chart(fig, height=400):
    bg = "#1e293b" if st.session_state.dark_mode else "white"
    tc = "#94a3b8" if st.session_state.dark_mode else "#475569"
    gc = "#334155" if st.session_state.dark_mode else "#f1f5f9"
    fig.update_layout(
        template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(family="Arial, sans-serif", size=12, color=tc),
        title_font=dict(size=15, color="#0f4c81"),
        margin=dict(l=20, r=20, t=50, b=30),
        height=height,
    )
    fig.update_xaxes(showgrid=False, linecolor=gc)
    fig.update_yaxes(showgrid=True, gridcolor=gc, linecolor=gc)
    return fig

# ── Opportunity scoring ───────────────────────────────────────────────────────
def compute_opportunity_score(row: pd.Series) -> float:
    """
    Multi-factor score (higher = more attractive buy opportunity).
    Factors:
      1. RSI distance from oversold (max +3)
      2. Dividend yield              (max +2)
      3. P/E value                   (max +2)
      4. SMA trend                   (max +2)
      5. Volume spike (relative)     (max +1)
    """
    score = 0.0
    rsi = row.get("RSI", 50) or 50
    score += max(0, (RSI_OVERSOLD + 20 - rsi) / 10)                      # (1)

    div = row.get("dividend_yield_recent", 0) or 0
    score += min(2, div / 5)                                               # (2)

    pe = row.get("price_earnings_ttm", 15) or 15
    if pe > 0:
        score += min(2, 15 / pe)                                           # (3)

    close  = row.get("close", 0)  or 0
    sma50  = row.get("SMA50")
    sma200 = row.get("SMA200")
    if pd.notna(sma50) and pd.notna(sma200) and close:
        if close > sma50 and sma50 > sma200:
            score += 2   # golden cross territory                          # (4)
        elif close > sma50:
            score += 1

    vol     = row.get("volume", 0)   or 0
    avg_vol = row.get("avg_volume", 0) or 0
    if avg_vol and vol > avg_vol * 1.5:
        score += 1                                                         # (5)

    return round(score, 2)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Language + dark mode row
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("🌐 العربية", use_container_width=True):
            st.session_state.language = "ar"
            st.rerun()
    with c2:
        if st.button("🇬🇧 English", use_container_width=True):
            st.session_state.language = "en"
            st.rerun()
    with c3:
        dark_icon = "🌙" if not st.session_state.dark_mode else "☀️"
        if st.button(dark_icon, use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")
    st.markdown(f"### 🔍 {t('filters')}")

    # Market selection
    all_market_labels = [_market_label(m) for m in GLOBAL_MARKETS]
    if not df.empty and "market" in df.columns:
        current_labels = set(df["market"].dropna().tolist())
        all_market_labels = [label for label in all_market_labels if label in current_labels]

    selected_markets = st.multiselect(t("markets"), all_market_labels, default=all_market_labels[:3])

    st.markdown("---")

    # RSI filter
    st.markdown(f"**{t('rsi_range')}**")
    col1, col2 = st.columns(2)
    with col1:
        rsi_min = st.number_input(t("min"), 0, 100, 0,  key="rsi_min_inp")
    with col2:
        rsi_max = st.number_input(t("max"), 0, 100, 100, key="rsi_max_inp")

    # P/E filter
    st.markdown(f"**{t('pe_range')}**")
    col1, col2 = st.columns(2)
    with col1:
        pe_min = st.number_input(t("min"), 0, 200, 0,   key="pe_min_inp")
    with col2:
        pe_max = st.number_input(t("max"), 0, 200, 100, key="pe_max_inp")

    div_min = st.slider(t("min_dividend"), 0.0, 20.0, 0.0, step=0.5)

    st.markdown("---")

    rsi_condition = st.radio(
        t("rsi_condition"),
        [t("all"), t("oversold"), t("overbought"), t("neutral")],
        horizontal=True,
    )

    st.markdown("---")

    # Stats + cache health
    if not df.empty:
        st.metric(t("total_stocks"), len(df))

    if cache_status and "_source" not in cache_status:
        st.markdown(f"**{t('cache_health')}**")
        for mkt, info in cache_status.items():
            pill_cls = "cache-pill-stale" if info.get("stale") else "cache-pill-fresh"
            label    = t("stale") if info.get("stale") else t("fresh")
            age_str  = f"{info['age']}m" if info.get("age") else "—"
            st.markdown(
                f'<span class="{pill_cls}">{_market_label(mkt)} · {age_str} · {label}</span>',
                unsafe_allow_html=True,
            )

    st.caption(f"🕐 {t('last_update')}: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    if fetcher_obj:
        if st.button(f"🔄 {t('refresh')}", use_container_width=True):
            with st.spinner(t("loading")):
                fetcher_obj.fetch_all()
            st.cache_data.clear()
            st.rerun()

# ── Filter data ───────────────────────────────────────────────────────────────
if df.empty:
    st.error(f"❌ {t('no_data')}")
    st.stop()

filtered_df = df.copy()
if selected_markets:
    filtered_df = filtered_df[filtered_df["market"].isin(selected_markets)]

filtered_df = filtered_df[
    filtered_df["RSI"].fillna(50).between(rsi_min, rsi_max) &
    filtered_df["price_earnings_ttm"].fillna(15).between(pe_min, pe_max) &
    (filtered_df["dividend_yield_recent"].fillna(0) >= div_min)
]

if rsi_condition == t("oversold"):
    filtered_df = filtered_df[filtered_df["RSI"] < RSI_OVERSOLD]
elif rsi_condition == t("overbought"):
    filtered_df = filtered_df[filtered_df["RSI"] > RSI_OVERBOUGHT]
elif rsi_condition == t("neutral"):
    filtered_df = filtered_df[filtered_df["RSI"].between(RSI_OVERSOLD, RSI_OVERBOUGHT)]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>{t('title')}</h1>
    <p><span class="live-dot"></span>{t('subtitle')}</p>
    <p style="font-size:.8rem;color:#94a3b8;">{t('version')}</p>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(f"### 📊 {t('overview')}")
n = max(len(filtered_df), 1)
up = int((filtered_df["change"] > 0).sum())
dn = int((filtered_df["change"] < 0).sum())
flat_count = n - up - dn
avg_rsi = filtered_df["RSI"].mean()
avg_ch = filtered_df["change"].mean()

# Sentiment bar
up_pct = up / n * 100
st.markdown(f"""
<div class="sentiment-bar">
    <div style="display:flex;justify-content:space-between;margin-bottom:.5rem;">
        <span style="color:#10b981;font-weight:700;">📈 {up} ({up_pct:.0f}%)</span>
        <span style="color:#64748b;font-weight:600;">{len(filtered_df)} {t('stocks_count')}</span>
        <span style="color:#ef4444;font-weight:700;">📉 {dn} ({dn/n*100:.0f}%)</span>
    </div>
    <div style="display:flex;border-radius:8px;overflow:hidden;height:10px;">
        <div style="width:{up_pct:.1f}%;background:linear-gradient(90deg,#10b981,#34d399);transition:width 1s;"></div>
        <div style="width:{flat_count/n*100:.1f}%;background:#94a3b8;"></div>
        <div style="width:{dn/n*100:.1f}%;background:linear-gradient(90deg,#f87171,#ef4444);transition:width 1s;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

cols = st.columns(5)
with cols[0]:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(filtered_df)}</div><div class="metric-label">{t("stocks_count")}</div></div>', unsafe_allow_html=True)

with cols[1]:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#10b981;">{up} ({up_pct:.0f}%)</div><div class="metric-label">📈 {t("advancers")}</div></div>', unsafe_allow_html=True)

with cols[2]:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444;">{dn} ({dn/n*100:.0f}%)</div><div class="metric-label">📉 {t("decliners")}</div></div>', unsafe_allow_html=True)

with cols[3]:
    rc = "#10b981" if avg_rsi < RSI_OVERSOLD else "#ef4444" if avg_rsi > RSI_OVERBOUGHT else "#f59e0b"
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{rc};">{avg_rsi:.1f}</div><div class="metric-label">{t("avg_rsi")}</div></div>', unsafe_allow_html=True)

with cols[4]:
    cc = "#10b981" if avg_ch > 0 else "#ef4444" if avg_ch < 0 else "#64748b"
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{cc};">{avg_ch:+.2f}%</div><div class="metric-label">{t("avg_change")}</div></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    f"📋 {t('stocks_tab')}",
    f"🎯 {t('opportunities_tab')}",
    f"📈 {t('analysis_tab')}",
    f"🏭 {t('sectors_tab')}",
    f"💼 {t('portfolio_tab')}",
    f"⭐ {t('watchlist_tab')}",
    f"🤖 {t('ai_tab')}",
    f"🏭 {t('commodities_tab')}",
])

# ═══════════════════════════════════════════════════════
# TAB 1 — Stocks list
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown(f"### 📋 {t('stocks_tab')}")

    # Search filter
    _search_label = "🔍 Search..." if st.session_state.language == "en" else "🔍 بحث..."
    search_query = st.text_input(_search_label, key="stock_search", label_visibility="collapsed", placeholder=_search_label)

    display_cols = ["name","market","close","change","RSI","price_earnings_ttm","dividend_yield_recent","sector"]
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    _search_base = filtered_df.copy()
    if search_query:
        _sq = search_query.strip().lower()
        _mask = _search_base["name"].fillna("").str.lower().str.contains(_sq, regex=False)
        if "sector" in _search_base.columns:
            _mask = _mask | _search_base["sector"].fillna("").str.lower().str.contains(_sq, regex=False)
        _search_base = _search_base[_mask]
    display_df = _search_base[available_cols].copy()

    if "close" in display_df.columns:
        display_df["close"] = _search_base.apply(
            lambda row: _format_price(row.get("close"), row.get("_market_code")),
            axis=1,
        )

    for col, rnd in [("change",2),("RSI",1),("price_earnings_ttm",1),("dividend_yield_recent",2)]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(rnd)

    rename = {
        "name": t("name"), "market": t("market"), "close": t("price"),
        "change": t("change"), "RSI": "RSI", "price_earnings_ttm": t("pe"),
        "dividend_yield_recent": t("dividend"), "sector": t("sector"),
    }
    display_df = display_df.rename(columns=rename)
    st.dataframe(display_df, use_container_width=True, height=500)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 {t('download')} CSV", csv, "eisax_stocks.csv", "text/csv")

    # Quick add to watchlist
    if "name" in filtered_df.columns:
        ticker_options = filtered_df["name"].dropna().unique().tolist()
        picked = st.selectbox(f"⭐ {t('add_to_watchlist')}", ["—"] + ticker_options)
        if picked != "—" and picked not in st.session_state.watchlist:
            if st.button(t("add"), key="add_wl_tab1"):
                st.session_state.watchlist.append(picked)
                _wl_add(picked)
                st.success(f"✅ {picked} → {t('watchlist_tab')}")

# ═══════════════════════════════════════════════════════
# TAB 2 — Opportunities
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### 🎯 {t('opp_title')}")
    st.caption(t("opp_subtitle"))

    opp_df = filtered_df.copy()
    opp_df["score"] = opp_df.apply(compute_opportunity_score, axis=1)
    top_opp = opp_df.nlargest(10, "score")

    for idx, (_, row) in enumerate(top_opp.iterrows(), 1):
        rsi = row.get("RSI", 50)
        if rsi < RSI_OVERSOLD:
            badge_cls, rsi_text = "badge-success", t("oversold")
        elif rsi > RSI_OVERBOUGHT:
            badge_cls, rsi_text = "badge-danger",  t("overbought")
        else:
            badge_cls, rsi_text = "badge-warning", t("neutral")

        close  = row.get("close", 0)
        sma50  = row.get("SMA50")
        sma200 = row.get("SMA200")
        if pd.notna(sma50) and pd.notna(sma200) and close:
            if close > sma50 and close > sma200:
                trend, trend_cls = f"🔼 {t('trend_up')}",   "badge-success"
            elif close < sma50 and close < sma200:
                trend, trend_cls = f"🔽 {t('trend_down')}",  "badge-danger"
            else:
                trend, trend_cls = f"➡️ {t('trend_mixed')}", "badge-warning"
        else:
            trend, trend_cls = "—", "badge-info"

        change_val = row.get("change", 0)
        ch_cls = "positive" if change_val > 0 else "negative"
        bar_w  = min(100, row["score"] * 10)

        st.markdown(f"""
        <div class="stock-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;">
            <div>
              <strong>#{idx} {row.get('name','—')}</strong><br>
              <small>{row.get('market','—')} | {row.get('sector','—')}</small>
            </div>
            <div>
              <span class="{badge_cls}">RSI: {rsi:.0f} ({rsi_text})</span>
              <span class="{trend_cls}" style="margin-inline-start:.5rem;">{trend}</span>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:.5rem;gap:.5rem;flex-wrap:wrap;">
            <div><span style="color:#64748b;">{t('price')}:</span> <strong>{_format_price(close, row.get('_market_code'))}</strong></div>
            <div><span style="color:#64748b;">{t('change')}:</span> <strong class="{ch_cls}">{change_val:+.2f}%</strong></div>
            <div><span style="color:#64748b;">{t('pe')}:</span> <strong>{row.get('price_earnings_ttm',0):.1f}</strong></div>
            <div><span style="color:#64748b;">{t('dividend')}:</span> <strong>{row.get('dividend_yield_recent',0):.2f}%</strong></div>
          </div>
          <div style="margin-top:.75rem;">
            <div style="background:#e2e8f0;border-radius:10px;height:6px;overflow:hidden;">
              <div style="background:#0f4c81;width:{bar_w}%;height:100%;border-radius:10px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:.25rem;">
              <small style="color:#64748b;">{t('score')}</small>
              <small style="color:#0f4c81;font-weight:bold;">{row['score']:.2f}/10</small>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 3 — Market Analysis
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### 📈 {t('analysis_tab')}")

    col1, col2 = st.columns(2)
    with col1:
        mkt_perf = filtered_df.groupby("market")["change"].mean().reset_index()
        fig1 = px.bar(mkt_perf, x="market", y="change",
                      title=f"📊 {t('chart_avg_change')}",
                      color="change", color_continuous_scale=COLOR_SCALE_CHANGE, text="change")
        fig1.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(style_chart(fig1), use_container_width=True)

    with col2:
        mkt_rsi = filtered_df.groupby("market")["RSI"].mean().reset_index()
        fig2 = px.bar(mkt_rsi, x="market", y="RSI",
                      title=f"🎯 {t('chart_avg_rsi')}",
                      color="RSI", color_continuous_scale=COLOR_SCALE_RSI)
        fig2.add_hline(y=RSI_OVERSOLD,   line_dash="dash", line_color="#10b981", annotation_text="30")
        fig2.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="#ef4444", annotation_text="70")
        st.plotly_chart(style_chart(fig2), use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 🏆 {t('chart_top_gainers_losers')}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**📈 {t('advancers')}**")
        gainers = filtered_df.nlargest(10, "change")[["name","market","close","change","RSI"]].copy()
        gainers["change"] = gainers["change"].apply(lambda x: f"▲ +{x:.2f}%")
        st.dataframe(gainers, use_container_width=True, hide_index=True)
    with col2:
        st.markdown(f"**📉 {t('decliners')}**")
        losers = filtered_df.nsmallest(10, "change")[["name","market","close","change","RSI"]].copy()
        losers["change"] = losers["change"].apply(lambda x: f"▼ {x:.2f}%")
        st.dataframe(losers, use_container_width=True, hide_index=True)

    # RSI vs P/E scatter (value screen)
    st.markdown("---")
    st.markdown(f"### 🔍 {t('chart_rsi_pe_scatter')}")
    st.caption("Stocks in the bottom-left quadrant (low RSI + low P/E) are classic value + momentum opportunities." if st.session_state.language == "en" else "الأسهم في الربع السفلي الأيسر (RSI منخفض + P/E منخفض) هي فرص قيمة وزخم كلاسيكية.")
    scatter_df = filtered_df[["name","market","RSI","price_earnings_ttm","dividend_yield_recent","close"]].dropna()
    if not scatter_df.empty:
        fig_sc = px.scatter(
            scatter_df, x="RSI", y="price_earnings_ttm",
            size="close", color="market",
            hover_data={"name":True,"dividend_yield_recent":True},
            title=f"🔍 {t('chart_rsi_pe_scatter')}",
            labels={"price_earnings_ttm": "P/E", "RSI": "RSI"},
            color_discrete_sequence=PIE_COLORS,
        )
        fig_sc.add_vline(x=RSI_OVERSOLD,   line_dash="dot", line_color="#10b981")
        fig_sc.add_vline(x=RSI_OVERBOUGHT, line_dash="dot", line_color="#ef4444")
        fig_sc.add_hline(y=15, line_dash="dot", line_color="#94a3b8")
        st.plotly_chart(style_chart(fig_sc, height=450), use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 4 — Sector Analysis
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown(f"### 🏭 {t('sectors_tab')}")

    col1, col2 = st.columns(2)
    with col1:
        sc = filtered_df["sector"].value_counts().head(8).reset_index()
        sc.columns = [t("sector"), "Count"]
        fig3 = px.pie(sc, values="Count", names=t("sector"),
                      title=f"📊 {t('chart_sector_dist')}",
                      color_discrete_sequence=PIE_COLORS)
        fig3.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_chart(fig3, 400), use_container_width=True)

    with col2:
        sp = filtered_df.groupby("sector")["change"].mean().sort_values(ascending=False).head(10).reset_index()
        sp.columns = [t("sector"), "Change %"]
        fig4 = px.bar(sp, x=t("sector"), y="Change %",
                      title=f"🏆 {t('chart_sector_perf')}",
                      color="Change %", color_continuous_scale=COLOR_SCALE_CHANGE)
        st.plotly_chart(style_chart(fig4, 400), use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 🌡️ {t('chart_rsi_heatmap')}")
    if "sector" in filtered_df.columns and "market" in filtered_df.columns:
        hm = filtered_df.pivot_table(index="sector", columns="market", values="RSI", aggfunc="mean").round(1)
        fig5 = px.imshow(hm, color_continuous_scale=COLOR_SCALE_RSI, aspect="auto", zmin=20, zmax=80, text_auto=True,
                         title=f"🌡️ {t('chart_rsi_heatmap')}")
        st.plotly_chart(style_chart(fig5, 500), use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 5 — Portfolio
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown(f"### 💼 {t('portfolio_title')}")
    st.caption(t("portfolio_desc"))

    # Guard: pipeline required
    if qe is None:
        st.warning(f"⚠️ Pipeline not available — portfolio analysis requires live data connection.")
    else:
        try:
            from portfolio import Portfolio
        except ImportError:
            st.error("❌ `portfolio` module not found. Please ensure portfolio.py is in the project root.")
            st.stop()

        with st.expander("ℹ️ " + ("How to enter holdings" if st.session_state.language == "en" else "كيفية إدخال الأسهم")):
            st.markdown(f"**{t('format_example')}**\n```\n{t('example')}\nuae:FAB 500 18.00\nksa:2222.SR 300 30.00\n```")

        holdings_text = st.text_area(t("holdings"),
                                     value="uae:EMAAR 1000 14.50\nuae:FAB 500 18.00\nksa:2222.SR 300 30.00",
                                     height=150)
        target_text = st.text_input(
            "🎯 " + ("Target sector allocation (optional)" if st.session_state.language == "en" else "توزيع القطاعات المستهدف (اختياري)"),
            placeholder="Finance:40 Energy:30 Technology:20",
        )

        if st.button(f"🔍 {t('analyze')}", type="primary", use_container_width=True):
            p = Portfolio(qe)
            parse_errors = []
            for line in holdings_text.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    parse_errors.append(f"⚠️ Invalid line: {line}")
                    continue
                raw = parts[0]
                try:
                    qty = float(parts[1])
                except ValueError:
                    parse_errors.append(f"⚠️ Invalid qty: {line}")
                    continue
                cost = float(parts[2]) if len(parts) >= 3 else None
                if ":" in raw:
                    market_code, ticker = raw.split(":", 1)
                else:
                    try:
                        from report_enhancer import _resolve_market
                        market_code = _resolve_market(raw) or "uae"
                    except ImportError:
                        market_code = "uae"
                    ticker = raw
                p.add(ticker, market=market_code, qty=qty, cost_basis=cost)

            for err in parse_errors:
                st.warning(err)

            target_weights = {}
            for item in target_text.split():
                if ":" in item:
                    sec, val = item.split(":", 1)
                    try:
                        target_weights[sec] = float(val)
                    except ValueError:
                        pass

            with st.spinner(t("loading")):
                summary = p.summary()

            pos_df = summary.get("positions", pd.DataFrame())
            if pos_df.empty or not summary.get("total_value"):
                st.error(f"❌ {t('error')}: No holdings found in cache")
            else:
                kc = st.columns(4)
                with kc[0]:
                    st.metric(f"💰 {t('total_value')}", f"{summary['total_value']:,.0f}")
                with kc[1]:
                    pnl = summary.get("total_pnl")
                    pnl_str = f"{pnl:+,.0f}" if pnl is not None else "—"
                    pnl_color = "green" if pnl and pnl > 0 else "red"
                    st.markdown(f"<div style='color:{pnl_color}'><strong>📊 {t('unrealized_pnl')}</strong><br>{pnl_str}</div>", unsafe_allow_html=True)
                with kc[2]:
                    risk = summary.get("risk_score", 50)
                    if risk > 70:
                        rt, rc2 = t("high_risk"),   "#ef4444"
                    elif risk > 40:
                        rt, rc2 = t("medium_risk"), "#f59e0b"
                    else:
                        rt, rc2 = t("low_risk"),    "#10b981"
                    st.markdown(f"<div style='color:{rc2}'><strong>⚠️ {t('risk_score')}</strong><br>{rt} ({risk}/100)</div>", unsafe_allow_html=True)
                with kc[3]:
                    st.metric(f"🌍 {t('diversification')}", summary.get("diversification", "—"))

                show_cols = [c for c in ["name","market","qty","price","value","pnl","pnl_pct","sector","RSI"] if c in pos_df.columns]
                st.dataframe(pos_df[show_cols].round(2), use_container_width=True, hide_index=True)

                md_report = p.to_markdown(target_weights or None)
                st.download_button(f"📥 {t('download')} Report", md_report.encode("utf-8"), "eisax_portfolio_report.md", "text/markdown")

# ═══════════════════════════════════════════════════════
# TAB 6 — Watchlist
# ═══════════════════════════════════════════════════════
with tab6:
    st.markdown(f"### ⭐ {t('watchlist_title')}")

    c1, c2 = st.columns([4, 1])
    with c1:
        new_ticker = st.text_input(t("watchlist_input"), key="wl_input", label_visibility="collapsed",
                                   placeholder=t("watchlist_input"))
    with c2:
        if st.button(f"➕ {t('add')}", use_container_width=True):
            ticker_clean = new_ticker.strip().upper()
            if ticker_clean and ticker_clean not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker_clean)
                _wl_add(ticker_clean)
                st.rerun()

    if not st.session_state.watchlist:
        st.info(t("watchlist_empty"))
    else:
        wl_df = pd.DataFrame()
        if not df.empty and "name" in df.columns:
            wl_df = df[df["name"].isin(st.session_state.watchlist)].copy()

        for ticker in st.session_state.watchlist:
            row_data = wl_df[wl_df["name"] == ticker].iloc[0] if not wl_df.empty and ticker in wl_df["name"].values else None
            with st.container():
                rc1, rc2, rc3 = st.columns([4, 3, 1])
                with rc1:
                    if row_data is not None:
                        ch = row_data.get("change", 0)
                        ch_cls = "positive" if ch > 0 else "negative"
                        st.markdown(f"**{ticker}** — {row_data.get('market','—')} | {row_data.get('sector','—')}")
                        price_str = _format_price(row_data.get('close', 0), row_data.get('_market_code'))
                        st.markdown(f"Price: **{price_str}** | <span class='{ch_cls}'>{ch:+.2f}%</span> | RSI: **{row_data.get('RSI',0):.1f}**", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{ticker}** *(not in current filter)*")
                with rc2:
                    if row_data is not None:
                        rsi_v = row_data.get("RSI", 50)
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=rsi_v,
                            gauge={"axis":{"range":[0,100]},
                                   "bar":{"color":"#0f4c81"},
                                   "steps":[{"range":[0,30],"color":"#d1fae5"},{"range":[70,100],"color":"#fee2e2"}],
                                   "threshold":{"line":{"color":"red","width":2},"thickness":.75,"value":RSI_OVERBOUGHT}},
                            title={"text":"RSI"},
                        ))
                        fig_gauge.update_layout(height=120, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{ticker}")
                with rc3:
                    if st.button(f"🗑️", key=f"rm_{ticker}"):
                        st.session_state.watchlist.remove(ticker)
                        _wl_remove(ticker)
                        st.rerun()
                st.markdown("---")

# ═══════════════════════════════════════════════════════
# TAB 7 — AI Assistant
# ═══════════════════════════════════════════════════════
with tab7:
    st.markdown(f"### 🤖 {t('ai_title')}")
    st.caption(t("ai_desc"))

    # Render chat history
    for msg in st.session_state.ai_history:
        role_cls = "chat-user" if msg["role"] == "user" else "chat-ai"
        st.markdown(f'<div class="{role_cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input row
    ci1, ci2 = st.columns([5, 1])
    with ci1:
        user_query = st.text_input(
            label="query", label_visibility="collapsed",
            placeholder=t("ai_placeholder"), key="ai_query_input",
        )
    with ci2:
        send_clicked = st.button(f"🚀 {t('ai_send')}", use_container_width=True, type="primary")

    col_a, col_b = st.columns([4, 1])
    with col_b:
        if st.button(f"🗑️ {t('ai_clear')}", use_container_width=True):
            st.session_state.ai_history = []
            st.rerun()

    if send_clicked and user_query.strip():
        st.session_state.ai_history.append({"role": "user", "content": user_query})
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.ai_history]
        market_context, selected_count = build_ai_market_context(
            filtered_df,
            user_query,
            max_rows=18,
        )

        with st.spinner(t("ai_thinking")):
            try:
                ai_reply = ask_eisa_ai(
                    messages=messages,
                    market_context=market_context,
                    stock_count=selected_count,
                    language=st.session_state.language,
                )
            except requests.exceptions.Timeout:
                ai_reply = "⏱️ Request timed out. Please try again." if st.session_state.language == "en" else "⏱️ انتهى وقت الانتظار. يرجى المحاولة مرة أخرى."
            except requests.exceptions.HTTPError:
                ai_reply = "❌ AI service error. Please try again in a moment." if st.session_state.language == "en" else "❌ حدث خطأ في خدمة الذكاء الاصطناعي. حاول مرة أخرى بعد قليل."
            except Exception as e:
                ai_reply = f"❌ Error: {str(e)}"

        note = f"_Context used: {selected_count} relevant rows from {len(filtered_df)} filtered stocks._"
        ai_reply = f"{ai_reply}\n\n{note}"
        st.session_state.ai_history.append({"role": "assistant", "content": ai_reply})
        st.rerun()

# ═══════════════════════════════════════════════════════
# TAB 8 — Commodities
# ═══════════════════════════════════════════════════════
with tab8:
    st.markdown(f"### 🏭 {t('commodities_tab')}")

    cache_obj, _, _ = _get_pipeline()
    df_com, ts_com = cache_obj.get_latest("commodities") if cache_obj is not None else (None, None)

    if df_com is None or df_com.empty:
        st.warning(t("commodities_unavailable"))
    else:
        df_com = df_com.copy()
        for col in ("close", "change"):
            if col in df_com.columns:
                df_com[col] = pd.to_numeric(df_com[col], errors="coerce")

        kpi_items = [
            ("Gold", "🥇 Gold"),
            ("Crude Oil (WTI)", "🛢️ Oil WTI"),
            ("Silver", "🥈 Silver"),
        ]
        kpi_cols = st.columns(3)
        for idx, (commodity_name, label) in enumerate(kpi_items):
            row = _commodity_row(df_com, commodity_name)
            value = _format_price(row.get("close"), "america") if row is not None else "—"
            delta = f"{(row.get('change', 0) or 0):+.2f}%" if row is not None else None
            with kpi_cols[idx]:
                st.metric(label, value, delta=delta)

        if ts_com:
            st.caption(f"{t('last_update')}: {ts_com}")

        table_df = df_com.copy()
        if "close" in table_df.columns:
            table_df["close"] = table_df["close"].apply(lambda v: _format_price(v, "america"))
        if "change" in table_df.columns:
            table_df["change"] = table_df["change"].round(2)

        table_df = table_df.rename(columns={
            "name": t("name"),
            "close": t("price"),
            "change": t("one_day_change"),
        })
        show_cols = [c for c in [t("name"), t("price"), t("one_day_change")] if c in table_df.columns]
        st.dataframe(table_df[show_cols], use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="eisax-footer">
    <span>📅 {now_dubai_str()}</span>
    <span>📊 {len(filtered_df)} {t('stocks_count')}</span>
    <span class="brand">🚀 EisaX Analytics v5.1</span>
</div>
""", unsafe_allow_html=True)
