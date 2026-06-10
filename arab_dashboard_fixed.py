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
from core.streamlit_auth import require_auth, show_user_badge
from core.portfolio_db import save_portfolio as _ptf_save, load_portfolios as _ptf_load, delete_portfolio as _ptf_delete

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
DEEPSEEK_MODEL = DEFAULT_MODEL or "deepseek-v4-flash"

PIE_COLORS       = ["#0f4c81","#0ea5a4","#3b82f6","#8b5cf6","#f59e0b","#10b981","#f97316","#ef4444"]
COLOR_SCALE_CHANGE = [(0.0,"#ef4444"),(0.5,"#f8fafc"),(1.0,"#10b981")]
COLOR_SCALE_RSI    = [(0.0,"#10b981"),(0.5,"#f59e0b"),(1.0,"#ef4444")]
DUBAI_TZ = ZoneInfo("Asia/Dubai")

# ── EisaX Agent (lazy singleton) ─────────────────────────────────────────────
_eisax_agent = None

@st.cache_resource
def _get_eisax_agent():
    """Load MultiAgentOrchestrator once per Streamlit process."""
    try:
        from core.orchestrator import MultiAgentOrchestrator
        return MultiAgentOrchestrator()
    except Exception as _e:
        return None

_REPORTS_DIR = Path("/home/ubuntu/investwise/static/reports")

def _clip_ai_context(text: str, limit: int = 12000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[Context truncated]"

def _is_portfolio_query(message: str) -> bool:
    msg = (message or "").lower()
    keywords = (
        "portfolio", "allocation", "rebalance", "holdings", "position", "risk",
        "محفظ", "توزيع", "مخاطر", "صفقات", "مراكز", "اعادة توازن", "إعادة توازن",
    )
    return any(keyword in msg for keyword in keywords)

def _should_use_agent_for_ai(message: str, file_block: str = "") -> bool:
    if (file_block or "").strip():
        return True
    msg = (message or "").lower()
    agent_keywords = (
        "pdf", "report", "export", "download",
        "تقرير", "تصدير", "تحميل",
    )
    return any(keyword in msg for keyword in agent_keywords)

def _agent_chat(uid: int, message: str, session_id: str,
                portfolio_ctx: str = "", file_block: str = "") -> tuple[str, str, dict | None]:
    """
    Call EisaX agent synchronously from Streamlit.
    Returns (reply, session_id, download_info | None).
    download_info = {"filename": "...", "path": Path(...)} when agent generates a PDF.
    Falls back to ("", session_id, None) if agent unavailable — caller handles.
    """
    import asyncio
    import concurrent.futures

    agent = _get_eisax_agent()
    if agent is None:
        return "", session_id, None

    # Only inject portfolio context when the question is actually about the
    # user's portfolio. Injecting it into every message confuses the router.
    parts = []
    if file_block:
        parts.append(f"[FILE ANALYSIS]\nFile content below:\n\n{_clip_ai_context(file_block)}")
    if portfolio_ctx and (_is_portfolio_query(message) or file_block):
        parts.append(f"[PORTFOLIO CONTEXT — user's actual positions]\n{portfolio_ctx}")
    parts.append(f"User message: {message}")
    full_message = "\n\n".join(parts)

    async def _call():
        return await agent.process_message(
            user_id=str(uid),          # ← real uid, matches PortfolioTracker in Tab 5
            message=full_message,
            session_id=session_id or None,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(asyncio.run, _call()).result(timeout=55)

        reply    = result.get("reply") or result.get("response") or ""
        new_sid  = result.get("session_id", session_id)

        # ── Extract download info if agent produced a PDF ────────────────
        dl_info  = None
        raw_url  = result.get("download_url") or result.get("url") or ""
        if raw_url:
            filename = raw_url.split("/")[-1]
            pdf_path = _REPORTS_DIR / filename
            if pdf_path.exists():
                dl_info = {"filename": filename, "path": pdf_path}

        return reply, new_sid, dl_info

    except Exception:
        return "", session_id, None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EisaX | الأسواق العربية | Arab Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate — must come after set_page_config ───────────────────────────────
_current_user = require_auth()   # stops + shows login if not authenticated
_uid = _current_user["id"]       # int user id, used for per-user data

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("language", "ar"),
    ("dark_mode", False),
    ("watchlist", _wl_load(str(_uid))),
    ("_wl_loaded", True),
    ("ai_history", []),
    ("ai_session_id", None),   # EisaX agent session — persists across Tab 7 messages
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
        "forex_tab": "العملات",
        "crypto_tab": "العملات الرقمية",
        "forex_unavailable": "بيانات الفوركس غير متاحة مؤقتاً",
        "crypto_unavailable": "بيانات العملات الرقمية غير متاحة مؤقتاً",
        "pair": "الزوج",
        "rate": "السعر",
        "prev_close": "إغلاق سابق",
        "change_pct": "التغيير %",
        "category": "الفئة",
        "arab_pairs": "أزواج عربية",
        "major_pairs": "أزواج رئيسية",
        "em_pairs": "أسواق ناشئة",
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
        "forex_tab": "Forex",
        "crypto_tab": "Crypto",
        "forex_unavailable": "Forex data temporarily unavailable",
        "crypto_unavailable": "Crypto data temporarily unavailable",
        "pair": "Pair",
        "rate": "Rate",
        "prev_close": "Prev Close",
        "change_pct": "Change %",
        "category": "Category",
        "arab_pairs": "Arab Pairs",
        "major_pairs": "Major Pairs",
        "em_pairs": "Emerging Markets",
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


def _build_portfolio_ai_context(uid: int, df: "pd.DataFrame") -> str:
    """
    Build a compact portfolio summary for injection into the AI system prompt.
    Uses PortfolioTracker positions + current prices from parquet cache df.
    Returns empty string if no positions found.
    """
    try:
        from core.portfolio_tracker import PortfolioTracker as _PT
        tracker_state = _PT().get_portfolio(str(uid))
        positions = tracker_state.get("positions", [])
        cash      = float(tracker_state.get("cash", 0.0) or 0.0)

        position_items: list[tuple[str, dict]] = []
        if isinstance(positions, dict):
            for ticker, pos in positions.items():
                if not ticker or not isinstance(pos, dict):
                    continue
                position_items.append((str(ticker).strip().upper(), pos))
        else:
            for pos in positions or []:
                if not isinstance(pos, dict):
                    continue
                ticker = str(pos.get("ticker", "")).strip().upper()
                if not ticker:
                    continue
                position_items.append((ticker, pos))

        if not position_items:
            return ""

        # Build price lookup from parquet df (ticker -> close)
        price_map: dict = {}
        if df is not None and not df.empty and "close" in df.columns:
            for _, row in df.iterrows():
                ticker_key = str(row.get("ticker", "") or row.get("name", "")).strip().upper()
                market_key = str(row.get("market", "") or "").strip().lower()
                price_val = float(row["close"] or 0)
                if ticker_key:
                    price_map[ticker_key] = price_val
                    if market_key:
                        price_map[f"{market_key}:{ticker_key}".upper()] = price_val

        rows, total_cost, total_val = [], 0.0, 0.0
        for ticker, pos in position_items:
            market = str(pos.get("market", "") or "").strip().lower()
            shares = float(pos.get("shares", pos.get("qty", 0)) or 0)
            cost_p = float(pos.get("purchase_price", pos.get("cost_basis", pos.get("price", 0))) or 0)
            cur_p  = price_map.get(f"{market}:{ticker}".upper()) or price_map.get(ticker.upper(), cost_p)
            pos_cost = shares * cost_p
            pos_val  = shares * cur_p
            pnl      = pos_val - pos_cost
            pnl_pct  = (pnl / pos_cost * 100) if pos_cost else 0
            total_cost += pos_cost
            total_val  += pos_val
            rows.append(f"| {ticker} | {shares:.0f} | {cost_p:,.2f} | {cur_p:,.2f} | {pos_val:,.0f} | {pnl_pct:+.1f}% |")

        if not rows:
            return ""

        total_pnl = total_val - total_cost
        total_pct = (total_pnl / total_cost * 100) if total_cost else 0
        grand_total = total_val + cash
        lines = [
            "## User's Actual Portfolio",
            f"*{len(rows)} positions — data as of latest cache*",
            "",
            "| Ticker | Shares | Avg Cost | Current | Mkt Value | P&L% |",
            "|--------|--------|----------|---------|-----------|------|",
            *rows,
            "",
            f"**Total Invested:** {total_cost:,.0f} | "
            f"**Market Value:** {total_val:,.0f} | "
            f"**Unrealized P&L:** {total_pnl:+,.0f} ({total_pct:+.1f}%)",
        ]
        if cash > 0:
            lines.append(f"**Cash:** {cash:,.0f} | **Total (incl. cash):** {grand_total:,.0f}")
        return "\n".join(lines)
    except Exception:
        return ""


def ask_eisa_ai(messages, market_context: str, stock_count: int, language: str,
                portfolio_context: str = "", file_context: str = "") -> str:
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

    macro_line     = f"\nMacro context: {macro_ctx}" if macro_ctx else ""
    portfolio_line = f"\n\n{portfolio_context}" if portfolio_context else ""
    file_line      = f"\n\nUploaded file context:\n{_clip_ai_context(file_context, limit=16000)}" if file_context else ""
    portfolio_instruction = (
        "\nThe user's ACTUAL portfolio is provided above — always refer to it when answering "
        "portfolio-related questions. Use real position sizes and P&L in your analysis."
        if portfolio_context else ""
    )
    file_instruction = (
        "\nAn uploaded user file is provided above. Use it as the primary source for holdings, weights, "
        "and portfolio structure whenever the user asks about that file."
        if file_context else ""
    )
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
{portfolio_instruction}{file_instruction}

Market data (filtered, {stock_count} stocks):
{market_context}
{macro_line}{portfolio_line}{file_line}
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
        bg           = "#080e1a"
        surface      = "#0f172a"
        surface2     = "#1e293b"
        text         = "#f1f5f9"
        subtext      = "#94a3b8"
        border_col   = "#1e293b"
        header_bg    = "linear-gradient(135deg,#080e1a 0%,#0d1f3c 60%,#080e1a 100%)"
        card_bg      = "rgba(15,23,42,0.8)"
        tab_bg       = "rgba(15,23,42,0.95)"
        grid_col     = "#1e293b"
        app_bg       = "#080e1a"
        glass_border = "rgba(56,189,248,0.12)"
        shadow_soft  = "rgba(0,0,0,0.5)"
        accent_glow  = "rgba(14,165,164,0.25)"
        input_bg     = "#0f172a"
        input_border = "#334155"
    else:
        bg           = "#f8fafc"
        surface      = "#ffffff"
        surface2     = "#f1f5f9"
        text         = "#0f172a"
        subtext      = "#475569"
        border_col   = "#e2e8f0"
        header_bg    = "linear-gradient(135deg,#ffffff 0%,#eff6ff 50%,#f0fdf4 100%)"
        card_bg      = "rgba(255,255,255,0.9)"
        tab_bg       = "rgba(255,255,255,0.95)"
        grid_col     = "#f1f5f9"
        app_bg       = "linear-gradient(160deg,#f8fafc 0%,#eff6ff 50%,#f0fdf4 100%)"
        glass_border = "rgba(15,76,129,0.1)"
        shadow_soft  = "rgba(15,23,42,0.08)"
        accent_glow  = "rgba(14,165,164,0.15)"
        input_bg     = "#ffffff"
        input_border = "#cbd5e1"

    return f'''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Tajawal:wght@300;400;500;700&display=swap');

        /* ── Base ── */
        .stApp {{
            background: {app_bg};
            font-family: 'Inter','Tajawal','Segoe UI',sans-serif;
        }}
        .stMainBlockContainer, [data-testid="stSidebarContent"] {{
            direction: {direction};
        }}

        /* ── ALL text inherits correctly ── */
        .stApp, .stApp p, .stApp li, .stApp span, .stApp div,
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {{
            color: {text};
        }}

        /* ── Headings ── */
        .stApp h1,.stApp h2,.stApp h3,.stApp h4 {{
            color: {text};
            font-weight: 700;
        }}

        /* ── Captions & labels ── */
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {subtext} !important;
        }}
        label, .stLabel, [data-baseweb="label"],
        .stTextInput label, .stSelectbox label,
        .stNumberInput label, .stTextArea label,
        .stSlider label, .stMultiSelect label {{
            color: {subtext} !important;
            font-size: .82rem !important;
            font-weight: 500 !important;
        }}

        /* ── Input fields ── */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {{
            background: {input_bg} !important;
            color: {text} !important;
            border: 1px solid {input_border} !important;
            border-radius: 10px !important;
        }}
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: #0ea5a4 !important;
            box-shadow: 0 0 0 2px rgba(14,165,164,0.2) !important;
        }}
        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {{
            color: {subtext} !important;
            opacity: 0.7;
        }}

        /* ── Select / Dropdown ── */
        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div > div {{
            background: {input_bg} !important;
            color: {text} !important;
            border-color: {input_border} !important;
            border-radius: 10px !important;
        }}
        [data-baseweb="popover"] [role="option"] {{
            background: {surface} !important;
            color: {text} !important;
        }}
        [data-baseweb="popover"] [role="option"]:hover {{
            background: {surface2} !important;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: {surface} !important;
            border-{border_side}: 2px solid {glass_border};
        }}
        [data-testid="stSidebar"] * {{ color: {text}; }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption {{ color: {subtext} !important; }}
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: {surface2} !important;
            color: {text} !important;
            border-color: {input_border} !important;
        }}

        /* ── Metric (st.metric) ── */
        [data-testid="stMetricValue"] {{ color: {text} !important; font-weight: 800; font-size: 1.6rem; }}
        [data-testid="stMetricLabel"] {{ color: {subtext} !important; font-size: .8rem; font-weight: 500; }}
        [data-testid="stMetricDelta"] {{ font-weight: 600; }}

        /* ── Expanders ── */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary {{
            background: {surface} !important;
            color: {text} !important;
            border-radius: 12px;
            border: 1px solid {glass_border};
            padding: .6rem 1rem;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid {glass_border} !important;
            border-radius: 12px !important;
            background: {surface} !important;
        }}
        [data-testid="stExpander"] > div > div {{ background: {surface}; color: {text}; }}

        /* ── Dataframe ── */
        .stDataFrame {{ border-radius: 14px; overflow: hidden; border: 1px solid {glass_border}; }}
        .stDataFrame [data-testid="stDataFrameResizable"] {{
            background: {surface} !important;
        }}
        .stDataFrame th {{
            background: {surface2} !important;
            color: {text} !important;
            font-weight: 600;
        }}
        .stDataFrame td {{ color: {text} !important; background: {surface} !important; }}

        /* ── Alerts / Info boxes ── */
        [data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid {glass_border};
        }}
        .stAlert [data-testid="stMarkdownContainer"] p {{ color: {text} !important; }}

        /* ── Main header ── */
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .main-header {{
            background: {header_bg};
            border-radius: 24px;
            padding: 2.2rem 2rem;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 40px {shadow_soft};
            border: 1px solid {glass_border};
            position: relative;
            overflow: hidden;
        }}
        .main-header::before {{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at 20% 50%, {accent_glow} 0%, transparent 60%),
                        radial-gradient(ellipse at 80% 20%, rgba(99,102,241,0.1) 0%, transparent 50%);
            pointer-events: none;
        }}
        .main-header::after {{
            content: '';
            position: absolute;
            bottom: 0; left: 10%; right: 10%;
            height: 1px;
            background: linear-gradient(90deg, transparent, #0ea5a4, transparent);
        }}
        .main-header h1 {{
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: .3rem;
            letter-spacing: -0.02em;
            position: relative;
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5a4 50%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .main-header p {{
            color: {subtext};
            font-size: .9rem;
            position: relative;
        }}

        /* ── Pulse ── */
        @keyframes pulse-live {{
            0%,100% {{ opacity:1; transform:scale(1); }}
            50%      {{ opacity:.5; transform:scale(1.4); }}
        }}
        .live-dot {{
            display:inline-block; width:8px; height:8px;
            background:#10b981; border-radius:50%;
            margin-inline-end:6px;
            animation:pulse-live 2s ease-in-out infinite;
            vertical-align:middle;
            box-shadow: 0 0 8px #10b981;
        }}

        /* ── Metric card ── */
        @keyframes fadeSlideUp {{
            from {{ opacity:0; transform:translateY(16px); }}
            to   {{ opacity:1; transform:translateY(0); }}
        }}
        .metric-card {{
            background: {card_bg};
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            padding: 1.3rem 1rem;
            text-align: center;
            box-shadow: 0 4px 24px {shadow_soft};
            border: 1px solid {glass_border};
            animation: fadeSlideUp 0.5s ease-out both;
            transition: transform .3s cubic-bezier(.4,0,.2,1), box-shadow .3s ease;
            position: relative;
            overflow: hidden;
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #0f4c81, #0ea5a4);
        }}
        .metric-card:hover {{
            transform: translateY(-6px);
            box-shadow: 0 16px 40px {shadow_soft};
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #38bdf8, #0ea5a4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .metric-label {{
            font-size: .78rem;
            color: {subtext};
            margin-top: .3rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: .06em;
        }}

        /* ── Sentiment bar ── */
        .sentiment-bar {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 1rem 1.4rem;
            margin-bottom: 1rem;
            border: 1px solid {glass_border};
            box-shadow: 0 2px 16px {shadow_soft};
        }}
        .sentiment-track {{
            height: 8px;
            border-radius: 8px;
            background: linear-gradient(90deg,#ef4444 0%,#f59e0b 45%,#10b981 100%);
            position: relative;
            overflow: visible;
        }}

        /* ── Stock card ── */
        .stock-card {{
            background: {card_bg};
            backdrop-filter: blur(8px);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin: .5rem 0;
            box-shadow: 0 2px 12px {shadow_soft};
            border-{card_border}: 3px solid #0ea5a4;
            border-top: 1px solid {glass_border};
            border-bottom: 1px solid {glass_border};
            border-{border_side}: 1px solid {glass_border};
            transition: transform .3s cubic-bezier(.4,0,.2,1), box-shadow .3s ease;
        }}
        .stock-card:hover {{
            transform: translateX({hover_tx});
            box-shadow: 0 8px 28px {shadow_soft};
        }}
        .stock-card * {{ color: {text} !important; }}

        /* ── Badges ── */
        .badge-success {{ background:rgba(16,185,129,.15); color:#10b981; border:1px solid rgba(16,185,129,.3); padding:.2rem .7rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}
        .badge-warning {{ background:rgba(245,158,11,.15); color:#f59e0b; border:1px solid rgba(245,158,11,.3); padding:.2rem .7rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}
        .badge-danger  {{ background:rgba(239,68,68,.15);  color:#ef4444; border:1px solid rgba(239,68,68,.3);  padding:.2rem .7rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}
        .badge-info    {{ background:rgba(56,189,248,.15); color:#38bdf8; border:1px solid rgba(56,189,248,.3); padding:.2rem .7rem; border-radius:20px; font-size:.75rem; font-weight:600; display:inline-block; }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap:.4rem;
            background: {tab_bg};
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: .4rem;
            border: 1px solid {glass_border};
            margin-bottom: 1.2rem;
            flex-wrap: wrap;
            box-shadow: 0 4px 16px {shadow_soft};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 12px;
            padding: .5rem 1.1rem;
            font-weight: 600;
            font-size: .88rem;
            color: {subtext} !important;
            transition: all .2s ease;
            border: 1px solid transparent;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: {surface2};
            color: {text} !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg,#0f4c81,#0ea5a4) !important;
            color: white !important;
            box-shadow: 0 4px 16px rgba(14,165,164,0.35);
            border-color: transparent !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background: linear-gradient(135deg,#0f4c81,#0ea5a4);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: .55rem 1.2rem;
            font-weight: 600;
            font-family: 'Inter','Tajawal',sans-serif;
            font-size: .9rem;
            transition: all .25s cubic-bezier(.4,0,.2,1);
            box-shadow: 0 2px 12px rgba(14,165,164,0.25);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(14,165,164,0.4);
        }}
        .stButton > button[kind="secondary"] {{
            background: {surface2} !important;
            color: {text} !important;
            box-shadow: none;
            border: 1px solid {glass_border} !important;
        }}

        /* ── Chat bubbles ── */
        .chat-user {{
            background: linear-gradient(135deg,#0f4c81,#0ea5a4);
            color: white !important;
            border-radius: 20px 20px 4px 20px;
            padding: .9rem 1.2rem;
            margin: .5rem 0 .5rem auto;
            max-width: 75%;
            box-shadow: 0 4px 16px rgba(14,165,164,0.3);
            font-size: .93rem;
            line-height: 1.6;
        }}
        .chat-ai {{
            background: {card_bg};
            backdrop-filter: blur(8px);
            color: {text} !important;
            border-radius: 20px 20px 20px 4px;
            padding: .9rem 1.2rem;
            margin: .5rem auto .5rem 0;
            max-width: 85%;
            border: 1px solid {glass_border};
            box-shadow: 0 4px 16px {shadow_soft};
            font-size: .93rem;
            line-height: 1.7;
        }}
        .chat-ai * {{ color: {text} !important; }}

        /* ── User badge card ── */
        .user-badge-card {{
            background: {surface2};
            border-radius: 14px;
            padding: .8rem 1rem;
            border: 1px solid {glass_border};
            margin-bottom: .5rem;
            display: flex;
            align-items: center;
            gap: .75rem;
        }}
        .user-badge-avatar {{
            width:38px; height:38px;
            background: linear-gradient(135deg,#0f4c81,#0ea5a4);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; color: white; font-size: .95rem; flex-shrink: 0;
        }}
        .user-badge-info {{ flex:1; min-width:0; }}
        .user-badge-name {{ font-weight:600; color:{text}; font-size:.88rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .user-badge-email {{ font-size:.72rem; color:{subtext}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

        /* ── Cache pills ── */
        .cache-pill-fresh {{ background:rgba(16,185,129,.15); color:#10b981; padding:.2rem .6rem; border-radius:20px; font-size:.72rem; border:1px solid rgba(16,185,129,.3); }}
        .cache-pill-stale {{ background:rgba(239,68,68,.15); color:#ef4444; padding:.2rem .6rem; border-radius:20px; font-size:.72rem; border:1px solid rgba(239,68,68,.3); }}

        /* ── Positive / Negative ── */
        .positive {{ color:#10b981 !important; font-weight:700; }}
        .negative {{ color:#ef4444 !important; font-weight:700; }}

        /* ── Divider ── */
        hr {{ margin:1.2rem 0; border:none; height:1px; background:linear-gradient(to right,transparent,{border_col},transparent); }}

        /* ── Footer ── */
        .eisax-footer {{
            background: {card_bg};
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: .8rem 1.5rem;
            margin-top: 1.5rem;
            border-top: 2px solid {glass_border};
            border: 1px solid {glass_border};
            box-shadow: 0 -2px 16px {shadow_soft};
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: .5rem;
        }}
        .eisax-footer span {{ font-size:.8rem; color:{subtext}; font-weight:500; }}
        .eisax-footer .brand {{
            font-weight:800;
            background: linear-gradient(135deg,#38bdf8,#0ea5a4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width:5px; height:5px; }}
        ::-webkit-scrollbar-track {{ background:{surface}; border-radius:10px; }}
        ::-webkit-scrollbar-thumb {{ background:linear-gradient(135deg,#0f4c81,#0ea5a4); border-radius:10px; }}

        /* ── Responsive ── */
        @media (max-width:768px) {{
            .main-header h1 {{ font-size:1.6rem; }}
            .metric-value {{ font-size:1.4rem; }}
            .eisax-footer {{ flex-direction:column; text-align:center; }}
            .chat-user,.chat-ai {{ max-width:95%; }}
        }}
    </style>
    '''

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
def style_chart(fig, height=400, show_legend=None):
    bg = "#1e293b" if st.session_state.dark_mode else "white"
    tc = "#94a3b8" if st.session_state.dark_mode else "#475569"
    gc = "#334155" if st.session_state.dark_mode else "#f1f5f9"
    layout_kwargs = dict(
        template="plotly_dark" if st.session_state.dark_mode else "plotly_white",
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(family="Inter, Tajawal, sans-serif", size=12, color=tc),
        title_font=dict(size=15, color="#0f4c81"),
        margin=dict(l=20, r=20, t=50, b=30),
        height=height,
    )
    if show_legend is not None:
        layout_kwargs["showlegend"] = show_legend
    fig.update_layout(**layout_kwargs)
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
    show_user_badge()
    # Agent status indicator
    agent_ok = _get_eisax_agent() is not None
    _status_color = '#10b981' if agent_ok else '#f59e0b'
    _status_text  = ('🟢 EisaX Agent متصل' if st.session_state.get('language','ar') == 'ar' else '🟢 EisaX Agent connected') if agent_ok else ('🟡 وضع احتياطي' if st.session_state.get('language','ar') == 'ar' else '🟡 Fallback mode')
    st.sidebar.markdown(f'<div style="font-size:.75rem;color:{_status_color};margin-bottom:.5rem">{_status_text}</div>', unsafe_allow_html=True)
    st.markdown("---")
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
        st.caption(("In cache — apply filters to narrow down" if st.session_state.language == "en"
                    else f"في الكاش — بعد الفلاتر: {len(filtered_df) if 'filtered_df' in dir() else '…'}"))

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
        <span style="color:#64748b;font-weight:600;">{len(filtered_df)} {t('stocks_count')} {"/ " + str(len(df)) + " total" if len(filtered_df) != len(df) else ""}</span>
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
    _fc_label = t("stocks_count") + (f" / {len(df)}" if len(filtered_df) != len(df) else "")
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(filtered_df)}</div><div class="metric-label">{_fc_label}</div></div>', unsafe_allow_html=True)

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
_is_admin = _current_user.get("role") == "admin"
_tab_labels = [
    f"📋 {t('stocks_tab')}",
    f"🎯 {t('opportunities_tab')}",
    f"📈 {t('analysis_tab')}",
    f"🏭 {t('sectors_tab')}",
    f"💼 {t('portfolio_tab')}",
    f"⭐ {t('watchlist_tab')}",
    f"🤖 {t('ai_tab')}",
    f"🏭 {t('commodities_tab')}",
    f"💱 {t('forex_tab')}",
    f"🪙 {t('crypto_tab')}",
]
if _is_admin:
    _tab_labels.append("🛡️ Admin")

_all_tabs = st.tabs(_tab_labels)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = _all_tabs[:10]
tab_admin = _all_tabs[10] if _is_admin else None

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
                _wl_add(picked, str(_uid))
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

        # ── Saved portfolios (Phase J) ─────────────────────────────────────────
        _saved = _ptf_load(_uid)
        _default_holdings = "uae:EMAAR 1000 14.50\nuae:FAB 500 18.00\nksa:2222.SR 300 30.00"

        if _saved:
            _ptf_names = ["— " + ("new" if st.session_state.language == "en" else "جديد") + " —"] + [p["name"] for p in _saved]
            _sel = st.selectbox(
                "📂 " + ("Load saved portfolio" if st.session_state.language == "en" else "تحميل محفظة محفوظة"),
                _ptf_names, key="ptf_load_sel"
            )
            if _sel != _ptf_names[0]:
                _match = next((p for p in _saved if p["name"] == _sel), None)
                if _match:
                    _default_holdings = _match["holdings"]

        holdings_text = st.text_area(t("holdings"), value=_default_holdings, height=150)
        target_text = st.text_input(
            "🎯 " + ("Target sector allocation (optional)" if st.session_state.language == "en" else "توزيع القطاعات المستهدف (اختياري)"),
            placeholder="Finance:40 Energy:30 Technology:20",
        )

        # ── Save portfolio row ─────────────────────────────────────────────────
        _sc1, _sc2, _sc3 = st.columns([3, 1, 1])
        with _sc1:
            _ptf_name_input = st.text_input(
                "💾 " + ("Save as..." if st.session_state.language == "en" else "احفظ باسم..."),
                placeholder="My Portfolio", label_visibility="collapsed",
                key="ptf_name_input"
            )
        with _sc2:
            if st.button("💾 " + ("Save" if st.session_state.language == "en" else "حفظ"), use_container_width=True):
                _name = _ptf_name_input.strip() or "محفظتي"
                if _ptf_save(_uid, _name, holdings_text):
                    st.success(f"✅ {'Saved' if st.session_state.language == 'en' else 'تم الحفظ'}: {_name}")
                    st.rerun()
        with _sc3:
            if _saved and st.session_state.get("ptf_load_sel") and st.session_state.ptf_load_sel not in ["— جديد —", "— new —"]:
                if st.button("🗑️ " + ("Delete" if st.session_state.language == "en" else "حذف"), use_container_width=True):
                    _ptf_delete(_uid, st.session_state.ptf_load_sel)
                    st.rerun()

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
                # Normalize: uppercase + strip whitespace
                ticker = ticker.strip().upper()
                market_code = market_code.strip().lower()
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

        # ══════════════════════════════════════════════════
        # POSITIONS TRACKER + ANALYZE FROM POSITIONS
        # ══════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### 📌 " + ("Positions Tracker" if st.session_state.language == "en" else "متابعة الصفقات"))

        try:
            from core.portfolio_tracker import PortfolioTracker
            _tracker  = PortfolioTracker()
            _ptf_data = _tracker.get_portfolio(str(_uid))
            _positions = _ptf_data.get("positions", [])

            # bilingual helper — available throughout the whole tracker block
            _ar = st.session_state.language == "ar"
            def _bi(ar_txt, en_txt): return ar_txt if _ar else en_txt

            # ── Add Position form ─────────────────────────────────────────────
            _MARKET_OPTIONS = ["uae","ksa","egypt","qatar","bahrain","kuwait",
                               "morocco","tunisia","america","commodities","crypto"]
            with st.expander("➕ " + ("Add Position" if st.session_state.language == "en" else "إضافة صفقة"),
                             expanded=not _positions):
                pf1, pf2, pf3, pf4, pf5 = st.columns([2, 1.5, 1, 1, 1])
                with pf1:
                    _pt_ticker = st.text_input("Ticker", placeholder="EMAAR", key="pt_ticker").upper().strip()
                with pf2:
                    _pt_market = st.selectbox("Market", _MARKET_OPTIONS, key="pt_market")
                with pf3:
                    _pt_shares = st.number_input("Shares", min_value=0.01, value=100.0, step=1.0, key="pt_shares")
                with pf4:
                    _pt_price  = st.number_input("Buy Price", min_value=0.001, value=10.0, step=0.01, key="pt_price")
                with pf5:
                    _pt_date   = st.date_input("Date", key="pt_date")

                if st.button("➕ " + ("Add" if st.session_state.language == "en" else "إضافة"), key="pt_add", type="primary"):
                    if _pt_ticker:
                        _tracker.add_position(str(_uid), _pt_ticker, _pt_shares, _pt_price,
                                              date=str(_pt_date), market=_pt_market)
                        st.success(f"✅ {_pt_ticker} ({_pt_market.upper()}) added")
                        st.rerun()
                    else:
                        st.error("أدخل رمز السهم")

            if not _positions:
                st.info("لا توجد صفقات مسجّلة بعد — أضف أولى صفقاتك أعلاه" if st.session_state.language == "ar"
                        else "No positions yet — add your first trade above.")
            else:
                # ── Build enriched position rows from parquet cache ───────────
                _total_cost = 0.0
                _total_val  = 0.0
                _pos_rows   = []
                _ptf_for_analysis = []  # (ticker, market, shares, cost)

                for _pos in _positions:
                    _tk   = _pos["ticker"]
                    _mkt  = _pos.get("market", "uae")
                    _sh   = _pos["shares"]
                    _cost = _pos["purchase_price"]

                    # look up current price in parquet
                    _cur_price, _rsi_val = None, None
                    if not df.empty and "name" in df.columns:
                        _m = df[df["name"] == _tk]
                        if not _m.empty:
                            _cur_price = _m.iloc[0].get("close")
                            _rsi_val   = _m.iloc[0].get("RSI")

                    _cost_total = _sh * _cost
                    _total_cost += _cost_total
                    if _cur_price:
                        _val   = _sh * float(_cur_price)
                        _pnl   = _val - _cost_total
                        _pnl_p = (_pnl / _cost_total * 100) if _cost_total else 0
                        _total_val += _val
                    else:
                        _val, _pnl, _pnl_p = None, None, None

                    _ptf_for_analysis.append({"ticker": _tk, "market": _mkt,
                                              "shares": _sh, "cost": _cost,
                                              "cur_price": _cur_price, "rsi": _rsi_val,
                                              "val": _val, "pnl": _pnl, "pnl_pct": _pnl_p})
                    _pos_rows.append({
                        "Ticker":     _tk,
                        "Market":     _mkt.upper(),
                        "Shares":     _sh,
                        "Buy":        round(_cost, 3),
                        "Current":    round(float(_cur_price), 3) if _cur_price else "—",
                        "Cost Basis": f"{_cost_total:,.2f}",
                        "P&L":        f"{_pnl:+,.2f} ({_pnl_p:+.1f}%)" if _pnl is not None else "—",
                        "RSI":        round(float(_rsi_val), 1) if _rsi_val else "—",
                        "Date":       _pos.get("purchase_date", "—"),
                    })

                # ── Cash field ────────────────────────────────────────────────
                _cash_val = float(_ptf_data.get("cash", 0.0))
                _cash_c1, _cash_c2 = st.columns([3, 1])
                with _cash_c1:
                    _new_cash = st.number_input(
                        "💵 " + ("Cash available (in portfolio currency)" if st.session_state.language == "en"
                                 else "السيولة النقدية المتاحة (بعملة المحفظة)"),
                        min_value=0.0, value=_cash_val, step=1000.0, key="pt_cash_input", format="%.2f")
                with _cash_c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("💾 " + ("Save" if st.session_state.language == "en" else "حفظ"),
                                 key="pt_cash_save", use_container_width=True):
                        _tracker.set_cash(str(_uid), _new_cash)
                        st.rerun()
                _cash_val = _new_cash  # use latest value

                # ── Summary Card ──────────────────────────────────────────────
                _total_with_cash = (_total_val + _cash_val) if _total_val else _cash_val
                _cash_pct  = (_cash_val / _total_with_cash * 100) if _total_with_cash else 0
                _unreal_gain = _total_val - _total_cost if _total_val else None
                _unreal_pct  = (_unreal_gain / _total_cost * 100) if (_unreal_gain is not None and _total_cost) else None

                _winners = sorted([p for p in _ptf_for_analysis if p["pnl_pct"] is not None],
                                  key=lambda x: x["pnl_pct"], reverse=True)
                _top_w   = _winners[0]  if _winners else None
                _top_l   = _winners[-1] if len(_winners) > 1 else None

                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                with sc1:
                    st.metric("💰 " + ("Invested" if st.session_state.language == "en" else "مستثمر"),
                              f"{_total_cost:,.0f}")
                with sc2:
                    _val_str = f"{_total_val:,.0f}" if _total_val else "—"
                    st.metric("📊 " + ("Market Value" if st.session_state.language == "en" else "قيمة السوق"),
                              _val_str)
                with sc3:
                    if _unreal_gain is not None:
                        _gc = "#10b981" if _unreal_gain >= 0 else "#ef4444"
                        st.markdown(f"<div><strong>{'P&L' if st.session_state.language == 'en' else 'ربح/خسارة'}</strong>"
                                    f"<br><span style='color:{_gc};font-size:1.15rem;font-weight:700'>"
                                    f"{_unreal_gain:+,.0f} ({_unreal_pct:+.1f}%)</span></div>",
                                    unsafe_allow_html=True)
                    else:
                        st.metric("P&L", "—")
                with sc4:
                    _cash_c = "#10b981" if _cash_pct < 20 else "#f59e0b" if _cash_pct < 40 else "#ef4444"
                    st.markdown(f"<div><strong>💵 {'Cash' if st.session_state.language == 'en' else 'سيولة'}</strong>"
                                f"<br><span style='color:{_cash_c};font-size:1.15rem;font-weight:700'>"
                                f"{_cash_val:,.0f} ({_cash_pct:.0f}%)</span></div>",
                                unsafe_allow_html=True)
                with sc5:
                    _wl_txt = ""
                    if _top_w:
                        _wl_txt += f"🏆 {_top_w['ticker']} {_top_w['pnl_pct']:+.1f}%"
                    if _top_l:
                        _wl_txt += f"\n🔻 {_top_l['ticker']} {_top_l['pnl_pct']:+.1f}%"
                    st.markdown(f"<div><strong>{'Leaders' if st.session_state.language == 'en' else 'الأبرز'}</strong>"
                                f"<br><pre style='font-size:.8rem;margin:0'>{_wl_txt or '—'}</pre></div>",
                                unsafe_allow_html=True)

                # ── Positions table ───────────────────────────────────────────
                st.dataframe(pd.DataFrame(_pos_rows), use_container_width=True, hide_index=True)

                # Remove position
                _rmc1, _rmc2 = st.columns([3, 1])
                with _rmc1:
                    _rm_ticker = st.selectbox("🗑️ " + ("Remove position" if st.session_state.language == "en" else "حذف صفقة"),
                                             ["—"] + [p["ticker"] for p in _positions], key="pt_rm")
                with _rmc2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if _rm_ticker != "—" and st.button("حذف", key="pt_rm_btn", type="secondary"):
                        _tracker.remove_position(str(_uid), _rm_ticker)
                        st.rerun()

                # ══════════════════════════════════════════════════════════════
                # ANALYZE FROM POSITIONS
                # ══════════════════════════════════════════════════════════════
                st.markdown("---")
                if st.button("🔍 " + ("Analyze Portfolio" if st.session_state.language == "en" else "تحليل المحفظة من الصفقات"),
                             type="primary", use_container_width=True, key="pt_analyze"):
                    st.session_state["pt_show_analysis"] = True

                if st.session_state.get("pt_show_analysis") and qe is not None:
                    with st.spinner("جاري التحليل..." if st.session_state.language == "ar" else "Analyzing..."):
                        try:
                            from portfolio import Portfolio as _Ptf
                            _p = _Ptf(qe)
                            for _pa in _ptf_for_analysis:
                                _p.add(_pa["ticker"], market=_pa["market"],
                                       qty=_pa["shares"], cost_basis=_pa["cost"])
                            _smry = _p.summary()
                        except Exception as _ae:
                            st.error(f"Analysis error: {_ae}")
                            _smry = None

                    if _smry:
                        _pos_df2 = _smry.get("positions", pd.DataFrame())

                        # ── KPI row ───────────────────────────────────────────
                        ak1, ak2, ak3, ak4 = st.columns(4)
                        with ak1:
                            st.metric("💰 " + t("total_value"),
                                      f"{_smry['total_value']:,.0f}" if _smry['total_value'] else "—")
                        with ak2:
                            _ap = _smry.get("total_pnl")
                            _ac = "green" if _ap and _ap > 0 else "red"
                            st.markdown(f"<div><strong>📊 Unrealized P&L</strong><br>"
                                        f"<span style='color:{_ac}'>{_ap:+,.0f}</span></div>"
                                        if _ap is not None else "<div>P&L: —</div>", unsafe_allow_html=True)
                        with ak3:
                            _risk = _smry.get("risk_score", 50)
                            _rc  = "#ef4444" if _risk > 70 else "#f59e0b" if _risk > 40 else "#10b981"
                            _rl  = (t("high_risk") if _risk > 70 else t("medium_risk") if _risk > 40 else t("low_risk"))
                            st.markdown(f"<div><strong>⚠️ Risk</strong><br>"
                                        f"<span style='color:{_rc}'>{_rl} ({_risk}/100)</span></div>",
                                        unsafe_allow_html=True)
                        with ak4:
                            st.metric("🌍 Diversification", _smry.get("diversification","—"))

                        # ── Allocation charts ─────────────────────────────────
                        _ch1, _ch2, _ch3 = st.columns(3)
                        with _ch1:
                            _sw = _smry.get("sector_weights", {})
                            if _sw:
                                _fig_s = px.pie(values=list(_sw.values()), names=list(_sw.keys()),
                                                title=_bi("توزيع القطاعات","Sector Allocation"),
                                                color_discrete_sequence=PIE_COLORS, hole=0.4)
                                style_chart(_fig_s, height=260)
                                _fig_s.update_layout(margin=dict(l=0,r=0,t=30,b=0),
                                                     paper_bgcolor="rgba(0,0,0,0)",
                                                     plot_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(_fig_s, use_container_width=True, key="ptf_sector_pie")
                        with _ch2:
                            _mw = _smry.get("market_weights", {})
                            if _mw:
                                _fig_m = px.pie(values=list(_mw.values()), names=list(_mw.keys()),
                                                title=_bi("توزيع الأسواق","Market Allocation"),
                                                color_discrete_sequence=PIE_COLORS, hole=0.4)
                                style_chart(_fig_m, height=260)
                                _fig_m.update_layout(margin=dict(l=0,r=0,t=30,b=0),
                                                     paper_bgcolor="rgba(0,0,0,0)",
                                                     plot_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(_fig_m, use_container_width=True, key="ptf_market_pie")
                        with _ch3:
                            # Asset class breakdown: Invested vs Cash
                            if _total_with_cash > 0:
                                _alloc_names = [_bi("مستثمر","Invested"), _bi("سيولة","Cash")]
                                _alloc_vals  = [_total_val or 0, _cash_val]
                                _fig_ac = px.pie(values=_alloc_vals, names=_alloc_names,
                                                 title=_bi("المحفظة الكلية","Total Allocation"),
                                                 color_discrete_sequence=["#0ea5a4","#f59e0b"], hole=0.4)
                                style_chart(_fig_ac, height=260)
                                _fig_ac.update_layout(margin=dict(l=0,r=0,t=30,b=0),
                                                      paper_bgcolor="rgba(0,0,0,0)",
                                                      plot_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(_fig_ac, use_container_width=True, key="ptf_cash_pie")

                        # ── Per-position recommendations ──────────────────────
                        st.markdown("##### 💡 " + ("Position Recommendations" if st.session_state.language == "en"
                                                   else "توصيات لكل صفقة"))

                        def _position_action(row, weight_pct):
                            reasons, action = [], _bi("⏸️ احتفظ", "⏸️ Hold")
                            rsi   = row.get("RSI")
                            pnl_p = row.get("pnl_pct")

                            if weight_pct and weight_pct > 35:
                                action = _bi("⚠️ خفّف", "⚠️ Reduce")
                                reasons.append(_bi(f"تركّز {weight_pct:.0f}% من المحفظة",
                                                   f"Concentration {weight_pct:.0f}% of portfolio"))
                            if rsi and rsi > 70:
                                action = _bi("🔴 جنّي أرباح", "🔴 Take Profits")
                                reasons.append(_bi(f"RSI في ذروة الشراء ({rsi:.0f})", f"RSI overbought ({rsi:.0f})"))
                            elif rsi and rsi < 30:
                                action = _bi("🟢 زيادة", "🟢 Add")
                                reasons.append(_bi(f"RSI في ذروة البيع ({rsi:.0f})", f"RSI oversold ({rsi:.0f})"))
                            if pnl_p and pnl_p > 50 and "احتفظ" in action or "Hold" in action:
                                action = _bi("💛 إعادة توازن", "💛 Rebalance")
                                reasons.append(_bi(f"ربح كبير {pnl_p:+.0f}% — فكر في تثبيت الأرباح",
                                                   f"Large gain {pnl_p:+.0f}% — consider locking in"))
                            if pnl_p and pnl_p < -20:
                                if _bi("احتفظ","Hold") in action:
                                    action = _bi("📋 راجع", "📋 Review")
                                reasons.append(_bi(f"خسارة {pnl_p:.0f}% — راجع مسوّغ الدخول",
                                                   f"Down {pnl_p:.0f}% — review thesis"))
                            if not reasons:
                                p2, s50, s200 = row.get("price"), row.get("SMA50"), row.get("SMA200")
                                if p2 and s50 and s200 and p2 > s50 > s200:
                                    reasons.append(_bi("اتجاه صاعد (السعر > SMA50 > SMA200)",
                                                       "Uptrend (price > SMA50 > SMA200)"))
                                else:
                                    reasons.append(_bi("ضمن المعدل الطبيعي", "Within normal parameters"))
                            return action, " | ".join(reasons)

                        _total_v_smry = _smry.get("total_value") or 1
                        _rec_rows = []
                        _portfolio_actions = []
                        if not _pos_df2.empty:
                            for _, _row in _pos_df2.iterrows():
                                _w = round((_row.get("value") or 0) / _total_v_smry * 100, 1)
                                _act, _why = _position_action(_row, _w)
                                _portfolio_actions.append(_act)
                                _rec_rows.append({
                                    _bi("السهم","Ticker"):   _row["ticker"],
                                    _bi("السوق","Market"):   _row.get("market","—").upper(),
                                    _bi("الوزن","Weight"):   f"{_w:.1f}%",
                                    "P&L":                   f"{_row['pnl_pct']:+.1f}%" if _row.get("pnl_pct") is not None else "—",
                                    "RSI":                   f"{_row['RSI']:.0f}" if _row.get("RSI") is not None else "—",
                                    _bi("القرار","Action"):  _act,
                                    _bi("السبب","Reason"):   _why,
                                })
                        if _rec_rows:
                            st.dataframe(pd.DataFrame(_rec_rows), use_container_width=True, hide_index=True)

                        # ── Portfolio-level risks ─────────────────────────────
                        st.markdown("##### 🚨 " + _bi("أبرز المخاطر", "Top Risks"))
                        _risks = []
                        _risk_reasons = []
                        _sw2 = _smry.get("sector_weights", {})
                        _mw2 = _smry.get("market_weights", {})
                        for _sec, _wt in _sw2.items():
                            if _wt > 50:
                                _risks.append(_bi(f"🔴 **تركّز قطاعي**: {_sec} = {_wt:.0f}% من المحفظة",
                                                  f"🔴 **Sector concentration**: {_sec} = {_wt:.0f}% of portfolio"))
                                _risk_reasons.append(_bi(f"تركّز مفرط في قطاع {_sec}", f"Over-exposed to {_sec} sector"))
                        for _mkt, _wt in _mw2.items():
                            if _wt > 60:
                                _risks.append(_bi(f"🔴 **تركّز سوقي**: {_mkt.upper()} = {_wt:.0f}%",
                                                  f"🔴 **Market concentration**: {_mkt.upper()} = {_wt:.0f}%"))
                                _risk_reasons.append(_bi(f"تركّز مفرط في سوق {_mkt.upper()}", f"Over-exposed to {_mkt.upper()} market"))
                        _avg_rsi = _smry.get("avg_rsi")
                        if _avg_rsi and _avg_rsi > 65:
                            _risks.append(_bi(f"🟡 **المحفظة في ذروة الشراء**: متوسط RSI = {_avg_rsi:.0f}",
                                             f"🟡 **Portfolio overbought**: avg RSI = {_avg_rsi:.0f}"))
                            _risk_reasons.append(_bi("ذروة شراء في المحفظة", "Portfolio overbought"))
                        if _smry.get("n_positions", 0) < 3:
                            _risks.append(_bi("🟡 **تنويع ضعيف**: عدد المراكز أقل من 3",
                                             "🟡 **Low diversification**: fewer than 3 positions"))
                            _risk_reasons.append(_bi("عدد مراكز قليل جداً", "Too few positions"))
                        if not _risks:
                            _risks.append(_bi("🟢 لا مخاطر رئيسية مكتشفة", "🟢 No major risk flags detected"))
                        for _r in _risks:
                            st.markdown(_r)

                        # ══════════════════════════════════════════════════════
                        # PORTFOLIO FINAL DECISION BOX
                        # ══════════════════════════════════════════════════════
                        # Determine overall decision
                        _risk_sc = _smry.get("risk_score", 50)
                        _n_pos   = _smry.get("n_positions", 0)
                        _reduce_count   = sum(1 for a in _portfolio_actions if "Reduce" in a or "خفّف" in a)
                        _rebalance_count = sum(1 for a in _portfolio_actions if "Rebalance" in a or "توازن" in a)
                        _review_count   = sum(1 for a in _portfolio_actions if "Review" in a or "راجع" in a)

                        if _risk_sc > 70 or _reduce_count >= 2:
                            _final_decision = _bi("تخفيف المراكز", "Reduce Exposure")
                            _decision_color = "#ef4444"
                            _decision_icon  = "🔴"
                            _confidence     = min(95, 60 + _risk_sc // 5)
                        elif _rebalance_count >= 1 or _risk_reasons:
                            _final_decision = _bi("إعادة التوازن", "Rebalance")
                            _decision_color = "#f59e0b"
                            _decision_icon  = "💛"
                            _confidence     = 75
                        elif _review_count >= 1:
                            _final_decision = _bi("مراجعة المراكز", "Review Positions")
                            _decision_color = "#f59e0b"
                            _decision_icon  = "📋"
                            _confidence     = 70
                        else:
                            _final_decision = _bi("احتفظ بالمحفظة", "Hold Portfolio")
                            _decision_color = "#10b981"
                            _decision_icon  = "✅"
                            _confidence     = 80

                        _decision_reason = " — ".join(_risk_reasons) if _risk_reasons else _bi("المحفظة متوازنة", "Portfolio within parameters")

                        st.markdown(f"""
<div style="background:{'#1e293b' if st.session_state.dark_mode else '#f8fafc'};
            border:2px solid {_decision_color};border-radius:12px;
            padding:1.2rem 1.5rem;margin-top:1.2rem;">
  <div style="font-size:1rem;color:#94a3b8;margin-bottom:.4rem;">
    {_bi('قرار المحفظة', 'Portfolio Decision')}
  </div>
  <div style="font-size:2rem;font-weight:800;color:{_decision_color};">
    {_decision_icon} {_final_decision}
  </div>
  <div style="margin-top:.5rem;font-size:.95rem;">
    <span style="background:{_decision_color}22;color:{_decision_color};
                 border-radius:20px;padding:.2rem .8rem;font-weight:600;">
      {_bi('درجة الثقة','Confidence')}: {_confidence}%
    </span>
  </div>
  <div style="margin-top:.6rem;color:{'#cbd5e1' if st.session_state.dark_mode else '#475569'};font-size:.9rem;">
    {_bi('السبب','Reason')}: {_decision_reason}
  </div>
</div>
""", unsafe_allow_html=True)

                        # ══════════════════════════════════════════════════════
                        # EXECUTABLE ACTION PLAN
                        # ══════════════════════════════════════════════════════
                        st.markdown("##### ⚡ " + _bi("خطة التنفيذ", "Execution Plan"))
                        _TARGET_WEIGHT = 20.0   # max single-position target %
                        _exec_actions  = []

                        if not _pos_df2.empty:
                            for _, _row in _pos_df2.iterrows():
                                _tk2  = _row["ticker"]
                                _w2   = round((_row.get("value") or 0) / _total_v_smry * 100, 1)
                                _rsi2 = _row.get("RSI")
                                _p2   = _row.get("price")
                                _sh2  = _row.get("qty")

                                # Over-concentrated: sell down to TARGET
                                if _w2 > _TARGET_WEIGHT + 5 and _p2 and _sh2:
                                    _target_val   = _total_v_smry * _TARGET_WEIGHT / 100
                                    _current_val  = (_row.get("value") or 0)
                                    _sell_val     = _current_val - _target_val
                                    _sell_shares  = int(_sell_val / float(_p2))
                                    if _sell_shares > 0:
                                        _exec_actions.append({
                                            "icon": "🔴",
                                            "action": _bi(f"بيع {_sell_shares:,} سهم من {_tk2}",
                                                         f"Sell {_sell_shares:,} shares of {_tk2}"),
                                            "detail": _bi(f"تخفيض من {_w2:.0f}% → {_TARGET_WEIGHT:.0f}% ({_sell_val:,.0f} بيع)",
                                                         f"Reduce from {_w2:.0f}% → {_TARGET_WEIGHT:.0f}% (sell {_sell_val:,.0f})"),
                                        })

                                # RSI overbought: suggest partial profit
                                elif _rsi2 and float(_rsi2) > 70 and _p2 and _sh2:
                                    _sell_shares = max(1, int(float(_sh2) * 0.25))
                                    _exec_actions.append({
                                        "icon": "🟡",
                                        "action": _bi(f"بيع 25% من {_tk2} ({_sell_shares:,} سهم)",
                                                     f"Sell 25% of {_tk2} ({_sell_shares:,} shares)"),
                                        "detail": _bi(f"RSI = {float(_rsi2):.0f} — جني أرباح جزئي",
                                                     f"RSI = {float(_rsi2):.0f} — partial profit taking"),
                                    })

                                # RSI oversold: suggest adding
                                elif _rsi2 and float(_rsi2) < 30:
                                    _exec_actions.append({
                                        "icon": "🟢",
                                        "action": _bi(f"زيادة في {_tk2}",
                                                     f"Add to {_tk2}"),
                                        "detail": _bi(f"RSI = {float(_rsi2):.0f} — فرصة دخول",
                                                     f"RSI = {float(_rsi2):.0f} — oversold entry opportunity"),
                                    })

                        # Cash deployment signal
                        if _cash_val > 0:
                            _under_sectors = [s for s, w in _sw2.items() if w < 10]
                            if _cash_pct > 30:
                                _exec_actions.append({
                                    "icon": "💵",
                                    "action": _bi(f"نشر السيولة ({_cash_val:,.0f})",
                                                 f"Deploy cash ({_cash_val:,.0f})"),
                                    "detail": _bi(f"السيولة {_cash_pct:.0f}% من المحفظة — قطاعات مقترحة: {', '.join(_under_sectors[:3]) or 'Diversify'}",
                                                 f"Cash is {_cash_pct:.0f}% of portfolio — suggested sectors: {', '.join(_under_sectors[:3]) or 'Diversify'}"),
                                })
                            else:
                                _exec_actions.append({
                                    "icon": "✅",
                                    "action": _bi("الاحتفاظ بالسيولة", "Keep cash reserve"),
                                    "detail": _bi(f"{_cash_pct:.0f}% سيولة — مستوى مناسب للطوارئ",
                                                 f"{_cash_pct:.0f}% cash — adequate buffer"),
                                })

                        # Diversification signal
                        _sw2_markets = _smry.get("market_weights", {})
                        if len(_sw2_markets) < 2:
                            _exec_actions.append({
                                "icon": "🌍",
                                "action": _bi("إضافة تنويع جغرافي", "Add geographic diversification"),
                                "detail": _bi("المحفظة في سوق واحد فقط — أضف UAE أو KSA أو Egypt",
                                             "Portfolio in single market — consider UAE / KSA / Egypt"),
                            })

                        if not _exec_actions:
                            _exec_actions.append({
                                "icon": "✅",
                                "action": _bi("لا إجراء مطلوب", "No action required"),
                                "detail": _bi("المحفظة متوازنة — راقب RSI أسبوعياً",
                                             "Portfolio is balanced — monitor RSI weekly"),
                            })

                        _dm = st.session_state.dark_mode
                        for _ea in _exec_actions:
                            st.markdown(
                                f"<div style='display:flex;gap:.8rem;align-items:flex-start;"
                                f"background:{'#1e293b' if _dm else '#f1f5f9'};"
                                f"border-radius:8px;padding:.7rem 1rem;margin-bottom:.5rem'>"
                                f"<span style='font-size:1.3rem'>{_ea['icon']}</span>"
                                f"<div><div style='font-weight:700;font-size:.95rem'>{_ea['action']}</div>"
                                f"<div style='color:#94a3b8;font-size:.83rem;margin-top:.15rem'>{_ea['detail']}</div>"
                                f"</div></div>",
                                unsafe_allow_html=True
                            )

                        # ══════════════════════════════════════════════════════
                        # BENCHMARK COMPARISON
                        # ══════════════════════════════════════════════════════
                        st.markdown("##### 📊 " + _bi("مقارنة بالمرجع", "Benchmark Comparison"))
                        try:
                            import yfinance as _yf
                            # Determine oldest position date for period
                            _dates = [p.get("purchase_date","") for p in _positions if p.get("purchase_date")]
                            _oldest = min(_dates) if _dates else "2024-01-01"
                            # Determine benchmark: america-heavy → SPY, else use ^TASI
                            _america_pct = _smry.get("market_weights",{}).get("america",
                                            _smry.get("market_weights",{}).get("AMERICA", 0))
                            _bench_ticker = "SPY" if _america_pct > 40 else "^TASI"
                            _bench_label  = "S&P 500 (SPY)" if _bench_ticker == "SPY" else "Tadawul (^TASI)"

                            with st.spinner(_bi(f"جلب بيانات {_bench_label}…", f"Fetching {_bench_label}…")):
                                _bdf = _yf.download(_bench_ticker, start=_oldest, progress=False, auto_adjust=True)

                            if not _bdf.empty:
                                _b_start = float(_bdf["Close"].iloc[0])
                                _b_end   = float(_bdf["Close"].iloc[-1])
                                _bench_ret = (_b_end / _b_start - 1) * 100

                                # Portfolio return (cost-weighted)
                                _ptf_ret = _unreal_pct if _unreal_pct is not None else 0.0
                                _alpha   = _ptf_ret - _bench_ret

                                bc1, bc2, bc3 = st.columns(3)
                                with bc1:
                                    _pc = "#10b981" if _ptf_ret >= 0 else "#ef4444"
                                    st.markdown(f"<div style='text-align:center'>"
                                                f"<div style='color:#94a3b8;font-size:.8rem'>"
                                                f"{_bi('عائد محفظتك','Your Return')}</div>"
                                                f"<div style='color:{_pc};font-size:1.6rem;font-weight:800'>"
                                                f"{_ptf_ret:+.1f}%</div></div>", unsafe_allow_html=True)
                                with bc2:
                                    _bc = "#10b981" if _bench_ret >= 0 else "#ef4444"
                                    st.markdown(f"<div style='text-align:center'>"
                                                f"<div style='color:#94a3b8;font-size:.8rem'>{_bench_label}</div>"
                                                f"<div style='color:{_bc};font-size:1.6rem;font-weight:800'>"
                                                f"{_bench_ret:+.1f}%</div></div>", unsafe_allow_html=True)
                                with bc3:
                                    _ac = "#10b981" if _alpha >= 0 else "#ef4444"
                                    _alabel = _bi("ألفا (تفوقك)", "Alpha (vs Benchmark)")
                                    st.markdown(f"<div style='text-align:center'>"
                                                f"<div style='color:#94a3b8;font-size:.8rem'>{_alabel}</div>"
                                                f"<div style='color:{_ac};font-size:1.6rem;font-weight:800'>"
                                                f"{_alpha:+.1f}%</div></div>", unsafe_allow_html=True)
                                st.caption(f"📅 {_bi('الفترة من','Period from')} {_oldest} {_bi('حتى اليوم','to today')} | Benchmark: {_bench_label}")
                            else:
                                st.caption(_bi("⚠️ لم تتوفر بيانات المرجع", "⚠️ Benchmark data unavailable"))
                        except Exception as _be:
                            st.caption(_bi(f"⚠️ المرجع غير متاح: {_be}", f"⚠️ Benchmark unavailable: {_be}"))

                        # ── Download report — passes all context ─────────────
                        st.markdown("<br>", unsafe_allow_html=True)
                        try:
                            # Build final_decision dict for report
                            _rpt_decision = {
                                "icon":       _decision_icon,
                                "decision":   _final_decision,
                                "confidence": _confidence,
                                "reasons":    _risk_reasons if _risk_reasons else ["Portfolio within parameters"],
                            }
                            # Build English-only exec plan for report
                            _rpt_exec = []
                            for _ea in _exec_actions:
                                _rpt_exec.append({
                                    "icon":   _ea["icon"],
                                    "action": _ea["action"],
                                    "detail": _ea["detail"],
                                })
                            # Benchmark dict (collected earlier if available)
                            _rpt_bench = None
                            if locals().get("_bench_ret") is not None and locals().get("_ptf_ret") is not None:
                                _rpt_bench = {
                                    "name":        locals().get("_bench_label","Benchmark"),
                                    "ptf_ret":     locals().get("_ptf_ret", 0),
                                    "bench_ret":   locals().get("_bench_ret", 0),
                                    "alpha":       locals().get("_alpha", 0),
                                    "period_start": min([p.get("purchase_date","") for p in _positions if p.get("purchase_date")], default="—"),
                                }
                            _md_r = _p.to_markdown(
                                cash=_cash_val,
                                final_decision=_rpt_decision,
                                execution_plan=_rpt_exec,
                                benchmark=_rpt_bench,
                            )
                            st.download_button("📥 " + _bi("تحميل التقرير","Download Report"),
                                               _md_r.encode(), "eisax_portfolio_report.md", "text/markdown")
                        except Exception as _rpt_err:
                            st.caption(f"Report generation error: {_rpt_err}")

                        # ══════════════════════════════════════════════════════
                        # F1 — MACRO SIMULATION
                        # ══════════════════════════════════════════════════════
                        with st.expander("🌍 " + _bi("محاكاة الاقتصاد الكلي", "Macro Simulation")):
                            try:
                                from core.macro_elasticities import MACRO_VAR_DEFAULTS, MACRO_VAR_RANGES, MACRO_VAR_LABELS
                                from core.macro_simulator import MacroScenario, simulate_portfolio as _sim_ptf

                                st.caption(_bi(
                                    "حرّك المتغيرات الاقتصادية وشاهد تأثيرها على محفظتك",
                                    "Adjust macro variables to see their impact on your portfolio"
                                ))
                                _mc1, _mc2 = st.columns(2)
                                _lang = st.session_state.language
                                with _mc1:
                                    _gdp   = st.slider(MACRO_VAR_LABELS["gdp_growth"][0 if _lang=="ar" else 1],
                                                       *MACRO_VAR_RANGES["gdp_growth"],
                                                       value=MACRO_VAR_DEFAULTS["gdp_growth"], key="mc_gdp")
                                    _infl  = st.slider(MACRO_VAR_LABELS["inflation"][0 if _lang=="ar" else 1],
                                                       *MACRO_VAR_RANGES["inflation"],
                                                       value=MACRO_VAR_DEFAULTS["inflation"], key="mc_infl")
                                    _rate  = st.slider(MACRO_VAR_LABELS["fed_rate"][0 if _lang=="ar" else 1],
                                                       *MACRO_VAR_RANGES["fed_rate"],
                                                       value=MACRO_VAR_DEFAULTS["fed_rate"], key="mc_rate")
                                with _mc2:
                                    _oil   = st.slider(MACRO_VAR_LABELS["oil_brent"][0 if _lang=="ar" else 1],
                                                       *MACRO_VAR_RANGES["oil_brent"],
                                                       value=MACRO_VAR_DEFAULTS["oil_brent"], key="mc_oil")
                                    _dxy   = st.slider(MACRO_VAR_LABELS["usd_index"][0 if _lang=="ar" else 1],
                                                       *MACRO_VAR_RANGES["usd_index"],
                                                       value=MACRO_VAR_DEFAULTS["usd_index"], key="mc_dxy")

                                if st.button("▶️ " + _bi("تشغيل المحاكاة", "Run Simulation"), key="run_macro_sim"):
                                    _scen = MacroScenario(gdp_growth=_gdp, inflation=_infl,
                                                          fed_rate=_rate, oil_brent=_oil, usd_index=_dxy)
                                    _msim = _sim_ptf(_pos_df2, _smry["total_value"], _scen)

                                    _ic1, _ic2, _ic3 = st.columns(3)
                                    _ic_col = "#10b981" if _msim["total_impact_pct"] >= 0 else "#ef4444"
                                    with _ic1:
                                        st.metric(_bi("التأثير الكلي %", "Total Impact %"),
                                                  f"{_msim['total_impact_pct']:+.2f}%")
                                    with _ic2:
                                        st.metric(_bi("تغيير القيمة", "Value Change"),
                                                  f"{_msim['total_impact_value']:+,.0f}")
                                    with _ic3:
                                        st.metric(_bi("القيمة الجديدة", "New Value"),
                                                  f"{_msim['new_portfolio_value']:,.0f}")

                                    _sec_imp = _msim["sector_impacts"]
                                    _fig_macro = px.bar(
                                        x=list(_sec_imp.keys()),
                                        y=[v * 100 for v in _sec_imp.values() if abs(v) > 0.001],
                                        labels={"x": _bi("القطاع","Sector"), "y": _bi("التأثير %","Impact %")},
                                        title=_bi("تأثير الاقتصاد الكلي على القطاعات","Macro Impact by Sector"),
                                        color=[v for v in _sec_imp.values() if abs(v) > 0.001],
                                        color_continuous_scale=["#ef4444","#94a3b8","#10b981"],
                                    )
                                    _sec_keys = [k for k, v in _sec_imp.items() if abs(v) > 0.001]
                                    _sec_vals = [v * 100 for k, v in _sec_imp.items() if abs(v) > 0.001]
                                    _fig_macro = px.bar(
                                        x=_sec_keys, y=_sec_vals,
                                        labels={"x": _bi("القطاع","Sector"), "y": _bi("التأثير %","Impact %")},
                                        title=_bi("تأثير الاقتصاد الكلي على القطاعات","Macro Impact by Sector"),
                                        color=_sec_vals,
                                        color_continuous_scale=["#ef4444","#94a3b8","#10b981"],
                                    )
                                    style_chart(_fig_macro, height=280)
                                    st.plotly_chart(_fig_macro, use_container_width=True, key="macro_sector_chart")

                                    if not _msim["position_impacts"].empty:
                                        st.dataframe(_msim["position_impacts"], use_container_width=True)
                            except Exception as _f1_err:
                                st.warning(f"Macro simulation error: {_f1_err}")

                        # ══════════════════════════════════════════════════════
                        # F2 — BUDGET PLANNER
                        # ══════════════════════════════════════════════════════
                        with st.expander("💰 " + _bi("مخطط الميزانية", "Budget Planner")):
                            try:
                                from core.budget_engine import compute_budget_allocation

                                st.caption(_bi(
                                    "أدخل ميزانيتك وأوزان القطاعات المستهدفة لمعرفة الأسهم التي يجب شراؤها أو بيعها",
                                    "Enter your budget and target sector weights to get exact buy/sell quantities"
                                ))
                                _bg1, _bg2 = st.columns(2)
                                with _bg1:
                                    _budget = st.number_input(
                                        _bi("الميزانية الإضافية", "Additional Budget"),
                                        min_value=0.0, value=0.0, step=1000.0,
                                        format="%.0f", key="budget_amount"
                                    )
                                with _bg2:
                                    _budget_note = st.caption(_bi(
                                        "يمكن أن تكون صفراً لإعادة التوازن فقط",
                                        "Can be 0 for rebalance-only"
                                    ))

                                st.markdown(_bi("**أوزان القطاعات المستهدفة % (المجموع = 100)**",
                                                "**Target Sector Weights % (sum = 100)**"))
                                _sw_existing = _smry.get("sector_weights", {})
                                _bg_sectors = {}
                                _bg_cols = st.columns(3)
                                _all_sectors = list(_sw_existing.keys()) or ["Finance", "Energy", "Real Estate"]
                                for _i, _sec in enumerate(_all_sectors):
                                    with _bg_cols[_i % 3]:
                                        _bg_sectors[_sec] = st.number_input(
                                            _sec, min_value=0.0, max_value=100.0,
                                            value=round(_sw_existing.get(_sec, 0), 1),
                                            step=1.0, key=f"bg_sec_{_sec}"
                                        )

                                if st.button("💰 " + _bi("احسب خطة الميزانية", "Compute Budget Plan"), key="run_budget"):
                                    _bplan = compute_budget_allocation(
                                        total_budget=_budget,
                                        target_sector_weights=_bg_sectors,
                                        positions_df=_pos_df2,
                                        total_value=_smry["total_value"],
                                    )
                                    _bm1, _bm2, _bm3, _bm4 = st.columns(4)
                                    with _bm1:
                                        st.metric(_bi("إجمالي الشراء", "Total Buy"), f"{_bplan['total_buy_cost']:,.0f}")
                                    with _bm2:
                                        st.metric(_bi("إجمالي البيع", "Total Sell"), f"{_bplan['total_sell_proceeds']:,.0f}")
                                    with _bm3:
                                        st.metric(_bi("صافي النقد المطلوب", "Net Cash Needed"), f"{_bplan['net_cash_required']:,.0f}")
                                    with _bm4:
                                        _fc = "✅" if _bplan["feasible"] else "❌"
                                        st.metric(_bi("النقد المتبقي", "Remaining Cash"),
                                                  f"{_fc} {_bplan['remaining_cash']:,.0f}")

                                    if _bplan["warnings"]:
                                        for _w in _bplan["warnings"]:
                                            st.warning(_w)

                                    if not _bplan["allocations"].empty:
                                        _display_cols = ["Ticker", "Sector", "Action", "Shares to Buy",
                                                         "Shares to Sell", "Est. Cost", "Target Weight %"]
                                        _dcols = [c for c in _display_cols if c in _bplan["allocations"].columns]
                                        st.dataframe(_bplan["allocations"][_dcols], use_container_width=True)
                            except Exception as _f2_err:
                                st.warning(f"Budget planner error: {_f2_err}")

                        # ══════════════════════════════════════════════════════
                        # F3 — FORWARD SCENARIO BUILDER
                        # ══════════════════════════════════════════════════════
                        with st.expander("🔭 " + _bi("سيناريو مستقبلي", "Forward Scenario")):
                            try:
                                from core.macro_elasticities import MACRO_VAR_DEFAULTS as _fwd_defaults, MACRO_VAR_RANGES as _fwd_ranges, MACRO_VAR_LABELS as _fwd_labels
                                from core.macro_simulator import MacroScenario as _FwdScen
                                from core.scenario_builder import build_forward_scenario

                                st.caption(_bi(
                                    "اضبط افتراضاتك الاقتصادية وشاهد توقعات محفظتك على 3 و 6 و 12 شهراً",
                                    "Set your macro assumptions and see 3/6/12 month portfolio projections"
                                ))
                                _fl = st.session_state.language
                                _fwd_c1, _fwd_c2 = st.columns(2)
                                with _fwd_c1:
                                    _fgdp  = st.slider(_fwd_labels["gdp_growth"][0 if _fl=="ar" else 1],
                                                       *_fwd_ranges["gdp_growth"],
                                                       value=_fwd_defaults["gdp_growth"], key="fwd_gdp")
                                    _finfl = st.slider(_fwd_labels["inflation"][0 if _fl=="ar" else 1],
                                                       *_fwd_ranges["inflation"],
                                                       value=_fwd_defaults["inflation"], key="fwd_infl")
                                    _frate = st.slider(_fwd_labels["fed_rate"][0 if _fl=="ar" else 1],
                                                       *_fwd_ranges["fed_rate"],
                                                       value=_fwd_defaults["fed_rate"], key="fwd_rate")
                                with _fwd_c2:
                                    _foil  = st.slider(_fwd_labels["oil_brent"][0 if _fl=="ar" else 1],
                                                       *_fwd_ranges["oil_brent"],
                                                       value=_fwd_defaults["oil_brent"], key="fwd_oil")
                                    _fdxy  = st.slider(_fwd_labels["usd_index"][0 if _fl=="ar" else 1],
                                                       *_fwd_ranges["usd_index"],
                                                       value=_fwd_defaults["usd_index"], key="fwd_dxy")

                                if st.button("🔭 " + _bi("احسب التوقعات", "Project Forward"), key="run_fwd_scenario"):
                                    with st.spinner(_bi("جاري الحساب…", "Computing projections…")):
                                        _fscen = _FwdScen(gdp_growth=_fgdp, inflation=_finfl,
                                                          fed_rate=_frate, oil_brent=_foil, usd_index=_fdxy)
                                        _fwd = build_forward_scenario(
                                            _pos_df2, _smry["total_value"], _fscen
                                        )

                                    st.caption(f"📌 {_bi('السيناريو','Scenario')}: {_fwd['scenario_label']}")

                                    _fhm1, _fhm2, _fhm3 = st.columns(3)
                                    for _col, _h in zip([_fhm1, _fhm2, _fhm3], [3, 6, 12]):
                                        _hdata = _fwd["horizons"].get(_h, {})
                                        _pv    = _hdata.get("projected_value", _smry["total_value"])
                                        _pc    = _hdata.get("pct_change", 0.0)
                                        _hcol  = "#10b981" if _pc >= 0 else "#ef4444"
                                        with _col:
                                            st.markdown(
                                                f"<div style='text-align:center'>"
                                                f"<div style='color:#94a3b8;font-size:.8rem'>{_h}M</div>"
                                                f"<div style='font-size:1.2rem;font-weight:700'>{_pv:,.0f}</div>"
                                                f"<div style='color:{_hcol}'>{_pc:+.1f}%</div>"
                                                f"</div>", unsafe_allow_html=True
                                            )

                                    # Line chart of projections
                                    _tv = _smry["total_value"]
                                    _fwd_chart_data = {
                                        _bi("الأفق (شهور)","Horizon (months)"): [0, 3, 6, 12],
                                        _bi("القيمة المتوقعة","Projected Value"): [
                                            _tv,
                                            _fwd["horizons"].get(3, {}).get("projected_value", _tv),
                                            _fwd["horizons"].get(6, {}).get("projected_value", _tv),
                                            _fwd["horizons"].get(12, {}).get("projected_value", _tv),
                                        ]
                                    }
                                    _fig_fwd = px.line(
                                        _fwd_chart_data,
                                        x=_bi("الأفق (شهور)","Horizon (months)"),
                                        y=_bi("القيمة المتوقعة","Projected Value"),
                                        markers=True,
                                        title=_bi("مسار المحفظة المتوقع","Projected Portfolio Path"),
                                    )
                                    style_chart(_fig_fwd, height=260)
                                    st.plotly_chart(_fig_fwd, use_container_width=True, key="fwd_line_chart")

                                    _proj_df = _fwd["horizons"].get(12, {}).get("position_projections", pd.DataFrame())
                                    if not _proj_df.empty:
                                        _show_cols = ["Ticker","Name","Sector","Current Value",
                                                      "Adjusted Return %","Proj 3m","Proj 6m","Proj 12m"]
                                        _show_cols = [c for c in _show_cols if c in _proj_df.columns]
                                        st.dataframe(_proj_df[_show_cols], use_container_width=True)
                            except Exception as _f3_err:
                                st.warning(f"Forward scenario error: {_f3_err}")

                        # ══════════════════════════════════════════════════════
                        # F4 — MONTE CARLO / VAR
                        # ══════════════════════════════════════════════════════
                        with st.expander("🎲 " + _bi("مونت كارلو / قيمة المخاطرة", "Monte Carlo / VaR")):
                            try:
                                from core.monte_carlo import run_portfolio_monte_carlo
                                import numpy as np

                                st.caption(_bi(
                                    "محاكاة آلاف المسارات لتقدير أقصى خسارة محتملة (VaR) وتوزيع النتائج",
                                    "Simulate thousands of paths to estimate maximum probable loss (VaR) and outcome distribution"
                                ))
                                _mc_c1, _mc_c2, _mc_c3 = st.columns(3)
                                with _mc_c1:
                                    _n_sim = st.selectbox(_bi("عدد المحاكاة","Simulations"),
                                                          [1000, 2000, 5000, 10000], index=2, key="mc_nsim")
                                with _mc_c2:
                                    _h_days = st.selectbox(_bi("الأفق الزمني","Horizon"),
                                                           [63, 126, 252], index=2,
                                                           format_func=lambda x: f"{x}d ({x//21}m)",
                                                           key="mc_horizon")
                                with _mc_c3:
                                    _loss_thr = st.slider(_bi("عتبة الخسارة %","Loss Threshold %"),
                                                          5, 50, 10, key="mc_loss_thr") / 100

                                if st.button("🎲 " + _bi("تشغيل المحاكاة", "Run Monte Carlo"), key="run_mc"):
                                    with st.spinner(_bi("جاري المحاكاة…", "Running simulation…")):
                                        _mc_res = run_portfolio_monte_carlo(
                                            _pos_df2, _smry["total_value"],
                                            n_simulations=_n_sim,
                                            horizon_days=_h_days,
                                            loss_threshold_pct=_loss_thr,
                                        )

                                    _mv1, _mv2, _mv3, _mv4 = st.columns(4)
                                    with _mv1:
                                        st.metric("VaR 95%", f"{_mc_res['var'].get(0.95, 0):+.1f}%")
                                    with _mv2:
                                        st.metric("VaR 99%", f"{_mc_res['var'].get(0.99, 0):+.1f}%")
                                    with _mv3:
                                        st.metric("CVaR 95%", f"{_mc_res['cvar'].get(0.95, 0):+.1f}%")
                                    with _mv4:
                                        st.metric(
                                            _bi(f"احتمال خسارة > {int(_loss_thr*100)}%",
                                                f"P(loss > {int(_loss_thr*100)}%)"),
                                            f"{_mc_res['prob_loss_gt_threshold']:.1f}%"
                                        )

                                    _mo1, _mo2, _mo3 = st.columns(3)
                                    with _mo1:
                                        st.metric(_bi("أفضل حالة (P90)","Best (P90)"),
                                                  f"{_mc_res['best_outcome']:,.0f}")
                                    with _mo2:
                                        st.metric(_bi("المتوسط (P50)","Median (P50)"),
                                                  f"{_mc_res['median_outcome']:,.0f}")
                                    with _mo3:
                                        st.metric(_bi("أسوأ حالة (P10)","Worst (P10)"),
                                                  f"{_mc_res['worst_outcome']:,.0f}")

                                    # Paths chart
                                    _paths = _mc_res.get("paths_sample")
                                    if _paths is not None and _paths.shape[0] > 1:
                                        _path_df = pd.DataFrame(_paths,
                                                                 columns=[f"sim_{i}" for i in range(_paths.shape[1])])
                                        _fig_mc = px.line(_path_df, title=_bi("مسارات المحاكاة","Simulation Paths"),
                                                          color_discrete_sequence=["rgba(14,165,164,0.08)"] * _paths.shape[1])
                                        # Overlay P10/P50/P90
                                        _tv0 = _smry["total_value"]
                                        _mc_ts = list(range(_paths.shape[0]))
                                        for _pct_lbl, _pct_val, _col in [
                                            ("P90", 90, "#10b981"), ("P50", 50, "#f59e0b"), ("P10", 10, "#ef4444")
                                        ]:
                                            _pct_line = np.percentile(_paths, _pct_val, axis=1)
                                            _fig_mc.add_scatter(x=_mc_ts, y=_pct_line,
                                                                mode="lines", name=_pct_lbl,
                                                                line=dict(color=_col, width=2))
                                        style_chart(_fig_mc, height=300)
                                        st.plotly_chart(_fig_mc, use_container_width=True, key="mc_paths_chart")

                                    # Histogram of terminal values
                                    _term = _mc_res.get("terminal_distribution")
                                    if _term is not None:
                                        _fig_hist = px.histogram(
                                            x=_term, nbins=60,
                                            title=_bi("توزيع القيمة النهائية","Terminal Value Distribution"),
                                            labels={"x": _bi("القيمة","Value"), "y": _bi("التكرار","Count")},
                                            color_discrete_sequence=["#0ea5a4"],
                                        )
                                        _var95_val = _mc_res["var"].get(0.95, 0) / 100
                                        _var95_abs = _smry["total_value"] * (1 + _var95_val)
                                        _fig_hist.add_vline(x=_var95_abs, line_color="#ef4444",
                                                            line_dash="dash",
                                                            annotation_text="VaR 95%",
                                                            annotation_position="top right")
                                        style_chart(_fig_hist, height=260)
                                        st.plotly_chart(_fig_hist, use_container_width=True, key="mc_hist_chart")
                            except Exception as _f4_err:
                                st.warning(f"Monte Carlo error: {_f4_err}")

                        # ══════════════════════════════════════════════════════
                        # F5 — MARKET REGIME COMPARISON
                        # ══════════════════════════════════════════════════════
                        with st.expander("🌐 " + _bi("مقارنة أنظمة السوق", "Market Regime Comparison")):
                            try:
                                from core.market_regimes import compare_regimes

                                st.caption(_bi(
                                    "قارن أداء محفظتك في ظل أربعة أنظمة سوق مختلفة",
                                    "Compare your portfolio performance under 4 market regimes"
                                ))
                                _rg_horizon = st.selectbox(
                                    _bi("الأفق الزمني", "Horizon"),
                                    [6, 12], index=1,
                                    format_func=lambda x: f"{x} " + _bi("شهراً","months"),
                                    key="regime_horizon"
                                )

                                if st.button("🌐 " + _bi("مقارنة الأنظمة", "Compare Regimes"), key="run_regimes"):
                                    with st.spinner(_bi("جاري الحساب…", "Analyzing regimes…")):
                                        _rg = compare_regimes(
                                            _pos_df2, _smry["total_value"],
                                            horizon_months=_rg_horizon,
                                        )

                                    _rg_names, _rg_vals, _rg_rets = [], [], []
                                    for _rname, _rdata in _rg["regimes"].items():
                                        _lbl = _rdata["label_ar"] if st.session_state.language == "ar" else _rdata["label_en"]
                                        _rg_names.append(_lbl)
                                        _rg_vals.append(_rdata["projected_value"])
                                        _rg_rets.append(_rdata["expected_return_pct"])

                                    _tv_base = _smry["total_value"]
                                    _fig_rg = px.bar(
                                        x=_rg_names, y=_rg_vals,
                                        color=_rg_rets,
                                        color_continuous_scale=["#ef4444","#94a3b8","#10b981"],
                                        labels={
                                            "x": _bi("نظام السوق","Market Regime"),
                                            "y": _bi("القيمة المتوقعة","Projected Value"),
                                        },
                                        title=_bi(
                                            f"المحفظة في أفق {_rg_horizon} شهراً",
                                            f"Portfolio at {_rg_horizon}-Month Horizon"
                                        ),
                                        text=[f"{r:+.1f}%" for r in _rg_rets],
                                    )
                                    _fig_rg.add_hline(y=_tv_base, line_dash="dot",
                                                      line_color="#94a3b8",
                                                      annotation_text=_bi("القيمة الحالية","Current Value"))
                                    style_chart(_fig_rg, height=320)
                                    st.plotly_chart(_fig_rg, use_container_width=True, key="regime_bar_chart")

                                    # Summary table
                                    _rg_rows = []
                                    for _rname, _rdata in _rg["regimes"].items():
                                        _lbl = _rdata["label_ar"] if st.session_state.language == "ar" else _rdata["label_en"]
                                        _rg_rows.append({
                                            _bi("النظام","Regime"):           _lbl,
                                            _bi("العائد %","Return %"):       f"{_rdata['expected_return_pct']:+.1f}%",
                                            _bi("القيمة المتوقعة","Proj Value"): f"{_rdata['projected_value']:,.0f}",
                                            "GDP %":                          _rdata["macro_profile"]["gdp_growth"],
                                            _bi("التضخم %","Inflation %"):    _rdata["macro_profile"]["inflation"],
                                            _bi("الفائدة %","Rate %"):        _rdata["macro_profile"]["fed_rate"],
                                            _bi("النفط $","Oil $"):           _rdata["macro_profile"]["oil_brent"],
                                        })
                                    st.dataframe(pd.DataFrame(_rg_rows), use_container_width=True, hide_index=True)

                                    _best_lbl  = _rg["regimes"][_rg["best_regime"]]["label_en"]
                                    _worst_lbl = _rg["regimes"][_rg["worst_regime"]]["label_en"]
                                    st.caption(
                                        f"✅ {_bi('الأفضل','Best')}: {_best_lbl}  |  "
                                        f"⚠️ {_bi('الأسوأ','Worst')}: {_worst_lbl}  |  "
                                        f"{_bi('الفارق','Spread')}: {_rg['regime_spread_pct']:+.1f}%"
                                    )
                            except Exception as _f5_err:
                                st.warning(f"Market regimes error: {_f5_err}")

                        # ══════════════════════════════════════════════════════
                        # F6 — SHARIAH COMPLIANCE SCREENING
                        # ══════════════════════════════════════════════════════
                        with st.expander("🕌 " + _bi("الفحص الشرعي", "Shariah Compliance Screening")):
                            try:
                                from core.shariah_screener import screen_portfolio

                                st.caption(_bi(
                                    "فحص حيازاتك حسب معايير AAOIFI: نسبة الدين، النشاط التجاري، الإيراد الحرام",
                                    "Screen your holdings against AAOIFI rules: debt ratio, business activity, haram income"
                                ))
                                if st.button("🕌 " + _bi("افحص المحفظة", "Screen Portfolio"), key="run_shariah"):
                                    with st.spinner(_bi("جاري الفحص…", "Screening holdings…")):
                                        _sh = screen_portfolio(_pos_df2)

                                    _sh1, _sh2, _sh3, _sh4 = st.columns(4)
                                    with _sh1:
                                        _crc = "#10b981" if _sh["compliance_rate_pct"] >= 95 else "#f59e0b" if _sh["compliance_rate_pct"] >= 70 else "#ef4444"
                                        st.markdown(
                                            f"<div style='text-align:center'>"
                                            f"<div style='color:#94a3b8;font-size:.8rem'>{_bi('نسبة الامتثال','Compliance Rate')}</div>"
                                            f"<div style='color:{_crc};font-size:1.6rem;font-weight:800'>{_sh['compliance_rate_pct']:.1f}%</div>"
                                            f"</div>", unsafe_allow_html=True
                                        )
                                    with _sh2:
                                        st.metric(_bi("✅ حلال", "✅ Halal"), _sh["halal_count"])
                                    with _sh3:
                                        st.metric(_bi("❌ حرام", "❌ Haram"), _sh["haram_count"])
                                    with _sh4:
                                        st.metric(_bi("❓ غير محدد", "❓ Unknown"), _sh["unknown_count"])

                                    st.markdown(f"**{_sh['summary']}**")

                                    if _sh["purification_estimate"] > 0:
                                        st.info(_bi(
                                            f"💰 تقدير التطهير المالي: {_sh['purification_estimate']:,.0f} (تبرع بهذا المبلغ)",
                                            f"💰 Purification estimate: {_sh['purification_estimate']:,.0f} (donate this amount)"
                                        ))

                                    if not _sh["results"].empty:
                                        st.dataframe(_sh["results"], use_container_width=True, hide_index=True)

                                    _vc1, _vc2, _vc3 = st.columns(3)
                                    _total = _sh["halal_count"] + _sh["haram_count"] + _sh["unknown_count"]
                                    if _total > 0:
                                        _pie_data = {
                                            "Verdict": [_bi("حلال","Halal"), _bi("حرام","Haram"), _bi("غير محدد","Unknown")],
                                            "Value":   [_sh["total_halal_value"], _sh["total_haram_value"], _sh["total_unknown_value"]],
                                        }
                                        _fig_sh = px.pie(
                                            _pie_data, values="Value", names="Verdict",
                                            color="Verdict",
                                            color_discrete_map={
                                                _bi("حلال","Halal"): "#10b981",
                                                _bi("حرام","Haram"): "#ef4444",
                                                _bi("غير محدد","Unknown"): "#94a3b8",
                                            },
                                            title=_bi("توزيع المحفظة حسب الامتثال","Portfolio Breakdown by Compliance"),
                                            hole=0.4,
                                        )
                                        style_chart(_fig_sh, height=260)
                                        st.plotly_chart(_fig_sh, use_container_width=True, key="shariah_pie")
                            except Exception as _f6_err:
                                st.warning(f"Shariah screener error: {_f6_err}")

                        # ══════════════════════════════════════════════════════
                        # F7 — PORTFOLIO OPTIMIZATION (EFFICIENT FRONTIER)
                        # ══════════════════════════════════════════════════════
                        with st.expander("📊 " + _bi("تحسين المحفظة", "Portfolio Optimization")):
                            try:
                                from core.portfolio_optimizer import optimize_portfolio, efficient_frontier

                                st.caption(_bi(
                                    "اعثر على الأوزان المثالية لمحفظتك (Markowitz): أقصى نسبة شارب، أقل تذبذب",
                                    "Find optimal portfolio weights (Markowitz): max Sharpe ratio or min variance"
                                ))
                                _po_c1, _po_c2, _po_c3 = st.columns(3)
                                with _po_c1:
                                    _po_obj = st.selectbox(
                                        _bi("الهدف","Objective"),
                                        ["max_sharpe", "min_variance"],
                                        format_func=lambda x: {
                                            "max_sharpe":   _bi("أقصى نسبة شارب","Max Sharpe"),
                                            "min_variance": _bi("أقل تذبذب","Min Variance"),
                                        }.get(x, x),
                                        key="po_obj",
                                    )
                                with _po_c2:
                                    _po_rf = st.slider(_bi("سعر الفائدة الخالي %","Risk-Free Rate %"),
                                                       0.0, 8.0, 4.0, 0.5, key="po_rf") / 100
                                with _po_c3:
                                    _po_maxw = st.slider(_bi("الحد الأقصى للوزن %","Max Weight %"),
                                                         10, 100, 40, 5, key="po_maxw") / 100

                                if st.button("📊 " + _bi("حسّن المحفظة", "Optimize Portfolio"), key="run_optimize"):
                                    with st.spinner(_bi("جاري التحسين…", "Optimizing…")):
                                        _po = optimize_portfolio(
                                            _pos_df2, objective=_po_obj,
                                            risk_free_rate=_po_rf, max_weight=_po_maxw,
                                        )

                                    if _po.get("error"):
                                        st.warning(_po["error"])
                                    else:
                                        _ps1, _ps2, _ps3 = st.columns(3)
                                        with _ps1:
                                            st.markdown(_bi("**الحالي**","**Current**"))
                                            st.metric(_bi("العائد %","Return %"), f"{_po['current_stats']['return']:+.2f}%")
                                            st.metric(_bi("التذبذب %","Volatility %"), f"{_po['current_stats']['volatility']:.2f}%")
                                            st.metric("Sharpe", f"{_po['current_stats']['sharpe']:.3f}")
                                        with _ps2:
                                            st.markdown(_bi("**المثالي**","**Optimal**"))
                                            st.metric(_bi("العائد %","Return %"), f"{_po['optimal_stats']['return']:+.2f}%")
                                            st.metric(_bi("التذبذب %","Volatility %"), f"{_po['optimal_stats']['volatility']:.2f}%")
                                            st.metric("Sharpe", f"{_po['optimal_stats']['sharpe']:.3f}")
                                        with _ps3:
                                            st.markdown(_bi("**التحسن**","**Improvement**"))
                                            _imp = _po["improvement"]
                                            _rc = "#10b981" if _imp["return_lift"] >= 0 else "#ef4444"
                                            _sc = "#10b981" if _imp["sharpe_lift"] >= 0 else "#ef4444"
                                            st.metric(_bi("زيادة العائد","Return Lift"), f"{_imp['return_lift']:+.2f}%")
                                            st.metric(_bi("تغيير التذبذب","Vol Change"), f"{_imp['vol_change']:+.2f}%")
                                            st.metric("Sharpe Δ", f"{_imp['sharpe_lift']:+.3f}")

                                        if not _po["rebalance_actions"].empty:
                                            st.markdown(_bi("**إجراءات إعادة التوازن**","**Rebalance Actions**"))
                                            st.dataframe(_po["rebalance_actions"], use_container_width=True, hide_index=True)

                                    # Efficient frontier chart
                                    with st.spinner(_bi("بناء الحدود الكفؤة…","Building efficient frontier…")):
                                        _ef = efficient_frontier(
                                            _pos_df2, n_points=20,
                                            risk_free_rate=_po_rf, max_weight=_po_maxw,
                                        )
                                    if not _ef.get("error") and not _ef["frontier"].empty:
                                        _fig_ef = px.scatter(
                                            _ef["frontier"], x="volatility_pct", y="return_pct",
                                            color="sharpe", color_continuous_scale="Viridis",
                                            title=_bi("الحدود الكفؤة","Efficient Frontier"),
                                            labels={
                                                "volatility_pct": _bi("التذبذب %","Volatility %"),
                                                "return_pct":     _bi("العائد %","Return %"),
                                            },
                                        )
                                        _ms = _ef["max_sharpe_point"]
                                        _mv = _ef["min_variance_point"]
                                        _cur = _ef["current_point"]
                                        _fig_ef.add_scatter(x=[_ms["volatility"]], y=[_ms["return"]],
                                                            mode="markers+text",
                                                            marker=dict(size=18, color="#10b981", symbol="star"),
                                                            text=["Max Sharpe"], textposition="top center",
                                                            name="Max Sharpe")
                                        _fig_ef.add_scatter(x=[_mv["volatility"]], y=[_mv["return"]],
                                                            mode="markers+text",
                                                            marker=dict(size=18, color="#0ea5a4", symbol="diamond"),
                                                            text=["Min Var"], textposition="top center",
                                                            name="Min Variance")
                                        _fig_ef.add_scatter(x=[_cur["volatility"]], y=[_cur["return"]],
                                                            mode="markers+text",
                                                            marker=dict(size=18, color="#ef4444", symbol="x"),
                                                            text=[_bi("الحالي","Current")], textposition="bottom center",
                                                            name=_bi("الحالي","Current"))
                                        style_chart(_fig_ef, height=320)
                                        st.plotly_chart(_fig_ef, use_container_width=True, key="ef_chart")
                            except Exception as _f7_err:
                                st.warning(f"Portfolio optimizer error: {_f7_err}")

                        # ══════════════════════════════════════════════════════
                        # F8 — DIVIDEND INCOME PROJECTION
                        # ══════════════════════════════════════════════════════
                        with st.expander("💸 " + _bi("توقع دخل الأرباح الموزعة", "Dividend Income Projection")):
                            try:
                                from core.dividend_engine import project_portfolio_income

                                st.caption(_bi(
                                    "توقع الدخل السنوي من الأرباح الموزعة، العائد على التكلفة، واستدامة التوزيعات",
                                    "Project annual dividend income, yield-on-cost, and payout sustainability"
                                ))
                                _di_c1, _di_c2 = st.columns(2)
                                with _di_c1:
                                    _di_contrib = st.number_input(
                                        _bi("إضافة سنوية","Annual Contribution"),
                                        min_value=0.0, value=0.0, step=1000.0,
                                        format="%.0f", key="di_contrib",
                                    )
                                with _di_c2:
                                    _di_growth = st.slider(_bi("افتراض نمو التوزيعات %","Dividend Growth %"),
                                                           -5.0, 15.0, 0.0, 0.5, key="di_growth")

                                if st.button("💸 " + _bi("احسب الدخل المتوقع", "Project Income"), key="run_dividend"):
                                    with st.spinner(_bi("جاري جلب بيانات التوزيعات…","Fetching dividend data…")):
                                        _di = project_portfolio_income(
                                            _pos_df2,
                                            annual_contribution=_di_contrib,
                                            growth_assumption_pct=_di_growth,
                                        )

                                    _dm1, _dm2, _dm3, _dm4 = st.columns(4)
                                    with _dm1:
                                        st.metric(_bi("الدخل السنوي","Annual Income"),
                                                  f"{_di['total_annual_income']:,.0f}")
                                    with _dm2:
                                        st.metric(_bi("المتوسط الشهري","Monthly Avg"),
                                                  f"{_di['monthly_average_income']:,.0f}")
                                    with _dm3:
                                        st.metric(_bi("عائد المحفظة %","Portfolio Yield %"),
                                                  f"{_di['portfolio_yield_pct']:.2f}%")
                                    with _dm4:
                                        _yoc = _di["yield_on_cost_pct"]
                                        st.metric(_bi("العائد على التكلفة %","Yield on Cost %"),
                                                  f"{_yoc:.2f}%" if _yoc is not None else "—")

                                    _dm5, _dm6, _dm7 = st.columns(3)
                                    with _dm5:
                                        _wp = _di["weighted_payout_ratio"]
                                        st.metric(_bi("نسبة التوزيع %","Payout Ratio %"),
                                                  f"{_wp:.1f}%" if _wp is not None else "—")
                                    with _dm6:
                                        _wg = _di["weighted_growth_rate"]
                                        st.metric(_bi("نمو التوزيعات %","Growth Rate %"),
                                                  f"{_wg:+.2f}%" if _wg is not None else "—")
                                    with _dm7:
                                        _ss = _di["sustainability_score"]
                                        _sc = "#10b981" if _ss >= 70 else "#f59e0b" if _ss >= 40 else "#ef4444"
                                        st.markdown(
                                            f"<div style='text-align:center'>"
                                            f"<div style='color:#94a3b8;font-size:.8rem'>{_bi('استدامة','Sustainability')}</div>"
                                            f"<div style='color:{_sc};font-size:1.4rem;font-weight:800'>{_ss}/100</div>"
                                            f"</div>", unsafe_allow_html=True
                                        )

                                    if _di["warnings"]:
                                        with st.expander(_bi("⚠️ تحذيرات","⚠️ Warnings")):
                                            for _w in _di["warnings"]:
                                                st.caption(_w)

                                    if not _di["positions"].empty:
                                        st.markdown(_bi("**حيازات تدفع توزيعات**","**Dividend-Paying Holdings**"))
                                        st.dataframe(_di["positions"], use_container_width=True, hide_index=True)

                                    # Monthly calendar chart
                                    if not _di["monthly_calendar"].empty:
                                        _fig_cal = px.bar(
                                            _di["monthly_calendar"], x="Month", y="Expected Income",
                                            title=_bi("التقويم الشهري للأرباح","Monthly Income Calendar"),
                                            color="Expected Income",
                                            color_continuous_scale="Tealgrn",
                                        )
                                        style_chart(_fig_cal, height=260)
                                        st.plotly_chart(_fig_cal, use_container_width=True, key="dividend_calendar")

                                    # 5-year projection
                                    if not _di["projection_5y"].empty:
                                        st.markdown(_bi("**توقع 5 سنوات**","**5-Year Projection**"))
                                        st.dataframe(_di["projection_5y"], use_container_width=True, hide_index=True)
                                        _fig_proj = px.line(
                                            _di["projection_5y"], x="Year", y="Annual Income",
                                            markers=True,
                                            title=_bi("نمو الدخل السنوي","Annual Income Growth"),
                                        )
                                        style_chart(_fig_proj, height=240)
                                        st.plotly_chart(_fig_proj, use_container_width=True, key="dividend_proj")
                            except Exception as _f8_err:
                                st.warning(f"Dividend engine error: {_f8_err}")

                elif st.session_state.get("pt_show_analysis") and qe is None:
                    st.warning("Pipeline unavailable — analysis requires live data connection.")

        except Exception as _te:
            st.warning(f"Positions tracker unavailable: {_te}")

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
                _wl_add(ticker_clean, str(_uid))
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
                        style_chart(fig_gauge, height=120)
                        fig_gauge.update_layout(margin=dict(l=10,r=10,t=30,b=10),
                                                paper_bgcolor="rgba(0,0,0,0)",
                                                plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{ticker}")
                with rc3:
                    if st.button(f"🗑️", key=f"rm_{ticker}"):
                        st.session_state.watchlist.remove(ticker)
                        _wl_remove(ticker, str(_uid))
                        st.rerun()
                st.markdown("---")

# ═══════════════════════════════════════════════════════
# TAB 7 — AI Assistant
# ═══════════════════════════════════════════════════════
with tab7:
    st.markdown(f"### 🤖 {t('ai_title')}")
    st.caption(t("ai_desc"))

    # ── File upload (CSV / Excel portfolio analysis) ──────────────────────────
    with st.expander("📎 " + ("رفع ملف محفظة (CSV / Excel)" if st.session_state.language == "ar" else "Upload Portfolio File (CSV / Excel)"), expanded=False):
        uploaded_file = st.file_uploader(
            "portfolio-file-upload", type=["csv", "xlsx", "xls"],
            label_visibility="collapsed", key="ai_file_upload"
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    _uf_df = pd.read_csv(uploaded_file)
                else:
                    _uf_df = pd.read_excel(uploaded_file)
                st.dataframe(_uf_df.head(10), use_container_width=True, hide_index=True)
                st.session_state["ai_file_block"] = _uf_df.to_markdown(index=False)
                st.success("✅ " + ("تم تحميل الملف — أرسل سؤالك الآن" if st.session_state.language == "ar" else "File loaded — send your question now"))
            except Exception as _fe:
                st.error(f"❌ {_fe}")
        elif "ai_file_block" in st.session_state:
            st.caption("📄 " + ("ملف محمّل مسبقاً — /clear لمسحه" if st.session_state.language == "ar" else "File already loaded — /clear to remove"))

    # Empty state — show suggested prompts when no chat history
    if not st.session_state.ai_history:
        lang = st.session_state.language
        suggestions = [
            ('حلّل محفظتي الحالية وأعطني توصيات', 'Analyze my current portfolio and give recommendations'),
            ('ما هي أفضل الفرص في السوق السعودي الآن؟', 'What are the best opportunities in Saudi market now?'),
            ('قيّم المخاطر في محفظتي', 'Assess the risks in my portfolio'),
            ('ابني محفظة متوازنة برأس مال 100,000 دولار', 'Build a balanced portfolio with $100,000 capital'),
        ]
        st.markdown('---')
        _bi_local = lambda ar, en: ar if lang == 'ar' else en
        st.caption(_bi_local('💡 جرّب أحد هذه الأسئلة:', '💡 Try one of these:'))
        _sc1, _sc2 = st.columns(2)
        for i, (ar_s, en_s) in enumerate(suggestions):
            _col = _sc1 if i % 2 == 0 else _sc2
            with _col:
                label = ar_s if lang == 'ar' else en_s
                if st.button(label, key=f'sugg_{i}', use_container_width=True):
                    st.session_state['ai_prefill'] = label
                    st.rerun()
        st.markdown('---')

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.ai_history:
        role_cls = "chat-user" if msg["role"] == "user" else "chat-ai"
        st.markdown(f'<div class="{role_cls}">{msg["content"]}</div>', unsafe_allow_html=True)
        # Show download button if message carries a PDF attachment
        if msg.get("pdf_path") and Path(msg["pdf_path"]).exists():
            with open(msg["pdf_path"], "rb") as _pf:
                st.download_button(
                    label="📥 " + ("تحميل التقرير PDF" if st.session_state.language == "ar" else "Download PDF Report"),
                    data=_pf.read(),
                    file_name=msg.get("pdf_filename", "report.pdf"),
                    mime="application/pdf",
                    key=f"dl_{msg['pdf_path']}",
                )

    # ── Input row ─────────────────────────────────────────────────────────────
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
            st.session_state.pop("ai_file_block", None)
            st.rerun()

    # Handle suggested prompt prefill
    if st.session_state.get('ai_prefill'):
        user_query = st.session_state.pop('ai_prefill')
        send_clicked = True

    if send_clicked and user_query.strip():
        st.session_state.ai_history.append({"role": "user", "content": user_query})
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.ai_history]
        portfolio_ctx = _build_portfolio_ai_context(_uid, filtered_df)
        file_block    = st.session_state.get("ai_file_block", "")

        with st.spinner(t("ai_thinking")):
            ai_reply  = ""
            dl_info   = None
            market_context, selected_count = build_ai_market_context(
                filtered_df, user_query, max_rows=18,
            )
            prefer_agent = _should_use_agent_for_ai(user_query, file_block=file_block)

            if prefer_agent:
                ai_reply, new_sid, dl_info = _agent_chat(
                    uid=_uid,
                    message=user_query,
                    session_id=st.session_state.ai_session_id,
                    portfolio_ctx=portfolio_ctx,
                    file_block=file_block,
                )
                if new_sid:
                    st.session_state.ai_session_id = new_sid
                if file_block and ai_reply:
                    st.session_state.pop("ai_file_block", None)

            if not ai_reply:
                try:
                    ai_reply = ask_eisa_ai(
                        messages=messages,
                        market_context=market_context,
                        stock_count=selected_count,
                        language=st.session_state.language,
                        portfolio_context=portfolio_ctx,
                        file_context=file_block,
                    )
                    ai_reply += f"\n\n_Context: {selected_count} stocks (live dashboard mode)_"
                    if file_block:
                        st.session_state.pop("ai_file_block", None)
                except requests.exceptions.Timeout:
                    ai_reply = "⏱️ انتهى وقت الانتظار. يرجى المحاولة مرة أخرى." if st.session_state.language == "ar" else "⏱️ Request timed out. Please try again."
                except Exception as _e:
                    ai_reply = f"❌ Error: {_e}"

        # Store message + optional PDF reference in history
        history_entry = {"role": "assistant", "content": ai_reply}
        if dl_info:
            history_entry["pdf_path"]     = str(dl_info["path"])
            history_entry["pdf_filename"] = dl_info["filename"]
        st.session_state.ai_history.append(history_entry)
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

# ═══════════════════════════════════════════════════════
# TAB 9 — Forex
# ═══════════════════════════════════════════════════════
with tab9:
    st.markdown(f"### 💱 {t('forex_tab')}")

    try:
        from core.forex import ForexFetcher, FOREX_PAIRS
        _fx_data = ForexFetcher().fetch(use_cache=True)
    except Exception as _fx_exc:
        _fx_data = []
        st.warning(f"{t('forex_unavailable')}: {_fx_exc}")

    if _fx_data:
        # ── KPI row: key Arab rates ────────────────────────────────────────────
        arab_kpis = [("USDAED=X","🇦🇪 USD/AED"),("USDSAR=X","🇸🇦 USD/SAR"),
                     ("USDEGP=X","🇪🇬 USD/EGP"),("USDKWD=X","🇰🇼 USD/KWD"),
                     ("USDQAR=X","🇶🇦 USD/QAR"),("USDBHD=X","🇧🇭 USD/BHD")]
        _fx_by_sym = {r["symbol"]: r for r in _fx_data}
        kpi_cols_fx = st.columns(len(arab_kpis))
        for idx, (sym, label) in enumerate(arab_kpis):
            row = _fx_by_sym.get(sym, {})
            price = row.get("price")
            chg   = row.get("change_pct")
            val   = f"{price:,.4f}" if price else "—"
            delta = f"{chg:+.2f}%" if chg is not None else None
            src   = row.get("source","")
            with kpi_cols_fx[idx]:
                st.metric(label, val, delta=delta)
                if src == "fallback":
                    st.caption("📌 est.")

        st.divider()

        # ── Grouped tables ─────────────────────────────────────────────────────
        _cat_labels = {
            "arab":  f"🌙 {t('arab_pairs')}",
            "major": f"🌍 {t('major_pairs')}",
            "em":    f"🌱 {t('em_pairs')}",
        }
        _cat_order = ["arab", "major", "em"]

        for cat in _cat_order:
            cat_rows = [r for r in _fx_data if r.get("category") == cat]
            if not cat_rows:
                continue
            st.markdown(f"#### {_cat_labels.get(cat, cat)}")
            df_cat = pd.DataFrame(cat_rows)
            display_cols = {
                "name":       t("pair"),
                "price":      t("rate"),
                "prev_close": t("prev_close"),
                "change_pct": t("change_pct"),
                "source":     "Source",
            }
            df_show = df_cat[[c for c in display_cols if c in df_cat.columns]].copy()
            df_show = df_show.rename(columns=display_cols)

            # Style change column
            def _style_chg(val):
                if val is None or pd.isna(val):
                    return ""
                return "color: #10b981" if val > 0 else ("color: #ef4444" if val < 0 else "")

            chg_col = t("change_pct")
            if chg_col in df_show.columns:
                df_show[chg_col] = df_show[chg_col].apply(
                    lambda v: f"{v:+.4f}%" if v is not None and not pd.isna(v) else "—"
                )
            rate_col = t("rate")
            if rate_col in df_show.columns:
                df_show[rate_col] = df_show[rate_col].apply(
                    lambda v: f"{v:,.6f}" if v is not None else "—"
                )
            prev_col = t("prev_close")
            if prev_col in df_show.columns:
                df_show[prev_col] = df_show[prev_col].apply(
                    lambda v: f"{v:,.6f}" if v is not None else "—"
                )

            st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Source timestamp
        if _fx_data:
            ts_fx = _fx_data[0].get("timestamp", "")
            if ts_fx:
                st.caption(f"🕐 {t('last_update')}: {ts_fx}")
    else:
        if not _fx_data:
            st.info(t("forex_unavailable"))


# ═══════════════════════════════════════════════════════
# TAB 10 — Crypto
# ═══════════════════════════════════════════════════════
with tab10:
    st.markdown(f"### 🪙 {t('crypto_tab')}")

    cache_obj_c, _, _ = _get_pipeline()
    df_crypto, ts_crypto = cache_obj_c.get_latest("crypto") if cache_obj_c is not None else (None, None)

    if df_crypto is None or df_crypto.empty:
        st.info(t("crypto_unavailable"))
    else:
        df_crypto = df_crypto.copy()
        for col in ("close", "change", "volume", "market_cap_basic"):
            if col in df_crypto.columns:
                df_crypto[col] = pd.to_numeric(df_crypto[col], errors="coerce")

        # ── KPI row: BTC / ETH / BNB ──────────────────────────────────────────
        kpi_crypto = [("BTCUSDT","₿ Bitcoin","BTC"),("ETHUSDT","Ξ Ethereum","ETH"),("BNBUSDT","⬡ BNB","BNB")]
        kpi_cols_c = st.columns(3)
        for idx, (sym, label, short) in enumerate(kpi_crypto):
            _row = None
            if "name" in df_crypto.columns:
                _matches = df_crypto[df_crypto["name"].str.upper().str.contains(short, na=False)]
                if _matches.empty:
                    _matches = df_crypto[df_crypto.index.astype(str).str.upper().str.contains(sym, na=False)]
                _row = _matches.iloc[0] if not _matches.empty else None
            price_c = _row["close"] if _row is not None and pd.notna(_row.get("close")) else None
            chg_c   = _row["change"] if _row is not None and pd.notna(_row.get("change")) else None
            val_c   = f"${price_c:,.2f}" if price_c else "—"
            delta_c = f"{chg_c:+.2f}%" if chg_c is not None else None
            with kpi_cols_c[idx]:
                st.metric(label, val_c, delta=delta_c)

        st.divider()

        # ── Top gainers / losers ───────────────────────────────────────────────
        if "change" in df_crypto.columns:
            col_g, col_l = st.columns(2)
            with col_g:
                st.markdown("#### 🟢 Top Gainers")
                df_g = df_crypto.nlargest(10, "change")[["name","close","change"]].copy()
                df_g.columns = [t("name"), t("price"), t("change")]
                df_g[t("price")] = df_g[t("price")].apply(lambda v: f"${v:,.4f}" if pd.notna(v) else "—")
                df_g[t("change")] = df_g[t("change")].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
                st.dataframe(df_g, use_container_width=True, hide_index=True)
            with col_l:
                st.markdown("#### 🔴 Top Losers")
                df_l = df_crypto.nsmallest(10, "change")[["name","close","change"]].copy()
                df_l.columns = [t("name"), t("price"), t("change")]
                df_l[t("price")] = df_l[t("price")].apply(lambda v: f"${v:,.4f}" if pd.notna(v) else "—")
                df_l[t("change")] = df_l[t("change")].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
                st.dataframe(df_l, use_container_width=True, hide_index=True)

        st.divider()

        # ── Full table ─────────────────────────────────────────────────────────
        st.markdown(f"#### 📊 All Cryptocurrencies ({len(df_crypto)} coins)")
        sort_opts = [c for c in ["market_cap_basic","change","volume","close"] if c in df_crypto.columns]
        if sort_opts:
            sort_by = st.selectbox("Sort by", sort_opts, index=0, key="crypto_sort")
            df_table = df_crypto.sort_values(sort_by, ascending=False)
        else:
            df_table = df_crypto

        show_cols_c = [c for c in ["name","close","change","volume","market_cap_basic"] if c in df_table.columns]
        df_show_c = df_table[show_cols_c].copy()
        rename_map = {"name": t("name"), "close": t("price"), "change": t("change"),
                      "volume": t("volume"), "market_cap_basic": t("cap")}
        df_show_c = df_show_c.rename(columns={k: v for k, v in rename_map.items() if k in df_show_c.columns})
        pr_col = t("price")
        if pr_col in df_show_c.columns:
            df_show_c[pr_col] = df_show_c[pr_col].apply(lambda v: f"${v:,.4f}" if pd.notna(v) else "—")
        ch_col = t("change")
        if ch_col in df_show_c.columns:
            df_show_c[ch_col] = df_show_c[ch_col].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        cap_col = t("cap")
        if cap_col in df_show_c.columns:
            df_show_c[cap_col] = df_show_c[cap_col].apply(
                lambda v: f"${v/1e9:.2f}B" if pd.notna(v) and v >= 1e9
                else (f"${v/1e6:.1f}M" if pd.notna(v) and v >= 1e6 else ("—" if not pd.notna(v) else f"${v:,.0f}"))
            )

        st.dataframe(df_show_c, use_container_width=True, hide_index=True)

        if ts_crypto:
            st.caption(f"🕐 {t('last_update')}: {ts_crypto}")


# ═══════════════════════════════════════════════════════
# TAB ADMIN — User Management (admin only)
# ═══════════════════════════════════════════════════════
if _is_admin and tab_admin is not None:
    with tab_admin:
        from core.user_db import list_users as _lu, create_user as _cu, update_user as _uu, delete_user as _du
        from core.auth import hash_password as _hp, generate_temp_password as _gtp

        st.markdown("### 🛡️ إدارة المستخدمين / User Management")

        # ── Current users table ───────────────────────────────────────────────
        _users = _lu()
        if _users:
            _u_df = pd.DataFrame(_users)[["id","name","email","role","is_active","must_change_pw","last_login"]]
            _u_df.columns = ["ID","Name","Email","Role","Active","Must Change PW","Last Login"]
            _u_df["Active"] = _u_df["Active"].map({1:"✅",0:"❌"})
            _u_df["Must Change PW"] = _u_df["Must Change PW"].map({1:"⚠️ Yes",0:"No"})
            st.dataframe(_u_df, use_container_width=True, hide_index=True)
        else:
            st.info("No users yet.")

        st.markdown("---")

        # ── Create new user ───────────────────────────────────────────────────
        with st.expander("➕ إضافة مستخدم جديد / Add New User", expanded=False):
            ac1, ac2 = st.columns(2)
            with ac1:
                _new_name  = st.text_input("الاسم / Name", key="adm_name")
                _new_email = st.text_input("البريد الإلكتروني / Email", key="adm_email")
            with ac2:
                _new_role  = st.selectbox("الدور / Role", ["user","admin"], key="adm_role")
                _new_pw    = st.text_input("كلمة المرور / Password (leave blank = auto)", key="adm_pw")

            if st.button("إنشاء / Create", type="primary", key="adm_create"):
                if not _new_name or not _new_email:
                    st.error("الاسم والبريد الإلكتروني مطلوبان")
                else:
                    _pw_final = _new_pw.strip() if _new_pw.strip() else _gtp()
                    try:
                        _cu(_new_email.strip(), _new_name.strip(), _hp(_pw_final),
                            role=_new_role, must_change_pw=not bool(_new_pw.strip()))
                        st.success(f"✅ تم إنشاء الحساب | كلمة المرور: `{_pw_final}`")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"❌ {_e}")

        # ── Manage existing user ──────────────────────────────────────────────
        with st.expander("⚙️ تعديل مستخدم / Manage User", expanded=False):
            _sel_user = st.selectbox(
                "اختر مستخدم / Select User",
                [f"{u['id']} — {u['email']}" for u in _users],
                key="adm_sel_user"
            )
            if _sel_user:
                _sel_id = int(_sel_user.split(" — ")[0])
                _sel_obj = next((u for u in _users if u["id"] == _sel_id), None)
                if _sel_obj:
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        _active_label = "✅ Active" if _sel_obj["is_active"] else "❌ Inactive"
                        if st.button(f"Toggle Active ({_active_label})", key="adm_toggle_active"):
                            _uu(_sel_id, is_active=0 if _sel_obj["is_active"] else 1)
                            st.rerun()
                    with mc2:
                        if st.button("🔑 Reset Password", key="adm_reset_pw"):
                            _tmp = _gtp()
                            _uu(_sel_id, password_hash=_hp(_tmp), must_change_pw=1)
                            st.success(f"كلمة المرور الجديدة: `{_tmp}`")
                    with mc3:
                        if _sel_id != _current_user["id"]:
                            if st.button("🗑️ Delete User", key="adm_del"):
                                _du(_sel_id)
                                st.success("تم الحذف")
                                st.rerun()
                        else:
                            st.caption("(لا يمكن حذف حسابك الحالي)")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="eisax-footer">
    <span>📅 {now_dubai_str()}</span>
    <span>📊 {len(filtered_df)} {t('stocks_count')}</span>
    <span class="brand">🚀 EisaX Analytics v5.1</span>
</div>
""", unsafe_allow_html=True)
