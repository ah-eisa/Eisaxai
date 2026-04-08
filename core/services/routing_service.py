"""
core/services/routing_service.py
──────────────────────────────────
Fast-path detection and file-analysis routing extracted from process_message.

Public API
──────────
    is_export_request(message)           -> bool
    is_file_analysis(message)            -> bool
    is_bond_request(message)             -> bool
    is_greeting(message)                 -> bool
    detect_arabic_ticker(message)        -> str | None
    detect_dfm_screen(message)           -> tuple[str, str] | None
        Returns (criterion, markdown_table) or None.
    handle_file_analysis(message, financial_agent, gemini_client,
                         gemini_model, gemini_api_key)
                                         -> tuple[str, str]
        Returns (reply_text, agent_label).
"""

from __future__ import annotations

import logging
import re as _re
from typing import Any

logger = logging.getLogger(__name__)


# ── Keyword sets ──────────────────────────────────────────────────────────────

_EXPORT_KEYWORDS: set[str] = {
    "export", "صدر", "save", "download", "pdf", "word", "docx",
}

_BOND_KEYWORDS: list[str] = [
    "bond", "bonds", "treasury", "treasuries", "sukuk", "fixed income",
    "t-bill", "t-bills", "sovereign debt", "yield curve", "coupon",
    "سندات", "سند", "صكوك", "أذونات", "دخل ثابت", "isin",
]

_GREETING_WORDS: set[str] = {
    "hi", "hello", "hey", "morning", "evening", "howdy", "sup", "yo", "greetings",
    "مرحبا", "ازيك", "ازى", "ازيكو", "هاي", "هالو", "اهلا", "أهلا", "سلام",
    "صباح", "مساء", "السلام", "هلا", "اهلين", "مرحبتين",
}

_ANALYZE_WORDS: set[str] = {
    "حلل", "حللي", "حللى", "analyze", "analysis", "تحليل",
    # Mode/speed modifiers — treat as noise in ticker detection
    "بسرعة", "سريع", "سريعة", "ملخص", "مختصر", "مختصرة",
    "quick", "brief", "summary", "short", "fast",
    "full", "detailed", "deep", "cio", "memo",
}

# Arabic name → ticker mapping (used by Arabic stock fast-path)
ARABIC_COMPANY_MAP: dict[str, str] = {
    "ارامكو": "2222.SR", "أرامكو": "2222.SR", "aramco": "2222.SR",
    "الراجحي": "1120.SR", "rajhi": "1120.SR",
    "سابك": "2010.SR", "sabic": "2010.SR",
    "stc": "7010.SR", "اتصالات السعودية": "7010.SR",
    "اعمار": "EMAAR.DU", "أعمار": "EMAAR.DU", "emaar": "EMAAR.DU",
    "فاب": "FAB.AE", "fab": "FAB.AE", "بنك ابوظبي": "FAB.AE",
    "ادنوك": "ADNOCDIST.AE", "أدنوك": "ADNOCDIST.AE", "adnoc": "ADNOCDIST.AE",
    "adnocgas": "ADNOCGAS.AE", "adnocdrill": "ADNOCDRILL.AE",
    "taqa": "TAQA.AE", "طاقة": "TAQA.AE",
    "ihc": "IHC.AE",
    "eand": "EAND.AE", "اتصالات": "EAND.AE",
    "بيتك": "KFH.KW", "kfh": "KFH.KW",
    "الوطني": "NBK.KW", "nbk": "NBK.KW",
    "زين": "ZAIN.KW", "zain": "ZAIN.KW",
    "qnb": "QNBK.QA", "قطر الوطني": "QNBK.QA",
    "cib": "COMI.CA", "كوميرشيال": "COMI.CA",
    "ابل": "AAPL", "آبل": "AAPL", "apple": "AAPL",
    "نيفيديا": "NVDA", "نفيديا": "NVDA", "nvidia": "NVDA",
    "تسلا": "TSLA", "tesla": "TSLA",
    "مايكروسوفت": "MSFT", "microsoft": "MSFT",
    "امازون": "AMZN", "amazon": "AMZN",
    "جوجل": "GOOGL", "google": "GOOGL",
    "ميتا": "META", "فيسبوك": "META", "meta": "META",
    "بيتكوين": "BTC-USD", "bitcoin": "BTC-USD",
}

_DFM_KEYWORDS: list[str] = ["dfm", "dubai financial", "سوق دبي", "البورصة الاماراتية"]

_DFM_SCREEN_MAP: dict[str, list[str]] = {
    "low_pe":   ["lowest pe", "low pe", "cheap stocks", "undervalued", "أرخص أسهم", "ارخص"],
    "high_pe":  ["highest pe", "growth stocks", "أغلى أسهم"],
    "low_beta": ["low risk", "stable stocks", "low beta", "أقل تذبذب", "defensive"],
}

# Words that look like tickers but aren't (used in file-analysis)
_SKIP_WORDS: frozenset[str] = frozenset({
    "AT", "OR", "IN", "OF", "TO", "BY", "VS", "AND", "THE", "FOR",
    "SAR", "USD", "AED", "EGP", "SR", "FILE", "USER", "CSV",
    "EPS", "ROE", "ETF", "YOY", "TTM", "ANALYSIS", "CONTENT",
})


# ── Detection helpers ─────────────────────────────────────────────────────────

def is_export_request(message: str) -> bool:
    ml = message.lower()
    return any(kw in ml for kw in _EXPORT_KEYWORDS)


def is_file_analysis(message: str) -> bool:
    return message.startswith("[FILE ANALYSIS]")


def is_bond_request(message: str) -> bool:
    ml = message.lower()
    # Check for ISIN first
    try:
        from core.fixed_income import extract_isin as _extract_isin
        if _extract_isin(message):
            return True
    except Exception:
        pass
    return any(kw in ml for kw in _BOND_KEYWORDS)


def is_greeting(message: str) -> bool:
    words = set(message.lower().strip().split())
    return bool(words & _GREETING_WORDS) and len(message.strip()) < 40


def detect_arabic_ticker(message: str) -> str | None:
    """
    Detect "حلل X" / "X" patterns that map to a known ticker.
    Returns the ticker string or None.
    """
    msg_stripped = message.strip()
    msg_parts    = msg_stripped.split()

    if len(msg_parts) <= 4:
        remaining = [w for w in msg_parts
                     if w.lower() not in _ANALYZE_WORDS
                     and w not in {"سهم", "شركة", "سعر"}]
        company_guess = " ".join(remaining).strip().lower()
        if company_guess in ARABIC_COMPANY_MAP:
            return ARABIC_COMPANY_MAP[company_guess]
        if len(remaining) == 1:
            single = remaining[0].lower()
            for key, val in ARABIC_COMPANY_MAP.items():
                if key in single or single in key:
                    return val

    # Bare company name (1-2 words, no verb)
    if len(msg_parts) <= 2:
        bare = msg_stripped.lower()
        if bare in ARABIC_COMPANY_MAP:
            return ARABIC_COMPANY_MAP[bare]

    return None


def detect_dfm_screen(message: str) -> tuple[str, str] | None:
    """
    Check if message is a DFM screening request.
    Returns (criterion, markdown_table_string) or None.
    """
    ml = message.lower()
    if not any(k in ml for k in _DFM_KEYWORDS):
        return None

    criterion = None
    for crit, kws in _DFM_SCREEN_MAP.items():
        if any(k in ml for k in kws):
            criterion = crit
            break
    if not criterion:
        return None

    try:
        from core.dfm_lookup import screen_dfm
        stocks = screen_dfm(criterion, top_n=10)
        text = f"## DFM Stock Screener: {criterion.replace('_', ' ').title()}\n\n"
        text += "| # | Company | Ticker | P/E | Beta | Market Cap | Avg Vol |\n"
        text += "|---|---------|--------|-----|------|------------|---------|\n"
        for i, s in enumerate(stocks, 1):
            text += (
                f"| {i} | {s['name']} | {s['ticker'] or 'N/A'} | "
                f"{s['pe_ratio'] or 'N/A'} | {s['beta'] or 'N/A'} | "
                f"{s['market_cap']} | {s['avg_vol_3m']} |\n"
            )
        return criterion, text
    except Exception as exc:
        logger.warning("[RoutingService] DFM screen failed: %s", exc)
        return None


# ── File analysis handler ─────────────────────────────────────────────────────

# Weight-column synonyms — if CSV has these it's a weight-based portfolio
_WEIGHT_COLS   = {"weight", "allocation", "alloc", "pct", "percent", "%", "proportion",
                  "value", "amount", "usd", "sar", "aed"}
# Share-count synonyms — if CSV has these it's a P&L portfolio (shares + cost)
_SHARES_COLS   = {"shares", "qty", "quantity", "units", "lots",
                  "avg_price", "avg price", "cost_basis", "cost basis", "avg_cost"}


def _try_parse_weight_csv(message: str) -> dict | None:
    """
    Try to extract a weight-based portfolio dict from [FILE ANALYSIS] message text.
    Returns {TICKER: weight_0_to_1, ...} or None if not parseable as weight portfolio.
    """
    try:
        import io, re as _r, pandas as _pd

        # Extract CSV block from message (between file content header and "User question:")
        csv_match = _r.search(
            r"File content \([^)]+\):\n\n(.+?)(?:\n\nUser question:|$)",
            message, _r.DOTALL
        )
        raw_csv = csv_match.group(1).strip() if csv_match else message

        # Try to parse as CSV
        df = _pd.read_csv(io.StringIO(raw_csv))
        if df.empty or len(df.columns) < 2:
            return None

        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

        # Identify ticker column
        ticker_col = next((c for c in cols if c in
                           {"ticker","symbol","stock","name","asset","security"}), cols[0])

        # Identify weight column
        weight_col = next((c for c in cols if c in _WEIGHT_COLS), None)

        # Identify if shares-based (P&L portfolio) — if so, return None
        has_shares = any(c in _SHARES_COLS for c in cols)
        if has_shares and not weight_col:
            return None  # route to _handle_cio_analysis instead

        if not weight_col:
            # Try second numeric column as weight
            numeric_cols = [c for c in cols if c != ticker_col
                            and _pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                return None
            weight_col = numeric_cols[0]

        # Build portfolio dict
        portfolio: dict = {}
        for _, row in df.iterrows():
            t = str(row[ticker_col]).strip().upper()
            if not t or t in {"NAN", "TICKER", "SYMBOL", ""}:
                continue
            try:
                w = float(str(row[weight_col]).replace("%", "").replace(",", ""))
            except Exception:
                continue
            if w <= 0:
                continue
            portfolio[t] = w

        if not portfolio:
            return None

        # Normalise: if weights look like percentages (sum > 1.5), divide by 100
        total = sum(portfolio.values())
        if total > 1.5:
            portfolio = {t: w / 100 for t, w in portfolio.items()}

        logger.info("[FileCSV] Parsed weight portfolio: %d tickers, sum=%.2f",
                    len(portfolio), sum(portfolio.values()))
        return portfolio

    except Exception as exc:
        logger.debug("[FileCSV] CSV parse failed: %s", exc)
        return None


def _call_portfolio_api(portfolio: dict, user_id: str = "anonymous") -> str | None:
    """
    Call the /v1/upload-portfolio endpoint via localhost with the parsed portfolio dict.
    Builds a synthetic CSV and POSTs it as a file upload.
    Returns the markdown analysis string or None on error.
    """
    try:
        import io, requests as _rq, os as _os
        from dotenv import load_dotenv
        from core.config import BASE_DIR as _BD
        load_dotenv(str(_BD / ".env"))

        token = _os.environ.get("SECURE_TOKEN", "")
        if not token:
            logger.warning("[FileCSV→API] No SECURE_TOKEN — cannot self-call")
            return None

        # Build minimal CSV from portfolio dict
        csv_rows = ["ticker,weight"] + [f"{t},{w}" for t, w in portfolio.items()]
        csv_bytes = "\n".join(csv_rows).encode()

        resp = _rq.post(
            "http://localhost:8000/v1/upload-portfolio",
            headers={"X-API-Key": token},
            files={"file": ("portfolio.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"user_id": user_id},
            timeout=180,
        )
        if resp.status_code == 200:
            data = resp.json()
            analysis = data.get("analysis") or data.get("report") or ""
            if analysis:
                logger.info("[FileCSV→API] Portfolio analysis received (%d chars)", len(analysis))
                return analysis
        logger.warning("[FileCSV→API] Bad status %d", resp.status_code)
        return None
    except Exception as exc:
        logger.error("[FileCSV→API] Self-call failed: %s", exc)
        return None


def handle_file_analysis(
    message:         str,
    financial_agent: Any | None,
    gemini_client:   Any | None,
    gemini_model:    str = "gemini-2.0-flash",
    gemini_api_key:  str = "",
    user_id:         str = "anonymous",
    session_id:      str = "default",
) -> tuple[str, str]:
    """
    Route a [FILE ANALYSIS] message:
      1. Weight-based CSV  → /v1/upload-portfolio (full institutional analysis)
      2. Shares+cost CSV   → _handle_cio_analysis (P&L analysis)
      3. Other files       → Gemini vision fallback

    Returns (reply_text, agent_label)
    """
    message_lower = message.lower()
    reply: str | None = None
    label = "EisaX Vision"

    # ── Route 1: Weight-based portfolio CSV ───────────────────────────────────
    portfolio = _try_parse_weight_csv(message)
    if portfolio:
        logger.info("[File→Portfolio] Detected weight CSV — routing to /v1/upload-portfolio")
        reply = _call_portfolio_api(portfolio, user_id=user_id)
        if reply:
            return reply, "EisaX Portfolio Engine"
        # If API call failed, fall through to CIO

    # ── Route 2: Shares+cost CIO analysis ────────────────────────────────────
    _portfolio_hdrs = [
        "ticker", "shares", "symbol", "qty", "quantity",
        "weight", "allocation", "avg price", "avg_price",
        "cost", "holding", "position",
    ]
    has_portfolio_kws = sum(1 for k in _portfolio_hdrs if k in message_lower) >= 1
    file_tickers = [
        t for t in set(_re.findall(r"\b([A-Z]{2,5})\b", message))
        if t not in _SKIP_WORDS and len(t) >= 2
    ]
    is_portfolio_file = has_portfolio_kws or len(file_tickers) >= 3

    if is_portfolio_file and financial_agent:
        try:
            result = financial_agent._handle_cio_analysis(message, session_id)
            reply  = result.get("reply") or None
            if reply:
                logger.info("[File→CIO] Shares/cost portfolio → CIO analysis")
                return reply, "EisaX CIO"
        except Exception as exc:
            logger.error("[File→CIO] Failed: %s", exc)

    # ── Route 3: Gemini fallback ──────────────────────────────────────────────
    if not reply:
        try:
            from google import genai as _genai
            gc   = _genai.Client(api_key=gemini_api_key)
            resp = gc.models.generate_content(
                model=gemini_model,
                contents=(
                    "You are EisaX financial analyst. Analyze this file content and provide "
                    "detailed financial insights. If it contains portfolio/stock data, analyze "
                    f"each position.\n\n{message[:12000]}"
                ),
            )
            reply = resp.text.strip() if resp.text else "❌ فشل تحليل الملف"
        except Exception as exc:
            reply = f"❌ خطأ: {str(exc)}"

    return reply or "❌", label
