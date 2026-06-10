"""
EisaX Proprietary ScoreCard v2
================================
نظام تسجيل محايد حقيقي — مش كل حاجة تاخد 100%.

المبادئ:
  - البيتا العالية = RISK = بتخصم، مش بتزيد
  - الـ Valuation مهم — P/E مرتفع جداً بيخصم
  - الـ Upside لازم يكون real مش inflated
  - الـ Sentiment بيتاخد من إشارات متعددة مش كلمة واحدة
  - في risk adjustment layer في الآخر
  - Max score حقيقي للسهم الممتاز: ~88-92 مش 99 دايماً
"""

import re
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# DATA SANITY LAYER — pre-render field validation
# ══════════════════════════════════════════════════════════════════════════════
# Low-correlation hedges that legitimately have near-zero or negative beta
_NEAR_ZERO_BETA_ALLOWED = frozenset({"TLT", "GLD", "SHY", "BIL", "SLV", "IAU", "SGOL", "GLDM"})

def sanitize_field(field_name: str, value, ticker: str = ""):
    """
    Validate a single data field before rendering.
    Returns cleaned value, or None if the value is unreliable.

    Rendering rule: if this returns None → display "N/A (data unverified)"
    """
    if value is None:
        return None

    try:
        v = float(value)
    except (ValueError, TypeError):
        return value  # non-numeric passthrough (strings, etc.)

    # ── Beta ──────────────────────────────────────────────────────────────
    if field_name == "beta":
        if abs(v) < 0.05 and ticker.upper() not in _NEAR_ZERO_BETA_ALLOWED:
            return None   # unreliable (e.g. Aramco -0.01)
        if v < -0.5:
            return None   # unrealistic
        if v > 5.0:
            return None   # data error

    # ── Dividend Yield (%) ────────────────────────────────────────────────
    if field_name in ("div_yield", "dividend_yield"):
        if v > 30:
            return None   # data error (>30% yield)
        if v < 0:
            return None   # impossible

    # ── P/E Ratio ─────────────────────────────────────────────────────────
    if field_name in ("pe", "forward_pe", "ttm_pe"):
        if v < 0:
            return None   # negative P/E = loss-making (show separately)
        if v > 1000:
            return None   # data error

    # ── Market Cap ────────────────────────────────────────────────────────
    if field_name in ("mktcap", "market_cap"):
        if v < 0:
            return None

    return v


def render_field(field_name: str, value, ticker: str = "", fmt: str = "{:.2f}") -> str:
    """Sanitize + format a field for display. Returns formatted string or N/A."""
    clean = sanitize_field(field_name, value, ticker)
    if clean is None:
        return "N/A (data unverified)"
    try:
        return fmt.format(clean)
    except (ValueError, TypeError):
        return str(clean)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def parse_report(text: str) -> Optional[dict]:
    """
    استخرج كل البيانات القابلة للقياس من التقرير.
    أي field مش موجود بيتملى بـ None — مش بـ default يخلى الـ score مرتفع.
    """
    try:
        data = {}

        # ── السعر الحالي ──
        p = re.search(r'Live Price.*?\$([\d\.,]+)', text, re.I | re.S)
        data['price'] = float(p.group(1).replace(',', '')) if p else None

        # ── السعر المستهدف ──
        t = re.search(r'(?:price target|Price Target|target price)[^$]*\$([\d\.,]+)', text, re.I | re.S)
        data['target'] = float(t.group(1).replace(',', '')) if t else None

        # ── البيتا ──
        b = re.search(r'Beta[:\s|]+([0-9]+\.?[0-9]*)', text, re.I)
        if not b:
            b = re.search(r'Beta of ([0-9]+\.?[0-9]*)', text, re.I)
        if not b:
            b = re.search(r'Beta[^0-9]*([0-9]+\.[0-9]+)', text, re.I)
        data['beta'] = float(b.group(1)) if b else None

        # ── Market Cap ──
        m = re.search(r'Market\s*Cap[^0-9]*([\d\.]+)\s*([BTM])', text, re.I | re.S)
        if not m:
            m = re.search(r'Market Cap[:\s]*\$?([\d\.]+)\s*([BTM])', text, re.I)
        if not m:
            # Try without currency
            m = re.search(r'Market Cap[^$]*\$?([\d\.]+)\s*([BTM])', text, re.I)
        if m:
            v, u = m.groups()
            mult = {'T': 1e12, 'B': 1e9, 'M': 1e6}.get(u.upper(), 1e9)
            data['mc'] = float(v) * mult
        else:
            data['mc'] = None

        # ── Quality Score ──
        q = re.search(r'Quality\s*Score.*?(\d+)/100', text, re.I)
        data['quality'] = int(q.group(1)) if q else None

        # ── Forward P/E ──
        fpe = re.search(r'Forward\s*P/?E[^0-9]*([\d\.]+)', text, re.I)
        data['forward_pe'] = float(fpe.group(1)) if fpe else None

        # ── TTM P/E ──
        ttm = re.search(r'(?:TTM|P/E\s*\(TTM\))[^0-9]*([\d\.]+)', text, re.I)
        data['ttm_pe'] = float(ttm.group(1)) if ttm else None

        # ── RSI ──
        rsi = re.search(r'RSI[^0-9]*([\d\.]+)', text, re.I)
        data['rsi'] = float(rsi.group(1)) if rsi else None

        # ── ADX ──
        adx = re.search(r'ADX[^0-9]*([\d\.]+)', text, re.I)
        data['adx'] = float(adx.group(1)) if adx else None

        # ── Net Margin ──
        nm = re.search(r'(?:Net\s*Margin|net margin)[^0-9]*([\d\.]+)%', text, re.I)
        data['net_margin'] = float(nm.group(1)) if nm else None

        # ── Gross Margin ──
        gm = re.search(r'[Gg]ross\s*[Mm]argin[^0-9]*([\d\.]+)%', text, re.I)
        data['gross_margin'] = float(gm.group(1)) if gm else None

        # ── ROE ──
        roe = re.search(r'ROE[^0-9]*([\d\.]+)%', text, re.I)
        data['roe'] = float(roe.group(1)) if roe else None

        # ── Revenue Growth ──
        rg = re.search(r'revenue growth[^0-9]*([\d\.]+)%', text, re.I)
        if not rg:
            rg = re.search(r'revenue growth of ([\d\.]+)%', text, re.I)
        data['rev_growth'] = float(rg.group(1)) if rg else None

        # ── Ticker ──
        tk = re.search(r'Intelligence Report:\s*([A-Z]{1,6})', text)
        data['ticker'] = tk.group(1) if tk else "N/A"

        # ── Sector ──
        sec = re.search(r'Sector:\s*([^\n|]+)', text, re.I)
        data['sector'] = sec.group(1).strip() if sec else "Unknown"

        # ── Verdict من التقرير ──
        v = re.search(r'\bVERDICT[:\s]*\*{0,2}(BUY|SELL|HOLD|STRONG BUY|STRONG SELL)\b', text, re.I)
        data['llm_verdict'] = v.group(1).upper().strip() if v else "HOLD"

        # ══════════════════════════
        # SIGNALS — متعددة مش كلمة واحدة
        # ══════════════════════════

        text_lower = text.lower()

        # Moat — لازم evidence حقيقية مش بس كلمة
        moat_signals = [
            "pricing power", "switching cost", "network effect",
            "dominant market", "market leader", "unassailable",
            "monopol", "moat", "fortress balance",
            "dominant competitive moat", "structural advantage",
            "best-in-class", "market share", "unmatched"
        ]
        data['moat_signals'] = sum(1 for s in moat_signals if s in text_lower)
        data['has_moat'] = data['moat_signals'] >= 2  # لازم أكتر من كلمة واحدة

        # Bearish signals — متعددة
        bearish_signals = [
            "momentum is bearish", "downtrend", "bearish crossover",
            "death cross", "below sma200", "rsi overbought"
        ]
        data['bearish_count'] = sum(1 for s in bearish_signals if s in text_lower)
        data['is_bearish'] = data['bearish_count'] >= 1

        # Bullish signals
        bullish_signals = [
            "uptrend", "bullish", "above sma200", "breakout",
            "strong buy", "accumulate", "golden cross",
            "trend is bullish", "bullish crossover", "rising sma"
        ]
        data['bullish_count'] = sum(1 for s in bullish_signals if s in text_lower)

        # Tech sector check — broader
        tech_keywords = [
            "technology", "semiconductor", "software", "cloud",
            "artificial intelligence", "ai infrastructure", "chip",
            "data center", "nvidia", "intel", "amd"
        ]
        data['is_tech'] = any(k in text_lower for k in tech_keywords)

        # Risks mentioned
        risk_keywords = [
            "concentration risk", "valuation risk", "multiple compression",
            "cyclical", "competition", "regulatory", "liquidity risk",
            "high beta", "interest rate"
        ]
        data['risk_count'] = sum(1 for r in risk_keywords if r in text_lower)

        return data

    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

