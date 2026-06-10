#!/usr/bin/env python3
"""
EisaX Fundamental Engine
1. Fundamental Analysis (FMP)
2. News Intelligence (LLM-powered)
3. Relative Strength vs Sector vs Market
"""
import os
import requests
import json
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

FMP_KEY = os.getenv("FMP_API_KEY", "")
AV_KEY  = os.getenv("ALPHA_VANTAGE_KEY", "")
logger  = logging.getLogger(__name__)

SECTOR_ETFS = {
    "Technology":            "XLK",
    "Healthcare":            "XLV",
    "Financials":            "XLF",
    "Energy":                "XLE",
    "Consumer Discretionary":"XLY",
    "Consumer Staples":      "XLP",
    "Industrials":           "XLI",
    "Materials":             "XLB",
    "Real Estate":           "XLRE",
    "Utilities":             "XLU",
    "Communication":         "XLC",
}

# ── 1. FUNDAMENTALS ───────────────────────────────────────────────────────────
def get_fundamentals(ticker: str) -> Dict:
    """P/E, EPS, Revenue, Margins, Debt - from yfinance (primary) + Finnhub (secondary)"""
    import time as _t_cache
    # ── Cache check: same ticker within 10 min → return cached result ──
    try:
        from core.agents.finance import _fundamentals_cache
        _cached = _fundamentals_cache.get(f"fund_{ticker.upper()}")
        if _cached:
            logger.info(f"[Fundamentals] Cache HIT for {ticker}")
            return _cached
    except Exception:
        pass
    result = {}
    ticker = ticker.upper()

    # ── PRIMARY: yfinance (free, no rate limits) ──────────────────────────
    try:
        import yfinance as yf, time as _t
        info = {}
        for _attempt in range(3):
            try:
                info = yf.Ticker(ticker).info or {}
                if info.get('symbol') or info.get('shortName') or info.get('regularMarketPrice'):
                    break  # valid response
                _t.sleep(1.5 * (_attempt + 1))  # wait for crumb to settle
            except Exception as _e401:
                if _attempt < 2:
                    _t.sleep(2 * (_attempt + 1))
                else:
                    raise

        # ── Helper: None-safe rounding (returns None if data missing, not 0) ──
        def _r(val, ndigits=2):
            """Round value, return None instead of 0 for missing data."""
            return round(val, ndigits) if val else None

        def _rpct(val, ndigits=1):
            """Round percentage (×100), return None instead of 0."""
            return round(val * 100, ndigits) if val else None

        result["pe_ratio"]         = _r(info.get("trailingPE"), 1)
        result["ps_ratio"]         = _r(info.get("priceToSalesTrailing12Months"), 1)
        result["pb_ratio"]         = _r(info.get("priceToBook"), 2)
        result["peg_ratio"]        = _r(info.get("pegRatio"), 2)
        result["gross_margin"]     = _rpct(info.get("grossMargins"))
        result["net_margin"]       = _rpct(info.get("profitMargins"))
        result["operating_margin"] = _rpct(info.get("operatingMargins"))
        result["current_ratio"]    = _r(info.get("currentRatio"), 2)
        result["debt_equity"]      = round((info.get("debtToEquity") or 0) / 100, 2) if info.get("debtToEquity") else None
        result["ev_ebitda"]        = _r(info.get("enterpriseToEbitda"), 2)
        result["roe"]              = _rpct(info.get("returnOnEquity"), 2)
        result["roa"]              = _rpct(info.get("returnOnAssets"), 2)
        result["roic"]             = _rpct(info.get("returnOnAssets"), 2)
        result["market_cap"]       = info.get("marketCap") or 0
        result["total_debt"]       = info.get("totalDebt") or 0
        result["cash"]             = info.get("totalCash") or 0
        result["net_debt"]         = (result["total_debt"] or 0) - (result["cash"] or 0)
        result["revenue"]          = info.get("totalRevenue") or 0
        result["eps"]              = _r(info.get("trailingEps"), 2)
        result["revenue_growth"]   = _rpct(info.get("revenueGrowth"))
        result["eps_growth"]       = _rpct(info.get("earningsGrowth"))
        # Sector — never leave as "Unknown": use ticker-based classifier as fallback
        _raw_sector = info.get("sector") or ""
        result["sector"]  = _raw_sector if _raw_sector and _raw_sector != "Unknown" else ""
        result["industry"] = info.get("industry") or ""
        # Beta — None if missing; NEVER default to 1.0 (that overstates risk for defensive stocks)
        _beta_val = info.get("beta")
        result["beta"] = round(float(_beta_val), 2) if (_beta_val is not None and float(_beta_val) != 0) else None
        result["company_name"]     = info.get("longName", ticker)
        result["employees"]        = info.get("fullTimeEmployees", 0)
        result["description"]      = (info.get("longBusinessSummary") or "")[:300]
        result["net_income"]       = info.get("netIncomeToCommon") or 0

        # Earnings
        result["last_eps_actual"]   = round(info.get("trailingEps") or 0, 2)
        result["last_eps_estimate"] = round(info.get("epsForward") or 0, 2)
        result["forward_pe"]         = round(float(info.get("forwardPE") or 0), 1)

        # Analyst target — stored here so it survives concurrent yfinance rate-limits
        _at = info.get("targetMeanPrice") or info.get("targetMedianPrice")
        if _at:
            result["analyst_target"] = round(float(_at), 4)
        # Analyst consensus + count
        _ac = info.get("recommendationKey", "")
        if _ac:
            result["analyst_consensus"] = _ac.replace("_", " ").title()
        _an = info.get("numberOfAnalystOpinions")
        if _an:
            result["analyst_count"] = int(_an)

        logger.info(f"[Fundamentals] yfinance OK for {ticker}: PE={result['pe_ratio']}, margin={result['net_margin']}%")
    except Exception as e:
        logger.warning(f"yfinance fundamentals failed: {e}")

    # ── SECONDARY: Finnhub (fills gaps if yfinance missing) ───────────────
    try:
        import os as _os
        fh_key = _os.getenv("FINNHUB_API_KEY", "")
        if fh_key and not result.get("pe_ratio"):
            r = requests.get(
                f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={fh_key}",
                timeout=8
            ).json()
            m = r.get("metric", {})
            if not result.get("pe_ratio"):
                result["pe_ratio"] = round(m.get("peBasicExclExtraTTM") or 0, 1)
            if not result.get("52w_high"):
                result["52w_high"] = m.get("52WeekHigh")
    except Exception as e:
        logger.warning(f"Finnhub fallback failed: {e}")

    # Earnings date from yfinance calendar
    try:
        import yfinance as yf
        from datetime import datetime as _dt
        import pandas as pd
        yt2 = yf.Ticker(ticker)
        today = _dt.now().date()
        next_earnings = None

        # Method 1: calendar dict (new yfinance format)
        try:
            cal = yt2.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earningsDate")
                if ed:
                    if isinstance(ed, (list, tuple)):
                        for d in ed:
                            try:
                                dt = pd.Timestamp(d).date()
                                if dt > today:
                                    next_earnings = str(dt)
                                    break
                            except: pass
                    else:
                        dt = pd.Timestamp(ed).date()
                        if dt > today:
                            next_earnings = str(dt)
        except: pass

        # Method 2: calendar DataFrame (old yfinance format)
        if not next_earnings:
            try:
                cal = yt2.calendar
                if hasattr(cal, "columns"):
                    row = cal.get("Earnings Date") or cal.iloc[0] if not cal.empty else None
                    if row is not None:
                        for val in (row if hasattr(row, "__iter__") else [row]):
                            try:
                                dt = pd.Timestamp(val).date()
                                if dt > today:
                                    next_earnings = str(dt)
                                    break
                            except: pass
            except: pass

        # Method 3: info earningsDate
        if not next_earnings:
            try:
                ed = yt2.info.get("earningsDate") or yt2.info.get("nextEarningsDate")
                if ed:
                    dt = pd.Timestamp(ed if not isinstance(ed, list) else ed[0]).date()
                    if dt > today:
                        next_earnings = str(dt)
            except: pass

        if next_earnings:
            result["last_earnings_date"] = next_earnings
            logger.debug(f"[Fundamentals] Next earnings: {next_earnings}")
        else:
            result["last_earnings_date"] = "TBD"
    except Exception as _e:
        logger.error(f"[Fundamentals] earnings date failed: {_e}")
        result["last_earnings_date"] = "TBD"

    # ── Fundamental Score 0-100 ───────────────────────────────────────────
    # Use (x or 0) pattern to safely handle None values from _r()/_rpct()
    score = 50
    _rg = result.get("revenue_growth") or 0
    _eg = result.get("eps_growth") or 0
    _nm = result.get("net_margin") or 0
    _roe = result.get("roe") or 0
    _de = result.get("debt_equity") if result.get("debt_equity") is not None else 99
    _pe = result.get("pe_ratio") or 0
    if _rg > 15:   score += 15
    elif _rg > 5:  score += 7
    if _eg > 20:   score += 15
    elif _eg > 5:  score += 7
    if _nm > 20:   score += 10
    if _roe > 15:  score += 10
    if _de < 0.5:  score += 5
    if _pe > 60:   score -= 10
    result["fundamental_score"] = min(100, max(0, score))

    # ── Sector classifier fallback (if yfinance returned empty sector) ───────
    if not result.get("sector"):
        result["sector"] = _classify_sector(ticker)

    # ── Excel lookup: override Unknown sector/industry/name for regional stocks ──
    try:
        from core.excel_stock_lookup import enrich_fund_dict
        result = enrich_fund_dict(ticker, result)
    except Exception:
        pass

    # ── Sanity: never output fundamental_score=0 (means data failed, not score=0) ──
    if result.get("fundamental_score", 0) <= 0:
        result.pop("fundamental_score", None)  # will show N/A, not corrupt quality calc

    # ── Cache save ──
    try:
        from core.agents.finance import _fundamentals_cache
        _fundamentals_cache.set(f"fund_{ticker}", result)
    except Exception:
        pass
    return result


def _classify_sector(ticker: str) -> str:
    """Ticker-based sector classification when yfinance returns empty sector."""
    t = ticker.upper().split('.')[0]
    # Crypto
    if t in ('BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT') \
            or ticker.upper().endswith('-USD'):
        return "Cryptocurrency"
    # Big Tech
    if t in ('MSFT', 'AAPL', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM',
             'TSM', 'ASML', 'CRM', 'ADBE', 'ORCL', 'NFLX', 'UBER'):
        return "Technology"
    if t in ('AMZN',): return "Consumer Cyclical"
    if t in ('TSLA',): return "Consumer Cyclical"
    # Energy
    if t in ('XOM', 'CVX', 'BP', 'SHEL', 'TTE') or '2222' in t:
        return "Energy"
    if any(x in t for x in ('ADNOC', 'TAQA', 'DANA', 'PETRO', 'OIL', 'GAS', 'ENRG')):
        return "Energy"
    # Financials
    if t in ('JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BRK') or \
            any(x in t for x in ('BANK', 'FAB', 'ENBD', 'NBK', 'QNB', 'DIB', 'CBK')):
        return "Financial Services"
    # Real Estate
    if any(x in t for x in ('EMAAR', 'EMAR', 'ALDAR', 'DAMAC', 'DEWA')):
        return "Real Estate"
    # Healthcare
    if t in ('JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'LLY'):
        return "Healthcare"
    # Regional suffix-based fallback
    if ticker.upper().endswith(('.SR',)):
        return "Energy"   # most Saudi heavyweights are energy-adjacent
    if ticker.upper().endswith(('.CA',)):
        return "Industrials"  # EGX default
    return ""  # unknown — don't invent a sector


def format_fundamentals(ticker: str, data: Dict) -> str:
    """تنسيق الـ fundamentals للعرض"""
    def fmt_num(n, billions=True):
        if not n: return "N/A"
        if billions: return f"${n/1e9:.1f}B"
        return f"{n:.1f}%"

    score = data.get("fundamental_score", 50)
    score_emoji = "🟢" if score >= 70 else "🟡" if score >= 45 else "🔴"

    lines = [
        f"## 📊 Fundamentals: {ticker}",
        f"**Fundamental Score:** {score_emoji} {score}/100\n",
        f"### Valuation",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| P/E Ratio | {data.get('pe_ratio', 'N/A')}x |",
        f"| P/S Ratio | {data.get('ps_ratio', 'N/A')}x |",
        f"| EV/EBITDA | {data.get('ev_ebitda', 'N/A')}x |",
        f"| Beta | {data.get('beta', 'N/A')} |",
        f"\n### Growth (YoY)",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Revenue Growth | {data.get('revenue_growth', 'N/A')}% |",
        f"| EPS Growth | {data.get('eps_growth', 'N/A')}% |",
        f"| EPS (TTM) | ${data.get('eps', 'N/A')} |",
        f"\n### Profitability",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Gross Margin | {data.get('gross_margin', 'N/A')}% |",
        f"| Net Margin | {data.get('net_margin', 'N/A')}% |",
        f"| ROE | {data.get('roe', 'N/A')}% |",
        f"\n### Financial Health",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Debt | {fmt_num(data.get('total_debt'))} |",
        f"| Cash | {fmt_num(data.get('cash'))} |",
        f"| Debt/Equity | {data.get('debt_equity', 'N/A')} |",
        f"| Current Ratio | {data.get('current_ratio', 'N/A')} |",
    ]

    return "\n".join(lines)


# ── 2. NEWS INTELLIGENCE ──────────────────────────────────────────────────────
def get_smart_news(ticker: str) -> Dict:
    """جيب الأخبار وحللها بـ LLM"""
    headlines = []

    # Alpha Vantage News
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={AV_KEY}&limit=8&sort=LATEST"
        r = requests.get(url, timeout=10).json()
        feed = r.get("feed", [])
        for item in feed[:8]:
            score = 0
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    score = float(ts.get("ticker_sentiment_score", 0))
            headlines.append({
                "title":   item.get("title", ""),
                "source":  item.get("source", ""),
                "time":    item.get("time_published", "")[:8],
                "score":   score,
                "summary": item.get("summary", "")[:200]
            })
    except Exception as e:
        logger.warning(f"AV news failed: {e}")

    if not headlines:
        return {"headlines": [], "intelligence": "No recent news available", "overall_sentiment": "neutral"}

    # Build LLM analysis prompt
    news_text = "\n".join([
        f"- [{h['time']}] {h['title']} (sentiment: {h['score']:+.2f})"
        for h in headlines
    ])

    # Use DeepSeek to analyze
    intelligence = _llm_analyze_news(ticker, news_text)

    avg_score = sum(h["score"] for h in headlines) / len(headlines) if headlines else 0
    overall = "Bullish 📈" if avg_score > 0.15 else "Bearish 📉" if avg_score < -0.15 else "Neutral ➡️"

    return {
        "headlines":          headlines[:5],
        "intelligence":       intelligence,
        "overall_sentiment":  overall,
        "avg_score":          round(avg_score, 3)
    }


def _llm_analyze_news(ticker: str, news_text: str) -> str:
    """DeepSeek يقرأ الأخبار ويحللها"""
    try:
        import httpx, asyncio

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return "News analysis unavailable (no DeepSeek key)"

        prompt = f"""You are EisaX, a CIO-level analyst. Analyze these recent news headlines for {ticker}:

{news_text}

In 3-4 sentences, provide:
1. Key catalyst or risk identified
2. Short-term market impact (bullish/bearish/neutral)
3. What an investor should watch next

Be specific and institutional-grade. No fluff."""

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.3
            },
            timeout=15
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"LLM news analysis failed: {e}")
        return "News analyzed - check headlines above for details"


def format_news_intelligence(ticker: str, data: Dict) -> str:
    """تنسيق الأخبار للعرض"""
    lines = [
        f"## 📰 News Intelligence: {ticker}",
        f"**Overall Sentiment:** {data.get('overall_sentiment', 'N/A')} (Score: {data.get('avg_score', 0):+.3f})\n",
        f"**🤖 EisaX Analysis:**",
        f"{data.get('intelligence', 'N/A')}\n",
        f"**Recent Headlines:**"
    ]
    for h in data.get("headlines", [])[:4]:
        emoji = "📈" if h["score"] > 0.15 else "📉" if h["score"] < -0.15 else "➡️"
        lines.append(f"- {emoji} [{h['time']}] {h['title']} _{h['source']}_")

    return "\n".join(lines)


# ── 3. RELATIVE STRENGTH ──────────────────────────────────────────────────────
def get_relative_strength(ticker: str, sector: str = None) -> Dict:
    """مقارنة أداء السهم بالقطاع والسوق"""
    from market_data import get_realtime_quote

    result = {"ticker": ticker}

    # Get sector ETF
    sector_etf = SECTOR_ETFS.get(sector, "SPY")
    benchmarks = {"S&P 500": "SPY", "Sector": sector_etf, "Nasdaq": "QQQ"}

    for name, etf in benchmarks.items():
        try:
            q = get_realtime_quote(etf)
            result[f"{name}_change"] = q.get("change_pct", 0)
        except Exception as _e:
            result[f"{name}_change"] = 0

    # Stock change
    try:
        q = get_realtime_quote(ticker)
        result["stock_change"] = q.get("change_pct", 0)
        result["stock_price"]  = q.get("price", 0)
    except Exception as _e:
        result["stock_change"] = 0

    # RS Score vs each benchmark
    stock_chg = result["stock_change"]
    result["rs_vs_sp500"]  = round(stock_chg - result["S&P 500_change"], 2)
    result["rs_vs_sector"] = round(stock_chg - result["Sector_change"], 2)
    result["rs_vs_nasdaq"] = round(stock_chg - result["Nasdaq_change"], 2)

    # Overall RS rating
    rs_scores = [result["rs_vs_sp500"], result["rs_vs_sector"], result["rs_vs_nasdaq"]]
    avg_rs = sum(rs_scores) / 3
    result["rs_rating"] = "Strong Outperformer 🚀" if avg_rs > 1 else \
                          "Outperformer ✅" if avg_rs > 0 else \
                          "Underperformer ⚠️" if avg_rs > -1 else \
                          "Weak Underperformer 🔴"
    result["avg_rs"] = round(avg_rs, 2)
    result["sector_etf"] = sector_etf

    return result


def format_relative_strength(ticker: str, data: Dict) -> str:
    """تنسيق الـ RS للعرض"""
    def rs_arrow(val):
        return f"{'▲' if val > 0 else '▼'} {abs(val):.2f}%"

    lines = [
        f"## 📈 Relative Strength: {ticker}",
        f"**RS Rating:** {data.get('rs_rating', 'N/A')}",
        f"**Today's Performance:**\n",
        f"| Benchmark | Change | {ticker} vs |",
        f"|-----------|--------|------------|",
        f"| S&P 500 (SPY) | {data.get('S&P 500_change', 0):+.2f}% | {rs_arrow(data.get('rs_vs_sp500', 0))} |",
        f"| Sector ({data.get('sector_etf', 'SPY')}) | {data.get('Sector_change', 0):+.2f}% | {rs_arrow(data.get('rs_vs_sector', 0))} |",
        f"| Nasdaq (QQQ) | {data.get('Nasdaq_change', 0):+.2f}% | {rs_arrow(data.get('rs_vs_nasdaq', 0))} |",
        f"\n**{ticker} Change Today:** {data.get('stock_change', 0):+.2f}%",
    ]
    return "\n".join(lines)


# ── MASTER: Full Analysis ─────────────────────────────────────────────────────
def get_full_analysis(ticker: str) -> Dict:
    """تحليل شامل - fundamentals + news + relative strength"""
    ticker = ticker.upper()

    fundamentals = get_fundamentals(ticker)
    news = get_smart_news(ticker)
    rs   = get_relative_strength(ticker, fundamentals.get("sector"))

    # Combined Score
    f_score = fundamentals.get("fundamental_score", 50)
    n_score = 70 if "Bullish" in news.get("overall_sentiment","") else \
              30 if "Bearish" in news.get("overall_sentiment","") else 50
    rs_score = 70 if "Outperformer" in rs.get("rs_rating","") else \
               30 if "Underperformer" in rs.get("rs_rating","") else 50

    combined = int((f_score * 0.5) + (n_score * 0.3) + (rs_score * 0.2))
    verdict = "STRONG BUY 🟢" if combined >= 75 else \
              "BUY ✅" if combined >= 60 else \
              "HOLD ⚠️" if combined >= 45 else \
              "REDUCE 🔴"

    return {
        "ticker":       ticker,
        "fundamentals": fundamentals,
        "news":         news,
        "rs":           rs,
        "combined_score": combined,
        "verdict":      verdict,
        "sector":       fundamentals.get("sector", "Unknown")
    }


def format_full_analysis(ticker: str, data: Dict) -> str:
    """تنسيق التحليل الكامل"""
    score = data["combined_score"]
    emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"

    header = (
        f"# EisaX Full Analysis: {ticker}\n"
        f"**Sector:** {data['sector']}\n"
        f"**Combined Score:** {emoji} {score}/100 | **Verdict:** {data['verdict']}\n\n"
        f"---\n\n"
    )

    f_text  = format_fundamentals(ticker, data["fundamentals"])
    n_text  = format_news_intelligence(ticker, data["news"])
    rs_text = format_relative_strength(ticker, data["rs"])

    return header + f_text + "\n\n---\n\n" + n_text + "\n\n---\n\n" + rs_text


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    logger.debug(f"Analyzing {ticker}...")
    data = get_full_analysis(ticker)
    logger.debug(format_full_analysis(ticker, data))