"""
core/agent_tools.py
────────────────────
EisaX Tool Registry — OpenAI-compatible function calling schemas + executors.

The ToolAgent (core/tool_agent.py) uses these to let DeepSeek decide WHAT data
to fetch for each query, rather than fetching everything blindly.

Adding a new tool
─────────────────
1. Add its JSON schema to TOOLS list.
2. Add its executor to _EXECUTORS dict.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS  (OpenAI function-calling format)
# ══════════════════════════════════════════════════════════════════════════════

TOOLS: list[dict] = [

    # ── 1. Live price ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Fetch the current live price and basic technicals (RSI, 52w high/low, % change) "
                "for a stock, crypto, commodity, or index ticker. "
                "Use this FIRST for any stock-related query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol e.g. AAPL, DAMAC.DU, BTC-USD, GC=F, EMAAR.DU",
                    }
                },
                "required": ["ticker"],
            },
        },
    },

    # ── 2. Fundamentals ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": (
                "Fetch fundamental financial data: P/E ratio, EPS, revenue, net margin, "
                "ROE, debt/equity, market cap, dividend yield, sector, beta. "
                "Use for valuation and financial health analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol e.g. AAPL, MSFT, EMAAR.DU",
                    }
                },
                "required": ["ticker"],
            },
        },
    },

    # ── 3. News ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": (
                "Fetch recent news headlines for a company, ticker, or topic. "
                "Use when user asks about news, catalysts, recent events, or market sentiment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name, ticker, or search query e.g. 'DAMAC Properties', 'NVDA AI chips'",
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional ticker for relevance filtering e.g. DAMAC.DU",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max headlines to return (default 5, max 10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },

    # ── 4. User memory ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": (
                "Retrieve the user's stored profile: risk tolerance, investment capital, "
                "preferred markets, watchlist, time horizon. "
                "Call this when personalizing any recommendation or portfolio advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    # ── 5. Portfolio calculator ──────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio_metrics",
            "description": (
                "Calculate portfolio metrics: expected return, volatility, Sharpe ratio, "
                "max drawdown estimate, and correlation summary for a given list of holdings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "description": "List of holdings with ticker and weight",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                                "weight": {"type": "number", "description": "Weight 0-1"},
                            },
                            "required": ["ticker", "weight"],
                        },
                    }
                },
                "required": ["holdings"],
            },
        },
    },

    # ── 6. Technical analysis ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_technical_analysis",
            "description": (
                "Get detailed technical analysis: RSI, MACD, Bollinger Bands, "
                "support/resistance levels, trend direction, volume analysis. "
                "Use for technical trading questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTORS
# ══════════════════════════════════════════════════════════════════════════════

def _exec_get_stock_price(ticker: str) -> dict:
    """Returns price + basic technicals for a ticker."""
    result = {"ticker": ticker, "price": None, "change_pct": None, "rsi": None,
              "week_52_high": None, "week_52_low": None, "error": None}
    try:
        # Try TradingView cache first (fastest, most accurate for MENA)
        from core.price_cache import get as _get_price
        price = _get_price(ticker)
        if price:
            result["price"] = price

        # yfinance for full data
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.fast_info
        if not result["price"]:
            result["price"] = float(getattr(info, "last_price", None) or 0) or None
        result["week_52_high"] = float(getattr(info, "year_high", None) or 0) or None
        result["week_52_low"]  = float(getattr(info, "year_low",  None) or 0) or None

        # RSI via 14-day close
        hist = t.history(period="30d", interval="1d")
        if len(hist) >= 15:
            closes = hist["Close"].tolist()
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [max(d, 0) for d in deltas[-14:]]
            losses = [abs(min(d, 0)) for d in deltas[-14:]]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss == 0:
                result["rsi"] = 100.0
            else:
                rs = avg_gain / avg_loss
                result["rsi"] = round(100 - (100 / (1 + rs)), 1)

        if len(hist) >= 2:
            prev = hist["Close"].iloc[-2]
            curr = hist["Close"].iloc[-1]
            result["change_pct"] = round((curr - prev) / prev * 100, 2)

    except Exception as e:
        result["error"] = str(e)
        logger.debug("[Tool:get_stock_price] %s: %s", ticker, e)

    return result


def _exec_get_fundamentals(ticker: str) -> dict:
    """Returns fundamental data for a ticker."""
    result = {"ticker": ticker, "error": None}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        result.update({
            "company_name":  info.get("longName") or info.get("shortName"),
            "sector":        info.get("sector"),
            "industry":      info.get("industry"),
            "market_cap":    info.get("marketCap"),
            "pe_ratio":      info.get("trailingPE") or info.get("forwardPE"),
            "eps":           info.get("trailingEps"),
            "revenue":       info.get("totalRevenue"),
            "net_margin":    info.get("profitMargins"),
            "roe":           info.get("returnOnEquity"),
            "debt_equity":   info.get("debtToEquity"),
            "dividend_yield":info.get("dividendYield"),
            "beta":          info.get("beta"),
            "analyst_target":info.get("targetMeanPrice"),
            "recommendation":info.get("recommendationKey"),
        })
    except Exception as e:
        result["error"] = str(e)
        logger.debug("[Tool:get_fundamentals] %s: %s", ticker, e)
    return result


def _exec_get_news(query: str, ticker: str = "", limit: int = 5) -> dict:
    """Returns news headlines."""
    limit = min(int(limit), 10)
    articles = []
    try:
        from core.news_aggregator import get_news as _agg_news
        raw = _agg_news(ticker=ticker or query, query=query, limit=limit)
        for a in raw[:limit]:
            articles.append({
                "title":  a.get("title", "")[:120],
                "url":    a.get("url", ""),
                "source": a.get("source", ""),
            })
    except Exception as e:
        logger.debug("[Tool:get_news] %s: %s", query, e)
    return {"query": query, "articles": articles, "count": len(articles)}


def _exec_get_user_profile(user_id: str = "") -> dict:
    """Returns user memory/profile."""
    if not user_id:
        return {"error": "user_id not provided"}
    try:
        from core.memory_manager import get_rich_user_context
        ctx = get_rich_user_context(user_id)
        return ctx or {"note": "No profile data found"}
    except Exception as e:
        return {"error": str(e)}


def _exec_calculate_portfolio_metrics(holdings: list) -> dict:
    """Returns basic portfolio metrics for a list of {ticker, weight} holdings."""
    if not holdings:
        return {"error": "No holdings provided"}
    try:
        # Bucket-based expected return (simple model)
        _bucket_returns = {
            "equity": 0.12, "bonds": 0.05, "gold": 0.07,
            "crypto": 0.18, "commodities": 0.08, "cash": 0.045,
        }
        _bucket_vol = {
            "equity": 0.18, "bonds": 0.07, "gold": 0.15,
            "crypto": 0.65, "commodities": 0.22, "cash": 0.01,
        }
        total_ret = 0.0
        total_vol = 0.0
        for h in holdings:
            w = float(h.get("weight", 0))
            tk = h.get("ticker", "").upper()
            # Classify bucket
            if any(x in tk for x in ["BTC","ETH","SOL","DOGE","IBIT","FETH"]): b = "crypto"
            elif any(x in tk for x in ["GLD","IAU","SGOL","GC=F","GOLD","XAU"]): b = "gold"
            elif any(x in tk for x in ["BND","TLT","AGG","LQD","SHY"]): b = "bonds"
            elif any(x in tk for x in ["GSG","USO","SLV","PDBC","DJP"]): b = "commodities"
            elif tk == "CASH": b = "cash"
            else: b = "equity"
            total_ret += w * _bucket_returns[b]
            total_vol += w * _bucket_vol[b]

        sharpe = (total_ret - 0.045) / total_vol if total_vol > 0 else 0
        return {
            "expected_annual_return_pct": round(total_ret * 100, 2),
            "estimated_volatility_pct":   round(total_vol * 100, 2),
            "sharpe_ratio":               round(sharpe, 2),
            "max_drawdown_estimate_pct":  round(total_vol * 100 * 1.5, 1),
            "holdings_count":             len(holdings),
        }
    except Exception as e:
        return {"error": str(e)}


def _exec_get_technical_analysis(ticker: str) -> dict:
    """Returns technical analysis for a ticker."""
    result = {"ticker": ticker, "error": None}
    try:
        import yfinance as yf
        import numpy as np
        t = yf.Ticker(ticker)
        hist = t.history(period="60d", interval="1d")
        if len(hist) < 20:
            return {"ticker": ticker, "error": "Insufficient data"}

        closes = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()

        # RSI 14
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [max(d, 0) for d in deltas[-14:]]
        losses = [abs(min(d, 0)) for d in deltas[-14:]]
        avg_g  = sum(gains) / 14
        avg_l  = sum(losses) / 14
        rsi    = 100.0 if avg_l == 0 else round(100 - (100 / (1 + avg_g / avg_l)), 1)

        # MACD (12/26/9)
        def _ema(data, n):
            k = 2 / (n + 1)
            e = [data[0]]
            for p in data[1:]: e.append(p * k + e[-1] * (1 - k))
            return e
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_line  = round(ema12[-1] - ema26[-1], 4)
        signal_arr = _ema([e12 - e26 for e12, e26 in zip(ema12[25:], ema26[25:])], 9)
        macd_signal = round(signal_arr[-1], 4) if signal_arr else None

        # 20-day Bollinger Bands
        sma20 = sum(closes[-20:]) / 20
        std20 = (sum((c - sma20)**2 for c in closes[-20:]) / 20) ** 0.5
        bb_upper = round(sma20 + 2 * std20, 4)
        bb_lower = round(sma20 - 2 * std20, 4)

        # Simple support/resistance (recent swing lows/highs)
        recent = closes[-20:]
        support    = round(min(recent), 4)
        resistance = round(max(recent), 4)

        # Trend
        sma50 = sum(closes[-min(50, len(closes)):]) / min(50, len(closes))
        trend = "BULLISH" if closes[-1] > sma50 else "BEARISH"

        result.update({
            "current_price": round(closes[-1], 4),
            "rsi_14":        rsi,
            "rsi_signal":    "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral",
            "macd":          macd_line,
            "macd_signal":   macd_signal,
            "macd_cross":    "Bullish" if macd_line > (macd_signal or 0) else "Bearish",
            "bb_upper":      bb_upper,
            "bb_lower":      bb_lower,
            "bb_position":   "Above upper" if closes[-1] > bb_upper else "Below lower" if closes[-1] < bb_lower else "Inside bands",
            "support":       support,
            "resistance":    resistance,
            "trend_50d":     trend,
            "avg_volume_5d": round(sum(volumes[-5:]) / 5),
        })
    except Exception as e:
        result["error"] = str(e)
        logger.debug("[Tool:get_technical_analysis] %s: %s", ticker, e)
    return result


# ── Dispatcher ────────────────────────────────────────────────────────────────

_EXECUTORS: dict[str, Any] = {
    "get_stock_price":            _exec_get_stock_price,
    "get_fundamentals":           _exec_get_fundamentals,
    "get_news":                   _exec_get_news,
    "get_user_profile":           _exec_get_user_profile,
    "calculate_portfolio_metrics": _exec_calculate_portfolio_metrics,
    "get_technical_analysis":     _exec_get_technical_analysis,
}


def execute_tool(name: str, arguments: dict, user_id: str = "") -> dict:
    """
    Execute a tool by name with given arguments.
    Injects user_id automatically for get_user_profile.
    Returns a dict result (always — never raises).
    """
    fn = _EXECUTORS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        if name == "get_user_profile":
            return fn(user_id=user_id)
        return fn(**arguments)
    except Exception as e:
        logger.error("[Tool:%s] execution error: %s", name, e)
        return {"error": str(e)}
