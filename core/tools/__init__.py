"""
EisaX Agent Tools
─────────────────
Discrete, callable tools the agent can invoke.
Each tool:
  - has a clear name + description (for LLM function-calling schema)
  - takes typed inputs
  - returns structured output
  - handles its own errors gracefully

Registry: TOOLS_REGISTRY maps tool name → callable
Schema:   TOOLS_SCHEMA is the JSON array passed to LLM as `tools`
"""

from .price       import get_price
from .fundamentals import get_fundamentals
from .news        import search_news
from .screen      import screen_market
from .calculator  import calculate_portfolio

# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOLS_REGISTRY = {
    "get_price":           get_price,
    "get_fundamentals":    get_fundamentals,
    "search_news":         search_news,
    "screen_market":       screen_market,
    "calculate_portfolio": calculate_portfolio,
}

# ── JSON Schema for LLM function calling ──────────────────────────────────────
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get the current live price, change%, and basic market data for any ticker (stocks, crypto, commodities, indices). Use this before any price-related question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol e.g. AAPL, BTC-USD, EMAAR.DU, GC=F"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get fundamental financial data for a stock: P/E, P/B, EPS, revenue, margins, beta, 52w range, sector, analyst targets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker e.g. NVDA, EMAAR.DU, 2222.SR"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search for recent news headlines about a company, ticker, or topic. Returns top relevant articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company name, ticker, or topic to search"},
                    "limit": {"type": "integer", "description": "Max articles to return (default 5)", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screen_market",
            "description": "Screen stocks by criteria: market (UAE/Saudi/US/Global), sector, min/max P/E, min dividend yield, market cap range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market":      {"type": "string", "description": "UAE | Saudi | US | Global | Crypto"},
                    "sector":      {"type": "string", "description": "Optional sector filter e.g. Technology, Energy, Real Estate"},
                    "max_pe":      {"type": "number", "description": "Max P/E ratio"},
                    "min_div":     {"type": "number", "description": "Min dividend yield %"},
                    "criteria":    {"type": "string", "description": "Free-text criteria e.g. 'oversold RSI < 30'"}
                },
                "required": ["market"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio",
            "description": "Calculate portfolio metrics: expected return, volatility, Sharpe ratio, max drawdown estimate, and allocation weights for a given set of assets.",
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
                                "weight": {"type": "number", "description": "Weight as decimal 0-1"}
                            }
                        }
                    },
                    "capital": {"type": "number", "description": "Total capital in USD (optional)"}
                },
                "required": ["holdings"]
            }
        }
    },
]
