"""Tool: get_price — live price for any ticker."""
import logging
logger = logging.getLogger(__name__)

def get_price(ticker: str) -> dict:
    """
    Returns current price, change%, volume, 52w range for a ticker.
    Resolves natural language (Arabic/English) to Yahoo Finance ticker automatically.
    """
    # Resolve natural language → Yahoo ticker
    try:
        from core.tools.ticker_resolver import resolve_ticker, get_asset_type
        resolved = resolve_ticker(ticker)
        if resolved:
            if resolved != ticker.upper().strip():
                logger.info("[get_price] Resolved '%s' → '%s'", ticker, resolved)
            ticker = resolved
        else:
            ticker = ticker.upper().strip()
    except Exception:
        ticker = ticker.upper().strip()
    result = {"ticker": ticker, "price": None, "change_pct": None,
              "volume": None, "high_52w": None, "low_52w": None,
              "currency": None, "source": None, "error": None}
    try:
        # 1. yfinance primary (reliable, fast)
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
        if price and price > 0:
            result["price"]      = round(price, 4)
            result["change_pct"] = round(getattr(info, "last_price_change_pct", 0) or 0, 2)
            result["high_52w"]   = getattr(info, "fifty_two_week_high", None)
            result["low_52w"]    = getattr(info, "fifty_two_week_low", None)
            result["currency"]   = getattr(info, "currency", None)
            result["volume"]     = getattr(info, "three_month_average_volume", None)
            result["source"]     = "yfinance"
            return result

        result["error"] = f"Price unavailable for {ticker}"
    except Exception as e:
        result["error"] = str(e)
        logger.warning("[Tool:get_price] %s: %s", ticker, e)
    return result
