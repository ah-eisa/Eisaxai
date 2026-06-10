"""Tool: screen_market — screen stocks by criteria."""
import logging
logger = logging.getLogger(__name__)

# Curated universe per market (tickers with reliable data)
_UNIVERSE = {
    "UAE":    ["EMAAR.DU", "ADNOCDIST.AE", "FAB.AE", "ADNOCGAS.AE", "TAQA.AE",
               "EAND.AE", "IHC.AE", "DIB.DU", "ENBD.DU", "ALDAR.AE"],
    "Saudi":  ["2222.SR", "1120.SR", "2010.SR", "7010.SR", "1010.SR",
               "2380.SR", "4030.SR", "1150.SR", "2330.SR", "4240.SR"],
    "US":     ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
               "JPM", "V", "JNJ", "XOM", "BRK-B", "UNH"],
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
               "ADA-USD", "DOGE-USD", "MATIC-USD"],
    "Global": ["AAPL", "MSFT", "NVDA", "EMAAR.DU", "2222.SR", "FAB.AE",
               "BTC-USD", "ETH-USD", "GLD", "TLT"],
}

def screen_market(
    market: str,
    sector: str = None,
    max_pe: float = None,
    min_div: float = None,
    criteria: str = None,
) -> dict:
    """
    Screen stocks in a market universe by given criteria.
    Returns matching tickers with key metrics.
    """
    import yfinance as yf

    market_key = market.strip().capitalize()
    if market_key not in _UNIVERSE:
        # Try partial match
        for k in _UNIVERSE:
            if market.lower() in k.lower():
                market_key = k
                break
        else:
            market_key = "Global"

    universe = _UNIVERSE[market_key]
    results  = []
    errors   = []

    for ticker in universe:
        try:
            info = yf.Ticker(ticker).info or {}
            # Sector filter
            if sector:
                t_sector = (info.get("sector") or info.get("industry") or "").lower()
                if sector.lower() not in t_sector:
                    continue
            # P/E filter
            pe = info.get("trailingPE")
            if max_pe and pe and pe > max_pe:
                continue
            # Dividend filter
            div = (info.get("dividendYield") or 0) * 100
            if min_div and div < min_div:
                continue

            price = info.get("currentPrice") or info.get("previousClose") or 0
            results.append({
                "ticker":   ticker,
                "name":     info.get("shortName") or info.get("longName") or ticker,
                "price":    round(price, 2) if price else None,
                "pe":       round(pe, 1) if pe else None,
                "div_yield":round(div, 2) if div else None,
                "beta":     round(info.get("beta") or 1.0, 2),
                "sector":   info.get("sector") or info.get("industry"),
                "rec":      info.get("recommendationKey"),
            })
        except Exception as e:
            errors.append(f"{ticker}: {e}")

    return {
        "market":   market_key,
        "criteria": {"sector": sector, "max_pe": max_pe, "min_div": min_div, "free_text": criteria},
        "matches":  results,
        "count":    len(results),
        "errors":   errors[:3] if errors else [],
    }
