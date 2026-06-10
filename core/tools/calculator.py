"""Tool: calculate_portfolio — portfolio metrics calculator."""
import logging
logger = logging.getLogger(__name__)

# Expected return estimates by asset bucket
_BUCKET_RETURNS = {
    "us_equity":  0.12,
    "gcc_equity": 0.13,
    "bonds":      0.05,
    "gold":       0.07,
    "commodities":0.08,
    "crypto":     0.18,
    "cash":       0.045,
}
_BUCKET_VOL = {
    "us_equity":  0.18,
    "gcc_equity": 0.20,
    "bonds":      0.08,
    "gold":       0.15,
    "commodities":0.22,
    "crypto":     0.70,
    "cash":       0.01,
}
_TICKER_BUCKET = {
    # US equity
    "AAPL":"us_equity","MSFT":"us_equity","NVDA":"us_equity","AMZN":"us_equity",
    "GOOGL":"us_equity","META":"us_equity","JPM":"us_equity","V":"us_equity",
    "XOM":"us_equity","JNJ":"us_equity","SPY":"us_equity","QQQ":"us_equity",
    # GCC
    "2222.SR":"gcc_equity","1120.SR":"gcc_equity","EMAAR.DU":"gcc_equity",
    "FAB.AE":"gcc_equity","ADNOCDIST.AE":"gcc_equity","ADNOCGAS.AE":"gcc_equity",
    "TAQA.AE":"gcc_equity","7010.SR":"gcc_equity","KFH.KW":"gcc_equity",
    # Bonds
    "TLT":"bonds","BND":"bonds","AGG":"bonds","SHY":"bonds","IEF":"bonds",
    # Gold
    "GLD":"gold","IAU":"gold","SGOL":"gold","XAUUSD":"gold","GC=F":"gold",
    # Commodities
    "GSG":"commodities","DJP":"commodities","SLV":"commodities",
    "CL=F":"commodities","USO":"commodities",
    # Crypto
    "BTC-USD":"crypto","ETH-USD":"crypto","BNB-USD":"crypto",
    "SOL-USD":"crypto","IBIT":"crypto","FETH":"crypto",
}

def _classify(ticker: str) -> str:
    t = ticker.upper()
    if t in _TICKER_BUCKET:
        return _TICKER_BUCKET[t]
    if t.endswith((".DU",".AE",".SR",".KW",".QA",".CA")):
        return "gcc_equity"
    if "USD" in t or t in ("BTC","ETH","SOL","BNB","ADA","XRP","DOGE"):
        return "crypto"
    return "us_equity"  # default

def calculate_portfolio(holdings: list, capital: float = None) -> dict:
    """
    Calculate portfolio metrics from a list of {ticker, weight} dicts.
    Returns expected return, volatility, Sharpe ratio, drawdown estimate.
    """
    if not holdings:
        return {"error": "No holdings provided"}

    # Normalize weights
    total_w = sum(float(h.get("weight", 0)) for h in holdings)
    if total_w <= 0:
        return {"error": "Invalid weights — all zero or negative"}
    normalized = [
        {"ticker": h["ticker"], "weight": float(h.get("weight", 0)) / total_w}
        for h in holdings
    ]

    # Compute weighted return and volatility
    w_return = 0.0
    w_vol    = 0.0
    buckets  = {}
    for h in normalized:
        bucket = _classify(h["ticker"])
        r = _BUCKET_RETURNS.get(bucket, 0.10)
        v = _BUCKET_VOL.get(bucket, 0.18)
        w_return += h["weight"] * r
        w_vol    += h["weight"] * v
        buckets[bucket] = buckets.get(bucket, 0) + h["weight"]

    risk_free = 0.045  # current approximate RF rate
    sharpe    = (w_return - risk_free) / w_vol if w_vol > 0 else 0
    max_dd    = w_vol * 1.5  # rough estimate: 1.5x annualized vol

    result = {
        "holdings":        normalized,
        "expected_return": round(w_return * 100, 2),   # %
        "annual_volatility":round(w_vol * 100, 2),     # %
        "sharpe_ratio":    round(sharpe, 2),
        "max_drawdown_est":round(max_dd * 100, 1),     # %
        "bucket_breakdown": {k: round(v*100,1) for k,v in buckets.items()},
        "capital":         capital,
        "note": "Returns are fundamental estimates; true portfolio volatility depends on correlation regime."
    }
    if capital:
        result["position_sizes"] = {
            h["ticker"]: round(h["weight"] * capital, 2)
            for h in normalized
        }
    return result
