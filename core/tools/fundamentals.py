"""Tool: get_fundamentals — financial fundamentals for a stock."""
import logging
logger = logging.getLogger(__name__)

def get_fundamentals(ticker: str) -> dict:
    """
    Returns P/E, P/B, EPS, revenue, margins, beta, sector,
    analyst targets, dividend yield for a stock ticker.
    Resolves natural language (Arabic/English) to Yahoo Finance ticker automatically.
    """
    try:
        from core.tools.ticker_resolver import resolve_ticker
        resolved = resolve_ticker(ticker)
        ticker = resolved if resolved else ticker.upper().strip()
    except Exception:
        ticker = ticker.upper().strip()
    result = {
        "ticker": ticker, "company_name": None, "sector": None,
        "pe_ttm": None, "pb": None, "eps_ttm": None,
        "revenue": None, "net_margin": None, "roe": None,
        "beta": None, "dividend_yield": None,
        "analyst_target": None, "analyst_rating": None,
        "market_cap": None, "float_shares": None,
        "high_52w": None, "low_52w": None,
        "error": None
    }
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        info = t.info or {}

        def _f(key, default=None):
            v = info.get(key, default)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        result["company_name"]    = info.get("longName") or info.get("shortName")
        result["sector"]          = info.get("sector") or info.get("industry")
        result["pe_ttm"]          = _f("trailingPE")
        result["pb"]              = _f("priceToBook")
        result["eps_ttm"]         = _f("trailingEps")
        result["revenue"]         = _f("totalRevenue")
        result["net_margin"]      = _f("profitMargins")
        result["roe"]             = _f("returnOnEquity")
        result["beta"]            = _f("beta")
        result["dividend_yield"]  = _f("dividendYield")
        result["analyst_target"]  = _f("targetMeanPrice")
        result["analyst_rating"]  = info.get("recommendationKey")
        result["market_cap"]      = _f("marketCap")
        result["high_52w"]        = _f("fiftyTwoWeekHigh")
        result["low_52w"]         = _f("fiftyTwoWeekLow")

        # Try StockAnalysis for GCC tickers if yfinance data is thin
        _is_gcc = any(ticker.endswith(s) for s in (".DU", ".AE", ".SR", ".KW", ".QA"))
        if _is_gcc and not result["pe_ttm"]:
            try:
                from core.agents.finance import FinancialAgent
                _fa = FinancialAgent()
                _dc = _fa._stockanalysis_uae(ticker) if hasattr(_fa, "_stockanalysis_uae") else {}
                if _dc:
                    def _dc_f(k):
                        v = _dc.get(k)
                        try: return float(str(v).strip()) if v not in (None, "", "N/A") else None
                        except: return None
                    result["pe_ttm"]       = result["pe_ttm"]      or _dc_f("pe_ratio")
                    result["pb"]           = result["pb"]           or _dc_f("pb_ratio")
                    result["dividend_yield"]= result["dividend_yield"] or _dc_f("dividend_yield")
                    result["beta"]         = result["beta"]         or _dc_f("beta")
                    result["company_name"] = result["company_name"] or _dc.get("company_name")
                    result["sector"]       = result["sector"]       or _dc.get("sector")
            except Exception as _gcc_e:
                logger.debug("[Tool:fundamentals] GCC supplement failed: %s", _gcc_e)

        # Round floats
        for k in ("pe_ttm","pb","eps_ttm","net_margin","roe","beta","dividend_yield"):
            if result[k] is not None:
                result[k] = round(result[k], 4)

    except Exception as e:
        result["error"] = str(e)
        logger.warning("[Tool:get_fundamentals] %s: %s", ticker, e)
    return result
