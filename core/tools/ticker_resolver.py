"""
core/tools/ticker_resolver.py
──────────────────────────────
Resolves natural language (Arabic/English) to Yahoo Finance tickers.

Usage:
    from core.tools.ticker_resolver import resolve_ticker
    resolve_ticker("gold")       → "GC=F"
    resolve_ticker("الذهب")      → "GC=F"
    resolve_ticker("bitcoin")    → "BTC-USD"
    resolve_ticker("ارامكو")     → "2222.SR"   (falls through to GCC map)
    resolve_ticker("AAPL")       → "AAPL"      (already a valid ticker)
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)

# ── Commodities ───────────────────────────────────────────────────────────────
_COMMODITIES: dict[str, str] = {
    # Gold
    "gold": "GC=F", "الذهب": "GC=F", "ذهب": "GC=F", "xauusd": "GC=F",
    "xau": "GC=F", "gold futures": "GC=F",

    # Silver
    "silver": "SI=F", "الفضة": "SI=F", "فضة": "SI=F", "xagusd": "SI=F",

    # Oil - Crude
    "oil": "CL=F", "crude": "CL=F", "crude oil": "CL=F", "النفط": "CL=F",
    "نفط": "CL=F", "خام": "CL=F", "النفط الخام": "CL=F", "wti": "CL=F",
    "brent": "BZ=F", "برنت": "BZ=F",

    # Natural Gas
    "natural gas": "NG=F", "gas": "NG=F", "الغاز": "NG=F", "غاز": "NG=F",
    "lng": "NG=F",

    # Copper
    "copper": "HG=F", "نحاس": "HG=F",

    # Wheat / Corn / Agriculture
    "wheat": "ZW=F", "قمح": "ZW=F",
    "corn": "ZC=F", "ذرة": "ZC=F",
    "soybeans": "ZS=F", "فول الصويا": "ZS=F",
    "sugar": "SB=F", "سكر": "SB=F",
    "coffee": "KC=F", "قهوة": "KC=F",

    # Platinum / Palladium
    "platinum": "PL=F", "بلاتين": "PL=F",
    "palladium": "PA=F", "بلاديوم": "PA=F",
}

# ── Crypto ────────────────────────────────────────────────────────────────────
_CRYPTO: dict[str, str] = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "بيتكوين": "BTC-USD",
    "bitcoin usd": "BTC-USD",

    "ethereum": "ETH-USD", "eth": "ETH-USD", "إيثيريوم": "ETH-USD",
    "ايثيريوم": "ETH-USD",

    "bnb": "BNB-USD", "binance coin": "BNB-USD", "بينانس": "BNB-USD",

    "solana": "SOL-USD", "sol": "SOL-USD", "سولانا": "SOL-USD",

    "xrp": "XRP-USD", "ripple": "XRP-USD", "ريبل": "XRP-USD",

    "cardano": "ADA-USD", "ada": "ADA-USD", "كاردانو": "ADA-USD",

    "dogecoin": "DOGE-USD", "doge": "DOGE-USD", "دوج": "DOGE-USD",

    "avalanche": "AVAX-USD", "avax": "AVAX-USD",

    "polkadot": "DOT-USD", "dot": "DOT-USD",

    "chainlink": "LINK-USD", "link": "LINK-USD",

    "litecoin": "LTC-USD", "ltc": "LTC-USD", "لايتكوين": "LTC-USD",

    "shiba": "SHIB-USD", "shib": "SHIB-USD",

    "toncoin": "TON-USD", "ton": "TON-USD",

    "bitcoin etf": "IBIT", "ibit": "IBIT",
}

# ── Indices ───────────────────────────────────────────────────────────────────
_INDICES: dict[str, str] = {
    # US
    "sp500": "^GSPC", "s&p500": "^GSPC", "s&p 500": "^GSPC",
    "s&p": "^GSPC", "spx": "^GSPC", "اس اند بي": "^GSPC",

    "nasdaq": "^IXIC", "ناسداك": "^IXIC", "nasdaq composite": "^IXIC",
    "nasdaq 100": "^NDX", "ndx": "^NDX", "qqq": "QQQ",

    "dow jones": "^DJI", "dow": "^DJI", "djia": "^DJI",
    "داو جونز": "^DJI", "داو": "^DJI",

    "russell 2000": "^RUT", "russell": "^RUT",

    "vix": "^VIX", "fear index": "^VIX", "مؤشر الخوف": "^VIX",

    # GCC
    "tadawul": "^TASI.SR", "تداول": "^TASI.SR", "تاسي": "^TASI.SR",
    "tasi": "^TASI.SR", "سوق السعودية": "^TASI.SR",

    "dfm": "^DFMGI", "دبي": "^DFMGI", "سوق دبي": "^DFMGI",
    "dubai financial market": "^DFMGI",

    "adx": "^FTFADGI", "أبوظبي": "^FTFADGI", "سوق ابوظبي": "^FTFADGI",

    # Global
    "ftse": "^FTSE", "ftse 100": "^FTSE",
    "dax": "^GDAXI", "داكس": "^GDAXI",
    "nikkei": "^N225", "نيكي": "^N225",
    "hang seng": "^HSI", "هانج سينج": "^HSI",
    "shanghai": "000001.SS",
}

# ── ETFs & Popular Instruments ────────────────────────────────────────────────
_ETFS: dict[str, str] = {
    # Equity ETFs
    "spy": "SPY", "s&p etf": "SPY",
    "qqq": "QQQ", "nasdaq etf": "QQQ",
    "iwm": "IWM",
    "dia": "DIA",
    "vti": "VTI", "total market": "VTI",
    "voo": "VOO",

    # Sector ETFs
    "xlk": "XLK", "tech etf": "XLK",
    "xlf": "XLF", "financials etf": "XLF",
    "xle": "XLE", "energy etf": "XLE",
    "xlv": "XLV", "healthcare etf": "XLV",

    # Bond ETFs
    "tlt": "TLT", "long bonds": "TLT", "سندات طويلة": "TLT",
    "ief": "IEF", "bnd": "BND", "agg": "AGG",
    "shy": "SHY", "short bonds": "SHY",

    # Gold ETFs
    "gld": "GLD", "gold etf": "GLD", "صندوق الذهب": "GLD",
    "iau": "IAU",

    # Volatility
    "uvxy": "UVXY", "svxy": "SVXY",

    # International
    "eem": "EEM", "emerging markets": "EEM", "الأسواق الناشئة": "EEM",
    "vwo": "VWO",
}

# ── US Stocks (common name → ticker) ─────────────────────────────────────────
_US_STOCKS: dict[str, str] = {
    "apple": "AAPL", "آبل": "AAPL", "أبل": "AAPL",
    "microsoft": "MSFT", "مايكروسوفت": "MSFT",
    "nvidia": "NVDA", "نفيديا": "NVDA", "انفيديا": "NVDA",
    "amazon": "AMZN", "أمازون": "AMZN", "امازون": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "جوجل": "GOOGL",
    "meta": "META", "facebook": "META", "ميتا": "META",
    "tesla": "TSLA", "تسلا": "TSLA",
    "netflix": "NFLX", "نتفليكس": "NFLX",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "berkshire": "BRK-B", "buffett": "BRK-B",
    "visa": "V", "فيزا": "V",
    "mastercard": "MA", "ماستركارد": "MA",
    "paypal": "PYPL", "بايبال": "PYPL",
    "salesforce": "CRM",
    "amd": "AMD", "advanced micro": "AMD",
    "intel": "INTC", "انتل": "INTC",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "disney": "DIS", "ديزني": "DIS",
    "coca cola": "KO", "كوكاكولا": "KO", "كوكا كولا": "KO",
    "pepsi": "PEP", "pepisco": "PEP", "بيبسي": "PEP",
    "mcdonalds": "MCD", "ماكدونالدز": "MCD",
    "starbucks": "SBUX", "ستاربكس": "SBUX",
    "johnson": "JNJ", "johnson & johnson": "JNJ",
    "pfizer": "PFE", "فايزر": "PFE",
    "exxon": "XOM", "اكسون": "XOM",
    "chevron": "CVX", "شيفرون": "CVX",
    "boeing": "BA", "بوينج": "BA",
    "walmart": "WMT", "وول مارت": "WMT",
    "bank of america": "BAC",
    "goldman sachs": "GS", "goldman": "GS",
    "morgan stanley": "MS",
    "citigroup": "C", "citi": "C",
    "uber": "UBER", "أوبر": "UBER",
    "airbnb": "ABNB",
    "palantir": "PLTR",
    "openai": "MSFT",  # OpenAI not public → closest proxy
    "arm": "ARM",
    "snowflake": "SNOW",
    "coinbase": "COIN", "كوينبيس": "COIN",
}

# ── GCC Stocks (merges with arabic_stocks map if available) ──────────────────
_GCC_STOCKS: dict[str, str] = {
    "aramco": "2222.SR", "ارامكو": "2222.SR", "أرامكو": "2222.SR",
    "sabic": "2010.SR", "سابك": "2010.SR",
    "stc": "7010.SR", "الاتصالات السعودية": "7010.SR",
    "al rajhi": "1120.SR", "الراجحي": "1120.SR",
    "snb": "1180.SR", "البنك الأهلي": "1180.SR",
    "emaar": "EMAAR.DU", "اعمار": "EMAAR.DU", "إعمار": "EMAAR.DU",
    "damac": "DAMAC.DU", "داماك": "DAMAC.DU",
    "aldar": "ALDAR.AE", "الدار": "ALDAR.AE",
    "adnoc": "ADNOCDIST.AE", "ادنوك": "ADNOCDIST.AE",
    "fab": "FAB.AE", "بنك ابوظبي الاول": "FAB.AE",
    "taqa": "TAQA.AE", "طاقة": "TAQA.AE",
    "etisalat": "ETISALAT.AE", "اتصالات": "ETISALAT.AE",
    "kfh": "KFH.KW", "بيت التمويل": "KFH.KW",
    "qnb": "QNBK.QA", "بنك قطر الوطني": "QNBK.QA",
}

# ── Forex (Currency Pairs) ────────────────────────────────────────────────────
_FOREX: dict[str, str] = {
    "eurusd": "EURUSD=X", "eur/usd": "EURUSD=X", "يورو دولار": "EURUSD=X",
    "gbpusd": "GBPUSD=X", "gbp/usd": "GBPUSD=X", "جنيه دولار": "GBPUSD=X",
    "usdjpy": "USDJPY=X", "usd/jpy": "USDJPY=X", "دولار ين": "USDJPY=X",
    "usdsar": "USDSAR=X", "dollar riyal": "USDSAR=X", "دولار ريال": "USDSAR=X",
    "usdaed": "USDAED=X", "dollar dirham": "USDAED=X", "دولار درهم": "USDAED=X",
    "dollar": "DX-Y.NYB",  # Dollar index
    "الدولار": "DX-Y.NYB", "مؤشر الدولار": "DX-Y.NYB", "dxy": "DX-Y.NYB",
}

# ── Merge all maps ────────────────────────────────────────────────────────────
_ALL_MAPS = {
    **_COMMODITIES,
    **_CRYPTO,
    **_INDICES,
    **_ETFS,
    **_US_STOCKS,
    **_GCC_STOCKS,
    **_FOREX,
}

# ── Known ticker suffixes (already valid Yahoo tickers) ──────────────────────
_TICKER_RE = re.compile(
    r'^[A-Z0-9]{1,6}$|'                        # AAPL, BTC
    r'^[A-Z0-9]{1,6}-[A-Z]{2,4}$|'            # BTC-USD
    r'^[A-Z0-9]+\.[A-Z]{2}$|'                  # 2222.SR, EMAAR.DU
    r'^\^[A-Z]{2,10}$|'                        # ^GSPC, ^VIX
    r'^[A-Z]{1,6}=[A-Z]$'                      # GC=F, CL=F
)


def resolve_ticker(query: str) -> str | None:
    """
    Resolve a natural language query (Arabic or English) to a Yahoo Finance ticker.

    Returns the ticker string or None if unresolvable.

    Examples:
        resolve_ticker("gold")        → "GC=F"
        resolve_ticker("الذهب")       → "GC=F"
        resolve_ticker("bitcoin")     → "BTC-USD"
        resolve_ticker("AAPL")        → "AAPL"
        resolve_ticker("sp500")       → "^GSPC"
        resolve_ticker("ارامكو")      → "2222.SR"
        resolve_ticker("xyz123abc")   → None
    """
    if not query:
        return None

    raw = query.strip()

    # 1. Lookup in map FIRST (natural language takes priority)
    key = raw.lower().strip()
    if key in _ALL_MAPS:
        return _ALL_MAPS[key]

    # 2. Already a valid Yahoo ticker format — only if input looks like a ticker
    # (all uppercase, or contains special chars like . = ^ -)
    upper = raw.upper()
    _looks_like_ticker = (
        raw == raw.upper()           # already uppercase: AAPL, BTC-USD
        or "." in raw                # 2222.SR, EMAAR.DU
        or "=" in raw                # GC=F, CL=F
        or "^" in raw                # ^GSPC, ^VIX
        or raw.startswith("^")
    )
    if _looks_like_ticker and _TICKER_RE.match(upper):
        return upper

    # 3. Try with common Arabic diacritics stripped
    key_nodiac = _strip_arabic_diacritics(key)
    for map_key, ticker in _ALL_MAPS.items():
        if _strip_arabic_diacritics(map_key.lower()) == key_nodiac:
            return ticker

    # 4. Partial match — useful for "aramco stock" → "2222.SR"
    for map_key, ticker in _ALL_MAPS.items():
        if map_key in key or key in map_key:
            return ticker

    logger.debug("[TickerResolver] No match for: %s", query)
    return None


def resolve_tickers(queries: list[str]) -> dict[str, str | None]:
    """Resolve a list of queries. Returns {query: ticker_or_None}."""
    return {q: resolve_ticker(q) for q in queries}


def get_asset_type(ticker: str) -> str:
    """
    Classify a ticker into asset type.
    Returns: 'commodity' | 'crypto' | 'index' | 'etf' | 'forex' | 'stock'
    """
    t = ticker.upper()
    if t in _COMMODITIES.values():       return "commodity"
    if t in _CRYPTO.values():            return "crypto"
    if t in _INDICES.values():           return "index"
    if t in _ETFS.values():              return "etf"
    if t in _FOREX.values():             return "forex"
    if t.endswith((".SR", ".AE", ".DU", ".KW", ".QA")): return "gcc_stock"
    return "stock"


def _strip_arabic_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) for fuzzy matching."""
    # Diacritics range: U+064B–U+065F, U+0670
    return re.sub(r'[\u064b-\u065f\u0670]', '', text)
