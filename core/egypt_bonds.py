"""
egypt_bonds.py — Egyptian Government Bonds & T-Bills Scraper
=============================================================
Fetches the full EGP yield curve + CBE policy rate from multiple sources
with a 4-hour in-process cache so repeated questions don't re-scrape.

Source priority (yield curve):
  1. worldgovernmentbonds.com  — via Playwright (JS-rendered, full curve)
  2. investing.com             — via cloudscraper (Cloudflare bypass)
  3. FRED API                  — Egypt 10-year anchor (series INTGSBEGM193N)
  4. Serper/Google search      — regex extraction from news snippets

CBE policy rate: FRED INTDSREGM193N → Serper search fallback
EGP/USD rate:    FMP API → open.er-api.com fallback
"""

import os
import re
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Optional
from dotenv import load_dotenv

# Ensure .env is loaded when this module is imported standalone (tests, CLI)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)

logger = logging.getLogger(__name__)

# ── In-process cache ──────────────────────────────────────────────────────────
_cache: Dict = {}
_CACHE_TTL = 4 * 3600          # 4 hours — bond yields are slow-moving

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Tenor display order
TENOR_ORDER = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_pct(text: str) -> Optional[float]:
    """Extract a float percentage from a string like '27.50%' or '27,50'."""
    if not text:
        return None
    clean = re.sub(r"[^\d.,\-]", "", text.strip().replace(",", "."))
    try:
        val = float(clean)
        # Sanity check: Egyptian yields should be between 1% and 50%
        return val if 1.0 <= val <= 50.0 else None
    except ValueError:
        return None


def _normalize_tenor(raw: str) -> Optional[str]:
    """Normalize various maturity strings to a canonical short label."""
    raw = raw.strip().lower()
    mapping = {
        "3 month": "3M",  "3month": "3M",  "91": "3M",   "3-month": "3M",
        "6 month": "6M",  "6month": "6M",  "182": "6M",  "6-month": "6M",
        "9 month": "9M",  "270": "9M",
        "1 year":  "1Y",  "1year": "1Y",   "364": "1Y",  "12 month": "1Y",
        "2 year":  "2Y",  "2year": "2Y",
        "3 year":  "3Y",  "3year": "3Y",
        "5 year":  "5Y",  "5year": "5Y",
        "7 year":  "7Y",  "7year": "7Y",
        "10 year": "10Y", "10year": "10Y",
        "15 year": "15Y", "15year": "15Y",
        "20 year": "20Y", "20year": "20Y",
        "30 year": "30Y", "30year": "30Y",
    }
    for key, label in mapping.items():
        if key in raw:
            return label
    # Regex fallback: "2y", "5yr", "10 yr"
    m = re.search(r"(\d+)\s*(?:y|yr|year)", raw)
    if m:
        n = int(m.group(1))
        return f"{n}Y" if n > 1 else "1Y"
    m = re.search(r"(\d+)\s*(?:m|mo|month)", raw)
    if m:
        return f"{m.group(1)}M"
    return None


# ── Source 1: World Government Bonds (Playwright — handles JS rendering) ──────

def _scrape_world_gov_bonds() -> dict:
    """
    Scrape https://www.worldgovernmentbonds.com/country/egypt/ using Playwright
    so JavaScript-rendered table rows are visible.
    Falls back to a plain requests attempt if Playwright is unavailable.
    """
    url = "http://www.worldgovernmentbonds.com/country/egypt/"

    def _parse_html(html: str) -> Dict[str, float]:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        yields: Dict[str, float] = {}

        # Strategy A: scan every <tr> for (maturity, yield%) pairs
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            tenor = _normalize_tenor(cells[0].get_text(strip=True))
            if not tenor:
                continue
            for cell in cells[1:]:
                pct = _parse_pct(cell.get_text(strip=True))
                if pct is not None:
                    yields[tenor] = pct
                    break

        # Strategy B: broad regex over page text
        if not yields:
            page_text = soup.get_text(" ")
            for m in re.finditer(
                r"(\d+[\s\-](?:year|yr|month|mo)[s]?)\s*[:\|]?\s*(\d{1,2}[.,]\d{1,2})\s*%",
                page_text, re.I,
            ):
                tenor = _normalize_tenor(m.group(1))
                pct = _parse_pct(m.group(2))
                if tenor and pct:
                    yields[tenor] = pct
        return yields

    # ── Playwright path ────────────────────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": _HEADERS["User-Agent"]})
            page.goto(url, wait_until="networkidle", timeout=25000)
            html = page.content()
            browser.close()

        yields = _parse_html(html)
        if yields:
            logger.info("[egypt_bonds] worldgovernmentbonds (Playwright): %d tenors", len(yields))
            return {"yields": yields, "source": "worldgovernmentbonds.com"}
        logger.warning("[egypt_bonds] worldgovernmentbonds Playwright: page loaded but no yields parsed")

    except Exception as e:
        logger.warning("[egypt_bonds] worldgovernmentbonds Playwright failed: %s", e)

    # ── Plain requests fallback (works if site later drops JS rendering) ───────
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        resp = scraper.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        yields = _parse_html(resp.text)
        if yields:
            logger.info("[egypt_bonds] worldgovernmentbonds (cloudscraper): %d tenors", len(yields))
            return {"yields": yields, "source": "worldgovernmentbonds.com"}
    except Exception as e:
        logger.warning("[egypt_bonds] worldgovernmentbonds cloudscraper failed: %s", e)

    return {"yields": {}}


# ── Source 2: Investing.com (cloudscraper) ────────────────────────────────────

def _scrape_investing_com() -> dict:
    """
    Scrape https://www.investing.com/rates-bonds/egypt-government-bonds
    via cloudscraper to bypass Cloudflare.
    """
    url = "https://www.investing.com/rates-bonds/egypt-government-bonds"
    try:
        import cloudscraper
        from bs4 import BeautifulSoup

        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        resp = scraper.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        yields: Dict[str, float] = {}

        # Investing.com bond tables use class "genTbl" or id "cr1"
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                name_text = cells[0].get_text(strip=True)
                tenor = _normalize_tenor(name_text)
                if not tenor:
                    continue
                # Yield is typically in cells[1] (last yield) or cells[2] (prev close)
                for idx in [1, 2, 3]:
                    if idx < len(cells):
                        pct = _parse_pct(cells[idx].get_text(strip=True))
                        if pct is not None:
                            yields[tenor] = pct
                            break

        if yields:
            logger.info("[egypt_bonds] investing.com: %d tenors scraped", len(yields))
            return {"yields": yields, "source": "investing.com"}

    except Exception as e:
        logger.warning("[egypt_bonds] investing.com failed: %s", e)

    return {"yields": {}}


# ── Source 3: FRED API — Egypt 10-Year Government Bond Yield ─────────────────

def _fred_yields() -> dict:
    """
    Pull Egypt government bond yields from FRED.
    Tries multiple series — FRED often has "." (missing) for recent months,
    so we walk back through observations to find the latest real value.

    Series tried:
      INTGSBEGM193N  — 10-Year Government Bond Yield (Egypt)
      IRLTLT01EGM156N — Long-Term Interest Rate (Egypt, monthly)
    """
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        return {"yields": {}}

    # (series_id, tenor_label)
    series_to_try = [
        ("INTGSBEGM193N",   "10Y"),
        ("IRLTLT01EGM156N", "10Y"),
    ]

    for series_id, tenor in series_to_try:
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": fred_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 12,        # fetch 12 months — skip "." until real value
                },
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            obs = resp.json().get("observations", [])
            for ob in obs:
                raw = ob.get("value", ".")
                if raw in (".", "N/A", ""):
                    continue              # FRED missing-data sentinel
                val = _parse_pct(raw)
                if val:
                    date = ob.get("date", "")[:7]
                    logger.info("[egypt_bonds] FRED %s %s: %.2f%% (as of %s)",
                                series_id, tenor, val, date)
                    return {"yields": {tenor: val}, "source": f"FRED/{series_id} (as of {date})"}
        except Exception as e:
            logger.warning("[egypt_bonds] FRED %s failed: %s", series_id, e)

    return {"yields": {}}


# ── Source 4: Serper search — parse yield curve from news snippets ────────────

def _serper_yields() -> dict:
    """
    Use Serper Google Search to extract current Egyptian bond/T-bill yields.
    Runs three targeted searches and applies broad regex to each response —
    flexible enough to match how financial sites phrase yield data.
    """
    serper_key = os.getenv("SERPER_API_KEY", "")
    if not serper_key:
        return {"yields": {}}

    # (search query, tenor hints to try in that response)
    queries = [
        ("Egypt 10 year government bond yield percent", ["10Y"]),
        ("Egypt T-bill 91 day 182 day 364 day yield auction percent", ["3M", "6M", "1Y"]),
        ("Egypt 3 year 5 year bond yield percent", ["3Y", "5Y"]),
    ]

    # Flexible extractor: maturity keyword + any nearby percentage
    # Matches: "10-year ... 28.1%", "28.1% ... 10 year", "yield 27.5 percent"
    _TENOR_PAT = {
        "3M":  re.compile(r"(?:91[\s\-]day|3[\s\-]month)\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
        "6M":  re.compile(r"(?:182[\s\-]day|6[\s\-]month)\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
        "1Y":  re.compile(r"(?:364[\s\-]day|1[\s\-]year|12[\s\-]month)\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
        "2Y":  re.compile(r"2[\s\-]year\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
        "3Y":  re.compile(r"3[\s\-]year\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
        "5Y":  re.compile(r"5[\s\-]year\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
        "7Y":  re.compile(r"7[\s\-]year\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
        "10Y": re.compile(r"10[\s\-]year\D{0,40}?(\d{2}[.,]\d{1,2})\s*%", re.I),
    }

    yields: Dict[str, float] = {}

    for query, target_tenors in queries:
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": 8},
                timeout=12,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            # Combine answer box + organic snippets
            full_text = (data.get("answerBox", {}).get("answer", "") + " " +
                         data.get("answerBox", {}).get("snippet", "") + " " +
                         " ".join(
                             r.get("snippet", "") + " " + r.get("title", "")
                             for r in data.get("organic", [])
                         ))

            for tenor in target_tenors:
                if tenor in yields:
                    continue
                m = _TENOR_PAT.get(tenor, re.compile("$")).search(full_text)
                if m:
                    val = _parse_pct(m.group(1))
                    if val:
                        yields[tenor] = val

        except Exception as e:
            logger.warning("[egypt_bonds] Serper query %r failed: %s", query[:40], e)

    if yields:
        logger.info("[egypt_bonds] Serper: %d tenors extracted", len(yields))
        return {"yields": yields, "source": "Serper/Google search"}

    return {"yields": {}}


# ── Source 5: CBE Policy Rate ─────────────────────────────────────────────────

def _get_cbe_rate() -> Optional[float]:
    """
    Fetch CBE overnight deposit rate.
    Tries FRED API first (series INTDSREGM193N = Egypt discount rate),
    then falls back to Serper search.
    """
    # 1. FRED API
    fred_key = os.getenv("FRED_API_KEY", "")
    if fred_key:
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "INTDSREGM193N",
                    "api_key": fred_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                obs = resp.json().get("observations", [])
                if obs:
                    val = _parse_pct(obs[0].get("value", ""))
                    if val:
                        logger.info("[egypt_bonds] CBE rate from FRED: %.2f%%", val)
                        return val
        except Exception as e:
            logger.warning("[egypt_bonds] FRED CBE rate failed: %s", e)

    # 2. Serper search fallback
    serper_key = os.getenv("SERPER_API_KEY", "")
    if serper_key:
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": "CBE Central Bank Egypt overnight deposit lending rate 2026 percent", "num": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                snippets = " ".join(
                    r.get("snippet", "") for r in resp.json().get("organic", [])
                )
                # Look for patterns like "27.25%" or "27.25 percent"
                m = re.search(r"(\d{2}[.,]\d{1,2})\s*(?:%|percent)", snippets)
                if m:
                    val = _parse_pct(m.group(1))
                    if val:
                        logger.info("[egypt_bonds] CBE rate from Serper: %.2f%%", val)
                        return val
        except Exception as e:
            logger.warning("[egypt_bonds] Serper CBE rate failed: %s", e)

    return None


# ── Inflation & EGP rate ─────────────────────────────────────────────────────

def _get_egypt_inflation() -> Optional[float]:
    """Fetch Egypt CPI inflation from FRED (series EGYPCPIALLMINMEI or similar)."""
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        return None
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "EGYPCPIALLMINMEI",
                "api_key": fred_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 2,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            obs = resp.json().get("observations", [])
            # FRED gives CPI index; compute YoY %
            if len(obs) >= 2:
                curr = float(obs[0]["value"])
                # We need 12 months ago — fetch separately for accuracy
                # For now return the raw index change approximation
                return None   # skip complex YoY calculation; Serper is faster
    except Exception:
        pass
    return None


def _get_egp_rate() -> Optional[float]:
    """Get USD/EGP exchange rate via FMP or a simple endpoint."""
    try:
        fmp_key = os.getenv("FMP_API_KEY", "")
        if fmp_key:
            resp = requests.get(
                f"https://financialmodelingprep.com/api/v3/fx/USDEGP",
                params={"apikey": fmp_key},
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    return data[0].get("ask") or data[0].get("bid")
    except Exception:
        pass
    # Fallback: open exchange rate (no auth needed for USD/EGP)
    try:
        resp = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            timeout=8,
        )
        if resp.status_code == 200:
            return resp.json().get("rates", {}).get("EGP")
    except Exception:
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_egypt_bond_data(force_refresh: bool = False) -> dict:
    """
    Returns a structured dict with the full EGP yield curve + macro context.
    Cached for 4 hours. Falls back gracefully between sources.

    Returns:
        {
          "yields":     {"3M": 27.5, "6M": 27.8, "1Y": 28.1, ...},  # % annual
          "cbe_rate":   27.25,        # CBE overnight deposit rate %
          "egp_usd":    50.5,         # USD/EGP spot
          "source":     "worldgovernmentbonds.com",
          "fetched_at": "2025-03-03 14:22 UTC",
          "error":      None,
        }
    """
    if not force_refresh and _cache.get("data") and (time.time() - _cache.get("fetched_at", 0)) < _CACHE_TTL:
        return _cache["data"]

    result = {"yields": {}, "cbe_rate": None, "egp_usd": None,
              "source": None, "fetched_at": None, "error": None}

    # --- Yield curve: cascade through sources until one works ---
    for _source_fn in (
        _scrape_world_gov_bonds,   # 1. cloudscraper → worldgovernmentbonds.com
        _scrape_investing_com,     # 2. cloudscraper → investing.com
        _fred_yields,              # 3. FRED API     → 10Y anchor
        _serper_yields,            # 4. Serper/Google → news-snippet extraction
    ):
        data = _source_fn()
        if data.get("yields"):
            result["yields"] = data["yields"]
            result["source"] = data.get("source", _source_fn.__name__)
            break

    if not result["yields"]:
        result["error"] = "Could not fetch yield curve from any source"
        logger.warning("[egypt_bonds] All yield curve sources failed")

    # --- Macro enrichment (parallel would be nice but keep simple) ---
    result["cbe_rate"] = _get_cbe_rate()
    result["egp_usd"]  = _get_egp_rate()
    result["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    _cache["data"] = result
    _cache["fetched_at"] = time.time()

    logger.info("[egypt_bonds] data ready — %d tenors, CBE=%.2f%%, EGP/USD=%.2f",
                len(result["yields"]),
                result["cbe_rate"] or 0,
                result["egp_usd"] or 0)
    return result


def format_egypt_bonds_for_prompt(data: dict) -> str:
    """
    Format scraped bond data as a concise block for LLM prompt injection.
    """
    lines = ["[EGYPT GOVERNMENT BONDS — LIVE DATA]"]

    if data.get("fetched_at"):
        lines.append(f"As of: {data['fetched_at']} (source: {data.get('source', 'N/A')})")

    yields = data.get("yields", {})
    if yields:
        lines.append("\nYield Curve (annualised %):")
        # Display in tenor order
        for tenor in TENOR_ORDER:
            if tenor in yields:
                lines.append(f"  {tenor:>4}:  {yields[tenor]:.2f}%")
        # Any extra tenors not in standard order
        for tenor, val in yields.items():
            if tenor not in TENOR_ORDER:
                lines.append(f"  {tenor:>4}:  {val:.2f}%")

    if data.get("cbe_rate"):
        lines.append(f"\nCBE Overnight Deposit Rate: {data['cbe_rate']:.2f}%")

    if data.get("egp_usd"):
        lines.append(f"USD/EGP Spot Rate:          {data['egp_usd']:.2f}")

    if data.get("error"):
        lines.append(f"\n⚠️  Note: {data['error']}")

    return "\n".join(lines)


# ── Keyword detection ─────────────────────────────────────────────────────────

EGYPT_BOND_KEYWORDS_EN = [
    "egypt bond", "egyptian bond", "egypt t-bill", "egypt tbill", "egypt treasury",
    "egypt yield", "egypt debt", "egb ", "egypt government bond", "egypt sukuk",
    "nile bond", "egypt fixed income", "cbe rate", "egypt interest rate",
    "egyptian treasury", "egypt t bill", "egypt 10 year", "egypt 5 year",
]

EGYPT_BOND_KEYWORDS_AR = [
    "سندات مصر", "سندات مصرية", "أذون خزانة مصر", "اذون خزانة",
    "سندات الخزانة المصرية", "عائد سندات مصر", "الدين المصري",
    "سعر الفائدة المصري", "البنك المركزي المصري",
]


def is_egypt_bond_query(message: str) -> bool:
    """Return True if the message is asking about Egyptian bonds/T-bills."""
    low = message.lower()
    for kw in EGYPT_BOND_KEYWORDS_EN:
        if kw in low:
            return True
    for kw in EGYPT_BOND_KEYWORDS_AR:
        if kw in message:
            return True
    return False
