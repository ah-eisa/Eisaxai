"""
bond_data_fetcher.py — EisaX Smart Global Bond Data Engine
Uses worldgovernmentbonds.com API (no Cloudflare block)
- Auto-detects country from message (Arabic + English)
- Fetches full yield curve + key metrics
- Caches in SQLite (4h TTL)
- Returns prompt_block ready for DeepSeek injection
"""

import time
import sqlite3
import logging
import cloudscraper
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from core.config import CORE_DB as _cfg_core_db
DB_PATH = _cfg_core_db
CACHE_TTL_HOURS = 4
BASE_URL = "http://www.worldgovernmentbonds.com"

# ═══════════════════════════════════════════════════════
#  COUNTRY REGISTRY
#  SYMBOL = worldgovernmentbonds.com internal country ID
# ═══════════════════════════════════════════════════════
BOND_COUNTRIES = {
    "egypt": {
        "name": "Egypt", "symbol": "30", "bandiera": "eg",
        "url_page": "egypt", "currency": "EGP",
        "aliases_en": ["egypt", "egyptian", "egp", "cbe", "cairo"],
        "aliases_ar": ["\u0645\u0635\u0631", "\u0645\u0635\u0631\u064a", "\u0645\u0635\u0631\u064a\u0629"],
        "context": (
            "- CBE policy rate & inflation are key drivers\n"
            "- EGP floated 2024 under IMF deal (~$8bn) — FX risk present\n"
            "- Yield curve currently INVERTED (short > long)\n"
            "- T-bills (EGP): 91/182/364 days — highest liquidity\n"
            "- Eurobonds (USD): eliminate FX risk, lower nominal yield\n"
            "- Credit rating: B (S&P) — sub-investment grade"
        ),
    },
    "usa": {
        "name": "United States", "symbol": "1", "bandiera": "us",
        "url_page": "united-states", "currency": "USD",
        "aliases_en": ["usa", "us ", "united states", "america", "american",
                       "treasury", "treasuries", "t-bill", "t-bond", "federal reserve"],
        "aliases_ar": ["\u0623\u0645\u0631\u064a\u0643\u0627", "\u0623\u0645\u0631\u064a\u0643\u064a"],
        "context": (
            "- Federal Reserve controls fed funds rate\n"
            "- Global risk-free benchmark\n"
            "- Yield curve shape signals recession risk when inverted\n"
            "- TIPS available for inflation protection"
        ),
    },
    "uae": {
        "name": "UAE", "symbol": "55", "bandiera": "ae",
        "url_page": "united-arab-emirates", "currency": "AED",
        "aliases_en": ["uae", "emirates", "dubai", "abu dhabi", "aed", "dirham"],
        "aliases_ar": ["\u0627\u0644\u0625\u0645\u0627\u0631\u0627\u062a", "\u062f\u0628\u064a", "\u0623\u0628\u0648\u0638\u0628\u064a", "\u062f\u0631\u0647\u0645"],
        "context": (
            "- AED pegged to USD — minimal FX risk\n"
            "- Abu Dhabi rated AA — very low default risk\n"
            "- Sukuk (Islamic bonds) widely available"
        ),
    },
    "saudi": {
        "name": "Saudi Arabia", "symbol": "47", "bandiera": "sa",
        "url_page": "saudi-arabia", "currency": "SAR",
        "aliases_en": ["saudi", "ksa", "riyadh", "sar", "sama"],
        "aliases_ar": ["\u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629", "\u0633\u0639\u0648\u062f\u064a", "\u0627\u0644\u0631\u064a\u0627\u0636", "\u0633\u0627\u0645\u0627"],
        "context": (
            "- SAR pegged to USD — low FX risk\n"
            "- Rated A1/A+ — strong sovereign balance sheet\n"
            "- Vision 2030 reducing oil dependency"
        ),
    },
    "uk": {
        "name": "United Kingdom", "symbol": "4", "bandiera": "gb",
        "url_page": "united-kingdom", "currency": "GBP",
        "aliases_en": ["uk", "britain", "british", "gilts", "gbp", "pound", "bank of england"],
        "aliases_ar": ["\u0628\u0631\u064a\u0637\u0627\u0646\u064a\u0627", "\u0625\u0646\u062c\u0644\u062a\u0631\u0627"],
        "context": (
            "- BoE sets base rate\n"
            "- Gilts: highly liquid UK government bonds\n"
            "- GBP FX risk for non-GBP investors"
        ),
    },
    "germany": {
        "name": "Germany", "symbol": "19", "bandiera": "de",
        "url_page": "germany", "currency": "EUR",
        "aliases_en": ["germany", "german", "bund", "bunds", "eur", "euro", "ecb"],
        "aliases_ar": ["\u0623\u0644\u0645\u0627\u0646\u064a\u0627", "\u064a\u0648\u0631\u0648"],
        "context": (
            "- ECB sets Eurozone policy\n"
            "- Bunds: European AAA safe-haven benchmark\n"
            "- Lowest yields in Europe"
        ),
    },
    "turkey": {
        "name": "Turkey", "symbol": "29", "bandiera": "tr",
        "url_page": "turkey", "currency": "TRY",
        "aliases_en": ["turkey", "turkish", "try", "lira", "ankara"],
        "aliases_ar": ["\u062a\u0631\u0643\u064a\u0627", "\u062a\u0631\u0643\u064a", "\u0644\u064a\u0631\u0629 \u062a\u0631\u0643\u064a\u0629"],
        "context": (
            "- TRY: major depreciation history — high FX risk\n"
            "- High nominal yields, often negative real yields\n"
            "- Sub-investment grade credit rating"
        ),
    },
    "india": {
        "name": "India", "symbol": "52", "bandiera": "in",
        "url_page": "india", "currency": "INR",
        "aliases_en": ["india", "indian", "inr", "rupee", "rbi"],
        "aliases_ar": ["\u0627\u0644\u0647\u0646\u062f", "\u0647\u0646\u062f\u064a", "\u0631\u0648\u0628\u064a\u0629"],
        "context": (
            "- RBI sets monetary policy\n"
            "- In JP Morgan EM bond index since 2024\n"
            "- Strong GDP growth supports fiscal position"
        ),
    },
    "brazil": {
        "name": "Brazil", "symbol": "12", "bandiera": "br",
        "url_page": "brazil", "currency": "BRL",
        "aliases_en": ["brazil", "brazilian", "brl", "selic"],
        "aliases_ar": ["\u0627\u0644\u0628\u0631\u0627\u0632\u064a\u0644", "\u0628\u0631\u0627\u0632\u064a\u0644\u064a"],
        "context": (
            "- SELIC rate — one of highest real yields globally\n"
            "- BRL: volatile — significant FX risk\n"
            "- Sub-investment grade fiscal concerns"
        ),
    },
    "japan": {
        "name": "Japan", "symbol": "7", "bandiera": "jp",
        "url_page": "japan", "currency": "JPY",
        "aliases_en": ["japan", "japanese", "jpy", "yen", "boj", "jgb"],
        "aliases_ar": ["\u0627\u0644\u064a\u0627\u0628\u0627\u0646", "\u064a\u0627\u0628\u0627\u0646\u064a", "\u064a\u0646"],
        "context": (
            "- BoJ ultra-loose policy — yield curve control\n"
            "- JGBs: lowest yields globally\n"
            "- JPY FX risk — yen volatility significant"
        ),
    },
    "china": {
        "name": "China", "symbol": "37", "bandiera": "cn",
        "url_page": "china", "currency": "CNY",
        "aliases_en": ["china", "chinese", "cny", "yuan", "renminbi", "pboc"],
        "aliases_ar": ["\u0627\u0644\u0635\u064a\u0646", "\u0635\u064a\u0646\u064a", "\u064a\u0648\u0627\u0646"],
        "context": (
            "- PBOC manages monetary policy\n"
            "- CNY: managed float — moderate FX risk\n"
            "- Growing inclusion in global bond indices"
        ),
    },
}


# ═══════════════════════════════════════════════════════
#  CACHE
# ═══════════════════════════════════════════════════════

def _init_cache():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bond_yield_cache (
            country    TEXT PRIMARY KEY,
            data_json  TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

_init_cache()


def _load_cache(country: str) -> Optional[dict]:
    import json
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT data_json, fetched_at FROM bond_yield_cache WHERE country=?",
            (country,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        age = datetime.now() - datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
        if age > timedelta(hours=CACHE_TTL_HOURS):
            return None
        return json.loads(row[0])
    except Exception as e:
        logger.warning(f"[BondCache] Load failed: {e}")
        return None


def _save_cache(country: str, data: dict):
    import json
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO bond_yield_cache (country, data_json, fetched_at) VALUES (?,?,?)",
            (country, json.dumps(data), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[BondCache] Save failed: {e}")


# ═══════════════════════════════════════════════════════
#  COUNTRY DETECTION
# ═══════════════════════════════════════════════════════

def detect_country(message: str) -> str:
    msg = message.lower()
    for key, info in BOND_COUNTRIES.items():
        for alias in info["aliases_en"] + info["aliases_ar"]:
            if alias in msg:
                logger.info(f"[BondDetect] '{key}' via '{alias}'")
                return key
    return "general"


# ═══════════════════════════════════════════════════════
#  SCRAPER
# ═══════════════════════════════════════════════════════

def _fetch_wgb(country_key: str, info: dict) -> dict:
    """Fetch full bond data from worldgovernmentbonds.com API"""
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    page_url = f"{BASE_URL}/country/{info['url_page']}/"

    # Step 1: GET page first to establish session/cookies
    scraper.get(page_url, timeout=15)
    time.sleep(0.5)

    # Step 2: POST to API
    payload = {
        "GLOBALVAR": {
            "JS_VARIABLE": "jsGlobalVars",
            "FUNCTION": "Country",
            "DOMESTIC": True,
            "ENDPOINT": f"{BASE_URL}/wp-json/country/v1/historical",
            "DATE_RIF": "2099-12-31",
            "OBJ": None,
            "COUNTRY1": {
                "SYMBOL": info["symbol"],
                "PAESE": info["name"],
                "PAESE_UPPERCASE": info["name"].upper(),
                "BANDIERA": info["bandiera"],
                "URL_PAGE": info["url_page"],
            },
            "COUNTRY2": None, "OBJ1": None, "OBJ2": None
        }
    }

    r = scraper.post(
        f"{BASE_URL}/wp-json/country/v1/main",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Origin": BASE_URL,
            "Referer": page_url,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        },
        timeout=25
    )

    if r.status_code != 200:
        logger.error(f"[BondFetch] {country_key}: HTTP {r.status_code}")
        return {}

    data = r.json()
    if not data.get("success"):
        logger.error(f"[BondFetch] {country_key}: API error {data}")
        return {}

    # Parse yield curve from mainTable
    yields = []
    soup = BeautifulSoup(data.get("mainTable", ""), "html.parser")
    for row in soup.find_all("tr"):
        cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        # Row has maturity + yield: ['', '3 months', '24.154%', ...]
        if len(cols) >= 3 and "%" in cols[2] and cols[1] not in ("", "ResidualMaturity"):
            try:
                yields.append({
                    "maturity": cols[1],
                    "yield":    cols[2],
                    "chg_1m":   cols[3] if len(cols) > 3 else "N/A",
                    "chg_6m":   cols[4] if len(cols) > 4 else "N/A",
                    "chg_12m":  cols[5] if len(cols) > 5 else "N/A",
                })
            except Exception:
                continue

    return {
        "yields":        yields,
        "bond10y":       data.get("bond10y", "N/A"),
        "cb_rate":       data.get("cbRateNumber", "N/A"),
        "cb_rate_date":  data.get("cbRateDate", "N/A"),
        "rating":        data.get("lastRatingValue", "N/A"),
        "spread_10y_2y": data.get("mainSpreadValue", "N/A"),
        "cds":           data.get("lastCds", "N/A"),
        "cds_default_prob": data.get("lastCdsDefaultProb", "N/A"),
        "last_update":   data.get("lastDataValDesc", "N/A"),
    }


# ═══════════════════════════════════════════════════════
#  PROMPT BLOCK BUILDER
# ═══════════════════════════════════════════════════════

def _build_prompt_block(data: dict) -> str:
    if not data.get("yields") and not data.get("bond10y"):
        return ""

    name     = data["country_name"]
    currency = data["currency"]
    updated  = data.get("fetched_at", "recent")

    lines = [
        f"## LIVE BOND DATA — {name} Government Bonds",
        f"*Source: WorldGovernmentBonds.com | As of: {updated} | Currency: {currency}*\n",
        f"**Key Metrics:**",
        f"- 10Y Yield: {data.get('bond10y', 'N/A')}%",
        f"- Central Bank Rate: {data.get('cb_rate', 'N/A')}% ({data.get('cb_rate_date', '')})",
        f"- Credit Rating: {data.get('rating', 'N/A')}",
        f"- 10Y-2Y Spread: {data.get('spread_10y_2y', 'N/A')} bp",
        f"- CDS (5Y): {data.get('cds', 'N/A')} bp | Default Prob: {data.get('cds_default_prob', 'N/A')}%\n",
    ]

    if data.get("yields"):
        lines += [
            "**Full Yield Curve:**",
            "| Maturity | Yield | Chg 1M | Chg 6M | Chg 12M |",
            "|----------|-------|--------|--------|---------|",
        ]
        for y in data["yields"]:
            lines.append(
                f"| {y['maturity']} | {y['yield']} | {y['chg_1m']} | {y['chg_6m']} | {y['chg_12m']} |"
            )

    if data.get("context"):
        lines.append(f"\n**Macro Context:**\n{data['context']}")

    lines.append(
        "\n*MANDATORY: Reference the live yield data above in your analysis. "
        "Cite specific maturities, yields, and the CDS/rating in your response.*"
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTION
# ═══════════════════════════════════════════════════════

def get_bond_data(message: str) -> dict:
    """
    Main entry point. Call this from _handle_bond_query.
    Returns dict with 'prompt_block' ready for DeepSeek injection.
    """
    country_key = detect_country(message)

    if country_key == "general":
        return {
            "country": "general", "country_name": "Global",
            "currency": "USD", "yields": [], "prompt_block": "",
            "source": "none", "fetched_at": datetime.now().isoformat(),
        }

    info = BOND_COUNTRIES[country_key]

    # Try cache
    cached = _load_cache(country_key)
    if cached:
        logger.info(f"[BondData] Cache hit: {country_key}")
        cached["source"] = "cache"
        cached["prompt_block"] = _build_prompt_block(cached)
        return cached

    # Live fetch
    logger.info(f"[BondData] Live fetch: {country_key}")
    raw = _fetch_wgb(country_key, info)

    result = {
        "country":      country_key,
        "country_name": info["name"],
        "currency":     info["currency"],
        "context":      info["context"],
        "fetched_at":   datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "source":       "live" if raw.get("yields") else "unavailable",
        **raw,
    }

    if raw.get("yields"):
        _save_cache(country_key, result)

    result["prompt_block"] = _build_prompt_block(result)
    return result


# ═══════════════════════════════════════════════════════
#  TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        "what is the best Egyptian bond right now?",
        "should I buy US treasuries?",
        "tell me about Saudi bonds",
        "UAE sukuk analysis",
    ]
    for msg in tests:
        print(f"\n{'='*55}\nQuery: {msg}")
        r = get_bond_data(msg)
        print(f"Country: {r['country_name']} | Source: {r['source']} | Maturities: {len(r.get('yields', []))}")
        if r.get("yields"):
            for y in r["yields"][:3]:
                print(f"  {y['maturity']}: {y['yield']}")
        print(f"  10Y: {r.get('bond10y')} | CB Rate: {r.get('cb_rate')} | Rating: {r.get('rating')}")
