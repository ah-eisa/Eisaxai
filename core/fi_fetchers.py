"""fi_fetchers.py -- EisaX Fixed Income: external API fetchers + get_instrument_data."""
from __future__ import annotations

import os
import re
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from core.fi_routing import (
    _validate_isin, _infer_country_code,
    VALID_ISIN_PREFIXES, _COUNTRY_RATINGS, _WGB_COUNTRY_DATA,
    _WGB_COUNTRY_SLUGS, _SUKUK_STRUCTURES, _HEADERS,
    _cache, _CACHE_TTL,
)

logger = logging.getLogger(__name__)

def _fetch_openfigi(isin: str) -> dict:
    """
    Map ISIN → instrument metadata via OpenFIGI v3.
    Free tier (no key): equities only.
    With OPENFIGI_API_KEY: fixed income instruments included.
    Returns empty dict on failure.
    """
    openfigi_key = os.getenv("OPENFIGI_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if openfigi_key:
        headers["X-OPENFIGI-APIKEY"] = openfigi_key

    try:
        resp = requests.post(
            "https://api.openfigi.com/v3/mapping",
            headers=headers,
            json=[{"idType": "ID_ISIN", "idValue": isin}],
            timeout=12,
        )
        if resp.status_code == 429:
            logger.warning("[fixed_income] OpenFIGI rate limited for %s", isin)
            return {}
        if resp.status_code != 200:
            logger.warning("[fixed_income] OpenFIGI HTTP %d for %s", resp.status_code, isin)
            return {}

        data = resp.json()
        if not data or not data[0].get("data"):
            err = data[0].get("error") if data else "no data"
            logger.info("[fixed_income] OpenFIGI no match for %s: %s", isin, err)
            return {}

        # Take the first result (most liquid / most relevant)
        item = data[0]["data"][0]
        logger.info("[fixed_income] OpenFIGI found: %s → %s", isin, item.get("name", ""))
        return item

    except Exception as e:
        logger.warning("[fixed_income] OpenFIGI failed for %s: %s", isin, e)
        return {}

def _serper_isin_lookup(isin: str) -> dict:
    """
    Use Serper Google Search to find bond/sukuk name and details from an ISIN.
    Returns dict with name, coupon, maturity, issuer if found.
    """
    serper_key = os.getenv("SERPER_API_KEY", "")

    try:
        full_text = ""
        organic = []

        # ── Try Serper first (if key available and has credits) ──
        _serper_ok = False
        if serper_key:
            try:
                resp = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": f'"{isin}" bond issuer country coupon maturity', "num": 6},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    organic = data.get("organic", [])
                    full_text = (
                        data.get("answerBox", {}).get("answer", "") + " " +
                        data.get("answerBox", {}).get("snippet", "") + " " +
                        " ".join(r.get("title", "") + " " + r.get("snippet", "") for r in organic)
                    )
                    _serper_ok = bool(organic)
                    logger.info("[fixed_income] Serper OK for %s: %d results", isin, len(organic))
                else:
                    logger.warning("[fixed_income] Serper %s for %s — trying DuckDuckGo",
                                   resp.status_code, isin)
            except Exception as _se:
                logger.warning("[fixed_income] Serper exception for %s: %s — trying DuckDuckGo", isin, _se)

        # ── DuckDuckGo HTML fallback (free, no key — returns real snippets) ──
        if not _serper_ok:
            try:
                import html as _html_mod
                ddg_resp = requests.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": f"{isin} bond coupon maturity issuer"},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    timeout=12,
                )
                if ddg_resp.status_code == 200:
                    page = ddg_resp.text
                    # Extract result snippets
                    raw_snippets = re.findall(
                        r'class="result__snippet"[^>]*>(.*?)</a', page, re.S
                    )
                    raw_titles = re.findall(
                        r'class="result__title"[^>]*>.*?<a[^>]*>(.*?)</a', page, re.S
                    )
                    def _strip_html(s: str) -> str:
                        s = re.sub(r'<[^>]+>', ' ', s)
                        return _html_mod.unescape(re.sub(r'\s+', ' ', s)).strip()
                    snippets_clean = [_strip_html(s) for s in raw_snippets]
                    titles_clean   = [_strip_html(s) for s in raw_titles]
                    full_text = " ".join(titles_clean + snippets_clean)
                    # Rebuild organic for title extraction
                    organic = [{"title": t, "snippet": s}
                               for t, s in zip(titles_clean, snippets_clean)]
                    logger.info("[fixed_income] DDG-HTML: %d results for %s, text_len=%d",
                                len(organic), isin, len(full_text))
            except Exception as _de:
                logger.warning("[fixed_income] DuckDuckGo-HTML failed for %s: %s", isin, _de)

        if not full_text.strip():
            return {}

        result = {}

        # Try to extract name from first organic result title
        if organic:
            title = organic[0].get("title", "")
            if isin in title or "sukuk" in title.lower() or "bond" in title.lower():
                result["name"] = title[:120]

        # Coupon
        coupon_m = re.search(r'(\d{1,2}\.?\d{0,4})\s*%', full_text)
        if coupon_m:
            try:
                c = float(coupon_m.group(1))
                if 0.5 <= c <= 25:   # sanity check
                    result["coupon"] = c
            except ValueError:
                pass

        # Maturity — try multiple formats (most specific first)
        _mon = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)')
        # 1. YYYY-MM-DD or YYYY/MM/DD
        dm1 = re.search(r'(\d{4})[\/\-](\d{2})[\/\-](\d{2})', full_text)
        # 2a. DD-Mon-YYYY  e.g. "15-Apr-2026" (common in Bloomberg/Reuters)
        dm2a = re.search(rf'(\d{{1,2}})[\/\-]({_mon})[\/\-](\d{{4}})', full_text, re.I)
        # 2b. DD Month YYYY  e.g. "15 March 2026"
        dm2b = re.search(rf'(\d{{1,2}})\s+({_mon})\s+(\d{{4}})', full_text, re.I)
        # 3. Month YYYY  e.g. "March 2026" (no day)
        dm3 = re.search(rf'({_mon})\s+(\d{{4}})', full_text, re.I)

        if dm1:
            result["maturity"] = f"{dm1.group(1)}-{dm1.group(2)}-{dm1.group(3)}"
        elif dm2a:
            try:
                dt = datetime.strptime(f"{dm2a.group(1)} {dm2a.group(2)[:3]} {dm2a.group(3)}", "%d %b %Y")
                result["maturity"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        elif dm2b:
            try:
                dt = datetime.strptime(f"{dm2b.group(1)} {dm2b.group(2)[:3]} {dm2b.group(3)}", "%d %b %Y")
                result["maturity"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        elif dm3:
            try:
                dt = datetime.strptime(f"01 {dm3.group(1)[:3]} {dm3.group(2)}", "%d %b %Y")
                result["maturity"] = dt.strftime("%Y-%m-01")
            except ValueError:
                pass

        # Sukuk flag
        if "sukuk" in full_text.lower() or "trust cert" in full_text.lower():
            result["is_sukuk"] = True

        # ── Country inference from issuer/text (critical for XS-prefix bonds) ──
        result["inferred_country_code"] = _infer_country_code(full_text)

        if result:
            logger.info("[fixed_income] Serper found for %s: %s", isin, result)
        return result

    except Exception as e:
        logger.warning("[fixed_income] Serper ISIN lookup failed for %s: %s", isin, e)
        return {}


def _parse_name_components(name: str) -> dict:
    """
    Extract coupon, maturity date, and issuer from instrument name.
    E.g. "EMIRATES NBD SUKUK 3.625% 04/10/2029" → {coupon: 3.625, maturity: "2029-04-10", issuer: "EMIRATES NBD"}
    """
    result = {"coupon": None, "maturity": None, "issuer": None, "is_sukuk": False, "sukuk_structure": None}
    if not name:
        return result

    name_up = name.upper()

    # Sukuk detection
    if "SUKUK" in name_up or "TRUST CERT" in name_up:
        result["is_sukuk"] = True
    for structure_key, structure_desc in _SUKUK_STRUCTURES.items():
        if structure_key in name_up:
            result["sukuk_structure"] = structure_desc
            result["is_sukuk"] = True
            break

    # Coupon: e.g. "3.625%", "4 7/8%", "4.875%", "3.00%"
    coupon_match = re.search(r'(\d{1,2}\.?\d{0,4})\s*%', name)
    if coupon_match:
        try:
            result["coupon"] = float(coupon_match.group(1))
        except ValueError:
            pass

    # Maturity date: MM/DD/YYYY or MM/YYYY or YYYY-MM-DD
    date_patterns = [
        (r'(\d{2}/\d{2}/\d{4})', "%m/%d/%Y"),
        (r'(\d{4}-\d{2}-\d{2})', "%Y-%m-%d"),
        (r'(\d{2}/\d{4})',        "%m/%Y"),
    ]
    for pat, fmt in date_patterns:
        m = re.search(pat, name)
        if m:
            try:
                dt = datetime.strptime(m.group(1), fmt)
                result["maturity"] = dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                pass

    # Issuer: everything before the coupon/maturity/SUKUK/BOND keywords
    issuer_end = len(name)
    for keyword in [" SUKUK", " BOND", " NOTE", " CERT", " MTN",
                    " SR ", " JR ", " SUB", " UNSECURED"]:
        idx = name_up.find(keyword)
        if idx > 0:
            issuer_end = min(issuer_end, idx)
    coupon_idx = name.find("%")
    if coupon_idx > 0:
        # Walk back to find the space before the number
        chunk = name[:coupon_idx]
        sp = chunk.rfind(" ")
        if sp > 0:
            issuer_end = min(issuer_end, sp)
    result["issuer"] = name[:issuer_end].strip().title()

    return result


# ── Source 2: FMP API — bond details ──────────────────────────────────────────

def _fetch_fmp_bond(isin: str) -> dict:
    """
    Try Financial Modeling Prep for bond fundamentals.
    Returns empty dict if not available.
    """
    fmp_key = os.getenv("FMP_API_KEY", "")
    if not fmp_key:
        return {}
    try:
        resp = requests.get(
            "https://financialmodelingprep.com/api/v4/bond/info",
            params={"isin": isin, "apikey": fmp_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.debug("[fixed_income] FMP bond fetch failed for %s: %s", isin, e)
    return {}


# ── Source 3: FRED benchmark yields ───────────────────────────────────────────

# FRED series for benchmark yields
_FRED_BENCHMARKS = {
    "US Treasury 3M":  "DGS3MO",
    "US Treasury 2Y":  "DGS2",
    "US Treasury 5Y":  "DGS5",
    "US Treasury 10Y": "DGS10",
    "US Treasury 30Y": "DGS30",
    "UK Gilt 10Y":     "IRLTLT01GBM156N",
    "UK Gilt 2Y":      "IRLTLT01GBM156N",  # approximate
    "Germany 10Y":     "IRLTLT01DEM156N",
}

def _fetch_fred_yield(series_id: str) -> Optional[float]:
    """Fetch latest observation from FRED for a given series."""
    fred_key = os.getenv("FRED_API_KEY", "")
    if not fred_key:
        return None
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": fred_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        for obs in resp.json().get("observations", []):
            val = obs.get("value", ".")
            if val not in (".", "N/A", ""):
                try:
                    return float(val)
                except ValueError:
                    pass
    except Exception as e:
        logger.debug("[fixed_income] FRED %s failed: %s", series_id, e)
    return None


def _fetch_benchmarks(currency: str, years_to_maturity: float) -> dict:
    """
    Fetch relevant benchmark yields for the given currency and duration.
    Returns dict of {label: yield_pct}.
    """
    benchmarks = {}
    fred_key = os.getenv("FRED_API_KEY", "")

    if not fred_key:
        # Hardcoded approximate values as last resort
        _fallback = {
            "US Treasury 3M": 5.25,
            "US Treasury 2Y": 4.70,
            "US Treasury 5Y": 4.50,
            "US Treasury 10Y": 4.65,
        }
        return _fallback

    # Pick benchmark series based on duration
    if years_to_maturity <= 1:
        series = [("US Treasury 3M", "DGS3MO"), ("US Treasury 1Y", "DGS1")]
    elif years_to_maturity <= 3:
        series = [("US Treasury 2Y", "DGS2"), ("US Treasury 3Y", "DGS3")]
    elif years_to_maturity <= 7:
        series = [("US Treasury 5Y", "DGS5"), ("US Treasury 7Y", "DGS7")]
    else:
        series = [("US Treasury 10Y", "DGS10"), ("US Treasury 30Y", "DGS30")]

    for label, sid in series:
        val = _fetch_fred_yield(sid)
        if val is not None:
            benchmarks[label] = val

    # Add local benchmark for GCC
    if currency in ("AED", "SAR"):
        # Gulf currencies are pegged to USD — US Treasuries are the relevant benchmark
        benchmarks["Note"] = "AED/SAR pegged to USD — US Treasuries are primary benchmark"

    return benchmarks


# ── Source 4: Sovereign CDS spreads (worldgovernmentbonds.com) ───────────────

def _fetch_sovereign_cds(country_code: str) -> Optional[float]:
    """
    Fetch 5-year sovereign CDS spread (bps) from worldgovernmentbonds.com.
    Uses their WP JSON API: POST wp-json/country/v1/main with country data.
    Returns float (bps) or None on failure. Cached in-process for 4h.
    """
    slug = _WGB_COUNTRY_SLUGS.get(country_code.upper())
    if not slug:
        logger.info("[fixed_income] CDS: no slug for country_code=%r — skipping", country_code)
        return None
    logger.info("[fixed_income] CDS: fetching for %s (slug=%s)", country_code, slug)

    cache_key = f"cds_{country_code}"
    entry = _cache.get(cache_key)
    if entry and time.time() - entry.get("_ts", 0) < 14400:   # 4h TTL
        return entry.get("cds")

    _wgb_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Content-Type": "application/json",
        "Referer": f"https://www.worldgovernmentbonds.com/country/{slug}/",
        "Origin": "https://www.worldgovernmentbonds.com",
    }

    try:
        # Step 1: GET the country page to extract SYMBOL (needed by the API)
        page_resp = requests.get(
            f"https://www.worldgovernmentbonds.com/country/{slug}/",
            headers={"User-Agent": _wgb_headers["User-Agent"]},
            timeout=12,
        )
        if page_resp.status_code != 200:
            logger.debug("[fixed_income] CDS: country page %d for %s", page_resp.status_code, slug)
            return None

        # Extract jsGlobalVars from the page
        sym_m   = re.search(r'"SYMBOL"\s*:\s*"(\d+)"',       page_resp.text)
        paese_m = re.search(r'"PAESE"\s*:\s*"([^"]+)"',       page_resp.text)
        band_m  = re.search(r'"BANDIERA"\s*:\s*"([^"]+)"',    page_resp.text)
        if not sym_m:
            logger.debug("[fixed_income] CDS: SYMBOL not found on page for %s", slug)
            return None

        symbol  = sym_m.group(1)
        paese   = paese_m.group(1) if paese_m else slug.replace("-", " ").title()
        bandiera = band_m.group(1) if band_m else country_code.lower()

        # Step 2: POST to the WP JSON API with the extracted metadata
        post_body = {
            "GLOBALVAR": {
                "JS_VARIABLE":  "jsGlobalVars",
                "FUNCTION":     "Country",
                "DOMESTIC":     True,
                "ENDPOINT":     "https://www.worldgovernmentbonds.com/wp-json/country/v1/historical",
                "DATE_RIF":     "2099-12-31",
                "OBJ":          None,
                "COUNTRY1": {
                    "SYMBOL":          symbol,
                    "PAESE":           paese,
                    "PAESE_UPPERCASE": paese.upper(),
                    "BANDIERA":        bandiera,
                    "URL_PAGE":        slug,
                },
                "COUNTRY2": None,
                "OBJ1":   None,
                "OBJ2":   None,
            }
        }
        api_resp = requests.post(
            "https://www.worldgovernmentbonds.com/wp-json/country/v1/main",
            headers=_wgb_headers,
            json=post_body,
            timeout=15,
        )
        if api_resp.status_code != 200:
            logger.debug("[fixed_income] CDS: API %d for %s", api_resp.status_code, country_code)
            return None

        data = api_resp.json()
        last_cds = data.get("lastCds")
        if last_cds not in (None, "", "---", "--"):
            bps = float(str(last_cds).replace(",", ""))
            if 1 <= bps <= 50000:
                # Store default probability if WGB provides it directly
                default_prob = data.get("lastCdsDefaultProb")
                _cache[cache_key] = {"cds": bps, "default_prob": default_prob, "_ts": time.time()}
                logger.info("[fixed_income] CDS %s: %.1f bps (WGB API), default_prob=%s",
                            country_code, bps, default_prob)
                # ── Parse rating table (agencies + dates) from WGB response ──────
                # ratingTable HTML has rows: Agency | Rating | Outlook | Date | Action
                rating_table_html = data.get("ratingTable", "")
                if rating_table_html:
                    # Extract all table cells in order
                    cells_raw = re.findall(r'<td[^>]*>(.*?)</td>', rating_table_html, re.S)
                    cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells_raw]
                    cells = [c for c in cells if c and c not in ("-", "--", "---", "")]
                    # Agencies appear in order: S&P, Moody's, Fitch, DBRS (5 cells per row)
                    # Row format: AgencyName | Rating | Outlook | Date | Action
                    _agency_map = {
                        "standard": ("S&P", "sp"), "s&p": ("S&P", "sp"),
                        "moody": ("Moody's", "moodys"), "fitch": ("Fitch", "fitch"),
                    }
                    i = 0
                    while i < len(cells) - 1:
                        cell_lower = cells[i].lower()
                        for kw, (ag_name, ag_key) in _agency_map.items():
                            if kw in cell_lower:
                                # Next non-empty cells: rating, outlook?, date
                                rating_val = cells[i+1] if i+1 < len(cells) else ""
                                date_val = ""
                                action_val = ""
                                # Look ahead up to 4 cells for a date pattern
                                for j in range(i+2, min(i+6, len(cells))):
                                    if re.search(r'\d{1,2}\s+\w+\s+20\d{2}|\w+\s+20\d{2}', cells[j]):
                                        date_val = cells[j]
                                    if "upgrade" in cells[j].lower() or "downgrade" in cells[j].lower():
                                        action_val = cells[j]
                                if rating_val and rating_val not in ("-", "--", "---"):
                                    cache_entry = {
                                        "rating": rating_val,
                                        "agency": ag_name,
                                        "date": date_val or None,
                                        "action": action_val or None,
                                        "_ts": time.time(),
                                    }
                                    _cache[f"wgb_rating_{country_code}_{ag_key}"] = cache_entry
                                    logger.info("[fixed_income] WGB rating %s %s: %s (date: %s, action: %s)",
                                                country_code, ag_name, rating_val, date_val, action_val)
                                break
                        i += 1
                    # Also store the "best" (S&P preferred) rating in the legacy key
                    sp_entry = _cache.get(f"wgb_rating_{country_code}_sp")
                    moodys_entry = _cache.get(f"wgb_rating_{country_code}_moodys")
                    best_entry = sp_entry or moodys_entry
                    if best_entry:
                        _cache[f"wgb_rating_{country_code}"] = best_entry
                elif data.get("lastRatingValue", "") not in ("-", "--", "---", ""):
                    # Fallback: just the lastRatingValue without date
                    _cache[f"wgb_rating_{country_code}"] = {
                        "rating": data["lastRatingValue"], "_ts": time.time()
                    }
                return bps

        logger.debug("[fixed_income] CDS %s: API returned empty lastCds", country_code)

    except Exception as e:
        logger.debug("[fixed_income] CDS fetch failed for %s: %s", country_code, e)

    return None


def _fetch_rating_with_date(country_code: str, issuer_name: str = "") -> dict:
    """
    Fetch current credit rating + date of last rating action.
    Sources (in order):
      1. In-process cache (from CDS API call — worldgovernmentbonds.com returns lastRatingValue)
      2. Serper Google Search (if SERPER_API_KEY set and has credits)
    Returns {"rating": str, "outlook": str, "date": str, "agency": str} or {}.
    """
    logger.info("[fixed_income] Rating: searching for %s/%s", country_code, issuer_name[:30] or "sovereign")

    # ── Check WGB rating cache first (populated by _fetch_sovereign_cds) ──────
    # WGB now caches per-agency entries with dates; prefer S&P, then Moody's, then Fitch
    for ag_key in ("sp", "moodys", "fitch"):
        wgb_entry = _cache.get(f"wgb_rating_{country_code}_{ag_key}")
        if wgb_entry and wgb_entry.get("rating") and wgb_entry.get("date"):
            result = {
                "rating":  wgb_entry["rating"],
                "agency":  wgb_entry.get("agency", "WGB"),
                "outlook": wgb_entry.get("outlook"),
                "date":    wgb_entry["date"],
                "action":  wgb_entry.get("action"),
            }
            logger.info("[fixed_income] Rating %s: %s %s (date: %s) from WGB cache",
                        country_code, wgb_entry.get("agency"), wgb_entry["rating"], wgb_entry["date"])
            return result

    wgb_fallback = _cache.get(f"wgb_rating_{country_code}")
    _wgb_rating = wgb_fallback.get("rating") if wgb_fallback else None
    # Continue to search for rating + date via DDG/Serper if no WGB entry with date

    serper_key = os.getenv("SERPER_API_KEY", "")

    cache_key = f"rating_{country_code}_{issuer_name[:20]}"
    entry = _cache.get(cache_key)
    if entry and time.time() - entry.get("_ts", 0) < 14400:
        return {k: v for k, v in entry.items() if k != "_ts"}

    try:
        import html as _html_mod2
        # Query: combine country/issuer with specific rating agency terms
        country_name = _WGB_COUNTRY_SLUGS.get(country_code, country_code).replace("-", " ").title()
        subject = issuer_name if issuer_name else country_name
        query = f'{subject} credit rating Moody\'s S&P Fitch downgrade upgrade 2024 2025 2026'

        full_text = ""

        # ── Try Serper first ──
        if serper_key:
            try:
                resp = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": query, "num": 6},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    full_text = (
                        data.get("answerBox", {}).get("answer", "") + " " +
                        data.get("answerBox", {}).get("snippet", "") + " " +
                        " ".join(r.get("snippet", "") + " " + r.get("title", "")
                                 for r in data.get("organic", []))
                    )
                    logger.info("[fixed_income] Rating Serper OK for %s", country_code)
                else:
                    logger.warning("[fixed_income] Rating Serper %d for %s — trying DDG",
                                   resp.status_code, country_code)
            except Exception as _se:
                logger.warning("[fixed_income] Rating Serper exception for %s: %s", country_code, _se)

        # ── DuckDuckGo HTML fallback ──
        if not full_text.strip():
            try:
                ddg_r = requests.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html",
                    },
                    timeout=12,
                )
                if ddg_r.status_code == 200:
                    raw_snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a', ddg_r.text, re.S)
                    raw_titls = re.findall(r'class="result__title"[^>]*>.*?<a[^>]*>(.*?)</a', ddg_r.text, re.S)
                    def _s(x): return _html_mod2.unescape(re.sub(r'<[^>]+>', ' ', x).strip())
                    full_text = " ".join(_s(x) for x in raw_titls + raw_snips)
                    logger.info("[fixed_income] Rating DDG-HTML: %d results for %s, len=%d",
                                len(raw_snips), country_code, len(full_text))
            except Exception as _de:
                logger.warning("[fixed_income] Rating DDG-HTML failed for %s: %s", country_code, _de)

        if not full_text.strip():
            return {}

        result = {}

        # Extract Moody's rating: Caa1, Caa2, Caa3, Ca, C, B1..Ba3, Baa1..Aaa
        # Broader pattern: find rating symbol near "Moody" anywhere in text
        moody_m = re.search(
            r'Moody\'?s?[^A-Za-z0-9]{0,30}'
            r'(Aaa|Aa[1-3]|A[1-3]|Baa[1-3]|Ba[1-3]|B[1-3]|Caa[1-3]|Ca|C)',
            full_text, re.I
        )
        if not moody_m:
            # Try reversed: rating then Moody's
            moody_m = re.search(
                r'(Aaa|Aa[1-3]|A[1-3]|Baa[1-3]|Ba[1-3]|B[1-3]|Caa[1-3]|Ca|C)'
                r'[^A-Za-z0-9]{0,30}Moody\'?s?',
                full_text, re.I
            )
        sp_m = re.search(
            r'(?:S&P|Standard\s+&?\s+Poor[\'s]{0,2})\s*(?:rates?|downgrades?|upgrades?)?[^A-Za-z0-9]{0,10}'
            r'(AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|D)',
            full_text, re.I
        )
        fitch_m = re.search(
            r'Fitch\s*(?:rates?|downgrades?|upgrades?)?[^A-Za-z0-9]{0,10}'
            r'(AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|RD|D)',
            full_text, re.I
        )

        if moody_m:
            result["rating"] = moody_m.group(1)
            result["agency"] = "Moody's"
        elif sp_m:
            result["rating"] = sp_m.group(1)
            result["agency"] = "S&P"
        elif fitch_m:
            result["rating"] = fitch_m.group(1)
            result["agency"] = "Fitch"

        # Extract outlook: Stable, Negative, Positive, Watch
        outlook_m = re.search(
            r'outlook\s+(?:is\s+)?(Stable|Negative|Positive|Developing|Watch\s+Negative|Watch\s+Positive)',
            full_text, re.I
        )
        if outlook_m:
            result["outlook"] = outlook_m.group(1).title()

        # Extract date of last action: "November 2024", "March 2025", "Jan 15, 2025"
        date_m = re.search(
            r'(?:downgrade|upgrade|affirm|assign|rate)[^\.]{0,60}'
            r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'(?:\s+\d{1,2},?)?\s+20\d{2})',
            full_text, re.I
        )
        if not date_m:
            # Simpler: just find a month+year near "rating"
            date_m = re.search(
                r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                r'(?:\s+\d{1,2},?)?\s+20\d{2})',
                full_text, re.I
            )
        if date_m:
            result["date"] = date_m.group(1).strip()

        # If no rating found from search, fall back to WGB-cached rating
        if not result.get("rating") and _wgb_rating:
            result["rating"] = _wgb_rating
            result["agency"] = "S&P"

        if result:
            result["_ts"] = time.time()
            _cache[cache_key] = result
            logger.info("[fixed_income] Rating %s/%s: %s", country_code, issuer_name[:20], result)
        return {k: v for k, v in result.items() if k != "_ts"}

    except Exception as e:
        logger.debug("[fixed_income] Rating fetch failed for %s: %s", country_code, e)
        # Fallback: return WGB cached rating without date
        if _wgb_rating:
            return {"rating": _wgb_rating, "agency": "S&P", "outlook": None, "date": None}
        return {}


def _fetch_market_price_and_ytm(isin: str, coupon: Optional[float],
                                 years_to_maturity: Optional[float]) -> dict:
    """
    Try to find the current market price and compute YTM.
    Uses Serper search to find yield/price from financial data sites.
    Falls back to approximate YTM formula if price found.

    YTM approximation:
      YTM ≈ [C + (F - P) / n] / [(F + P) / 2]
      where C = annual coupon, F = 100 (par), P = market price, n = years to maturity

    Returns {"price": float|None, "ytm_pct": float|None, "ytm_source": str}
    """
    result = {"price": None, "ytm_pct": None, "ytm_source": None}
    if not coupon:
        return result

    serper_key = os.getenv("SERPER_API_KEY", "")
    if serper_key:
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": f'"{isin}" bond price yield 2025 2026 site:bloomberg.com OR site:investing.com OR site:cbonds.com', "num": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                full_text = (
                    data.get("answerBox", {}).get("answer", "") + " " +
                    " ".join(r.get("snippet", "") + " " + r.get("title", "")
                             for r in data.get("organic", []))
                )

                # Look for "yield: X.XX%" or "YTM: X.XX%" patterns
                ytm_m = re.search(r'(?:yield|ytm|yld)[^0-9]{0,15}(\d{1,2}\.\d{1,3})\s*%', full_text, re.I)
                if ytm_m:
                    ytm_val = float(ytm_m.group(1))
                    if coupon * 0.3 <= ytm_val <= coupon * 3:   # sanity: within 3x coupon
                        result["ytm_pct"] = round(ytm_val, 3)
                        result["ytm_source"] = "Web (Bloomberg/Investing)"
                        logger.info("[fixed_income] YTM from Serper for %s: %.3f%%", isin, ytm_val)
                        return result

                # Look for "price: XX.XX" or "bid: XX.XX"
                price_m = re.search(r'(?:price|bid|ask|last)[^0-9]{0,10}(\d{2,3}\.?\d{0,3})', full_text, re.I)
                if price_m:
                    price_val = float(price_m.group(1))
                    if 40 <= price_val <= 130:   # sanity: reasonable bond price
                        result["price"] = price_val

        except Exception as e:
            logger.debug("[fixed_income] Price/YTM Serper failed for %s: %s", isin, e)

    # Approximate YTM from price (if found) or skip
    price = result.get("price")
    if price and coupon and years_to_maturity and years_to_maturity > 0:
        f = 100.0     # par value
        c = coupon    # annual coupon %
        p = price
        n = years_to_maturity
        # Standard approximation
        ytm_approx = (c + (f - p) / n) / ((f + p) / 2)
        result["ytm_pct"]    = round(ytm_approx, 3)
        result["ytm_source"] = f"Calculated (price={price:.2f})"
        logger.info("[fixed_income] YTM approx for %s: %.3f%% (price=%.2f)", isin, ytm_approx, price)

    return result


# ── Source 5: FX rates ────────────────────────────────────────────────────────

def _get_fx_rate(currency: str) -> Optional[float]:
    """Get USD/{currency} exchange rate."""
    if currency == "USD":
        return 1.0
    # Hard-coded pegs
    pegs = {"AED": 3.6725, "SAR": 3.75, "KWD": 0.3070, "QAR": 3.64, "BHD": 0.376, "OMR": 0.385}
    if currency in pegs:
        return pegs[currency]
    try:
        resp = requests.get(
            f"https://open.er-api.com/v6/latest/USD",
            timeout=8,
        )
        if resp.status_code == 200:
            return resp.json().get("rates", {}).get(currency)
    except Exception:
        pass
    return None


# ── Main data assembler ───────────────────────────────────────────────────────

def get_instrument_data(isin: str, force_refresh: bool = False,
                        hint_text: str = "") -> dict:
    """
    Fetch and assemble all data for a given ISIN.
    Cached for 1 hour.

    Returns:
    {
      "isin": str,
      "name": str,
      "issuer": str,
      "security_type": str,        # "Corp Bond", "Govt Bond", etc.
      "market_sector": str,        # "Corp", "Govt", "Mtge"
      "exchange": str,             # "DIFX", "LSE", etc.
      "currency": str,
      "coupon": float | None,
      "maturity": str | None,      # "YYYY-MM-DD"
      "years_to_maturity": float | None,
      "is_sukuk": bool,
      "sukuk_structure": str | None,
      "country_code": str,         # 2-letter ISIN prefix
      "country_rating": str,
      "credit_score": int,         # 0-30
      "benchmarks": dict,          # {label: yield_pct}
      "fx_rate": float | None,     # USD/{currency}
      "source": str,
      "fetched_at": str,
      "error": str | None,
    }
    """
    cache_key = f"fi_{isin}"
    if not force_refresh and cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["_ts"] < _CACHE_TTL:
            # Even on cache hit: re-apply country inference from hint_text
            # so XS bonds get correct country_code if user mentioned country
            if hint_text and entry.get("country_code") in ("XS", "XD"):
                inferred = _infer_country_code(hint_text)
                if inferred:
                    entry = dict(entry)  # shallow copy to avoid mutating cache
                    entry["country_code"] = inferred
                    logger.info("[fixed_income] Cache hit: country re-inferred from hint: XS → %s", inferred)
            return entry

    result = {
        "isin": isin,
        "name": None,
        "issuer": None,
        "security_type": None,
        "market_sector": None,
        "exchange": None,
        "currency": "USD",  # default to USD (most Eurobonds/Sukuk are USD)
        "coupon": None,
        "maturity": None,
        "years_to_maturity": None,
        "is_sukuk": False,
        "sukuk_structure": None,
        "country_code": isin[:2].upper(),
        "country_rating": "--",
        "rating_date": None,        # NEW: date of last rating action
        "rating_outlook": None,     # NEW: Stable / Negative / Positive
        "rating_agency": None,      # NEW: Moody's / S&P / Fitch
        "cds_spread_bps": None,     # NEW: 5-year sovereign CDS in bps
        "market_price": None,       # NEW: current market price (% of par)
        "ytm_pct": None,            # NEW: yield to maturity %
        "ytm_source": None,         # NEW: how YTM was derived
        "credit_score": 15,
        "benchmarks": {},
        "fx_rate": None,
        "source": None,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "error": None,
        "_ts": time.time(),
    }

    # ── Step 0: Seed data from hint_text (user's message) ────────────────────
    # Extracts coupon, maturity, country, issuer from what the user typed.
    # This is free, instant, and works even when all APIs are unavailable.
    if hint_text:
        hint_parsed = _parse_name_components(hint_text)
        # Coupon: e.g. user wrote "7.95%" or "7.95% coupon"
        if hint_parsed.get("coupon"):
            result["coupon"] = hint_parsed["coupon"]
            logger.info("[fixed_income] Coupon seeded from hint_text: %.4f%%", result["coupon"])
        # Maturity: e.g. user wrote "matures 2026" or "due April 2026"
        if hint_parsed.get("maturity"):
            result["maturity"] = hint_parsed["maturity"]
            logger.info("[fixed_income] Maturity seeded from hint_text: %s", result["maturity"])
        # Country inference for XS/XD ISINs
        if result["country_code"] in ("XS", "XD"):
            inferred = _infer_country_code(hint_text)
            if inferred:
                logger.info("[fixed_income] Country inferred from hint_text: %s → %s",
                            result["country_code"], inferred)
                result["country_code"] = inferred

    # ── Step 1: OpenFIGI ──────────────────────────────────────────────────────
    figi_data = _fetch_openfigi(isin)
    if figi_data:
        result["name"]          = figi_data.get("name", "")
        result["security_type"] = figi_data.get("securityType", "")
        result["market_sector"] = figi_data.get("marketSector", "")
        result["exchange"]      = figi_data.get("exchCode", "")
        result["source"]        = "OpenFIGI"

        # Parse name for components
        parsed = _parse_name_components(result["name"])
        result["issuer"]          = parsed.get("issuer") or result["issuer"]
        result["coupon"]          = parsed.get("coupon") or result["coupon"]
        result["maturity"]        = parsed.get("maturity") or result["maturity"]
        result["is_sukuk"]        = parsed.get("is_sukuk", False)
        result["sukuk_structure"] = parsed.get("sukuk_structure")

    # ── Step 2: FMP enrichment (coupon/maturity if missing) ───────────────────
    if not result["coupon"] or not result["maturity"]:
        fmp_data = _fetch_fmp_bond(isin)
        if fmp_data:
            result["coupon"]   = result["coupon"]   or fmp_data.get("coupon")
            result["maturity"] = result["maturity"] or fmp_data.get("maturityDate")
            result["currency"] = fmp_data.get("currency") or result["currency"]
            if not result["name"]:
                result["name"] = fmp_data.get("name", "")
            if not result["source"]:
                result["source"] = "FMP"

    # ── Step 3: Serper web search fallback (when APIs return nothing) ─────────
    _serper_ran = False
    if not result["name"] or not result["coupon"]:
        serper_data = _serper_isin_lookup(isin)
        _serper_ran = True
        if serper_data:
            result["name"]     = result["name"]     or serper_data.get("name", "")
            result["coupon"]   = result["coupon"]   or serper_data.get("coupon")
            result["maturity"] = result["maturity"] or serper_data.get("maturity")
            if serper_data.get("is_sukuk"):
                result["is_sukuk"] = True
            # Country inference: override "XS" prefix with real country if detectable
            inferred_cc = serper_data.get("inferred_country_code")
            if inferred_cc and result["country_code"] in ("XS", "XD"):
                logger.info("[fixed_income] Country inferred from Serper: %s → %s", result["country_code"], inferred_cc)
                result["country_code"] = inferred_cc
            if not result["source"]:
                result["source"] = "Web Search"
            elif result["source"]:
                result["source"] += " + Web Search"
    # Always run Serper for country inference on XS bonds even if we have name/coupon
    if not _serper_ran and result["country_code"] in ("XS", "XD"):
        serper_data = _serper_isin_lookup(isin)
        inferred_cc = serper_data.get("inferred_country_code") if serper_data else None
        if inferred_cc:
            logger.info("[fixed_income] Country inferred (XS supplemental): %s", inferred_cc)
            result["country_code"] = inferred_cc

    # ── Step 4: Years to maturity ─────────────────────────────────────────────
    if result["maturity"]:
        try:
            mat_dt = datetime.strptime(result["maturity"][:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta = mat_dt - datetime.now(timezone.utc)
            result["years_to_maturity"] = round(delta.days / 365.25, 2)
        except Exception:
            pass

    # ── Step 5: Country / sovereign rating ───────────────────────────────────
    cc = result["country_code"]
    if cc in _COUNTRY_RATINGS:
        rating, score = _COUNTRY_RATINGS[cc]
        result["country_rating"] = rating
        result["credit_score"]   = score
    elif cc == "XS":
        # International — try to infer from exchange
        exch = (result["exchange"] or "").upper()
        if "DIFX" in exch or "NASDAQ DUBAI" in exch:
            result["country_rating"] = "AA (UAE issuer inferred)"
            result["credit_score"]   = 26
        elif "LSE" in exch:
            result["country_rating"] = "AA (UK-listed)"
            result["credit_score"]   = 27

    # ── Step 5b: CDS spread — overrides static credit_score if available ──────
    cds_bps = _fetch_sovereign_cds(cc)
    result["cds_spread_bps"] = cds_bps
    # Pull WGB-provided default probability (more accurate than formula)
    cds_cache = _cache.get(f"cds_{cc}", {})
    result["cds_default_prob_5y"] = cds_cache.get("default_prob")
    if cds_bps is not None:
        # Adjust credit_score: CDS is a market-based signal, more accurate than static table
        if   cds_bps < 30:    result["credit_score"] = min(30, result["credit_score"] + 2)
        elif cds_bps < 50:    pass                     # no change
        elif cds_bps < 100:   result["credit_score"] = max(0, result["credit_score"] - 1)
        elif cds_bps < 200:   result["credit_score"] = max(0, result["credit_score"] - 3)
        elif cds_bps < 500:   result["credit_score"] = max(0, result["credit_score"] - 6)
        elif cds_bps < 1000:  result["credit_score"] = max(0, result["credit_score"] - 10)
        else:                 result["credit_score"] = max(0, result["credit_score"] - 14)

    # ── Step 5c: Rating with date (Serper) ────────────────────────────────────
    rating_info = _fetch_rating_with_date(cc, result.get("issuer") or "")
    if rating_info:
        if rating_info.get("rating"):
            # Prefer live rating over static table
            result["country_rating"] = rating_info["rating"]
        result["rating_date"]    = rating_info.get("date")
        result["rating_outlook"] = rating_info.get("outlook")
        result["rating_agency"]  = rating_info.get("agency")

    # ── Step 6: Benchmark yields ──────────────────────────────────────────────
    ytm = result["years_to_maturity"] or 5.0
    result["benchmarks"] = _fetch_benchmarks(result["currency"], ytm)

    # ── Step 6b: Market price & YTM ───────────────────────────────────────────
    price_ytm = _fetch_market_price_and_ytm(
        isin, result["coupon"], result["years_to_maturity"]
    )
    result["market_price"] = price_ytm.get("price")
    result["ytm_pct"]      = price_ytm.get("ytm_pct")
    result["ytm_source"]   = price_ytm.get("ytm_source")

    # ── Step 7: FX rate ───────────────────────────────────────────────────────
    result["fx_rate"] = _get_fx_rate(result["currency"])

    # ── Step 8: Sukuk fallback detection from ISIN prefix ─────────────────────
    # UAE/Saudi Eurobonds are often Sukuk even if OpenFIGI doesn't label them
    if not result["is_sukuk"] and cc in ("AE", "SA", "KW", "QA", "BH", "OM"):
        sec_type = (result["security_type"] or "").lower()
        name_up  = (result["name"] or "").upper()
        if "sukuk" in name_up or "trust" in name_up:
            result["is_sukuk"] = True

    _cache[cache_key] = result
    logger.info(
        "[fixed_income] %s: %s | %s | coupon=%.2f%% | maturity=%s | sukuk=%s",
        isin,
        result["name"] or "?",
        result["security_type"] or "?",
        result["coupon"] or 0,
        result["maturity"] or "?",
        result["is_sukuk"],
    )
    return result
