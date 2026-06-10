"""
Fix UAE Stock Data
1. Find DAMAC investing.com ID
2. Add live price scraper for UAE stocks
3. Patch market_data_engine.py
"""
import cloudscraper
from bs4 import BeautifulSoup
import re, json, time

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Step 1: Find investing.com IDs for missing stocks ─────────────────────────

MISSING_STOCKS = {
    "DAMAC.DU": "damac-properties",
    "DUBAITAXI.DU": "dubai-taxi-company",
    "SPINNEYS.DU": "spinneys-1961-holding",
    "ALEC.DU": "alec-holdings",
}

def find_investing_id(slug: str) -> dict:
    """Get curr_id from investing.com equities page."""
    url = f"https://www.investing.com/equities/{slug}"
    try:
        r = scraper.get(url, headers=HEADERS, timeout=15)
        # Look for pair_id in page source
        match = re.search(r'"pairId"\s*:\s*(\d+)', r.text)
        if match:
            return {"id": match.group(1), "ref": f"{slug}-historical-data"}
        # Try another pattern
        match = re.search(r"pair_id\s*=\s*(\d+)", r.text)
        if match:
            return {"id": match.group(1), "ref": f"{slug}-historical-data"}
        # Try data attribute
        soup = BeautifulSoup(r.text, "html.parser")
        el = soup.find(attrs={"data-pair-id": True})
        if el:
            return {"id": el["data-pair-id"], "ref": f"{slug}-historical-data"}
        print(f"  ⚠ Could not find ID for {slug}")
        return {}
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {}


# ── Step 2: Live price from investing.com quote page ──────────────────────────

def get_live_price_investing(ticker: str, ref: str) -> float:
    """
    Scrape live/last price from investing.com equities page.
    More reliable than Yahoo for DFM/ADX stocks.
    """
    slug = ref.replace("-historical-data", "")
    url  = f"https://www.investing.com/equities/{slug}"
    try:
        r = scraper.get(url, headers={**HEADERS,
            "Referer": "https://www.investing.com/"}, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")

        # Try multiple selectors
        selectors = [
            '[data-test="instrument-price-last"]',
            '.instrument-price_last__KQzyA',
            'span[class*="last-price"]',
            '[class*="priceText"]',
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and el.text.strip():
                val = el.text.strip().replace(",", "")
                try:
                    return float(val)
                except ValueError:
                    continue

        # Fallback: regex search for price pattern in JSON/script
        match = re.search(r'"last"\s*:\s*"?([\d.]+)"?', r.text)
        if match:
            return float(match.group(1))

        return 0.0
    except Exception as e:
        print(f"  Live price error {ticker}: {e}")
        return 0.0


# ── Step 3: Patch market_data_engine.py ───────────────────────────────────────

def patch_engine():
    path = "/home/ubuntu/investwise/market_data_engine.py"
    try:
        with open(path, "r") as f:
            src = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return

    # Find missing IDs
    new_entries = {}
    for ticker, slug in MISSING_STOCKS.items():
        if f'"{ticker}"' not in src:
            print(f"🔍 Finding ID for {ticker} ({slug})...")
            info = find_investing_id(slug)
            if info:
                new_entries[ticker] = info
                print(f"  ✅ {ticker}: id={info['id']}")
            time.sleep(2)

    if new_entries:
        # Insert after last .DU entry
        insert_block = ""
        for ticker, info in new_entries.items():
            insert_block += f'    "{ticker}": {{"id": "{info["id"]}", "ref": "{info["ref"]}"}},\n'

        # Find insertion point — after EMAAR.DU line
        src = src.replace(
            '    "EMAAR.DU": {"id": "1055159", "ref": "emaar-properties-historical-data"},',
            f'    "EMAAR.DU": {{"id": "1055159", "ref": "emaar-properties-historical-data"}},\n{insert_block}'
        )
        with open(path, "w") as f:
            f.write(src)
        print(f"\n✅ Added {len(new_entries)} stocks to UAE_INVESTING")
    else:
        print("\n✅ All stocks already in UAE_INVESTING")


# ── Step 4: Patch market_data.py — UAE live price fallback ───────────────────

def patch_market_data():
    path = "/home/ubuntu/investwise/market_data.py"
    try:
        with open(path, "r") as f:
            src = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return

    # Add UAE live price function if not already there
    if "get_uae_live_price" in src:
        print("✅ UAE live price already patched")
        return

    uae_func = '''
import cloudscraper as _cloudscraper
import re as _re
from bs4 import BeautifulSoup as _BS

def get_uae_live_price(ticker: str) -> float:
    """Scrape live price for UAE stocks from investing.com."""
    try:
        from market_data_engine import UAE_INVESTING
        info = UAE_INVESTING.get(ticker)
        if not info:
            return 0.0
        slug = info["ref"].replace("-historical-data", "")
        url  = f"https://www.investing.com/equities/{slug}"
        sc   = _cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        r = sc.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Accept-Language": "en-US,en;q=0.9",
        }, timeout=12)
        soup = _BS(r.text, "html.parser")
        for sel in ['[data-test="instrument-price-last"]',
                    '.instrument-price_last__KQzyA',
                    'span[class*="last-price"]']:
            el = soup.select_one(sel)
            if el and el.text.strip():
                try:
                    return float(el.text.strip().replace(",", ""))
                except ValueError:
                    pass
        m = _re.search(r\'"last"\\s*:\\s*"?([\\d.]+)"?\', r.text)
        if m:
            return float(m.group(1))
        return 0.0
    except Exception:
        return 0.0

'''

    # Patch get_realtime_quote to use UAE live price for DFM/ADX
    old_func = '''def get_realtime_quote(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)'''

    new_func = '''def get_realtime_quote(ticker):
    # UAE stocks: use Investing.com scraper (Yahoo doesn't support DFM/ADX well)
    if ticker.endswith(".DU") or ticker.endswith(".AE"):
        live = get_uae_live_price(ticker)
        if live > 0:
            return {"ticker": ticker, "price": round(live, 2),
                    "change_pct": 0.0, "source": "investing.com"}
    try:
        ticker_obj = yf.Ticker(ticker)'''

    src = uae_func + src.replace(old_func, new_func)

    with open(path, "w") as f:
        f.write(src)
    print("✅ market_data.py patched — UAE live price via Investing.com")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔧 Fixing UAE Stock Data...\n")

    print("Step 1: Finding missing investing.com IDs...")
    patch_engine()

    print("\nStep 2: Patching live price for UAE stocks...")
    patch_market_data()

    print("\nStep 3: Testing DAMAC.DU live price...")
    # Test with known stock first
    price = get_live_price_investing("EMAAR.DU", "emaar-properties-historical-data")
    print(f"  EMAAR.DU test: {price} AED")

    price = get_live_price_investing("DAMAC.DU", "damac-properties-historical-data")
    print(f"  DAMAC.DU: {price} AED")

    print("\n✅ Done!")
