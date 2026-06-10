"""
build_investing_ids.py
======================
يقرأ الـ Excel files ويبني UAE_INVESTING كامل
Strategy: يستخدم الأسعار من الـ Excel كـ fingerprint لمطابقة الـ IDs

Steps:
1. استخرج أسهم UAE مع أسعارها من الـ Excel
2. ابني slug من اسم الشركة وجرب fetch الصفحة على investing.com
3. لو مش قادر يجيب ID من الصفحة، يعمل price probe
4. يحفظ النتايج في investing_ids_uae.json
"""
import cloudscraper, re, json, time, sys
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

# ─── Known IDs (confirmed) ────────────────────────────────────────────────────
KNOWN_IDS = {
    # DFM — already in UAE_INVESTING
    "AIRARABI.DU":  {"id": "12530",   "ref": "airarabi-historical-data"},
    "EMAAR.DU":     {"id": "1055159", "ref": "emaar-properties-historical-data"},
    "EMAARDEV.DU":  {"id": "1055160", "ref": "emaar-development-historical-data"},
    "DIB.DU":       {"id": "941311",  "ref": "dubai-islamic-bank-historical-data"},
    "ENBD.DU":      {"id": "12548",   "ref": "emirates-nbd-historical-data"},
    "DEWA.DU":      {"id": "941326",  "ref": "dubai-electricity-water-historical-data"},
    "SALIK.DU":     {"id": "1194944", "ref": "salik-historical-data"},
    "DU.DU":        {"id": "941321",  "ref": "emirates-integrated-telecommunications-historical-data"},
    "CBD.DU":       {"id": "941308",  "ref": "commercial-bank-of-dubai-historical-data"},
    "MASQ.DU":      {"id": "941323",  "ref": "mashreqbank-historical-data"},
    "NCC.DU":       {"id": "941325",  "ref": "national-cement-historical-data"},
    "SALAMA.DU":    {"id": "941328",  "ref": "islamic-arab-insurance-historical-data"},
    "TABREED.DU":   {"id": "941329",  "ref": "national-central-cooling-tabreed-historical-data"},
    "TECOM.DU":     {"id": "1192698", "ref": "tecom-historical-data"},
    "EMPOWER.DU":   {"id": "1197172", "ref": "empower-historical-data"},
    "TAALEEM.DU":   {"id": "1198050", "ref": "taaleem-historical-data"},
    "ALANSARI.DU":  {"id": "1201945", "ref": "alansari-historical-data"},
    "DUBAITAXI.DU": {"id": "1209220", "ref": "dubaitaxi-historical-data"},
    "PARKIN.DU":    {"id": "1212798", "ref": "parkin-historical-data"},
    "SPINNEYS.DU":  {"id": "1214529", "ref": "spinneys-historical-data"},
    "TALABAT.DU":   {"id": "1224079", "ref": "talabat-historical-data"},
    "SHUAA.DU":     {"id": "12557",   "ref": "shuaa-historical-data"},
    "AMLAK.DU":     {"id": "40413",   "ref": "amlak-historical-data"},
    "DEYAAR.DU":    {"id": "12538",   "ref": "deyaar-historical-data"},
    "DFM.DU":       {"id": "12539",   "ref": "dfm-historical-data"},
    "DINV.DU":      {"id": "12540",   "ref": "dubai-investments-historical-data"},
    # ADX — confirmed
    "FAB.AE":       {"id": "999060",  "ref": "first-abu-dhabi-bank-historical-data"},
    "ALDAR.AE":     {"id": "941317",  "ref": "aldar-properties-historical-data"},
    "TAQA.AE":      {"id": "941330",  "ref": "taqa-historical-data"},
    "ADNOCDIST.AE": {"id": "1055158", "ref": "national-oil-historical-data"},
}

def name_to_slug(name):
    """يحول اسم الشركة لـ slug على investing.com"""
    slug = name.lower()
    slug = re.sub(r'\s+pjsc|\s+psc|\s+plc|\s+p\.j\.s\.c\.?|\s+p\.s\.c\.?', '', slug)
    slug = re.sub(r'\s+co\.?$|\s+company$|\s+corp\.?$|\s+corporation$', '', slug)
    slug = re.sub(r'\band\b', 'and', slug)
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    return slug

def get_id_from_page(slug):
    """يجيب الـ curr_id من صفحة investing.com"""
    urls = [
        f"https://www.investing.com/equities/{slug}",
        f"https://www.investing.com/equities/{slug}-historical-data",
    ]
    for url in urls:
        try:
            r = scraper.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                "Accept": "text/html,application/xhtml+xml",
                "Referer": "https://www.investing.com/equities/united-arab-emirates",
            }, timeout=12)
            if r.status_code != 200:
                continue
            html = r.text
            # Try multiple patterns
            for pat in [
                r'data-pair-id=["\'](\d+)["\']',
                r'"pairId"\s*:\s*(\d+)',
                r'curr_id.*?value=["\'](\d+)["\']',
                r'"id"\s*:\s*(\d{6,7})',
            ]:
                m = re.search(pat, html)
                if m:
                    cid = m.group(1)
                    if 10000 <= int(cid) <= 9999999:
                        # Verify it's not the generic site ID
                        if cid not in ["2006651", "2000000"]:
                            return cid, slug
            time.sleep(0.5)
        except:
            pass
    return None, None

def probe_verify(curr_id, expected_price, tolerance=0.30):
    """يتحقق من الـ ID بمقارنة السعر"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://www.investing.com/equities/",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    # Use recent dates
    data = {"curr_id": str(curr_id),
            "st_date": "01/01/2026", "end_date": "03/10/2026",
            "interval_sec": "Daily", "sort_col": "date", "sort_ord": "DESC",
            "action": "historical_data"}
    try:
        r = scraper.post("https://www.investing.com/instruments/HistoricalDataAjax",
                         headers=headers, data=data, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("#curr_table tbody tr")
        if rows:
            cols = rows[0].find_all("td")
            if len(cols) >= 3 and "No results" not in cols[0].text:
                price = float(cols[1].text.strip().replace(",", ""))
                # Check if price is within tolerance
                if expected_price > 0:
                    diff = abs(price - expected_price) / expected_price
                    if diff <= tolerance:
                        return True, price
                return False, price
    except:
        pass
    return False, 0

def price_probe_range(id_range, expected_price, tolerance=0.15, delay=0.06):
    """يجرب range من IDs ويلاقي اللي سعره بالقرب من المتوقع"""
    for curr_id in id_range:
        ok, price = probe_verify(curr_id, expected_price, tolerance)
        if ok:
            return curr_id, price
        time.sleep(delay)
    return None, 0


# ─── Read Excel ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Reading Excel files...")
df = pd.read_excel('/home/ubuntu/investwise/Final_Stocks_Report_All_Countries.xlsx')
uae = df[df['Exchange'].str.contains('Abu Dhabi|Dubai', na=False, case=False)].copy()
uae['price_aed'] = uae['Last Trade Price'].astype(str).str.extract(r'([\d.,]+)').astype(float).values
uae = uae.dropna(subset=['Ticker', 'price_aed'])

print(f"UAE stocks: {len(uae)}")
print()

# ─── Build mapping ─────────────────────────────────────────────────────────────
results = dict(KNOWN_IDS)
pending = []

for _, row in uae.iterrows():
    ticker_raw = str(row['Ticker']).strip().upper()
    name = str(row['Name']).strip()
    price = float(row['price_aed'])
    exchange = str(row['Exchange'])

    # Determine .DU vs .AE
    if 'Dubai' in exchange and 'Abu Dhabi' not in exchange:
        ticker = f"{ticker_raw}.DU"
    elif 'Abu Dhabi' in exchange and 'Dubai' not in exchange:
        ticker = f"{ticker_raw}.AE"
    else:
        # Both — check ticker list
        if ticker_raw in ['IHC','FAB','ALDAR','TAQA','ADNOCDIST','ADNOCGAS','ADNOCDRILL',
                           'ADIB','ADCB','EAND','ALPHADHABI','FERTIGLOBE','NMDC','DANA',
                           'RAKBANK','AGTHIA','WAHA','IH','ADNIC','ADNTC','MODON','ARAM',
                           'NOGA','ESHRAQ','MANAZEL','HILY','ADNH','CBI','BANKSHJ','APEX',
                           'RAKPROP','RAKCEC2','NBQ','GPI','GCC','METHAQ','NBF','UAB','FH',
                           'EIC','ADAVIATION','HAYAH','QCC','EDC','BNI','NBM','ADSB','EMSTEEL',
                           'BOS','SCRJ','RAKCEC','GHITHA','STALLION','EASYLEASE','PALMS','ADIC',
                           'SIC','AFNIC','AWNIC','AAIA','UI','AKI','RAPCO','FCI','SUDATEL',
                           'OOREDOO','OEH','FBI','NMDC','ARAM','MODON','NOGA']:
            ticker = f"{ticker_raw}.AE"
        else:
            ticker = f"{ticker_raw}.DU"

    if ticker in results:
        print(f"  ✅ Already have: {ticker}")
        continue

    pending.append({
        'ticker': ticker,
        'ticker_raw': ticker_raw,
        'name': name,
        'price': price,
        'slug': name_to_slug(name),
    })

print(f"\nNeed to find IDs for {len(pending)} tickers")
print()

# ─── Find IDs ─────────────────────────────────────────────────────────────────
not_found = []

for i, stock in enumerate(pending):
    ticker = stock['ticker']
    name = stock['name']
    price = stock['price']
    slug = stock['slug']

    print(f"[{i+1}/{len(pending)}] {ticker} ({name}) price={price:.2f} AED")

    # 1. Try page fetch
    cid, found_slug = get_id_from_page(slug)
    if cid:
        ok, actual = probe_verify(cid, price, tolerance=0.35)
        if ok:
            results[ticker] = {"id": cid, "ref": f"{found_slug}-historical-data"}
            print(f"    ✅ Page fetch: id={cid}, price={actual:.2f}")
            continue
        else:
            print(f"    ⚠ Got id={cid} but price mismatch ({actual:.2f} vs {price:.2f})")

    time.sleep(0.5)

    # 2. Try alternate slugs
    alt_slugs = [
        stock['ticker_raw'].lower(),
        f"abu-dhabi-{slug}" if '.AE' in ticker else f"dubai-{slug}",
        slug.replace('-pjsc','').replace('-psc',''),
    ]
    found = False
    for alt in alt_slugs:
        cid, found_slug = get_id_from_page(alt)
        if cid:
            ok, actual = probe_verify(cid, price, tolerance=0.35)
            if ok:
                results[ticker] = {"id": cid, "ref": f"{found_slug}-historical-data"}
                print(f"    ✅ Alt slug '{alt}': id={cid}, price={actual:.2f}")
                found = True
                break
        time.sleep(0.3)

    if not found:
        not_found.append(stock)
        print(f"    ❌ Not found via page fetch")

    # Save intermediate results
    if i % 10 == 0:
        with open('/home/ubuntu/investwise/investing_ids_uae.json', 'w') as f:
            json.dump(results, f, indent=2)

# ─── Price probe for remaining not_found ─────────────────────────────────────
print(f"\n\n=== Price probe for {len(not_found)} remaining stocks ===")

# Cluster by price to find ranges
price_probe_map = {
    # Range → ID search range (based on previous probing)
    (0.1, 2.0):   list(range(941270, 941350)) + list(range(999050, 999200)),
    (2.0, 5.0):   list(range(941270, 941350)) + list(range(999050, 999200)) + list(range(1055140, 1055250)),
    (5.0, 15.0):  list(range(941270, 941350)) + list(range(999050, 999200)),
    (15.0, 50.0): list(range(941270, 941350)) + list(range(999050, 999200)),
    (50.0, 500.0):list(range(941270, 941350)) + list(range(999050, 999200)),
}

used_ids = set(int(v["id"]) for v in results.values())

for stock in not_found:
    ticker = stock['ticker']
    price = stock['price']
    print(f"\n  Probing for {ticker} (price={price:.2f} AED)")

    # Determine which ranges to probe
    probe_ids = []
    for (lo, hi), id_list in price_probe_map.items():
        if lo <= price <= hi:
            probe_ids = id_list
            break

    if not probe_ids:
        probe_ids = list(range(999050, 999200)) + list(range(941270, 941360))

    for curr_id in probe_ids:
        if curr_id in used_ids:
            continue
        ok, actual = probe_verify(curr_id, price, tolerance=0.12)
        if ok:
            results[ticker] = {"id": str(curr_id), "ref": f"ae-{ticker.replace('.AE','').replace('.DU','').lower()}-historical-data"}
            used_ids.add(curr_id)
            print(f"    ✅ Found: id={curr_id}, price={actual:.2f}")
            break
        time.sleep(0.06)
    else:
        print(f"    ❌ Still not found")

# ─── Final save ───────────────────────────────────────────────────────────────
with open('/home/ubuntu/investwise/investing_ids_uae.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\n{'='*60}")
print(f"DONE: Found {len(results)} IDs total")
print(f"Missing: {[s['ticker'] for s in not_found if s['ticker'] not in results]}")
print(f"Results saved to investing_ids_uae.json")
