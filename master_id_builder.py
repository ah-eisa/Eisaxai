"""
master_id_builder.py
====================
يبني UAE_INVESTING كامل باستخدام:
1. أسعار 2026 من الـ Excel كـ fingerprint
2. يعمل probe للـ ID ranges المعروفة بتاريخ حالي
3. يطابق الأسعار → يحدد كل ID لكل سهم
4. يحفظ النتايج ويحدث market_data_engine.py
"""
import cloudscraper, time, json, sys, re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

RESULTS_FILE = "/home/ubuntu/investwise/all_uae_ids.json"

# ─── Probe function ───────────────────────────────────────────────────────────
def probe_current(curr_id):
    """يجيب آخر سعر من investing.com (2025-2026)"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://www.investing.com/equities/",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "curr_id": str(curr_id),
        "st_date": "01/01/2026", "end_date": "03/12/2026",
        "interval_sec": "Daily", "sort_col": "date", "sort_ord": "DESC",
        "action": "historical_data"
    }
    try:
        r = scraper.post("https://www.investing.com/instruments/HistoricalDataAjax",
                         headers=headers, data=data, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("#curr_table tbody tr")
        if rows:
            cols = rows[0].find_all("td")
            if len(cols) >= 3 and "No results" not in cols[0].text:
                close_str = cols[1].text.strip().replace(",", "")
                try:
                    return float(close_str)
                except:
                    pass
    except:
        pass
    return None

# ─── Read Excel ───────────────────────────────────────────────────────────────
print("Reading Excel files...")
df = pd.read_excel('/home/ubuntu/investwise/Final_Stocks_Report_All_Countries.xlsx')

uae_df = df[df['Exchange'].str.contains('Abu Dhabi|Dubai', na=False, case=False)].copy()
uae_df['price'] = uae_df['Last Trade Price'].astype(str).str.extract(r'([\d.,]+)').iloc[:,0].str.replace(',','').astype(float)
uae_df = uae_df.dropna(subset=['Ticker','price'])

# Build price map: ticker → price
# Map Excel tickers to our .DU/.AE format
ADX_TICKERS = {
    'IHC','TAQA','ADNOCGAS','FAB','EAND','ADCB','ADIB','ALPHADHABI','ADNOCDRILL',
    'BOROUGE','ALDAR','ADNOCDIST','ADAVIATION','FERTIGLB','ADPORTS','PRESIGHT',
    'RAKBANK','ADNIC','UAB','NMDC','NMDCENR','APEX','NBF','NBQ','ADNH','ADNHC',
    'ADSB','GHITHA','PALMS','ESG','RAKCEC','WAHA','AGTHIA','ADNTC','BNI','EMSTEEL',
    'BOS','SCRJ','RAKCEC2','GPI','GCC','METHAQ','AKI','FCI','ESHRAQ','MANAZEL',
    'ARAM','MODON','NOGA','FH','EIC','HAYAH','QCC','DRIVE','FBI','OEH','OOREDOO',
    'UI','RAPCO','AAIA','SIC','AFNIC','AWNIC','ADIC','EASYLEASE','STALLION','DANA',
    'PUREHEALTH','ICAP','2POINTZERO','SPACE42','LULU','ALEFEDT','BURJEEL','INVICTUS',
    'AGILITY','ASM','UNIONCOOP','PHX','MAIR','CBI','JULPHAR','ALPHADATA','GMPC',
}

DFM_TICKERS = {
    'ENBD','EMAR','DEWAA','DISB','SALIK','DU','CBD','MASB','EMAARDEV','DFM','TABR',
    'EMPOWER','ALANSARI','PARKIN','SPINNEYS','TALABAT','TAALEEM','ALEC','DINV',
    'GNAV','ARMX','AMLK','DEYR','AJBNK','SUKOON','DRC','DINC','NCC','UPRO','AMANT',
    'E7','DTC','TECOM','AIRA','ASM2',
}

uae_price_map = {}
for _, row in uae_df.iterrows():
    t = str(row['Ticker']).strip().upper()
    p = float(row['price'])
    if t in ADX_TICKERS:
        uae_price_map[f"{t}.AE"] = p
    elif t in DFM_TICKERS:
        uae_price_map[f"{t}.DU"] = p
    else:
        # Guess based on price/name — default to .AE
        uae_price_map[f"{t}.AE"] = p

print(f"UAE stocks to map: {len(uae_price_map)}")

# ─── Known confirmed IDs ───────────────────────────────────────────────────────
confirmed = {
    # DFM — from UAE_INVESTING (already verified)
    "AIRARABI.DU":  {"id": "12530",   "ref": "airarabi-historical-data"},
    "AJMANBANK.DU": {"id": "12531",   "ref": "ajmanbank-historical-data"},
    "ALANSARI.DU":  {"id": "1201945", "ref": "alansari-historical-data"},
    "ALEC.DU":      {"id": "1215000", "ref": "alec-holdings-historical-data"},
    "AMANAT.DU":    {"id": "945149",  "ref": "amanat-historical-data"},
    "AMLAK.DU":     {"id": "40413",   "ref": "amlak-historical-data"},
    "ARMX.DU":      {"id": "12534",   "ref": "armx-historical-data"},
    "CBD.DU":       {"id": "941308",  "ref": "commercial-bank-of-dubai-historical-data"},
    "DEWA.DU":      {"id": "941326",  "ref": "dubai-electricity-water-historical-data"},
    "DEYAAR.DU":    {"id": "12538",   "ref": "deyaar-historical-data"},
    "DFM.DU":       {"id": "12539",   "ref": "dfm-historical-data"},
    "DIB.DU":       {"id": "941311",  "ref": "dubai-islamic-bank-historical-data"},
    "DINV.DU":      {"id": "12540",   "ref": "dubai-investments-historical-data"},
    "DSI.DU":       {"id": "12544",   "ref": "dsi-historical-data"},
    "DU.DU":        {"id": "941321",  "ref": "emirates-integrated-telecommunications-historical-data"},
    "DUBAITAXI.DU": {"id": "1209220", "ref": "dubaitaxi-historical-data"},
    "EMAAR.DU":     {"id": "1055159", "ref": "emaar-properties-historical-data"},
    "EMAARDEV.DU":  {"id": "1055160", "ref": "emaar-development-historical-data"},
    "EMPOWER.DU":   {"id": "1197172", "ref": "empower-historical-data"},
    "ENBD.DU":      {"id": "12548",   "ref": "emirates-nbd-historical-data"},
    "GNAV.DU":      {"id": "12550",   "ref": "gulf-navigation-historical-data"},
    "MASQ.DU":      {"id": "941323",  "ref": "mashreqbank-historical-data"},
    "NCC.DU":       {"id": "941325",  "ref": "national-cement-historical-data"},
    "NGI.DU":       {"id": "19060",   "ref": "ngi-historical-data"},
    "PARKIN.DU":    {"id": "1212798", "ref": "parkin-historical-data"},
    "SALAMA.DU":    {"id": "941328",  "ref": "islamic-arab-insurance-historical-data"},
    "SALIK.DU":     {"id": "1194944", "ref": "salik-historical-data"},
    "SHUAA.DU":     {"id": "12557",   "ref": "shuaa-historical-data"},
    "SPINNEYS.DU":  {"id": "1214529", "ref": "spinneys-historical-data"},
    "SUKOON.DU":    {"id": "40416",   "ref": "sukoon-historical-data"},
    "TAALEEM.DU":   {"id": "1198050", "ref": "taaleem-historical-data"},
    "TABREED.DU":   {"id": "941329",  "ref": "national-central-cooling-tabreed-historical-data"},
    "TALABAT.DU":   {"id": "1224079", "ref": "talabat-historical-data"},
    "TECOM.DU":     {"id": "1192698", "ref": "tecom-historical-data"},
    # ADX — confirmed
    "FAB.AE":       {"id": "999060",  "ref": "first-abu-dhabi-bank-historical-data"},
    "ALDAR.AE":     {"id": "941317",  "ref": "aldar-properties-historical-data"},
    "TAQA.AE":      {"id": "941330",  "ref": "taqa-historical-data"},
    "ADNOCDIST.AE": {"id": "1055158", "ref": "national-oil-historical-data"},
}

# ─── Probe ALL ID ranges with current 2026 dates ────────────────────────────
print("\nStarting comprehensive ID probe with 2026 prices...")
print("This builds a map of {investing_id: current_price}")

# UAE stock IDs tend to live in these ranges:
ID_RANGES = (
    list(range(941270, 941350)),    # Old ADX stocks
    list(range(999050, 999200)),    # 2017-era ADX stocks
    list(range(1055140, 1055220)),  # Nov 2017 IPOs
    list(range(1070000, 1080000)),  # 2018 era?
    list(range(1130000, 1145000)),  # 2020-2021 era
    list(range(1155000, 1160000)),  # 2021 era (Dubai bonds?)
    list(range(1185000, 1198000)),  # 2021-2022 era
    list(range(1199000, 1230000)),  # 2022-2024 IPOs
)

# Flatten and deduplicate
all_ids = []
seen = set()
for rng in ID_RANGES:
    for i in rng:
        if i not in seen:
            all_ids.append(i)
            seen.add(i)

# Remove already confirmed IDs
used_ids = {int(v["id"]) for v in confirmed.values()}
all_ids = [i for i in all_ids if i not in used_ids]

print(f"Will probe {len(all_ids)} IDs...")

# Probe and build price map
id_to_price = {}
total = len(all_ids)
for idx, curr_id in enumerate(all_ids):
    price = probe_current(curr_id)
    if price is not None and 0.1 <= price <= 5000:
        id_to_price[curr_id] = price
    if idx % 200 == 0:
        pct = idx * 100 // total
        print(f"  [{pct}%] {idx}/{total} probed, {len(id_to_price)} found data")
        # Intermediate save
        with open('/home/ubuntu/investwise/id_price_map_2026.json', 'w') as f:
            json.dump(id_to_price, f)
    time.sleep(0.055)

print(f"\n✅ Probed {total} IDs, got prices for {len(id_to_price)}")

# Save the raw price map
with open('/home/ubuntu/investwise/id_price_map_2026.json', 'w') as f:
    json.dump(id_to_price, f, indent=2)

# ─── Match prices → tickers ──────────────────────────────────────────────────
print("\nMatching prices to UAE stocks...")

# Sort stocks by price specificity (very unique prices matched first)
pending = [(ticker, price) for ticker, price in uae_price_map.items()
           if ticker not in confirmed]

# For each stock, find the best matching ID
used_ids = {int(v["id"]) for v in confirmed.values()}
matched = dict(confirmed)

TOLERANCE = 0.08  # 8% price tolerance

for ticker, expected_price in sorted(pending, key=lambda x: x[1], reverse=True):
    best_id = None
    best_diff = float('inf')

    for curr_id, actual_price in id_to_price.items():
        if curr_id in used_ids:
            continue
        diff = abs(actual_price - expected_price) / expected_price
        if diff < TOLERANCE and diff < best_diff:
            best_diff = diff
            best_id = curr_id

    if best_id:
        matched[ticker] = {
            "id": str(best_id),
            "ref": f"{ticker.replace('.AE','').replace('.DU','').lower()}-historical-data"
        }
        used_ids.add(best_id)
        print(f"  ✅ {ticker}: id={best_id} (expected={expected_price:.2f}, actual={id_to_price[best_id]:.2f}, diff={best_diff*100:.1f}%)")
    else:
        print(f"  ❌ {ticker}: no match for price={expected_price:.2f}")

# ─── Save final results ────────────────────────────────────────────────────────
with open(RESULTS_FILE, 'w') as f:
    json.dump(matched, f, indent=2)

print(f"\n{'='*60}")
print(f"FINAL: {len(matched)} IDs mapped ({len(matched)-len(confirmed)} new)")
still_missing = [t for t in uae_price_map if t not in matched]
print(f"Missing: {still_missing}")
print(f"Saved to {RESULTS_FILE}")
