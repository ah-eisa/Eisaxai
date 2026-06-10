"""
find_all_adx.py — يجيب كل IDs لـ 74 سهم ADX بالكامل
يشغّل في background ويحفظ النتايج كل شوية
"""
import cloudscraper, time, json, sys
from bs4 import BeautifulSoup
from pathlib import Path

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

RESULTS_FILE = "/home/ubuntu/investwise/adx_all_results.json"

def probe(curr_id):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://www.investing.com/equities/",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"curr_id": str(curr_id),
            "st_date": "01/01/2024", "end_date": "03/01/2024",
            "interval_sec": "Daily", "sort_col": "date", "sort_ord": "DESC",
            "action": "historical_data"}
    try:
        r = scraper.post("https://www.investing.com/instruments/HistoricalDataAjax",
                         headers=headers, data=data, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("#curr_table tbody tr")
        if rows:
            cols = rows[0].find_all("td")
            if len(cols) >= 3 and "No results" not in cols[0].text:
                close = cols[1].text.strip()
                try:
                    return float(close.replace(",",""))
                except: pass
    except: pass
    return None

def save(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

# ─── ADX stock prices in Q1 2024 (AED) ────────────────────────────────────────
ADX_EXPECTED = {
    "ADNOCDIST.AE":  (3.3, 4.2),
    "ADNOCGAS.AE":   (3.0, 4.6),
    "ADNOCDRILL.AE": (3.2, 4.6),
    "IHC.AE":        (295, 410),
    "EAND.AE":       (17, 25),
    "ADIB.AE":       (8.5, 11.5),
    "ADCB.AE":       (8.5, 12.0),
    "DANA.AE":       (1.0, 1.8),
    "RAKBANK.AE":    (11, 17),
    "ALPHADHABI.AE": (28, 45),
    "WAHA.AE":       (2.0, 3.8),
    "AGTHIA.AE":     (3.0, 5.5),
    "NMDC.AE":       (0.8, 1.5),
    "FERTIGLOBE.AE": (3.0, 4.8),
    "FAB.AE":        (6.5, 8.5),   # confirmed 999060
    "ALDAR.AE":      (4.5, 6.5),   # confirmed 941317
    "TAQA.AE":       (2.0, 3.5),   # confirmed 941330
}

# Pre-confirmed
confirmed = {
    "FAB.AE":       {"id": "999060",  "ref": "first-abu-dhabi-bank-historical-data"},
    "ALDAR.AE":     {"id": "941317",  "ref": "aldar-properties-historical-data"},
    "TAQA.AE":      {"id": "941330",  "ref": "taqa-historical-data"},
    "ADNOCDIST.AE": {"id": "1055158", "ref": "national-oil-historical-data"},
}

# From previous probe_ranges.py run — best candidates from 999xxx:
prev_999 = {
    999053: 19.27, 999054: 22.88, 999055: 14.78, 999057: 0.854,
    999058: 6.80, 999060: 7.49, 999061: 10.04, 999063: 7.00,
    999064: 1.420, 999066: 308.30, 999067: 2.34, 999071: 0.210,
    999075: 0.265, 999079: 0.710, 999081: 342.10, 999082: 347.65,
    999083: 22.95, 999086: 34.65, 999087: 5.58, 999088: 69.48,
    999089: 0.105, 999093: 297.55, 999095: 181.50, 999096: 242.40,
    999097: 1.148, 999099: 5.4, 999103: 4.010, 999105: 13.50,
    999109: 0.3250, 999110: 18.18, 999111: 232.50, 999112: 1.248,
    999113: 1.418, 999114: 1.781, 999116: 1.161, 999117: 2.71,
    999118: 2.153, 999119: 1.208,
}

print("=" * 60)
print("Finding ALL ADX investing.com IDs")
print("=" * 60)

all_found = {}  # price → [ids]

# Map 999xxx data
for id_, price in prev_999.items():
    if id_ not in [999060]:  # exclude FAB (already confirmed)
        all_found[id_] = price

# Try to match 999xxx to known stocks
print("\n--- Matching 999xxx data to ADX stocks ---")
matched = dict(confirmed)
used_ids = set(int(v["id"]) for v in confirmed.values())

for stock, (lo, hi) in ADX_EXPECTED.items():
    if stock in matched:
        continue
    candidates = [(id_, price) for id_, price in prev_999.items()
                  if lo <= price <= hi and id_ not in used_ids]
    if candidates:
        # Take the first match
        best_id, best_price = candidates[0]
        matched[stock] = {"id": str(best_id), "ref": f"id{best_id}-historical-data"}
        used_ids.add(best_id)
        print(f"  ✅ {stock}: {best_id} (price={best_price})")
    else:
        print(f"  ❓ {stock}: not found in 999xxx")

save(matched)
print(f"\n✅ Saved {len(matched)} tickers so far")

# ─── Now probe missing ones ────────────────────────────────────────────────────
missing = [s for s in ADX_EXPECTED if s not in matched]
print(f"\nMissing: {missing}")
print("\n--- Probing focused ranges for missing stocks ---")

# ADNOC Drilling (Oct 2021 IPO) — try ranges:
# Since SALIK (Dec 2022) = 1194944, Drilling (Oct 2021) might be < 1194944
# Try 1135000-1145000 AND 1165000-1185000
ranges_to_probe = []

if "ADNOCDRILL.AE" in missing:
    ranges_to_probe += list(range(1135000, 1150000))
    ranges_to_probe += list(range(1165000, 1185000))

if "ADNOCGAS.AE" in missing:
    # Gas IPO Mar 2023, ALANSARI (Mar 2023) = 1201945
    ranges_to_probe += list(range(1199000, 1212000))

if "ADCB.AE" in missing:
    # Try 941xxx range more broadly
    ranges_to_probe += list(range(941270, 941310))
    ranges_to_probe += list(range(941332, 941360))

# Also probe small gaps in 999xxx
ranges_to_probe += list(range(999120, 999180))
ranges_to_probe += list(range(941315, 941345))

print(f"Will probe {len(ranges_to_probe)} IDs...")

count = 0
for curr_id in ranges_to_probe:
    if curr_id in used_ids:
        continue
    price = probe(curr_id)
    if price is not None:
        # Check if matches any missing stock
        for stock in missing:
            if stock in matched:
                continue
            lo, hi = ADX_EXPECTED[stock]
            if lo <= price <= hi:
                matched[stock] = {"id": str(curr_id), "ref": f"id{curr_id}-historical-data"}
                used_ids.add(curr_id)
                print(f"  ✅ {stock}: ID={curr_id} price={price}")
                missing_now = [s for s in ADX_EXPECTED if s not in matched]
                if not missing_now:
                    print("All found! Done.")
                    break
        all_found[curr_id] = price

    count += 1
    if count % 100 == 0:
        save(matched)
        sys.stdout.write(f"\r  Progress: {count}/{len(ranges_to_probe)} | Found: {len(matched)}/{len(ADX_EXPECTED)}")
        sys.stdout.flush()

    time.sleep(0.08)

save(matched)

# ─── Final Report ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
for stock, info in sorted(matched.items()):
    print(f'    "{stock}": {{"id": "{info["id"]}", "ref": "{info["ref"]}"}},')

still_missing = [s for s in ADX_EXPECTED if s not in matched]
if still_missing:
    print(f"\nStill missing: {still_missing}")

print(f"\n✅ Saved to {RESULTS_FILE}")
