"""
probe_ranges.py — يبحث عن IDs بتجربة ranges حول تواريخ الـ IPOs المعروفة
يستخدم HistoricalDataAjax مباشرة ويشوف أي ID يرجع بيانات UAE
"""
import cloudscraper
from bs4 import BeautifulSoup
import time, sys

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

def probe_id(curr_id, ref="uae-equities"):
    """يختبر ID واحد - يرجع أول row من البيانات أو None"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": f"https://www.investing.com/equities/{ref}",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "curr_id": str(curr_id),
        "st_date": "01/01/2024",
        "end_date": "03/01/2024",
        "interval_sec": "Daily",
        "sort_col": "date",
        "sort_ord": "DESC",
        "action": "historical_data",
    }
    try:
        r = scraper.post(
            "https://www.investing.com/instruments/HistoricalDataAjax",
            headers=headers, data=data, timeout=10
        )
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("#curr_table tbody tr")
        if rows and len(rows) > 0:
            cols = rows[0].find_all("td")
            if len(cols) >= 3 and "No results" not in cols[0].text:
                close = cols[1].text.strip()
                # Filter: AED range for UAE stocks (0.1 to 200 usually)
                try:
                    price = float(close.replace(",", ""))
                    if 0.1 <= price <= 500:
                        return {
                            "date": cols[0].text.strip(),
                            "close": close,
                            "open": cols[2].text.strip() if len(cols) > 2 else "",
                        }
                except:
                    return {"date": cols[0].text.strip(), "close": close, "open": ""}
    except Exception as e:
        pass
    return None

# ADNOC Distribution (IPO Nov 2017) — try ranges near 1055159 (EMAAR date)
# Also try ranges for other ADNOC tickers

RANGES_TO_PROBE = {
    # Around EMAAR/EMAARDEV (both Nov 2017)
    "ADNOCDIST.AE?": range(1055140, 1055200),
    # Try earlier range (same period as FAB.AE 999060)
    "ADNOCDIST.AE_v2?": range(999000, 999120),
    # ADNOC Drilling (Oct 2021) — around when other 2021 IPOs got IDs
    # DEWA.DU is 941326 (IPO 2022!), SALIK 1194944 (Dec 2022)
    # So 2021 IPOs might be in 1100000-1190000 range
    "ADNOCDRILL.AE?": range(1155000, 1160000),
    # ADNOC Gas (Mar 2023) — after ALANSARI 1201945 (Mar 2023)
    "ADNOCGAS.AE?": range(1200000, 1207000),
    # IHC (got big in 2021)
    "IHC.AE?": range(1070000, 1080000),
}

# Test a known good ID first
print("Testing known good ID (EMAAR.DU = 1055159)...")
result = probe_id(1055159, "emaar-properties-historical-data")
if result:
    print(f"✅ Known good ID works: {result}")
else:
    print("❌ Known good ID FAILED - something is wrong")
    sys.exit(1)

time.sleep(1)
print()

all_found = {}

for name, id_range in RANGES_TO_PROBE.items():
    print(f"\nProbing {name}: {id_range.start}-{id_range.stop-1} ({len(id_range)} IDs)")
    found_in_range = []

    for i, curr_id in enumerate(id_range):
        result = probe_id(curr_id)
        if result:
            print(f"  ✅ ID {curr_id}: date={result['date']}, close={result['close']}, open={result['open']}")
            found_in_range.append({"id": curr_id, **result})
            all_found[curr_id] = result
        else:
            if i % 50 == 0:
                sys.stdout.write(f"\r  Tested: {i}/{len(id_range)}...")
                sys.stdout.flush()
        time.sleep(0.15)  # be nice to server

    print(f"\n  Found {len(found_in_range)} tickers in this range")

print("\n" + "=" * 60)
print(f"TOTAL FOUND: {len(all_found)} IDs across all ranges")
for curr_id, data in all_found.items():
    print(f"  ID {curr_id}: {data}")

with open("/home/ubuntu/investwise/probe_results.txt", "w") as f:
    f.write("Probe Results\n")
    for curr_id, data in all_found.items():
        f.write(f"ID {curr_id}: {data}\n")

print("\n✅ Results saved to probe_results.txt")
