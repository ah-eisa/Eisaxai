"""
verify_adx.py — يتحقق من IDs وجد الباقي
ADNOCDIST → 1055158 (3.70 AED in Mar 2024 ✅)
"""
import cloudscraper
from bs4 import BeautifulSoup
import time, sys

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

def probe(curr_id, st="01/01/2023", en="01/03/2024"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://www.investing.com/equities/",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"curr_id": str(curr_id), "st_date": st, "end_date": en,
            "interval_sec": "Daily", "sort_col": "date", "sort_ord": "DESC",
            "action": "historical_data"}
    try:
        r = scraper.post("https://www.investing.com/instruments/HistoricalDataAjax",
                         headers=headers, data=data, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("#curr_table tbody tr")
        if rows:
            cols = rows[0].find_all("td")
            if len(cols) >= 3 and "No results" not in cols[0].text:
                return {"date": cols[0].text.strip(),
                        "close": cols[1].text.strip(),
                        "open": cols[2].text.strip(),
                        "rows": len(rows)}
    except: pass
    return None

# ─── Step 1: Verify known IDs ─────────────────────────────────────────────────
print("=== VERIFICATION ===")
known = {
    "FAB.AE": 999060,
    "ALDAR.AE": 941317,
    "TAQA.AE": 941330,
    "EMAAR.DU": 1055159,
    "ADNOCDIST.AE?": 1055158,
}
for name, id_ in known.items():
    r = probe(id_)
    print(f"  {id_} [{name}]: {r}")
    time.sleep(0.5)

# ─── Step 2: Probe for remaining ADX stocks ───────────────────────────────────
print("\n=== PROBING ADX TICKERS ===")

# ADNOC Gas (IPO Mar 2023) — probe around 1200000-1206000
# ALANSARI.DU: 1201945, DUBAITAXI.DU: 1209220
# ADNOCGAS might be 1200xxx or 1201xxx
# ADNOC Gas price in Mar 2024: ~3-4 AED

# ADNOC Drilling (IPO Oct 2021) — probe around 1155000-1195000
# ADNOCDRILL price in Mar 2024: ~3-4 AED
# Salik: 1194944 (Dec 2022), Empower: 1197172 (Nov 2022)
# So Drilling (Oct 2021) → might be 1100000-1155000 range

# IHC.AE — price in Mar 2024: ~300-380 AED, older stock
# ADIB.AE — price in Mar 2024: ~9-11 AED
# ADCB.AE — price in Mar 2024: ~9-11 AED
# EAND.AE — price in Mar 2024: ~18-22 AED

# Strategy: probe targeted ranges, filter by UAE price range
# All ADX stocks trade in AED (0.1 to 500 range typically)

# For ADNOCDRILL: IPO Oct 2021
# Other 2021 IPOs: EMPOWER (Nov 2022 = 1197172), SALIK (Dec 2022 = 1194944)
# Drilling was IPO before both → try 1140000-1155000
drill_ranges = list(range(1130000, 1145000, 1))

# For ADNOC Gas: IPO Mar 2023
# ALANSARI was Mar 2023 = 1201945, DUBAITAXI Mar 2023 = 1209220
# Gas IPO was Mar 2023 → try 1200000-1212000
gas_ranges = list(range(1200000, 1212000, 1))

# For IHC: became prominent 2021, but older company → maybe 999xxx or 1055xxx
# IHC price Mar 2024: ~310-350 AED
# From 999xxx probe: 999081=342.10, 999082=347.65
ihc_candidates = [999081, 999082, 999083, 999093, 999095, 999096]

# For ADIB, ADCB, EAND: older ADX stocks
# ADIB price Mar 2024: ~9.2-10 AED
# ADCB price Mar 2024: ~9.5-10.5 AED
# EAND price Mar 2024: ~19-22 AED
older_candidates = list(range(941000, 941350, 1))  # same range as TAQA/ALDAR

print("\n1) Checking IHC candidates (high price ~300-380 AED):")
for id_ in ihc_candidates:
    r = probe(id_)
    if r:
        print(f"  ID {id_}: {r}")
    time.sleep(0.3)

print("\n2) Probing older ADX range 941000-941350 (ADIB/ADCB/EAND ~9-22 AED):")
aed_range_found = []
for curr_id in range(941000, 941350):
    r = probe(curr_id)
    if r:
        try:
            price = float(r['close'].replace(',',''))
            if 5 <= price <= 30:  # AED price range for banks/telecom
                print(f"  ✅ ID {curr_id}: close={r['close']} open={r['open']} ({r['date']})")
                aed_range_found.append({'id': curr_id, 'close': r['close']})
        except: pass
    time.sleep(0.12)
print(f"  Found {len(aed_range_found)} in 941xxx range")

print("\n3) Probing ADNOC Drilling range 1130000-1145000 (~3-4 AED):")
drill_found = []
for curr_id in range(1130000, 1145000):
    r = probe(curr_id)
    if r:
        try:
            price = float(r['close'].replace(',',''))
            if 2 <= price <= 6:
                print(f"  ✅ ID {curr_id}: close={r['close']} ({r['date']})")
                drill_found.append({'id': curr_id, 'close': r['close']})
        except: pass
    time.sleep(0.12)
print(f"  Found {len(drill_found)} in Drilling range")

# Summary
print("\n=== SUMMARY ===")
print("Likely ADNOCDIST.AE: 1055158 (3.70 AED) ← ADNOC Dist price was ~3.7 in Mar 2024")
print("IHC candidates:", ihc_candidates)
print("Drilling candidates:", drill_found)
