"""
match_adx.py — يتحقق من IDs ويطابقها مع أسهم ADX بالأسعار المعروفة
"""
import cloudscraper, time, json
from bs4 import BeautifulSoup

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

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
                         headers=headers, data=data, timeout=10)
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

# ─── Known ADX prices in Q1 2024 (AED) ───────────────────────────────────────
# These are approximate — enough to identify which ID is which stock
ADX_PRICES_2024 = {
    # Stock: [min_price, max_price, investing_id_hint]
    "ADNOCDIST.AE": [3.3, 4.1],   # IPO 2017, ~3.7 AED
    "ADNOCGAS.AE":  [3.0, 4.5],   # IPO Mar 2023, ~3.5-4 AED
    "ADNOCDRILL.AE":[3.2, 4.5],   # IPO Oct 2021, ~3.6-4 AED
    "IHC.AE":       [300, 400],   # ~350 AED
    "EAND.AE":      [17, 24],     # ~20 AED
    "ADIB.AE":      [8.5, 11],    # ~9.5 AED
    "ADCB.AE":      [8.5, 11.5],  # ~10 AED
    "DANA.AE":      [1.0, 1.8],   # ~1.3 AED
    "RAKBANK.AE":   [12, 17],     # ~14 AED
    "ALPHADHABI.AE":[30, 45],     # ~36 AED
    "WAHA.AE":      [2.0, 3.5],   # ~2.7 AED
    "AGTHIA.AE":    [3.0, 5.0],   # ~4 AED
    "NMDC.AE":      [0.8, 1.5],   # ~1.1 AED (NMDC Energy)
    "FERTIGLOBE.AE":[3.0, 4.5],   # ~3.7 AED
}

# ─── Priority IDs to check (from previous probe runs) ────────────────────────
# These were found to have data in the 999xxx range with AED-like prices
CANDIDATES = {
    # From 999xxx probe
    999053: "EAND.AE?",      # 19.27 → matches EAND range!
    999054: "EAND.AE_v2?",   # 22.88 → also in EAND range
    999055: "RAKBANK.AE?",   # 14.78 → matches RAKBANK
    999061: "ADIB.AE?",      # 10.04 → matches ADIB
    999063: "ADCB.AE?",      # 7.00 → maybe?
    999064: "DANA.AE?",      # 1.420 → matches DANA
    999066: "IHC.AE_v1?",    # 308.30 → IHC range
    999081: "IHC.AE_v2?",    # 342.10 → IHC range
    999082: "IHC.AE_v3?",    # 347.65 → matches IHC!
    999083: "EAND.AE_v3?",   # 22.95 → EAND range
    999086: "ALPHADHABI.AE?",# 34.65 → matches Alpha Dhabi
    999087: "WAHA.AE?",      # 5.58 → WAHA range? (2-3.5 expected)
    999097: "WAHA.AE_v2?",   # 1.148 → NMDC range?
    999103: "AGTHIA.AE?",    # 4.010 → matches Agthia
    999105: "ADIB.AE_v2?",   # 13.50 → ADIB range
    999109: "NMDC.AE?",      # 0.3250 → too low for NMDC?
    999112: "FERTIGLOBE.AE?",# 1.248 → too low
    999113: "DANA.AE_v2?",   # 1.418 → matches DANA!
    999116: "FERTIGLOBE_v2?",# 1.161 → similar
    999117: "WAHA.AE_v3?",   # 2.71 → matches WAHA!
    # From 1055xxx probe
    1055140: "?",             # 0.46
    1055158: "ADNOCDIST.AE", # 3.700 ← CONFIRMED CANDIDATE
}

print("=== Verifying candidate IDs for ADX stocks ===\n")
print(f"{'ID':<12} {'Price':<10} {'Best Match':<20} {'Note'}")
print("-" * 60)

matches = {}
for curr_id, hint in CANDIDATES.items():
    price = probe(curr_id)
    if price is not None:
        best_match = ""
        for stock, (lo, hi) in ADX_PRICES_2024.items():
            if lo <= price <= hi:
                if stock not in matches:
                    best_match = stock
                    break
        print(f"{curr_id:<12} {price:<10.3f} {best_match:<20} [{hint}]")
        if best_match:
            matches[best_match] = curr_id
    time.sleep(0.4)

# ─── Now probe 941xxx range for ADIB/ADCB/EAND (older ADX stocks) ─────────────
print("\n\n=== Probing 941xxx for older ADX stocks (ADIB ~9.5, ADCB ~10, EAND ~20 AED) ===")
print("(Checking 941300-941340 and 941350-941400)")

older_found = []
for curr_id in list(range(941300, 941355)) + list(range(941270, 941310)):
    price = probe(curr_id)
    if price is not None:
        # Filter to AED price ranges
        matches_stock = ""
        for stock, (lo, hi) in ADX_PRICES_2024.items():
            if lo <= price <= hi and stock not in matches:
                matches_stock = stock
                break
        status = f"← {matches_stock}" if matches_stock else ""
        print(f"  {curr_id}: {price:.3f} AED  {status}")
        if matches_stock:
            matches[matches_stock] = curr_id
        older_found.append(curr_id)
    time.sleep(0.12)

# ─── Probe for ADNOCDRILL (IPO Oct 2021) around 1185000-1195000 ───────────────
print("\n\n=== Probing for ADNOC Drilling 1185000-1194000 (~3.5-4.5 AED) ===")
for curr_id in range(1185000, 1193000):
    price = probe(curr_id)
    if price and 3.0 <= price <= 5.0:
        print(f"  ✅ {curr_id}: {price:.3f} AED ← likely ADNOCDRILL")
        if "ADNOCDRILL.AE" not in matches:
            matches["ADNOCDRILL.AE"] = curr_id
    time.sleep(0.08)

# ─── Probe for ADNOC Gas (IPO Mar 2023) around 1200000-1206000 ───────────────
print("\n\n=== Probing for ADNOC Gas 1200000-1206000 (~3-4.5 AED) ===")
for curr_id in range(1200000, 1206000):
    price = probe(curr_id)
    if price and 2.5 <= price <= 5.0:
        print(f"  ✅ {curr_id}: {price:.3f} AED ← likely ADNOCGAS")
        if "ADNOCGAS.AE" not in matches:
            matches["ADNOCGAS.AE"] = curr_id
    time.sleep(0.08)

# ─── Final Results ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("CONFIRMED MATCHES:")
for stock, id_ in matches.items():
    print(f"  {stock}: {id_}")

# Save results
results = {stock: {"id": str(id_)} for stock, id_ in matches.items()}
with open("/home/ubuntu/investwise/adx_confirmed.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✅ Saved to adx_confirmed.json")
