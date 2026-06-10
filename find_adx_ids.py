"""
find_adx_ids.py — يبحث عن investing.com IDs لكل أسهم ADX (أبوظبي)
يكتب النتايج في adx_ids_found.txt
"""
import cloudscraper
from bs4 import BeautifulSoup
import json, time, sys

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

# نجرب الـ search API على investing.com
def search_investing(query):
    """يستخدم investing.com search لإيجاد الـ ID"""
    url = "https://www.investing.com/search/service/searchTopBar"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.investing.com/",
    }
    try:
        r = scraper.get(url, params={"search_text": query}, headers=headers, timeout=15)
        data = r.json()
        quotes = data.get("quotes", [])
        return quotes
    except Exception as e:
        print(f"  search error: {e}")
        return []

# نتحقق من ID عن طريق جلب سطر بيانات منه
def verify_id(curr_id, ref):
    """يتحقق إن الـ ID صح بجلب بيانات حديثة"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": f"https://www.investing.com/equities/{ref}",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "curr_id": str(curr_id),
        "st_date": "01/01/2024",
        "end_date": "01/03/2024",
        "interval_sec": "Daily",
        "sort_col": "date",
        "sort_ord": "DESC",
        "action": "historical_data",
    }
    try:
        r = scraper.post(
            "https://www.investing.com/instruments/HistoricalDataAjax",
            headers=headers, data=data, timeout=15
        )
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("#curr_table tbody tr")
        if rows and "No results" not in rows[0].text and len(rows[0].find_all("td")) >= 3:
            first = [c.text.strip() for c in rows[0].find_all("td")[:3]]
            return first
    except:
        pass
    return None

# ADX tickers نريد إيجاد IDs لهم + الـ ref slugs المحتملة
ADX_TARGETS = {
    "ADNOCDIST.AE": [
        "national-oil",
        "adnoc-distribution",
        "abu-dhabi-national-oil-co-for-distribution",
    ],
    "ADNOCGAS.AE": [
        "adnoc-gas",
        "abu-dhabi-national-oil-co-gas-historical-data",
    ],
    "ADNOCDRILL.AE": [
        "adnoc-drilling",
        "abu-dhabi-national-oil-co-drilling",
    ],
    "IHC.AE": [
        "international-holding",
        "international-holding-company",
        "ihc",
    ],
    "ADIB.AE": [
        "abu-dhabi-islamic-bank",
        "adib",
    ],
    "ADCB.AE": [
        "abu-dhabi-commercial-bank",
        "adcb",
    ],
    "EAND.AE": [
        "emirates-telecommunications",
        "etisalat",
        "eand",
        "e-and",
    ],
    "ALPHADHABI.AE": [
        "alpha-dhabi",
        "alpha-dhabi-holding",
    ],
    "FERTIGLOBE.AE": [
        "fertiglobe",
        "fertiglobe-plc",
    ],
    "NMDC.AE": [
        "nmdc",
        "nmdc-energy",
    ],
    "DANA.AE": [
        "dana-gas",
        "dana-gas-pjsc",
    ],
    "RAKBANK.AE": [
        "rak-bank",
        "national-bank-ras-al-khaimah",
        "rakbank",
    ],
    "AGTHIA.AE": [
        "agthia",
        "agthia-group",
    ],
    "WAHA.AE": [
        "waha-capital",
        "waha",
    ],
}

# Search query لكل ticker
SEARCH_QUERIES = {
    "ADNOCDIST.AE": "ADNOC Distribution",
    "ADNOCGAS.AE": "ADNOC Gas",
    "ADNOCDRILL.AE": "ADNOC Drilling",
    "IHC.AE": "International Holding Company",
    "ADIB.AE": "Abu Dhabi Islamic Bank",
    "ADCB.AE": "Abu Dhabi Commercial Bank",
    "EAND.AE": "Emirates Telecommunications",
    "ALPHADHABI.AE": "Alpha Dhabi Holding",
    "FERTIGLOBE.AE": "Fertiglobe",
    "NMDC.AE": "NMDC Energy",
    "DANA.AE": "Dana Gas",
    "RAKBANK.AE": "National Bank Ras Al Khaimah",
    "AGTHIA.AE": "Agthia Group",
    "WAHA.AE": "Waha Capital",
}

results = {}

print("=" * 60)
print("Searching for ADX ticker IDs on investing.com")
print("=" * 60)

for ticker, query in SEARCH_QUERIES.items():
    print(f"\n[{ticker}] Searching: '{query}'")
    quotes = search_investing(query)
    time.sleep(1.5)

    found = False
    for q in quotes:
        q_id = q.get("id") or q.get("pairId") or q.get("pair_id")
        q_name = q.get("name", "")
        q_exchange = q.get("exchange", "")
        q_symbol = q.get("symbol", "")
        q_url = q.get("link", "") or q.get("url", "")

        # فلتر: نريد أسهم ADX/Abu Dhabi فقط
        if "abu dhabi" in q_exchange.lower() or "adx" in q_exchange.lower() or "UAE" in q_exchange.upper():
            print(f"  → ID={q_id} | {q_name} | {q_exchange} | symbol={q_symbol} | url={q_url}")
            if q_id:
                # استخرج الـ ref من الـ URL
                ref_slug = ""
                if q_url:
                    parts = q_url.strip("/").split("/")
                    ref_slug = parts[-1] if parts else ""
                results[ticker] = {"id": str(q_id), "ref": ref_slug or f"{q_symbol.lower()}-historical-data"}
                print(f"  ✅ FOUND: id={q_id}, ref={ref_slug}")
                found = True
                break

    if not found:
        # جرب بدون فلتر exchange
        for q in quotes[:5]:
            q_id = q.get("id") or q.get("pairId") or q.get("pair_id")
            q_name = q.get("name", "")
            q_exchange = q.get("exchange", "")
            q_symbol = q.get("symbol", "")
            q_url = q.get("link", "") or q.get("url", "")
            print(f"  ? ID={q_id} | {q_name} | {q_exchange} | symbol={q_symbol} | url={q_url}")
        print(f"  ⚠ Not confirmed for {ticker}")

# اكتب النتايج
print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
for ticker, info in results.items():
    print(f'    "{ticker}": {{"id": "{info["id"]}", "ref": "{info["ref"]}"}},')

with open("/home/ubuntu/investwise/adx_ids_found.txt", "w") as f:
    f.write("ADX Investing.com IDs\n")
    f.write("=" * 60 + "\n\n")
    f.write("# For UAE_INVESTING dict:\n")
    for ticker, info in results.items():
        f.write(f'    "{ticker}": {{"id": "{info["id"]}", "ref": "{info["ref"]}"}},\n')
    f.write("\n\nFull search results (raw JSON for manual inspection):\n")

print(f"\n✅ Results written to adx_ids_found.txt")
print(f"Found {len(results)}/{len(SEARCH_QUERIES)} tickers")
