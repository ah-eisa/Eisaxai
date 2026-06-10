"""
deep_crawl_adx.py — يجيب investing.com IDs لكل أسهم ADX
يفتح صفحة كل سهم ويستخرج الـ curr_id من الـ HTML
"""
import cloudscraper
from bs4 import BeautifulSoup
import re, json, time, sys

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

def extract_id_from_page(url, ref_slug):
    """يفتح صفحة investing.com ويستخرج الـ curr_id"""
    headers = {**BASE_HEADERS, "Referer": "https://www.investing.com/equities/united-arab-emirates"}

    try:
        r = scraper.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return None

        html = r.text

        # Method 1: data-pair-id attribute
        m = re.search(r'data-pair-id=["\'](\d+)["\']', html)
        if m:
            return m.group(1)

        # Method 2: pairId in JSON
        m = re.search(r'"pairId"\s*:\s*(\d+)', html)
        if m:
            return m.group(1)

        # Method 3: pair_id in JS
        m = re.search(r'pair_id\s*[=:]\s*["\']?(\d+)["\']?', html)
        if m:
            return m.group(1)

        # Method 4: curr_id in JS
        m = re.search(r'curr_id\s*[=:]\s*["\']?(\d+)["\']?', html)
        if m:
            return m.group(1)

        # Method 5: data-id near the stock symbol
        m = re.search(r'data-id=["\'](\d+)["\']', html)
        if m:
            return m.group(1)

        # Method 6: HistoricalDataAjax form
        m = re.search(r'name=["\']curr_id["\'][^>]*value=["\'](\d+)["\']', html)
        if m:
            return m.group(1)
        m = re.search(r'value=["\'](\d+)["\'][^>]*name=["\']curr_id["\']', html)
        if m:
            return m.group(1)

        # Method 7: Search in script tags for any window object
        for tag in BeautifulSoup(html, "html.parser").find_all("script"):
            if tag.string:
                m = re.search(r'"id"\s*:\s*(\d{5,7})', tag.string)
                if m:
                    candidate = m.group(1)
                    # Filter: valid investing.com ID range
                    if 10000 <= int(candidate) <= 9999999:
                        print(f"    Script tag candidate: {candidate}")
                        return candidate

        print(f"    No ID found in HTML ({len(html)} bytes)")
        return None

    except Exception as e:
        print(f"    Error: {e}")
        return None

def verify_id_works(curr_id, ref_slug):
    """يتحقق إن الـ ID يرجع بيانات حقيقية"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": f"https://www.investing.com/equities/{ref_slug}",
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
            headers=headers, data=data, timeout=15
        )
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("#curr_table tbody tr")
        if rows and len(rows) > 0:
            cols = rows[0].find_all("td")
            if len(cols) >= 3 and "No results" not in cols[0].text:
                return [c.text.strip() for c in cols[:4]]
    except Exception as e:
        print(f"    verify error: {e}")
    return None

# تعريف الأسهم المطلوبة مع URLs المحتملة
ADX_TARGETS = {
    "ADNOCDIST.AE": {
        "urls": [
            "https://www.investing.com/equities/national-oil",
            "https://www.investing.com/equities/adnoc-distribution",
        ],
        "ref": "national-oil",
        "name": "ADNOC Distribution",
    },
    "ADNOCGAS.AE": {
        "urls": [
            "https://www.investing.com/equities/adnoc-gas",
        ],
        "ref": "adnoc-gas",
        "name": "ADNOC Gas",
    },
    "ADNOCDRILL.AE": {
        "urls": [
            "https://www.investing.com/equities/adnoc-drilling",
        ],
        "ref": "adnoc-drilling",
        "name": "ADNOC Drilling",
    },
    "IHC.AE": {
        "urls": [
            "https://www.investing.com/equities/international-holding",
            "https://www.investing.com/equities/international-holding-company",
        ],
        "ref": "international-holding",
        "name": "International Holding Company",
    },
    "ADIB.AE": {
        "urls": [
            "https://www.investing.com/equities/abu-dhabi-islamic-bank",
        ],
        "ref": "abu-dhabi-islamic-bank",
        "name": "Abu Dhabi Islamic Bank",
    },
    "ADCB.AE": {
        "urls": [
            "https://www.investing.com/equities/abu-dhabi-commercial-bank",
        ],
        "ref": "abu-dhabi-commercial-bank",
        "name": "Abu Dhabi Commercial Bank",
    },
    "EAND.AE": {
        "urls": [
            "https://www.investing.com/equities/emirate-telecom",
            "https://www.investing.com/equities/emirates-telecommunications",
            "https://www.investing.com/equities/e-and",
        ],
        "ref": "emirate-telecom",
        "name": "Emirates Telecommunications (EAND)",
    },
    "ALPHADHABI.AE": {
        "urls": [
            "https://www.investing.com/equities/alpha-dhabi-holding",
            "https://www.investing.com/equities/alphadhabi-holding",
        ],
        "ref": "alpha-dhabi-holding",
        "name": "Alpha Dhabi Holding",
    },
    "FERTIGLOBE.AE": {
        "urls": [
            "https://www.investing.com/equities/fertiglobe",
        ],
        "ref": "fertiglobe",
        "name": "Fertiglobe",
    },
    "NMDC.AE": {
        "urls": [
            "https://www.investing.com/equities/nmdc",
            "https://www.investing.com/equities/nmdc-energy",
        ],
        "ref": "nmdc",
        "name": "NMDC Energy",
    },
    "DANA.AE": {
        "urls": [
            "https://www.investing.com/equities/dana-gas",
        ],
        "ref": "dana-gas",
        "name": "Dana Gas",
    },
    "RAKBANK.AE": {
        "urls": [
            "https://www.investing.com/equities/national-bank-ras-al-khaimah",
            "https://www.investing.com/equities/rak-bank",
        ],
        "ref": "national-bank-ras-al-khaimah",
        "name": "RAKBANK",
    },
    "AGTHIA.AE": {
        "urls": [
            "https://www.investing.com/equities/agthia-group",
        ],
        "ref": "agthia-group",
        "name": "Agthia Group",
    },
    "WAHA.AE": {
        "urls": [
            "https://www.investing.com/equities/waha-capital",
        ],
        "ref": "waha-capital",
        "name": "Waha Capital",
    },
}

results = {}
failed = []

print("=" * 60)
print("Deep crawling investing.com for ADX stock IDs")
print("=" * 60)

# أول شيء: احصل على الـ cookies بزيارة الصفحة الرئيسية
print("\nInitializing session...")
try:
    r = scraper.get("https://www.investing.com/equities/united-arab-emirates",
                    headers=BASE_HEADERS, timeout=20)
    print(f"Session initialized: {r.status_code}")
    time.sleep(2)
except Exception as e:
    print(f"Warning: {e}")

for ticker, info in ADX_TARGETS.items():
    print(f"\n[{ticker}] {info['name']}")
    found_id = None
    found_ref = info["ref"]

    for url in info["urls"]:
        print(f"  Trying: {url}")
        found_id = extract_id_from_page(url, info["ref"])
        if found_id:
            # Extract the actual ref from URL
            ref_from_url = url.split("/equities/")[-1]
            found_ref = ref_from_url
            print(f"  📍 Found ID: {found_id} from {url}")

            # Verify it works
            check = verify_id_works(found_id, found_ref)
            if check:
                print(f"  ✅ Verified! Data: {check}")
                results[ticker] = {"id": found_id, "ref": f"{found_ref}-historical-data"}
                break
            else:
                print(f"  ⚠ ID found but no data returned, trying next...")
                found_id = None
        time.sleep(1.5)

    if not found_id:
        failed.append(ticker)
        print(f"  ❌ No ID found for {ticker}")

    time.sleep(1)

# اكتب النتايج
print("\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
print(f"Found: {len(results)}/{len(ADX_TARGETS)}")
print(f"\n# Add these to UAE_INVESTING in market_data_engine.py:")
for ticker, info in results.items():
    print(f'    "{ticker}": {{"id": "{info["id"]}", "ref": "{info["ref"]}"}},')

if failed:
    print(f"\nFailed: {failed}")

with open("/home/ubuntu/investwise/adx_ids_found.txt", "w") as f:
    f.write("ADX Investing.com IDs — Deep Crawl Results\n")
    f.write("=" * 60 + "\n\n")
    f.write("# Add to UAE_INVESTING in market_data_engine.py:\n")
    for ticker, info in results.items():
        f.write(f'    "{ticker}": {{"id": "{info["id"]}", "ref": "{info["ref"]}"}},\n')
    if failed:
        f.write(f"\n# Not found: {failed}\n")

print("\n✅ Results saved to adx_ids_found.txt")
