
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd, time, logging, sys
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DATA_DIR = Path("/home/ubuntu/investwise/data")

UAE_TICKERS = {
    "EMAAR.DU":     {"id": "1055159", "ref": "emaar-properties-historical-data"},
    "FAB.AE":       {"id": "999060",  "ref": "first-abu-dhabi-bank-historical-data"},
    "ALDAR.AE":     {"id": "941317",  "ref": "aldar-properties-historical-data"},
    "TAQA.AE":      {"id": "941330",  "ref": "taqa-historical-data"},
}

def _parse_volume(v):
    v = v.replace(",","").strip()
    if "M" in v: return float(v.replace("M",""))*1_000_000
    if "K" in v: return float(v.replace("K",""))*1_000
    try: return float(v)
    except: return 0

def fetch_uae(ticker, info, start="2015-01-01"):
    safe = ticker.replace(".","_")
    path = DATA_DIR/"historical"/"AE"/f"{safe}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.today()
    chunks = []
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=365), end_dt)
        chunks.append((cur.strftime("%m/%d/%Y"), chunk_end.strftime("%m/%d/%Y")))
        cur = chunk_end + timedelta(days=1)

    scraper = cloudscraper.create_scraper(browser={"browser":"chrome","platform":"windows","mobile":False})
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": f"https://www.investing.com/equities/{info['ref']}",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    all_records = []
    for st, en in chunks:
        data = {"curr_id": info["id"], "st_date": st, "end_date": en,
                "interval_sec": "Daily", "sort_col": "date", "sort_ord": "DESC", "action": "historical_data"}
        try:
            r = scraper.post("https://www.investing.com/instruments/HistoricalDataAjax",
                             headers=headers, data=data, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select("#curr_table tbody tr")
            logger.info(f"  {ticker} {st}-{en}: {len(rows)} rows")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 5:
                    try:
                        all_records.append({
                            "Date": pd.to_datetime(cols[0].text.strip()),
                            "Close": float(cols[1].text.strip().replace(",","")),
                            "Open":  float(cols[2].text.strip().replace(",","")),
                            "High":  float(cols[3].text.strip().replace(",","")),
                            "Low":   float(cols[4].text.strip().replace(",","")),
                            "Volume": _parse_volume(cols[5].text.strip()) if len(cols)>5 else 0,
                        })
                    except: continue
            time.sleep(1.5)
        except Exception as e:
            logger.error(f"Chunk error {ticker} {st}-{en}: {e}")

    if not all_records:
        logger.warning(f"No data for {ticker}")
        return

    df = pd.DataFrame(all_records).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(path, index=True)
    logger.info(f"Saved {ticker} ({len(df)} rows)")

def fetch_all_uae(start="2015-01-01"):
    logger.info(f"UAE via Investing.com")
    for ticker, info in UAE_TICKERS.items():
        logger.info(f"-> {ticker}")
        fetch_uae(ticker, info, start=start)

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"
    fetch_all_uae(start)
