"""
data_fetcher.py — EisaX Market Data Fetcher
يجيب Historical + EOD data للأسواق السعودي، المصري، الإماراتي
المصادر: yfinance (SA/EG) + Stooq fallback (AE)
"""

import yfinance as yf
import pandas as pd
import requests
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from io import StringIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Ticker Lists (من local_tickers.py) ─────────────────────────────────────

SA_TICKERS = [
    "2222.SR", "1120.SR", "1180.SR", "2010.SR", "2380.SR", "4200.SR",
    "4110.SR", "4030.SR", "2330.SR", "2350.SR", "1010.SR", "1020.SR",
    "1030.SR", "1050.SR", "1060.SR", "1080.SR", "1090.SR", "1111.SR",
    "1140.SR", "1150.SR", "1210.SR", "1211.SR", "2001.SR", "2020.SR",
    "2030.SR", "2050.SR", "2060.SR", "2100.SR", "2160.SR", "2170.SR",
    "2200.SR", "2210.SR", "2220.SR", "2240.SR", "2250.SR", "2290.SR",
    "4001.SR", "4002.SR", "4003.SR", "4020.SR", "4031.SR", "4050.SR",
    "4051.SR", "4061.SR", "4100.SR", "4130.SR", "4160.SR", "4180.SR",
    "4190.SR", "4210.SR", "4230.SR", "4240.SR", "4250.SR", "4260.SR",
    "4261.SR", "4300.SR", "4321.SR", "4330.SR", "4340.SR",
    # Indices
    "^TASI", "^NOMU",
]

EG_TICKERS = [
    "COMI.CA", "HRHO.CA", "ETEL.CA", "OTMT.CA", "EKHO.CA", "ESRS.CA",
    "SWDY.CA", "PHDC.CA", "OCDI.CA", "EFIC.CA", "MNHD.CA", "ABUK.CA",
    "SUGR.CA", "AMER.CA", "ACGC.CA", "ORWE.CA", "FWRY.CA",
    # Index
    "^EGX30",
]

AE_TICKERS = {
    # ADX
    "ETISALAT": "ETISALAT.AE",
    "ADNOCDIST": "ADNOCDIST.AE",
    "FAB": "FAB.AE",
    "ADIB": "ADIB.AE",
    "ALDAR": "ALDAR.AE",
    "TAQA": "TAQA.AE",
    "IHC": "IHC.AE",
    "FERTIGLOBE": "FERTIGLOBE.AE",
    # DFM
    "DIB": "DIB.DU",
    "EMIRATES_NBD": "ENBD.DU",
    "DU": "DU.DU",
    "EMAAR": "EMAAR.DU",
    "DEWA": "DEWA.DU",
    "SALIK": "SALIK.DU",
    "PARKIN": "PARKIN.DU",
    "EMAARDEV": "EMAARDEV.DU",
    # Index
    "ADX": "^ADX",
    "DFM": "^DFMGI",
}

AE_STOOQ_MAP = {
    "EMAAR.DU":    "emaar.ae",
    "DIB.DU":      "dib.ae",
    "ENBD.DU":     "enbd.ae",
    "DU.DU":       "du.ae",
    "DEWA.DU":     "dewa.ae",
    "SALIK.DU":    "salik.ae",
    "EMAARDEV.DU": "emaardev.ae",
    "ETISALAT.AE": "etisalat.ae",
    "FAB.AE":      "fab.ae",
    "ADIB.AE":     "adib.ae",
    "ALDAR.AE":    "aldar.ae",
    "TAQA.AE":     "taqa.ae",
}

from core.config import DATA_DIR as _cfg_data_dir
DATA_DIR = _cfg_data_dir


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_parquet(df: pd.DataFrame, market: str, ticker: str):
    """يحفظ DataFrame كـ Parquet"""
    safe = ticker.replace(".", "_").replace("^", "IDX_")
    path = DATA_DIR / "historical" / market / f"{safe}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    logger.info(f"✅ Saved {ticker} → {path} ({len(df)} rows)")


def _load_parquet(market: str, ticker: str) -> Optional[pd.DataFrame]:
    safe = ticker.replace(".", "_").replace("^", "IDX_")
    path = DATA_DIR / "historical" / market / f"{safe}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _last_saved_date(market: str, ticker: str) -> Optional[str]:
    df = _load_parquet(market, ticker)
    if df is not None and not df.empty:
        return df.index[-1].strftime("%Y-%m-%d")
    return None


# ─── yfinance Fetcher (SA + EG) ───────────────────────────────────────────────

def fetch_yfinance(ticker: str, market: str, start: str = "2015-01-01", end: str = None):
    """يجيب historical data من yfinance ويحفظها"""
    end = end or datetime.today().strftime("%Y-%m-%d")

    # لو عندنا داتا، ابدأ من آخر تاريخ محفوظ + يوم
    last = _last_saved_date(market, ticker)
    if last:
        new_start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        if new_start >= end:
            logger.info(f"⏭️  {ticker} already up to date ({last})")
            return
        start = new_start
        mode = "append"
    else:
        mode = "full"

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            logger.warning(f"⚠️  No data for {ticker}")
            return

        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # لو append، ادمج مع الموجود
        if mode == "append":
            old = _load_parquet(market, ticker)
            if old is not None:
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        _save_parquet(df, market, ticker)
        time.sleep(0.3)  # rate limiting

    except Exception as e:
        logger.error(f"❌ yfinance error for {ticker}: {e}")


# ─── Stooq Fetcher (AE fallback) ─────────────────────────────────────────────

def fetch_stooq(ticker: str, stooq_sym: str, market: str = "AE", start: str = "2015-01-01"):
    """يجيب داتا UAE من Stooq كـ fallback"""
    last = _last_saved_date(market, ticker)
    if last:
        start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&d1={start.replace('-','')}&d2={datetime.today().strftime('%Y%m%d')}&i=d"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or "No data" in r.text or len(r.text) < 50:
            logger.warning(f"⚠️  Stooq no data for {ticker}")
            return

        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"], index_col="Date")
        df.columns = [c.strip().capitalize() for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.sort_index()

        if _last_saved_date(market, ticker):
            old = _load_parquet(market, ticker)
            if old is not None:
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        _save_parquet(df, market, ticker)
        time.sleep(0.5)

    except Exception as e:
        logger.error(f"❌ Stooq error for {ticker}: {e}")


# ─── Bulk Historical Download ─────────────────────────────────────────────────

def bulk_download(start: str = "2015-01-01"):
    """
    يحمّل كل الداتا التاريخية من 2015 لحد النهارده
    شغّله مرة واحدة — ياخد 10-20 دقيقة
    """
    logger.info("🚀 Starting BULK DOWNLOAD...")
    logger.info(f"📅 From: {start} → Today")

    # السعودي
    logger.info(f"\n🇸🇦 Saudi — {len(SA_TICKERS)} tickers")
    for t in SA_TICKERS:
        fetch_yfinance(t, "SA", start=start)

    # المصري
    logger.info(f"\n🇪🇬 Egypt — {len(EG_TICKERS)} tickers")
    for t in EG_TICKERS:
        fetch_yfinance(t, "EG", start=start)

    # الإماراتي — yfinance أول، Stooq للباقي
    logger.info(f"\n🇦🇪 UAE — {len(AE_TICKERS)} tickers")
    for name, ticker in AE_TICKERS.items():
        if ticker.startswith("^"):
            fetch_yfinance(ticker, "AE", start=start)
        elif ticker in AE_STOOQ_MAP:
            fetch_stooq(ticker, AE_STOOQ_MAP[ticker], start=start)
        else:
            fetch_yfinance(ticker, "AE", start=start)

    logger.info("\n✅ BULK DOWNLOAD COMPLETE!")
    _print_summary()


# ─── Daily EOD Update ─────────────────────────────────────────────────────────

def daily_update():
    """
    يحدّث الداتا بآخر يوم تداول
    شغّله كل يوم بعد إغلاق السوق (cron job)
    """
    today = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"📅 Daily EOD Update — {today}")

    for t in SA_TICKERS:
        fetch_yfinance(t, "SA")

    for t in EG_TICKERS:
        fetch_yfinance(t, "EG")

    for name, ticker in AE_TICKERS.items():
        if ticker.startswith("^"):
            fetch_yfinance(ticker, "AE")
        elif ticker in AE_STOOQ_MAP:
            fetch_stooq(ticker, AE_STOOQ_MAP[ticker])
        else:
            fetch_yfinance(ticker, "AE")

    logger.info("✅ Daily update complete!")


# ─── Summary ─────────────────────────────────────────────────────────────────

def _print_summary():
    """يطبع إحصائيات الداتا المحفوظة"""
    logger.info("\n📊 DATA SUMMARY:")
    for market in ["SA", "EG", "AE"]:
        path = DATA_DIR / "historical" / market
        if path.exists():
            files = list(path.glob("*.parquet"))
            total_rows = sum(len(pd.read_parquet(f)) for f in files)
            logger.info(f"  {market}: {len(files)} files, {total_rows:,} total rows")

# ─── EODHD Fetcher (AE) ───────────────────────────────────────────────────────

AE_EODHD_TICKERS = [
    "EMAAR.DFM", "ENBD.DFM", "DIB.DFM", "DU.DFM", "DEWA.DFM",
    "SALIK.DFM", "PARKIN.DFM", "EMAARDEV.DFM",
    "FAB.XADS", "ADIB.XADS", "ALDAR.XADS", "TAQA.XADS",
    "IHC.XADS", "EAND.XADS", "ADNOCDIST.XADS",
]

def fetch_eodhd(ticker: str, market: str = "AE", start: str = "2015-01-01"):
    """يجيب UAE data من EODHD"""
    import os
    api_key = os.getenv("EODHD_API_KEY", "69a490505cfaf4.80968728")

    last = _last_saved_date(market, ticker)
    if last:
        start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"https://eodhd.com/api/eod/{ticker}?from={start}&to={datetime.today().strftime('%Y-%m-%d')}&period=d&api_token={api_key}&fmt=json"

    try:
        r = requests.get(url, timeout=15)
        data = r.json()

        if not data or isinstance(data, dict):
            logger.warning(f"⚠️ EODHD no data for {ticker}")
            return

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df = df[["Open","High","Low","Close","Volume"]].dropna()

        if last:
            old = _load_parquet(market, ticker)
            if old is not None:
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        _save_parquet(df, market, ticker)
        time.sleep(0.5)

    except Exception as e:
        logger.error(f"❌ EODHD error for {ticker}: {e}")


def fetch_all_uae(start: str = "2015-01-01"):
    """يجيب كل الإماراتي من EODHD"""
    logger.info(f"\n🇦🇪 UAE via EODHD — {len(AE_EODHD_TICKERS)} tickers")
    for t in AE_EODHD_TICKERS:
        fetch_eodhd(t, start=start)
# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bulk"

    if cmd == "bulk":
        start_year = sys.argv[2] if len(sys.argv) > 2 else "2015-01-01"
        bulk_download(start=start_year)
    elif cmd == "daily":
        daily_update()
    elif cmd == "summary":
        _print_summary()
    else:
        print("Usage: python data_fetcher.py [bulk|daily|summary] [start_date]")
