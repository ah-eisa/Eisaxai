"""
market_data_engine.py — EisaX Hybrid Data Engine
Cache-first: يجيب من المخزن لو موجود، أو يسكرب ويخزن
"""

import sys
sys.setrecursionlimit(5000)
import yfinance as yf
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import json
import time
import logging

# ── RapidAPI fallback لأي ticker مش في UAE_INVESTING ──────────────────────────
try:
    from rapidapi_client import fetch_ohlcv_with_fallback as _rapidapi_fetch, get_quote as _rapidapi_quote
    _RAPIDAPI_AVAILABLE = True
except ImportError:
    _RAPIDAPI_AVAILABLE = False
    _rapidapi_fetch  = None
    _rapidapi_quote  = None
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

from core.config import HISTORICAL_DATA_DIR as _cfg_hist, CORE_DB as _cfg_core_db
DATA_DIR = _cfg_hist
DB_PATH  = _cfg_core_db

# ─── Ticker Maps ─────────────────────────────────────────────────────────────

# السعودي والمصري والكويتي والقطري عبر yfinance
YFINANCE_MARKETS = {"SA": ".SR", "EG": ".CA", "KW": ".KW", "QA": ".QA"}

# الإماراتي (DFM + ADX) عبر Investing.com — 144 سهم
UAE_INVESTING = {
    "2POINTZERO.AE": {"id": "999113", "ref": "2pointzero-historical-data"},
    "ABNIC.DU": {"id": "999128", "ref": "abnic-historical-data"},
    "ADAVIATION.AE": {"id": "999184", "ref": "adaviation-historical-data"},
    "ADCB.AE": {"id": "1070165", "ref": "abu-dhabi-commercial-bank-historical-data"},
    "ADIB.AE": {"id": "941307", "ref": "abu-dhabi-islamic-bank-historical-data"},
    "ADNH.AE": {"id": "999163", "ref": "adnh-historical-data"},
    "ADNHC.AE": {"id": "1070354", "ref": "adnhc-historical-data"},
    "ADNIC.AE": {"id": "999058", "ref": "adnic-historical-data"},
    "ADNOCDIST.AE": {"id": "1055158", "ref": "national-oil-historical-data"},
    "ADNOCDRILL.AE": {"id": "999132", "ref": "adnoc-drilling-historical-data"},
    "ADNOCGAS.AE": {"id": "999193", "ref": "adnoc-gas-historical-data"},
    "ADNOCLS.DU": {"id": "999196", "ref": "adnocls-historical-data"},
    "ADPORTS.AE": {"id": "941289", "ref": "adports-historical-data"},
    "ADSB.AE": {"id": "1070374", "ref": "adsb-historical-data"},
    "AGILITY.AE": {"id": "1070159", "ref": "agility-historical-data"},
    "AGTHIA.AE": {"id": "941309", "ref": "agthia-group-historical-data"},
    "AIRA.DU": {"id": "999165", "ref": "aira-historical-data"},
    "AIRARABI.DU": {"id": "12530", "ref": "airarabi-historical-data"},
    "AJBNK.DU": {"id": "1070503", "ref": "ajbnk-historical-data"},
    "ALAIN.DU": {"id": "1070136", "ref": "alain-historical-data"},
    "ALANSARI.DU": {"id": "1201945", "ref": "alansari-historical-data"},
    "ALDAR.AE": {"id": "941317", "ref": "aldar-properties-historical-data"},
    "ALEFEDT.AE": {"id": "1070464", "ref": "alefedt-historical-data"},
    "ALFH.DU": {"id": "1070370", "ref": "alfh-historical-data"},
    "ALPHADATA.AE": {"id": "1070130", "ref": "alphadata-historical-data"},
    "ALPHADHABI.AE": {"id": "1070061", "ref": "alpha-dhabi-holding-historical-data"},
    "ALRAMZ.DU": {"id": "999185", "ref": "alramz-historical-data"},
    "AMAN.DU": {"id": "999140", "ref": "aman-historical-data"},
    "AMANT.DU": {"id": "941310", "ref": "amant-historical-data"},
    "AMCREIT.DU": {"id": "999057", "ref": "amcreit-historical-data"},
    "AMLAK.DU": {"id": "40413", "ref": "amlak-historical-data"},
    "AMLK.DU": {"id": "941295", "ref": "amlk-historical-data"},
    "APEX.AE": {"id": "999169", "ref": "apex-historical-data"},
    "ARAM.AE": {"id": "999137", "ref": "aram-historical-data"},
    "ARMX.DU": {"id": "12534", "ref": "armx-historical-data"},
    "ASM.DU": {"id": "1070300", "ref": "asm-historical-data"},
    "AWNIC.AE": {"id": "999130", "ref": "awnic-historical-data"},
    "BHMCAPITAL.DU": {"id": "941327", "ref": "bhmcapital-historical-data"},
    "BOROUGE.AE": {"id": "1070362", "ref": "borouge-historical-data"},
    "BOS.AE": {"id": "1070360", "ref": "bos-historical-data"},
    "BURJEEL.AE": {"id": "1070400", "ref": "burjeel-historical-data"},
    "CBD.DU": {"id": "941308", "ref": "commercial-bank-of-dubai-historical-data"},
    "CBI.AE": {"id": "941281", "ref": "cbi-historical-data"},
    "DANA.AE": {"id": "941285", "ref": "dana-gas-historical-data"},
    "DEWA.DU": {"id": "941326", "ref": "dubai-electricity-water-historical-data"},
    "DEWAA.DU": {"id": "999136", "ref": "dewaa-historical-data"},
    "DEYAAR.DU": {"id": "12538", "ref": "deyaar-historical-data"},
    "DEYR.DU": {"id": "941320", "ref": "deyr-historical-data"},
    "DAMAC.DU": {"id": "12558", "ref": "damac-properties-dubai-co-psc-historical-data"},
    "DFM.DU": {"id": "12539", "ref": "dfm-historical-data"},
    "DHAFRA.DU": {"id": "999067", "ref": "dhafra-historical-data"},
    "DIB.DU": {"id": "941311", "ref": "dubai-islamic-bank-historical-data"},
    "DINC.DU": {"id": "1070271", "ref": "dinc-historical-data"},
    "DINV.DU": {"id": "999117", "ref": "dubai-investments-historical-data"},
    "DISB.DU": {"id": "941294", "ref": "disb-historical-data"},
    "DNIN.DU": {"id": "999181", "ref": "dnin-historical-data"},
    "DRC.DU": {"id": "1070227", "ref": "drc-historical-data"},
    "DRIVE.AE": {"id": "999197", "ref": "drive-historical-data"},
    "DSI.DU": {"id": "1070403", "ref": "dsi-historical-data"},
    "DTC.DU": {"id": "999116", "ref": "dtc-historical-data"},
    "DU.DU": {"id": "941321", "ref": "emirates-integrated-telecommunications-historical-data"},
    "DUBAIRESI.DU": {"id": "941314", "ref": "dubairesi-historical-data"},
    "DUBAITAXI.DU": {"id": "1209220", "ref": "dubaitaxi-historical-data"},
    "E7.DU": {"id": "999148", "ref": "e7-historical-data"},
    "EAND.AE": {"id": "1070214", "ref": "emirates-telecommunications-historical-data"},
    "EASYLEASE.AE": {"id": "1070286", "ref": "easylease-historical-data"},
    "EIBAN.DU": {"id": "1055210", "ref": "eiban-historical-data"},
    "EIC.AE": {"id": "1070228", "ref": "eic-historical-data"},
    "EMAAR.DU": {"id": "1055159", "ref": "emaar-properties-historical-data"},
    "EMAARDEV.DU": {"id": "1055160", "ref": "emaar-development-historical-data"},
    "EMAR.DU": {"id": "1070039", "ref": "emar-historical-data"},
    "EMPOWER.DU": {"id": "1197172", "ref": "empower-historical-data"},
    "EMSTEEL.AE": {"id": "941313", "ref": "emsteel-historical-data"},
    "ENBD.DU": {"id": "12548", "ref": "emirates-nbd-historical-data"},
    "ENBDREIT.DU": {"id": "1070191", "ref": "enbdreit-historical-data"},
    "ERC.DU": {"id": "999158", "ref": "erc-historical-data"},
    "ESG.AE": {"id": "1070195", "ref": "esg-historical-data"},
    "FAB.AE": {"id": "999060", "ref": "first-abu-dhabi-bank-historical-data"},
    "FBI.DU": {"id": "999174", "ref": "fbi-historical-data"},
    "FERTIGLB.AE": {"id": "941324", "ref": "fertiglobe-historical-data"},
    "FH.AE": {"id": "999119", "ref": "fh-historical-data"},
    "GCEM.DU": {"id": "999129", "ref": "gcem-historical-data"},
    "GHITHA.AE": {"id": "999110", "ref": "ghitha-historical-data"},
    "GMPC.AE": {"id": "941300", "ref": "gmpc-historical-data"},
    "GNAV.DU": {"id": "12550", "ref": "gulf-navigation-historical-data"},
    "HH.DU": {"id": "1070103", "ref": "hh-historical-data"},
    "ICAP.AE": {"id": "941297", "ref": "icap-historical-data"},
    "IH.DU": {"id": "941293", "ref": "ih-historical-data"},
    "IHC.AE": {"id": "941296", "ref": "international-holding-historical-data"},
    "INVICTUS.AE": {"id": "1070190", "ref": "invictus-historical-data"},
    "JULPHAR.AE": {"id": "999141", "ref": "julphar-historical-data"},
    "KICO.DU": {"id": "941305", "ref": "kico-historical-data"},
    "LULU.AE": {"id": "999126", "ref": "lulu-historical-data"},
    "MAIR.AE": {"id": "941303", "ref": "mair-historical-data"},
    "MANAZEL.AE": {"id": "1070379", "ref": "manazel-historical-data"},
    "MASB.DU": {"id": "999082", "ref": "masb-historical-data"},
    "MASQ.DU": {"id": "941323", "ref": "mashreqbank-historical-data"},
    "MODON.AE": {"id": "941299", "ref": "modon-historical-data"},
    "NBF.AE": {"id": "941292", "ref": "nbf-historical-data"},
    "NBQ.AE": {"id": "999127", "ref": "nbq-historical-data"},
    "NCC.DU": {"id": "941325", "ref": "national-cement-historical-data"},
    "NCTH.DU": {"id": "999133", "ref": "ncth-historical-data"},
    "NGIN.DU": {"id": "1070264", "ref": "ngin-historical-data"},
    "NMDC.AE": {"id": "1070213", "ref": "nmdc-historical-data"},
    "NMDCENR.AE": {"id": "999198", "ref": "nmdcenr-historical-data"},
    "PALMS.AE": {"id": "1070259", "ref": "palms-historical-data"},
    "PARKIN.DU": {"id": "1212798", "ref": "parkin-historical-data"},
    "PHX.AE": {"id": "999135", "ref": "phx-historical-data"},
    "PRESIGHT.AE": {"id": "999134", "ref": "presight-historical-data"},
    "PUREHEALTH.AE": {"id": "999114", "ref": "purehealth-historical-data"},
    "QIC.DU": {"id": "999192", "ref": "qic-historical-data"},
    "RAKBANK.AE": {"id": "1070318", "ref": "national-bank-ras-al-khaimah-historical-data"},
    "RAKCEC.AE": {"id": "999168", "ref": "rakcec-historical-data"},
    "RAKNIC.DU": {"id": "999183", "ref": "raknic-historical-data"},
    "RAKPROP.AE": {"id": "999147", "ref": "rakprop-historical-data"},
    "RAKWCT.DU": {"id": "941302", "ref": "rakwct-historical-data"},
    "RAPCO.AE": {"id": "941332", "ref": "rapco-historical-data"},
    "RPM.DU": {"id": "1070149", "ref": "rpm-historical-data"},
    "SALAMA.DU": {"id": "1070469", "ref": "islamic-arab-insurance-historical-data"},
    "SALIK.DU": {"id": "1194944", "ref": "salik-historical-data"},
    "SCIDC.DU": {"id": "999151", "ref": "scidc-historical-data"},
    "SHUAA.DU": {"id": "12557", "ref": "shuaa-historical-data"},
    "SIB.DU": {"id": "999103", "ref": "sib-historical-data"},
    "SICO.DU": {"id": "999187", "ref": "sico-historical-data"},
    "SPINNEYS.DU": {"id": "1214529", "ref": "spinneys-historical-data"},
    "SSUD.DU": {"id": "1070498", "ref": "ssud-historical-data"},
    "SUDATEL.DU": {"id": "999145", "ref": "sudatel-historical-data"},
    "SUKOON.DU": {"id": "999167", "ref": "sukoon-historical-data"},
    "SUKOONT.DU": {"id": "999097", "ref": "sukoont-historical-data"},
    "TAALEEM.DU": {"id": "1198050", "ref": "taaleem-historical-data"},
    "TABR.DU": {"id": "941306", "ref": "tabr-historical-data"},
    "TABREED.DU": {"id": "941329", "ref": "national-central-cooling-tabreed-historical-data"},
    "TALABAT.DU": {"id": "1224079", "ref": "talabat-historical-data"},
    "TAQA.AE": {"id": "941330", "ref": "taqa-historical-data"},
    "TECOM.DU": {"id": "1192698", "ref": "tecom-historical-data"},
    "TKFE.DU": {"id": "999150", "ref": "tkfe-historical-data"},
    "TKFL.DU": {"id": "941280", "ref": "tkfl-historical-data"},
    "UAB.AE": {"id": "999152", "ref": "uab-historical-data"},
    "UFC.DU": {"id": "1070253", "ref": "ufc-historical-data"},
    "UNIK.DU": {"id": "1070086", "ref": "unik-historical-data"},
    "UNION.DU": {"id": "999155", "ref": "union-historical-data"},
    "UNIONCOOP.AE": {"id": "999190", "ref": "unioncoop-historical-data"},
    "UPRO.DU": {"id": "999144", "ref": "upro-historical-data"},
    "WAHA.AE": {"id": "999186", "ref": "waha-capital-historical-data"},
    "WATANIA.DU": {"id": "999109", "ref": "watania-historical-data"},
}

# ─── DB Setup ─────────────────────────────────────────────────────────────────

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_reports (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT NOT NULL,
            market    TEXT NOT NULL,
            report    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_cache_meta (
            ticker     TEXT PRIMARY KEY,
            market     TEXT,
            last_updated TEXT,
            rows       INTEGER
        )
    """)
    conn.commit()
    conn.close()

_init_db()

# ─── Cache Helpers ────────────────────────────────────────────────────────────

def _parquet_path(ticker: str, market: str) -> Path:
    safe = ticker.replace(".", "_").replace("^", "IDX_")
    return DATA_DIR / market / f"{safe}.parquet"

def _load_cache(ticker: str, market: str) -> Optional[pd.DataFrame]:
    path = _parquet_path(ticker, market)
    if path.exists():
        return pd.read_parquet(path)
    return None

def _save_cache(df: pd.DataFrame, ticker: str, market: str):
    path = _parquet_path(ticker, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    # حدّث الـ meta
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO data_cache_meta (ticker, market, last_updated, rows)
        VALUES (?, ?, ?, ?)
    """, (ticker, market, datetime.today().strftime("%Y-%m-%d"), len(df)))
    conn.commit()
    conn.close()
    logger.info(f"✅ Cached {ticker} ({len(df)} rows)")

def _is_fresh(ticker: str, market: str) -> bool:
    """الداتا fresh لو آخر تحديث كان النهارده"""
    path = _parquet_path(ticker, market)
    if not path.exists():
        return False
    df = pd.read_parquet(path)
    if df.empty:
        return False
    last = pd.to_datetime(df.index[-1])
    # السوق مغلق الجمعة والسبت — نحسب آخر يوم تداول
    today = datetime.today()
    days_back = 1 if today.weekday() not in [4, 5] else (today.weekday() - 3)
    last_trading = today - timedelta(days=days_back)
    return last.date() >= last_trading.date()

# ─── yfinance Scraper (SA + EG) ───────────────────────────────────────────────

def _fetch_yfinance(ticker: str, market: str, start: str = "2018-01-01") -> Optional[pd.DataFrame]:
    old = _load_cache(ticker, market)
    if old is not None:
        last = pd.to_datetime(old.index[-1])
        start = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            return old
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)

        if old is not None:
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()

        # ── Sanity check: reject corrupted data ──────────────────────
        _last_close = df["Close"].iloc[-1] if not df.empty else 0
        _market_limits = {"SA": (0.5, 5000), "EG": (0.1, 50000), 
                          "KW": (0.01, 5000), "QA": (0.1, 500)}
        _lo, _hi = _market_limits.get(market, (0.001, 500000))
        if not (_lo <= _last_close <= _hi):
            logger.error(
                f"[SanityCheck] REJECTED {ticker}/{market}: "
                f"last_close={_last_close:,.2f} out of range [{_lo}, {_hi}]. "
                f"Likely yfinance returned wrong ticker data."
            )
            return old  # رجّع الداتا القديمة ولا تحفظ الفاسدة
        # ─────────────────────────────────────────────────────────────
        _save_cache(df, ticker, market)
        return df
    except Exception as e:
        logger.error(f"yfinance error {ticker}: {e}")
        return old

# ─── Investing.com Scraper (AE) ───────────────────────────────────────────────

def _parse_volume(v: str) -> float:
    v = v.replace(",", "").strip()
    if "M" in v: return float(v.replace("M", "")) * 1_000_000
    if "K" in v: return float(v.replace("K", "")) * 1_000
    try: return float(v)
    except: return 0

def _fetch_investing(ticker: str, info: dict, start: str = "2018-01-01") -> Optional[pd.DataFrame]:
    old = _load_cache(ticker, "AE")
    if old is not None:
        last = pd.to_datetime(old.index[-1])
        start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
        if start >= datetime.today().strftime("%Y-%m-%d"):
            return old

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.today()

    # chunks بحجم سنة
    chunks = []
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=365), end_dt)
        chunks.append((cur.strftime("%m/%d/%Y"), chunk_end.strftime("%m/%d/%Y")))
        cur = chunk_end + timedelta(days=1)

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": f"https://www.investing.com/equities/{info['ref']}",
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    all_records = []
    for st, en in chunks:
        try:
            r = scraper.post(
                "https://www.investing.com/instruments/HistoricalDataAjax",
                headers=headers,
                data={"curr_id": info["id"], "st_date": st, "end_date": en,
                      "interval_sec": "Daily", "sort_col": "date",
                      "sort_ord": "DESC", "action": "historical_data"},
                timeout=20,
            )
            soup = BeautifulSoup(r.text, "html.parser")
            for row in soup.select("#curr_table tbody tr"):
                cols = row.find_all("td")
                if len(cols) >= 5 and "No results" not in cols[0].text:
                    try:
                        all_records.append({
                            "Date":   pd.to_datetime(cols[0].text.strip()),
                            "Close":  float(cols[1].text.strip().replace(",", "")),
                            "Open":   float(cols[2].text.strip().replace(",", "")),
                            "High":   float(cols[3].text.strip().replace(",", "")),
                            "Low":    float(cols[4].text.strip().replace(",", "")),
                            "Volume": _parse_volume(cols[5].text.strip()) if len(cols) > 5 else 0,
                        })
                    except Exception as _e:
                        continue
            time.sleep(1.2)
        except Exception as e:
            logger.error(f"Investing.com chunk error {ticker} {st}-{en}: {e}")

    if not all_records:
        return old

    df = pd.DataFrame(all_records).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if old is not None:
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    _save_cache(df, ticker, "AE")
    return df

# ─── Main API ─────────────────────────────────────────────────────────────────

def get_stock_data(
    ticker: str,
    market: str,
    period: str = "1y",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    الواجهة الرئيسية — cache-first
    
    Args:
        ticker: "2222.SR", "COMI.CA", "EMAAR.DU"
        market: "SA", "EG", "AE"
        period: "1y", "6m", "3m", "2y", "5y", "all"
        force_refresh: يجيب فريش حتى لو عندنا cache
    
    Returns:
        DataFrame: OHLCV مفلتر بالـ period
    """
    # إذا الداتا fresh ومش محتاج refresh
    if not force_refresh and _is_fresh(ticker, market):
        df = _load_cache(ticker, market)
    else:
        # جيب وخزّن
        if market == "AE" and ticker in UAE_INVESTING:
            # ✅ الطريق الأصلي — investing.com عبر cloudscraper
            df = _fetch_investing(ticker, UAE_INVESTING[ticker])
            # لو فشل جرب RapidAPI
            if (df is None or df.empty) and _RAPIDAPI_AVAILABLE:
                logger.info(f"Cloudscraper failed for {ticker}, trying RapidAPI...")
                df = _rapidapi_fetch(ticker, market)

        elif market == "AE" and ticker not in UAE_INVESTING:
            # ✅ UAE ticker مش في mapping → RapidAPI مباشرة
            if _RAPIDAPI_AVAILABLE:
                logger.info(f"{ticker} not in UAE_INVESTING → RapidAPI fallback")
                df = _rapidapi_fetch(ticker, market)
            else:
                logger.warning(f"{ticker} not in UAE_INVESTING and RapidAPI unavailable")
                df = _load_cache(ticker, market)

        elif market in ("SA", "EG", "KW", "QA"):
            # ✅ yfinance أولاً، RapidAPI لو فشل
            df = _fetch_yfinance(ticker, market)
            if (df is None or df.empty) and _RAPIDAPI_AVAILABLE and market == "EG":
                logger.info(f"yfinance failed for {ticker} (EG), trying RapidAPI...")
                df = _rapidapi_fetch(ticker, market)

        else:
            logger.warning(f"Unknown market/ticker: {ticker}/{market}")
            df = _load_cache(ticker, market)

    if df is None or df.empty:
        return None

    # فلتر الـ period
    df.index = pd.to_datetime(df.index)
    today = datetime.today()
    periods = {
        "1m": 30, "3m": 90, "6m": 180,
        "1y": 365, "2y": 730, "5y": 1825, "all": 99999,
    }
    days = periods.get(period, 365)
    return df[df.index >= today - timedelta(days=days)]


def get_latest_price(ticker: str, market: str) -> Optional[dict]:
    """آخر سعر للسهم — historical أولاً، RapidAPI Quote لو فشل"""
    df = get_stock_data(ticker, market, period="1m")

    if df is not None and len(df) >= 2:
        last, prev = df.iloc[-1], df.iloc[-2]
        change = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
        currency = (
            "SAR" if market == "SA" else
            "EGP" if market == "EG" else
            "KWF" if market == "KW" else
            "QAR" if market == "QA" else
            "AED"
        )
        return {
            "ticker": ticker, "market": market,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "close": round(last["Close"], 2),
            "open": round(last["Open"], 2),
            "high": round(last["High"], 2),
            "low": round(last["Low"], 2),
            "volume": int(last["Volume"]),
            "change_pct": round(change, 2),
            "currency": currency,
        }

    # ── Fallback: RapidAPI Quote مباشرة (للأسهم اللي مفيش ليها history) ──────
    if _RAPIDAPI_AVAILABLE:
        logger.info(f"get_latest_price: no history for {ticker}, trying RapidAPI Quote")
        q = _rapidapi_quote(ticker, market)
        if q and q.get("price", 0) > 0:
            return {
                "ticker":     ticker,
                "market":     market,
                "date":       datetime.today().strftime("%Y-%m-%d"),
                "close":      round(q["price"], 2),
                "open":       round(q.get("open", q["price"]), 2),
                "high":       round(q.get("high", q["price"]), 2),
                "low":        round(q.get("low",  q["price"]), 2),
                "volume":     q.get("volume", 0),
                "change_pct": round(q.get("change_pct", 0), 2),
                "currency":   q.get("currency", "AED"),
                "source":     "rapidapi_quote",
            }

    return None


def get_returns(ticker: str, market: str, period: str = "1y") -> Optional[pd.Series]:
    """Daily returns للـ Portfolio Optimizer"""
    df = get_stock_data(ticker, market, period=period)
    if df is None:
        return None
    return df["Close"].pct_change().dropna()


# ─── Report Cache ─────────────────────────────────────────────────────────────

def save_report(ticker: str, market: str, report: str):
    """يحفظ تقرير التحليل في SQLite"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analysis_reports (ticker, market, report) VALUES (?, ?, ?)",
        (ticker, market, report)
    )
    conn.commit()
    conn.close()


def get_latest_report(ticker: str, market: str, max_age_hours: int = 24) -> Optional[str]:
    """يجيب آخر تقرير محفوظ لو مش قديم"""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT report, created_at FROM analysis_reports
        WHERE ticker=? AND market=?
        ORDER BY created_at DESC LIMIT 1
    """, (ticker, market)).fetchone()
    conn.close()

    if not row:
        return None

    report, created_at = row
    age = datetime.now() - datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
    if age.total_seconds() / 3600 > max_age_hours:
        return None
    return report


def get_data_summary() -> dict:
    """إحصائيات الداتا المخزنة"""
    summary = {}
    for market in ["SA", "EG", "AE"]:
        path = DATA_DIR / market
        if path.exists():
            files = list(path.glob("*.parquet"))
            total = sum(len(pd.read_parquet(f)) for f in files)
            summary[market] = {"tickers": len(files), "rows": total}
        else:
            summary[market] = {"tickers": 0, "rows": 0}
    return summary


# ─── للـ Finance Agent ────────────────────────────────────────────────────────

def prepare_analysis_context(ticker: str, market: str, period: str = "1y") -> dict:
    """
    يحضّر كل الداتا اللي DeepSeek محتاجها للتحليل
    Returns dict جاهز تبعته في الـ prompt
    """
    df = get_stock_data(ticker, market, period=period)
    if df is None:
        return {"error": f"No data for {ticker}"}

    latest = get_latest_price(ticker, market)
    returns = df["Close"].pct_change().dropna()

    # إحصائيات أساسية
    stats = {
        "ticker": ticker,
        "market": market,
        "period": period,
        "latest_price": latest,
        "data_points": len(df),
        "date_range": {
            "from": df.index[0].strftime("%Y-%m-%d"),
            "to": df.index[-1].strftime("%Y-%m-%d"),
        },
        "price_stats": {
            "high_52w": round(df["Close"].max(), 2),
            "low_52w": round(df["Close"].min(), 2),
            "avg_volume": int(df["Volume"].mean()),
        },
        "returns_stats": {
            "total_return_pct": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
            "volatility_annual": round(returns.std() * (252 ** 0.5) * 100, 2),
            "sharpe_approx": round((returns.mean() / returns.std()) * (252 ** 0.5), 2) if returns.std() > 0 else 0,
        },
        "recent_prices": {str(k): v for k, v in df["Close"].tail(30).round(2).to_dict().items()},
    }

    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ticker = sys.argv[1] if len(sys.argv) > 1 else "EMAAR.DU"
    market = sys.argv[2] if len(sys.argv) > 2 else "AE"

    print(f"\n📊 Testing: {ticker} / {market}")
    ctx = prepare_analysis_context(ticker, market, period="1y")
    import json
    print(json.dumps(ctx, indent=2, default=str))
