#!/usr/bin/env python3
"""
populate_egx_fundamentals.py — Populate Egypt (EGX) stock fundamentals
into egx_fundamentals table using yfinance + StockAnalysis EGX.

Usage:
    python3 scripts/populate_egx_fundamentals.py

Sources:
1. yfinance        → PE, Beta, MarketCap, 52W range, name
2. StockAnalysis   → PE, Revenue, Growth, Margins, ROE (egx exchange)
3. egx_lookup      → Existing sector/name context
"""

import sys, os, time, re, json, sqlite3, logging
sys.path.insert(0, "/home/ubuntu/investwise")
os.chdir("/home/ubuntu/investwise")

from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("populate_egx")

DB_PATH = Path("/home/ubuntu/investwise/core/investwise.db")

# Full EGX ticker list from StockAnalysis.com (224 tickers)
EGX_TICKERS = sorted([
    "AALR.CA", "ABUK.CA", "ACAMD.CA", "ACAP.CA", "ACGC.CA", "ACTF.CA",
    "ADCI.CA", "ADIB.CA", "ADPC.CA", "AFDI.CA", "AFMC.CA", "AIDC.CA",
    "AIHC.CA", "AJWA.CA", "ALCN.CA", "ALRA.CA", "ALUM.CA", "AMER.CA",
    "AMES.CA", "AMIA.CA", "AMOC.CA", "ANFI.CA", "APSW.CA", "ARAB.CA",
    "ARCC.CA", "AREH.CA", "ARVA.CA", "ASCM.CA", "ASPI.CA", "ATLC.CA",
    "ATQA.CA", "AXPH.CA", "BINV.CA", "BIOC.CA", "BONY.CA", "BTFH.CA",
    "CAED.CA", "CANA.CA", "CCAP.CA", "CCAPP.CA", "CCRS.CA", "CEFM.CA",
    "CERA.CA", "CFGH.CA", "CICH.CA", "CIEB.CA", "CIRA.CA", "CLHO.CA",
    "CNFN.CA", "COMI.CA", "COPR.CA", "COSG.CA", "CPCI.CA", "CPME.CA",
    "CRST.CA", "CSAG.CA", "DAPH.CA", "DCCC.CA", "DEIN.CA", "DGTZ.CA",
    "DOMT.CA", "DSCW.CA", "DTPP.CA", "EALR.CA", "EASB.CA", "EAST.CA",
    "EBSC.CA", "ECAP.CA", "EDFM.CA", "EEII.CA", "EFIC.CA", "EFID.CA",
    "EFIH.CA", "EGAL.CA", "EGAS.CA", "EGBE.CA", "EGCH.CA", "EGSA.CA",
    "EGTS.CA", "EHDR.CA", "ELEC.CA", "ELKA.CA", "ELNA.CA", "ELSH.CA",
    "ELWA.CA", "EMFD.CA", "ENGC.CA", "EOSB.CA", "EPCO.CA", "EPPK.CA",
    "ETEL.CA", "ETRS.CA", "EXPA.CA", "FAIT.CA", "FAITA.CA", "FERC.CA",
    "FWRY.CA", "GBCO.CA", "GDWA.CA", "GGCC.CA", "GGRN.CA", "GIHD.CA",
    "GMCI.CA", "GOUR.CA", "GPIM.CA", "GPPL.CA", "GRCA.CA", "GSSC.CA",
    "GTEX.CA", "GTWL.CA", "HDBK.CA", "HELI.CA", "HRHO.CA", "ICID.CA",
    "ICLE.CA", "IDRE.CA", "IFAP.CA", "INFI.CA", "IRON.CA", "ISMA.CA",
    "ISMQ.CA", "ISPH.CA", "JUFO.CA", "KABO.CA", "KRDI.CA", "KWIN.CA",
    "KZPC.CA", "LCSW.CA", "LUTS.CA", "MAAL.CA", "MASR.CA", "MBSC.CA",
    "MCQE.CA", "MCRO.CA", "MEGM.CA", "MENA.CA", "MEPA.CA", "MFPC.CA",
    "MFSC.CA", "MHOT.CA", "MICH.CA", "MILS.CA", "MIPH.CA", "MMAT.CA",
    "MOED.CA", "MOIL.CA", "MOIN.CA", "MOSC.CA", "MPCI.CA", "MPCO.CA",
    "MPRC.CA", "MTIE.CA", "NAHO.CA", "NAPR.CA", "NARE.CA", "NCCW.CA",
    "NDRL.CA", "NEDA.CA", "NHPS.CA", "NINH.CA", "NIPH.CA", "OBRI.CA",
    "OCDI.CA", "OCPH.CA", "ODIN.CA", "OFH.CA", "OIH.CA", "OLFI.CA",
    "ORAS.CA", "ORHD.CA", "ORWE.CA", "PHAR.CA", "PHDC.CA", "PHGC.CA",
    "PHTV.CA", "POCO.CA", "POUL.CA", "PRCL.CA", "PRDC.CA", "PRMH.CA",
    "QNBE.CA", "RACC.CA", "RAKT.CA", "RAYA.CA", "RMDA.CA", "ROTO.CA",
    "RREI.CA", "RTVC.CA", "RUBX.CA", "SAIB.CA", "SAUD.CA", "SCEM.CA",
    "SCFM.CA", "SCTS.CA", "SDTI.CA", "SEIG.CA", "SEIGA.CA", "SIPC.CA",
    "SKPC.CA", "SMFR.CA", "SNFC.CA", "SPHT.CA", "SPIN.CA", "SPMD.CA",
    "SUGR.CA", "SVCE.CA", "SWDY.CA", "TALM.CA", "TANM.CA", "TAQA.CA",
    "TMGH.CA", "TRTO.CA", "UBEE.CA", "UEFM.CA", "UEGC.CA", "UNIP.CA",
    "UNIT.CA", "VALU.CA", "VLMR.CA", "VLMRA.CA", "WCDF.CA", "WKOL.CA",
    "ZEOT.CA", "ZMID.CA",
])


def upgrade_schema(conn):
    """Add missing columns to egx_fundamentals."""
    new_cols = {
        "div_yield":        "REAL",
        "net_income":       "REAL",
        "shares_out":       "REAL",
        "week_52_high":     "REAL",
        "week_52_low":      "REAL",
        "price":            "REAL",
        "source":           "TEXT",
        "industry":         "TEXT",
        "forward_pe":       "REAL",
        "net_margin":       "REAL",
        "gross_margin":     "REAL",
        "roe":              "REAL",
        "debt_equity":      "REAL",
        "revenue_growth":   "REAL",
        "earnings_growth":  "REAL",
        "eps":              "REAL",
        "sector":           "TEXT",
        "company_name":     "TEXT",
    }
    existing = {row[1] for row in conn.execute("PRAGMA table_info(egx_fundamentals)")}
    for col, dtype in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE egx_fundamentals ADD COLUMN {col} {dtype}")
            logger.info(f"  Added column: {col} ({dtype})")
    conn.commit()


def fetch_yfinance(ticker: str) -> dict:
    """Fetch fundamentals via yfinance."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        mc = info.get("marketCap")
        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        eps = info.get("trailingEps")
        beta = info.get("beta")
        div = info.get("dividendYield")
        rev = info.get("totalRevenue")
        ni = info.get("netIncomeToCommon")
        sector = info.get("sector")
        industry = info.get("industry")
        name = info.get("longName") or info.get("shortName")
        w52h = info.get("fiftyTwoWeekHigh")
        w52l = info.get("fiftyTwoWeekLow")
        roe = info.get("returnOnEquity")
        gross_m = info.get("grossMargins")
        net_m = info.get("profitMargins")
        de = info.get("debtToEquity")
        rev_growth = info.get("revenueGrowth")
        earn_growth = info.get("earningsGrowth")
        shares = info.get("sharesOutstanding")
        avg_vol = info.get("averageVolume3Month")

        result = {}
        if price:      result["price"] = round(float(price), 3)
        if mc:         result["market_cap"] = float(mc)
        if pe:         result["pe_ratio"] = round(float(pe), 2)
        if fpe:        result["forward_pe"] = round(float(fpe), 2)
        if eps:        result["eps"] = round(float(eps), 4)
        if beta:       result["beta"] = round(float(beta), 3)
        if div:
            dv = float(div)
            result["div_yield"] = round(dv if dv < 1 else dv / 100, 4)
        if rev:        result["revenue"] = float(rev)
        if ni:         result["net_income"] = float(ni)
        if sector:     result["sector"] = sector
        if industry:   result["industry"] = industry
        if name:
            # yfinance sometimes returns composite garbage like "TICKER,0P0000XXXX,123456"
            # Filter these out — keep only real company names
            _nm = str(name).strip()
            if ',' not in _nm and not re.match(r'^[A-Z]+\.[A-Z]+,', _nm):
                result["company_name"] = _nm
        if w52h:       result["week_52_high"] = round(float(w52h), 3)
        if w52l:       result["week_52_low"] = round(float(w52l), 3)
        if roe:        result["roe"] = round(float(roe) * 100, 2)
        if gross_m:    result["gross_margin"] = round(float(gross_m) * 100, 2)
        if net_m:      result["net_margin"] = round(float(net_m) * 100, 2)
        if de:         result["debt_equity"] = round(float(de) / 100, 3)
        if rev_growth: result["revenue_growth"] = round(float(rev_growth) * 100, 2)
        if earn_growth:result["earnings_growth"] = round(float(earn_growth) * 100, 2)
        if shares:     result["shares_out"] = float(shares)
        if avg_vol:    result["avg_vol_3m"] = f"{avg_vol/1e6:.1f}M" if avg_vol > 1e6 else str(avg_vol)

        # Compute net margin from revenue + net_income if not provided
        if not result.get("net_margin") and result.get("net_income") and result.get("revenue") and result["revenue"] > 0:
            result["net_margin"] = round((result["net_income"] / result["revenue"]) * 100, 2)

        result["source"] = "yfinance (EGX)"
        return result

    except Exception as e:
        logger.warning(f"  [{ticker}] yfinance error: {e}")
        return {}


def fetch_stockanalysis_egx(ticker: str) -> dict:
    """Fetch additional data from StockAnalysis EGX exchange."""
    try:
        import requests
        slug = ticker.upper().replace(".CA", "").lower()
        url = f"https://stockanalysis.com/quote/egx/{slug}/"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            logger.debug(f"  [{ticker}] StockAnalysis EGX: {r.status_code}")
            return {}
        text = r.text

        def _rx(pattern, grp=1):
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(grp).strip() if m else None

        def _size_to_float(val):
            if not val: return None
            s = str(val).strip()
            try:
                if 'T' in s: return float(re.sub(r'[^\d.]', '', s.split('T')[0])) * 1e12
                if 'B' in s: return float(re.sub(r'[^\d.]', '', s.split('B')[0])) * 1e9
                if 'M' in s: return float(re.sub(r'[^\d.]', '', s.split('M')[0])) * 1e6
                return float(re.sub(r'[^\d.]', '', s))
            except:
                return None

        def _pct_to_float(val):
            if not val: return None
            try:
                return float(str(val).replace('%', '').replace('+', '').strip())
            except:
                return None

        result = {}

        mc_raw = _rx(r'marketCap[\"\':\s]+([0-9.e+\-]+)')
        if mc_raw:
            mc = float(mc_raw)
            result["market_cap"] = mc

        pe_raw = _rx(r'peRatio[\"\':\s]+([0-9.]+)')
        if pe_raw: result["pe_ratio"] = round(float(pe_raw), 2)

        rev_raw = _rx(r'"revenue"[:\s]+([0-9.e+\-]+)')
        if rev_raw: result["revenue"] = float(rev_raw)

        # Company name from page title
        title = _rx(r'<title>([^<]+)</title>')
        if title:
            name_m = re.match(r'^([^(]+)', title)
            if name_m:
                nm = name_m.group(1).strip()
                if len(nm) > 3:
                    result["company_name"] = nm

        return {k: v for k, v in result.items() if v is not None}

    except Exception as e:
        logger.warning(f"  [{ticker}] StockAnalysis EGX error: {e}")
        return {}


def get_egx_context(ticker: str) -> dict:
    """Get existing sector/name from egx_lookup."""
    try:
        from core.egx_lookup import get_egx_context as _get
        ctx = _get(ticker)
        if ctx:
            return {
                "company_name": ctx.get("name"),
                "sector":       ctx.get("sector"),
                "industry":     ctx.get("industry"),
            }
    except Exception:
        pass
    return {}


def populate():
    conn = sqlite3.connect(str(DB_PATH))
    upgrade_schema(conn)

    total = len(EGX_TICKERS)
    logger.info(f"🚀 Starting Egypt fundamentals population for {total} tickers")

    success = 0
    failed = 0
    skipped = 0

    for i, ticker in enumerate(EGX_TICKERS, 1):
        logger.info(f"[{i}/{total}] {ticker}...")

        # Primary: yfinance
        yf_data = fetch_yfinance(ticker)

        # Secondary: StockAnalysis EGX
        sa_data = fetch_stockanalysis_egx(ticker)

        # Tertiary: egx_lookup for sector/name context
        ctx_data = get_egx_context(ticker)

        # Merge: yfinance wins, SA fills gaps, ctx fills remaining gaps
        merged = {}
        merged.update(ctx_data)
        merged.update(sa_data)
        merged.update(yf_data)  # yfinance has priority

        if not merged:
            logger.warning(f"  ⚠️ {ticker}: No data from any source")
            skipped += 1
            continue

        company_name = merged.get("company_name") or ticker.replace(".CA", "")
        # Format market_cap and revenue as TEXT strings (egx_fundamentals uses TEXT for these)
        mc_val = merged.get("market_cap")
        mc_display = None
        if mc_val:
            if mc_val >= 1e12:   mc_display = f"{mc_val/1e12:.2f}T EGP"
            elif mc_val >= 1e9:  mc_display = f"{mc_val/1e9:.2f}B EGP"
            elif mc_val >= 1e6:  mc_display = f"{mc_val/1e6:.0f}M EGP"
            else:                mc_display = str(mc_val)

        rev_val = merged.get("revenue")
        rev_display = None
        if rev_val:
            if rev_val >= 1e12:   rev_display = f"{rev_val/1e12:.2f}T EGP"
            elif rev_val >= 1e9:  rev_display = f"{rev_val/1e9:.1f}B EGP"
            elif rev_val >= 1e6:  rev_display = f"{rev_val/1e6:.0f}M EGP"
            else:                 rev_display = str(rev_val)

        useful_fields = sum(1 for v in [
            mc_val, merged.get("pe_ratio"), merged.get("beta"),
            merged.get("eps"), merged.get("div_yield"), merged.get("revenue"),
            merged.get("net_margin"), merged.get("roe")
        ] if v)

        try:
            conn.execute("""
                INSERT INTO egx_fundamentals
                (ticker, name, company_name, market_cap, revenue, pe_ratio, beta,
                 eps, div_yield, forward_pe, net_margin, gross_margin,
                 roe, debt_equity, revenue_growth, earnings_growth,
                 net_income, shares_out, sector, industry,
                 week_52_high, week_52_low, price, source, avg_vol_3m, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name            = COALESCE(excluded.name, egx_fundamentals.name),
                    company_name    = COALESCE(excluded.company_name, egx_fundamentals.company_name),
                    market_cap      = COALESCE(excluded.market_cap, egx_fundamentals.market_cap),
                    revenue         = COALESCE(excluded.revenue, egx_fundamentals.revenue),
                    pe_ratio        = COALESCE(excluded.pe_ratio, egx_fundamentals.pe_ratio),
                    beta            = COALESCE(excluded.beta, egx_fundamentals.beta),
                    eps             = COALESCE(excluded.eps, egx_fundamentals.eps),
                    div_yield       = COALESCE(excluded.div_yield, egx_fundamentals.div_yield),
                    forward_pe      = COALESCE(excluded.forward_pe, egx_fundamentals.forward_pe),
                    net_margin      = COALESCE(excluded.net_margin, egx_fundamentals.net_margin),
                    gross_margin    = COALESCE(excluded.gross_margin, egx_fundamentals.gross_margin),
                    roe             = COALESCE(excluded.roe, egx_fundamentals.roe),
                    debt_equity     = COALESCE(excluded.debt_equity, egx_fundamentals.debt_equity),
                    revenue_growth  = COALESCE(excluded.revenue_growth, egx_fundamentals.revenue_growth),
                    earnings_growth = COALESCE(excluded.earnings_growth, egx_fundamentals.earnings_growth),
                    net_income      = COALESCE(excluded.net_income, egx_fundamentals.net_income),
                    shares_out      = COALESCE(excluded.shares_out, egx_fundamentals.shares_out),
                    sector          = COALESCE(excluded.sector, egx_fundamentals.sector),
                    industry        = COALESCE(excluded.industry, egx_fundamentals.industry),
                    week_52_high    = COALESCE(excluded.week_52_high, egx_fundamentals.week_52_high),
                    week_52_low     = COALESCE(excluded.week_52_low, egx_fundamentals.week_52_low),
                    price           = COALESCE(excluded.price, egx_fundamentals.price),
                    source          = COALESCE(excluded.source, egx_fundamentals.source),
                    avg_vol_3m      = COALESCE(excluded.avg_vol_3m, egx_fundamentals.avg_vol_3m),
                    updated_at      = excluded.updated_at
            """, (
                ticker, company_name, company_name,
                mc_display, rev_display,
                merged.get("pe_ratio"), merged.get("beta"),
                merged.get("eps"), merged.get("div_yield"),
                merged.get("forward_pe"), merged.get("net_margin"), merged.get("gross_margin"),
                merged.get("roe"), merged.get("debt_equity"),
                merged.get("revenue_growth"), merged.get("earnings_growth"),
                merged.get("net_income"), merged.get("shares_out"),
                merged.get("sector"), merged.get("industry"),
                merged.get("week_52_high"), merged.get("week_52_low"),
                merged.get("price"), merged.get("source"),
                merged.get("avg_vol_3m"),
                datetime.now().isoformat()
            ))
            conn.commit()
            success += 1
            logger.info(f"  ✅ {ticker} [{company_name}]: {useful_fields} fields | "
                        f"PE={merged.get('pe_ratio')} | ROE={merged.get('roe')} | "
                        f"Beta={merged.get('beta')} | Sector={merged.get('sector')}")
        except Exception as e:
            logger.error(f"  ❌ {ticker}: DB insert failed: {e}")
            failed += 1

        # Rate limit
        if i < total:
            time.sleep(1.0)

    conn.close()

    logger.info(f"""
{'='*60}
✅ Egypt Fundamentals Population Complete
{'='*60}
Total tickers:  {total}
Success:        {success}
Failed:         {failed}
Skipped:        {skipped}
{'='*60}
""")

    # Coverage summary
    conn2 = sqlite3.connect(str(DB_PATH))
    stats = conn2.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pe_ratio   IS NOT NULL THEN 1 ELSE 0 END) as has_pe,
            SUM(CASE WHEN beta       IS NOT NULL THEN 1 ELSE 0 END) as has_beta,
            SUM(CASE WHEN div_yield  IS NOT NULL THEN 1 ELSE 0 END) as has_div,
            SUM(CASE WHEN revenue    IS NOT NULL THEN 1 ELSE 0 END) as has_rev,
            SUM(CASE WHEN net_margin IS NOT NULL THEN 1 ELSE 0 END) as has_margin,
            SUM(CASE WHEN sector     IS NOT NULL THEN 1 ELSE 0 END) as has_sector,
            SUM(CASE WHEN roe        IS NOT NULL THEN 1 ELSE 0 END) as has_roe
        FROM egx_fundamentals
    """).fetchone()
    conn2.close()

    logger.info(f"""
📊 Egypt Coverage Summary:
  Total rows:    {stats[0]}
  Has PE:        {stats[1]}
  Has Beta:      {stats[2]}
  Has Dividend:  {stats[3]}
  Has Revenue:   {stats[4]}
  Has Margin:    {stats[5]}
  Has Sector:    {stats[6]}
  Has ROE:       {stats[7]}
""")


if __name__ == "__main__":
    populate()
