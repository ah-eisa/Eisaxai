#!/usr/bin/env python3
"""
populate_egypt_fundamentals.py — Egypt (EGX) fundamentals via StockAnalysis.com
"""
import sys, os, time, re, sqlite3, logging
sys.path.insert(0, "/home/ubuntu/investwise")
os.chdir("/home/ubuntu/investwise")
from datetime import datetime
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("populate_egypt")
DB_PATH = Path("/home/ubuntu/investwise/core/investwise.db")

# ── EGX tickers — comprehensive list of major Egyptian stocks ─────────────
EGYPT_TICKERS = [
    # EGX 30 Blue Chips
    "COMI.CA",   # Commercial International Bank (CIB)
    "HRHO.CA",   # Heliopolis Housing & Development
    "TMGH.CA",   # Talaat Moustafa Group
    "SWDY.CA",   # Sidi Kerir Petrochemicals (SIDPEC)
    "ETEL.CA",   # Telecom Egypt
    "OCDI.CA",   # Orascom Development Egypt
    "PHDC.CA",   # Palm Hills Development
    "EAST.CA",   # Eastern Company
    "ABUK.CA",   # Abu Qir Fertilizers
    "ORAS.CA",   # Orascom Construction
    "MNHD.CA",   # Madinat Nasr Housing
    "EKHO.CA",   # Egyptian Kuwaiti Holding
    "MFPC.CA",   # Misr Fertilizers Production (MOPCO)
    "JUFO.CA",   # Juhayna Food Industries
    "GTHE.CA",   # Ghabbour Auto
    "CLHO.CA",   # Click (CI Capital)
    "EFIH.CA",   # Egyptian Financial & Industrial
    "DOMT.CA",   # Domty
    "CIRA.CA",   # Cairo Investment & Real Estate
    "ALCN.CA",   # Alexandria Container & Cargo Handling
    "BTFH.CA",   # Beltone Financial Holding
    "SMFR.CA",   # Six of October Development (SODIC)
    "ACGC.CA",   # Arab Cotton Ginning
    "PIOH.CA",   # Pioneers Holding
    "AMOC.CA",   # Alexandria Mineral Oils
    "SKPC.CA",   # Sika (Misr Cement / Wadi Degla?)
    "SUGR.CA",   # Egyptian Sugar & Integrated Industries
    "SPMD.CA",   # Speed Medical
    "PORT.CA",   # Alexandria Port Development
    "RAIA.CA",   # Raya Holding
    # EGX 70 and other active stocks
    "CCAP.CA",   # Cairo Capital
    "ISPH.CA",   # International Pharmaceuticals (IBNSINA Pharma)
    "PHAR.CA",   # Pharos Holding
    "MPCO.CA",   # Misr Petroleum
    "SAIB.CA",   # SAIB Bank
    "ADIB.CA",   # Abu Dhabi Islamic Bank Egypt
    "EGBE.CA",   # Egyptian Gulf Bank
    "EFIC.CA",   # Egyptian Financial & Industrial
    "HELI.CA",   # Heliopolis Company for Housing
    "EGTS.CA",   # Egyptian Transport (EGYTRANS)
    "ARCO.CA",   # Arabian Cotton Ginning
    "UEGC.CA",   # Upper Egypt General Contracting
    "QNBA.CA",   # QNB Al Ahli Bank
    "CSAG.CA",   # Contact Financial Holding
    "MTIE.CA",   # Misr Technology & Information (e-finance)
    "EGIC.CA",   # Egypt Insurance Group
    "MNCO.CA",   # Medinet Nasr for Housing
    "NASR.CA",   # NASR City Housing
    "NBEK.CA",   # National Bank of Egypt
    "ITCE.CA",   # International Telecom Egypt
    "ENPA.CA",   # Edita Food Industries
    "ALAW.CA",   # Al Ahly Capital
    "BIND.CA",   # Bisco Misr
    "ELEC.CA",   # Egyptian Electricity
    "EGCP.CA",   # Egyptian Chemical
    "ESRS.CA",   # Ezz Steel (Ezz El Dekheila)
    "IRON.CA",   # Iron & Steel Egyptian
    "ALEX.CA",   # Alexandria Spinning & Weaving
    "AMRK.CA",   # Americana (Egypt listing)
    "CAID.CA",   # Cairo Development
    "CEPC.CA",   # Cairo Electric Poles
    "CIEB.CA",   # CIB (already COMI?)
    "DKHL.CA",   # Dice (Domiaty Holding?)
    "ECAP.CA",   # Egyptian Contracting
    "EGME.CA",   # Egyptian Media Production City
    "EKPC.CA",   # East Cairo Power
    "GDCO.CA",   # Gulf Development for Construction
    "GHAB.CA",   # Ghabbour (already GTHE?)
    "HDBK.CA",   # Housing & Development Bank
    "HWEL.CA",   # Helnan Hotels
    "INCO.CA",   # Industrial Commercial Bank
    "JUHD.CA",   # Juhayna Dairy
    "KAFR.CA",   # Kafr El-Zayat Pesticides
    "LGTH.CA",   # Lecico Egypt
    "MCQE.CA",   # Middle East Glass
    "MICE.CA",   # Misr Insurance
    "MISE.CA",   # Misr Exterior Trade
    "MITS.CA",   # Misr Insurance Holding
    "MOHG.CA",   # Moharram & Partners
    "NCGC.CA",   # National Cement
    "NEEM.CA",   # National Egyptian Export
    "NPTS.CA",   # North Pipe
    "OCPH.CA",   # Orascom Pharma
    "ORHD.CA",   # Orascom Hotels & Dev
    "ORCO.CA",   # Orascom Construction (dup?)
    "POLE.CA",   # Misr Alexandria Glass
    "POLY.CA",   # Egyptian Polypropylene
    "RDAL.CA",   # Raya Data
    "SDCM.CA",   # Six of October for Dry Cleaning
    "SFCO.CA",   # Sinai Cement
    "SFID.CA",   # Saudi Finance
    "SHTH.CA",   # Sharm El Sheikh Tourism
    "SIPH.CA",   # Sinai Petroleum
    "SMIC.CA",   # South Misr Spinning
    "SNCO.CA",   # Suez Canal Company
    "SOFA.CA",   # Sofa Interior
    "SOMA.CA",   # South Mining
    "STEI.CA",   # Steel Industries
    "STPC.CA",   # Setai Capital
    "TALM.CA",   # Talaat Group
    "TEBA.CA",   # Teba (Clothing)
    "TELB.CA",   # Telecom Egypt (dup?)
    "TICO.CA",   # Tico Trade
    "TRAC.CA",   # Trac (Transport)
    "UASG.CA",   # United Arab Shipping
    "UHIA.CA",   # Upper Egypt Housing
    "UORM.CA",   # Upper Egypt Mills
    "WATA.CA",   # Water Ways Tourism
    "WEMC.CA",   # Wadi El-Nil Real Estate
    "ZCOM.CA",   # Zamalek Comm
    "EGCH.CA",   # Egyptian Chemical Holding
    "GAPD.CA",   # El Nasr Transformers
    "NCIT.CA",   # National Company IT
    "CIEB.CA",   # CIB Egypt (same as COMI?)
    "ECME.CA",   # Egyptian Company Medical Equipment
    "SAIL.CA",   # Sail
    "SFER.CA",   # Sfeir (textiles)
    "SGBR.CA",   # Suez Gear Boxes
    "SPPD.CA",   # Sinai Portland Cement
    "SRGE.CA",   # Egyptian Resorts
]
EGYPT_TICKERS = list(dict.fromkeys(EGYPT_TICKERS))


def fetch_sa(ticker: str) -> dict:
    slug = ticker.upper().replace(".CA","").lower()
    url  = f"https://stockanalysis.com/quote/egx/{slug}/"
    try:
        r = requests.get(url, headers={
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120",
            "Accept-Language":"en-US,en;q=0.9"}, timeout=12)
        if r.status_code != 200:
            logger.warning(f"  {ticker} → HTTP {r.status_code}")
            return {}
        text = r.text
    except Exception as e:
        logger.warning(f"  {ticker} fetch error: {e}")
        return {}

    def _rx(pat):
        m = re.search(pat, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    # HTML format: key:"value" — unquoted keys, quoted or raw values
    out = {"ticker": ticker.upper(), "source": "StockAnalysis/EGX"}
    for key, pat in [
        ("mc_str",       r'marketCap:"([^"]+)"'),
        ("eps",          r'eps:"([^"]+)"'),
        ("pe_ratio",     r'peRatio:"([^"]+)"'),
        ("forward_pe",   r'forwardPE:"([^"]+)"'),
        ("beta",         r'beta:"([^"]+)"'),
        ("rev_str",      r'revenue:"([^"]+)"'),
        ("ni_str",       r'netIncome:"([^"]+)"'),
        ("sh_str",       r'sharesOut:"([^"]+)"'),
        ("div_yield",    r'dividendYield:"([^"]+)"'),
        ("rev_growth",   r'revenueGrowth:([-\d.]+)'),
        ("eps_growth",   r'epsGrowth:([-\d.]+)'),
        ("sector",       r'"sector"\s*:\s*"([^"]+)"'),
        ("industry",     r'"industry"\s*:\s*"([^"]+)"'),
        ("company_name", r'<title>([^|<(]+)'),
    ]:
        v = _rx(pat)
        if v:
            sv = str(v).strip().lower()
            if sv not in ('n/a', 'na', '', '-', 'none', 'null'):
                out[key] = v

    if out.get("company_name"):
        out["company_name"] = out["company_name"].strip().split(" Stock")[0].split(" (")[0]

    # Convert size strings → floats
    def _sz(val):
        if not val: return None
        s = str(val).strip().replace(',','')
        try:
            if 'T' in s: return float(re.sub(r'[^\d.]','',s.split('T')[0])) * 1e12
            if 'B' in s: return float(re.sub(r'[^\d.]','',s.split('B')[0])) * 1e9
            if 'M' in s: return float(re.sub(r'[^\d.]','',s.split('M')[0])) * 1e6
            return float(re.sub(r'[^\d.]','',s))
        except: return None

    out["market_cap"] = _sz(out.pop("mc_str", None))
    out["revenue"]    = _sz(out.pop("rev_str", None))
    out["net_income"] = _sz(out.pop("ni_str", None))
    out["shares_out"] = _sz(out.pop("sh_str", None))

    # Net margin
    try:
        rev = out.get("revenue"); ni = out.get("net_income")
        if rev and ni and rev > 0: out["net_margin"] = round(ni/rev*100, 2)
    except: pass

    # Div yield: "4.93%" → 0.0493
    try:
        dv = str(out.get("div_yield","")).replace('%','').strip()
        if dv: out["div_yield"] = float(dv)/100 if float(dv) > 1 else float(dv)
    except: pass

    return {k:v for k,v in out.items() if v is not None}


def get_price_yf(ticker):
    try:
        import yfinance as yf, warnings
        warnings.filterwarnings('ignore')
        p = getattr(yf.Ticker(ticker).fast_info, 'last_price', None)
        return round(float(p),3) if p and p > 0 else 0.0
    except: return 0.0


def upgrade_schema(conn):
    new_cols = {"forward_pe":"REAL","net_margin":"REAL","gross_margin":"REAL",
                "roe":"REAL","debt_equity":"REAL","revenue_growth":"REAL",
                "earnings_growth":"REAL","sector":"TEXT","industry":"TEXT",
                "net_income":"REAL","shares_out":"REAL","week_52_high":"REAL",
                "week_52_low":"REAL","price":"REAL","company_name":"TEXT","source":"TEXT"}
    existing = {row[1] for row in conn.execute("PRAGMA table_info(uae_fundamentals)")}
    for col, dtype in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE uae_fundamentals ADD COLUMN {col} {dtype}")
    conn.commit()


def populate():
    total = len(EGYPT_TICKERS)
    logger.info(f"🚀 Egypt fundamentals — {total} tickers")
    conn = sqlite3.connect(str(DB_PATH))
    upgrade_schema(conn)
    ok = fail = skip = 0

    for i, ticker in enumerate(EGYPT_TICKERS, 1):
        logger.info(f"[{i}/{total}] {ticker}")
        sa = fetch_sa(ticker)
        price = get_price_yf(ticker)

        def _f(k):
            try: return float(sa[k]) if sa.get(k) is not None else None
            except: return None

        eps=_f("eps"); pe=_f("pe_ratio"); fpe=_f("forward_pe"); beta=_f("beta")
        mc=_f("market_cap"); rev=_f("revenue"); ni=_f("net_income")
        dv=_f("div_yield"); rg=_f("rev_growth"); h52=_f("w52_high"); l52=_f("w52_low")
        nm=_f("net_margin")

        if pe is None and eps and price and eps > 0:
            pe = round(price/eps, 2)

        name = sa.get("company_name") or ticker.replace(".CA","")
        useful = sum(1 for v in [mc,pe,beta,eps,dv,rev,nm] if v)

        if useful < 1 and not price:
            logger.warning(f"  ⚠️  no data — skip")
            skip += 1
        else:
            try:
                conn.execute("""
                    INSERT INTO uae_fundamentals
                    (ticker,name,company_name,market_cap,pe_ratio,beta,eps,div_yield,
                     revenue,forward_pe,net_margin,revenue_growth,net_income,
                     sector,industry,week_52_high,week_52_low,price,source,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        name=COALESCE(excluded.name,uae_fundamentals.name),
                        company_name=COALESCE(excluded.company_name,uae_fundamentals.company_name),
                        market_cap=COALESCE(excluded.market_cap,uae_fundamentals.market_cap),
                        pe_ratio=COALESCE(excluded.pe_ratio,uae_fundamentals.pe_ratio),
                        beta=COALESCE(excluded.beta,uae_fundamentals.beta),
                        eps=COALESCE(excluded.eps,uae_fundamentals.eps),
                        div_yield=COALESCE(excluded.div_yield,uae_fundamentals.div_yield),
                        revenue=COALESCE(excluded.revenue,uae_fundamentals.revenue),
                        forward_pe=COALESCE(excluded.forward_pe,uae_fundamentals.forward_pe),
                        net_margin=COALESCE(excluded.net_margin,uae_fundamentals.net_margin),
                        revenue_growth=COALESCE(excluded.revenue_growth,uae_fundamentals.revenue_growth),
                        net_income=COALESCE(excluded.net_income,uae_fundamentals.net_income),
                        sector=COALESCE(excluded.sector,uae_fundamentals.sector),
                        industry=COALESCE(excluded.industry,uae_fundamentals.industry),
                        week_52_high=COALESCE(excluded.week_52_high,uae_fundamentals.week_52_high),
                        week_52_low=COALESCE(excluded.week_52_low,uae_fundamentals.week_52_low),
                        price=COALESCE(excluded.price,uae_fundamentals.price),
                        source=COALESCE(excluded.source,uae_fundamentals.source),
                        updated_at=excluded.updated_at
                """, (ticker,name,name,mc,pe,beta,eps,dv,rev,fpe,nm,rg,ni,
                      sa.get("sector"),sa.get("industry"),h52,l52,price,
                      sa.get("source","StockAnalysis/EGX"),datetime.now().isoformat()))
                conn.commit()
                ok += 1
                logger.info(f"  ✅ {name}: PE={pe} Beta={beta} Price={price}")
            except Exception as e:
                logger.error(f"  ❌ DB error: {e}")
                fail += 1

        if i < total: time.sleep(1.5)

    conn.close()
    logger.info(f"\n{'='*50}\n✅ Egypt Done: {ok} ok / {skip} skip / {fail} fail\n{'='*50}")

    conn2 = sqlite3.connect(str(DB_PATH))
    r = conn2.execute("SELECT COUNT(*),SUM(CASE WHEN pe_ratio IS NOT NULL THEN 1 ELSE 0 END),SUM(CASE WHEN beta IS NOT NULL THEN 1 ELSE 0 END),SUM(CASE WHEN sector IS NOT NULL THEN 1 ELSE 0 END) FROM uae_fundamentals WHERE ticker LIKE '%.CA'").fetchone()
    conn2.close()
    logger.info(f"📊 Egypt DB: {r[0]} rows | PE={r[1]} | Beta={r[2]} | Sector={r[3]}")

if __name__ == "__main__":
    populate()
