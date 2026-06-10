#!/usr/bin/env python3
"""
populate_saudi_fundamentals.py — Saudi (Tadawul) fundamentals via StockAnalysis.com
"""
import sys, os, time, re, sqlite3, logging
sys.path.insert(0, "/home/ubuntu/investwise")
os.chdir("/home/ubuntu/investwise")
from datetime import datetime
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("populate_saudi")
DB_PATH = Path("/home/ubuntu/investwise/core/investwise.db")

SAUDI_TICKERS = [
    # Banks
    "1010.SR","1020.SR","1030.SR","1050.SR","1060.SR","1080.SR",
    "1120.SR","1140.SR","1150.SR","1180.SR",
    # Petrochemicals / Industrial
    "2010.SR","2020.SR","2030.SR","2060.SR","2070.SR","2090.SR",
    "2150.SR","2160.SR","2180.SR","2200.SR","2210.SR","2220.SR",
    "2222.SR","2230.SR","2240.SR","2250.SR","2270.SR","2280.SR",
    "2290.SR","2300.SR","2310.SR","2320.SR","2330.SR","2340.SR",
    "2360.SR","2370.SR","2380.SR","2382.SR","2383.SR","2384.SR","2385.SR",
    # Cement
    "3002.SR","3003.SR","3005.SR","3008.SR","3010.SR","3020.SR",
    "3030.SR","3040.SR","3050.SR","3060.SR","3080.SR","3090.SR",
    "3091.SR","3092.SR","3093.SR",
    # Retail / Services / Healthcare / Real Estate
    "4002.SR","4003.SR","4005.SR","4007.SR","4008.SR","4009.SR",
    "4010.SR","4011.SR","4020.SR","4031.SR","4040.SR","4050.SR",
    "4051.SR","4061.SR","4065.SR","4080.SR","4100.SR","4110.SR",
    "4130.SR","4140.SR","4150.SR","4160.SR","4161.SR","4162.SR",
    "4163.SR","4164.SR","4165.SR","4166.SR","4170.SR","4180.SR",
    "4190.SR","4200.SR","4210.SR","4220.SR","4230.SR","4240.SR",
    "4250.SR","4261.SR","4262.SR","4263.SR","4264.SR","4270.SR",
    "4280.SR","4290.SR","4291.SR","4321.SR","4322.SR","4330.SR",
    "4331.SR","4332.SR","4333.SR","4334.SR","4335.SR","4336.SR",
    "4338.SR","4339.SR","4341.SR","4342.SR","4344.SR","4345.SR","4346.SR",
    # Electricity
    "5110.SR",
    # Telecom
    "7010.SR","7020.SR","7030.SR","7200.SR",
    # Insurance
    "8010.SR","8012.SR","8020.SR","8030.SR","8040.SR","8050.SR",
    "8060.SR","8070.SR","8100.SR","8120.SR","8150.SR","8160.SR",
    "8170.SR","8180.SR","8200.SR","8210.SR","8230.SR","8240.SR",
    "8250.SR","8260.SR","8270.SR","8280.SR","8300.SR","8311.SR",
]
SAUDI_TICKERS = list(dict.fromkeys(SAUDI_TICKERS))

def fetch_sa(ticker: str) -> dict:
    slug = ticker.upper().replace(".SR","").lower()
    url  = f"https://stockanalysis.com/quote/tadawul/{slug}/"
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

    # HTML format: key:"value" (unquoted keys, quoted or unquoted values)
    # e.g. marketCap:"6.55T", eps:"1.44", beta:"0.10", revenueGrowth:-7.242
    def _rxn(pat):
        """Match a numeric value — quoted or unquoted after the key."""
        m = re.search(pat, text, re.IGNORECASE)
        if not m: return None
        v = m.group(1).strip().strip('"')
        if v.lower() in ('n/a', 'na', '', '-', 'none', 'null'): return None
        return v

    out = {"ticker": ticker.upper(), "source": "StockAnalysis/Tadawul"}
    for key, pat in [
        ("mc_str",      r'marketCap:"([^"]+)"'),
        ("eps",         r'eps:"([^"]+)"'),
        ("pe_ratio",    r'peRatio:"([^"]+)"'),
        ("forward_pe",  r'forwardPE:"([^"]+)"'),
        ("beta",        r'beta:"([^"]+)"'),
        ("rev_str",     r'revenue:"([^"]+)"'),
        ("ni_str",      r'netIncome:"([^"]+)"'),
        ("sh_str",      r'sharesOut:"([^"]+)"'),
        ("div_yield",   r'dividendYield:"([^"]+)"'),
        ("rev_growth",  r'revenueGrowth:([-\d.]+)'),
        ("eps_growth",  r'epsGrowth:([-\d.]+)'),
        ("sector",      r'"sector"\s*:\s*"([^"]+)"'),
        ("industry",    r'"industry"\s*:\s*"([^"]+)"'),
        ("company_name",r'<title>([^|<(]+)'),
    ]:
        v = _rx(pat)
        if v: out[key] = v

    # Clean name
    if out.get("company_name"):
        out["company_name"] = out["company_name"].strip().split(" Stock")[0].split(" (")[0]

    # Convert size strings → floats (e.g. "6.55T" → 6.55e12, "1.67T" → 1.67e12)
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
    total = len(SAUDI_TICKERS)
    logger.info(f"🚀 Saudi fundamentals — {total} tickers")
    conn = sqlite3.connect(str(DB_PATH))
    upgrade_schema(conn)
    ok = fail = skip = 0

    for i, ticker in enumerate(SAUDI_TICKERS, 1):
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

        name = sa.get("company_name") or ticker.replace(".SR","")
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
                      sa.get("source","StockAnalysis/Tadawul"),datetime.now().isoformat()))
                conn.commit()
                ok += 1
                logger.info(f"  ✅ {name}: PE={pe} Beta={beta} Price={price}")
            except Exception as e:
                logger.error(f"  ❌ DB error: {e}")
                fail += 1

        if i < total: time.sleep(1.5)

    conn.close()
    logger.info(f"\n{'='*50}\n✅ Saudi Done: {ok} ok / {skip} skip / {fail} fail\n{'='*50}")

    conn2 = sqlite3.connect(str(DB_PATH))
    r = conn2.execute("SELECT COUNT(*),SUM(CASE WHEN pe_ratio IS NOT NULL THEN 1 ELSE 0 END),SUM(CASE WHEN beta IS NOT NULL THEN 1 ELSE 0 END),SUM(CASE WHEN sector IS NOT NULL THEN 1 ELSE 0 END) FROM uae_fundamentals WHERE ticker LIKE '%.SR'").fetchone()
    conn2.close()
    logger.info(f"📊 Saudi DB: {r[0]} rows | PE={r[1]} | Beta={r[2]} | Sector={r[3]}")

if __name__ == "__main__":
    populate()
