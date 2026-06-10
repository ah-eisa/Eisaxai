#!/usr/bin/env python3
"""
populate_sa_fundamentals.py — Populate Saudi Arabia (Tadawul) stock fundamentals
into uae_fundamentals table using yfinance + StockAnalysis.

Usage:
    python3 scripts/populate_sa_fundamentals.py

Sources:
1. yfinance        → PE, fwdPE, Beta, EPS, Dividend, Revenue, NetIncome, MarketCap, Sector, 52W
2. StockAnalysis   → Growth rates, margins (tadawul exchange)
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
logger = logging.getLogger("populate_sa")

DB_PATH = Path("/home/ubuntu/investwise/core/investwise.db")

# Full Tadawul ticker list from StockAnalysis.com (381 tickers)
SA_TICKERS = sorted([
    "1010.SR", "1020.SR", "1030.SR", "1050.SR", "1060.SR", "1080.SR",
    "1111.SR", "1120.SR", "1140.SR", "1150.SR", "1180.SR", "1182.SR",
    "1183.SR", "1201.SR", "1202.SR", "1210.SR", "1211.SR", "1212.SR",
    "1213.SR", "1214.SR", "1301.SR", "1302.SR", "1303.SR", "1304.SR",
    "1320.SR", "1321.SR", "1322.SR", "1323.SR", "1324.SR", "1810.SR",
    "1820.SR", "1830.SR", "1831.SR", "1832.SR", "1833.SR", "1834.SR",
    "1835.SR", "2001.SR", "2010.SR", "2020.SR", "2030.SR", "2040.SR",
    "2050.SR", "2060.SR", "2070.SR", "2080.SR", "2081.SR", "2082.SR",
    "2083.SR", "2084.SR", "2090.SR", "2100.SR", "2110.SR", "2120.SR",
    "2130.SR", "2140.SR", "2150.SR", "2160.SR", "2170.SR", "2180.SR",
    "2190.SR", "2200.SR", "2210.SR", "2220.SR", "2222.SR", "2223.SR",
    "2230.SR", "2240.SR", "2250.SR", "2270.SR", "2280.SR", "2281.SR",
    "2282.SR", "2283.SR", "2284.SR", "2285.SR", "2286.SR", "2287.SR",
    "2288.SR", "2290.SR", "2300.SR", "2310.SR", "2320.SR", "2330.SR",
    "2340.SR", "2350.SR", "2360.SR", "2370.SR", "2380.SR", "2381.SR",
    "2382.SR", "3002.SR", "3003.SR", "3004.SR", "3005.SR", "3007.SR",
    "3008.SR", "3010.SR", "3020.SR", "3030.SR", "3040.SR", "3050.SR",
    "3060.SR", "3080.SR", "3090.SR", "3091.SR", "3092.SR", "4001.SR",
    "4002.SR", "4003.SR", "4004.SR", "4005.SR", "4006.SR", "4007.SR",
    "4008.SR", "4009.SR", "4011.SR", "4012.SR", "4013.SR", "4014.SR",
    "4015.SR", "4016.SR", "4017.SR", "4018.SR", "4019.SR", "4020.SR",
    "4021.SR", "4030.SR", "4031.SR", "4040.SR", "4050.SR", "4051.SR",
    "4061.SR", "4070.SR", "4071.SR", "4072.SR", "4080.SR", "4081.SR",
    "4082.SR", "4083.SR", "4084.SR", "4090.SR", "4100.SR", "4110.SR",
    "4130.SR", "4140.SR", "4141.SR", "4142.SR", "4143.SR", "4144.SR",
    "4145.SR", "4146.SR", "4147.SR", "4148.SR", "4150.SR", "4160.SR",
    "4161.SR", "4162.SR", "4163.SR", "4164.SR", "4165.SR", "4170.SR",
    "4180.SR", "4190.SR", "4191.SR", "4192.SR", "4193.SR", "4194.SR",
    "4200.SR", "4210.SR", "4220.SR", "4230.SR", "4240.SR", "4250.SR",
    "4260.SR", "4261.SR", "4262.SR", "4263.SR", "4264.SR", "4265.SR",
    "4270.SR", "4280.SR", "4290.SR", "4291.SR", "4292.SR", "4300.SR",
    "4310.SR", "4320.SR", "4321.SR", "4322.SR", "4323.SR", "4324.SR",
    "4325.SR", "4326.SR", "4327.SR", "4330.SR", "4331.SR", "4333.SR",
    "4335.SR", "4336.SR", "4340.SR", "4344.SR", "5110.SR", "6001.SR",
    "6002.SR", "6004.SR", "6010.SR", "6012.SR", "6013.SR", "6014.SR",
    "6015.SR", "6016.SR", "6017.SR", "6018.SR", "6019.SR", "6020.SR",
    "6040.SR", "6050.SR", "6060.SR", "6070.SR", "6090.SR", "7010.SR",
    "7020.SR", "7030.SR", "7040.SR", "7200.SR", "7201.SR", "7202.SR",
    "7203.SR", "7204.SR", "7211.SR", "8010.SR", "8012.SR", "8020.SR",
    "8030.SR", "8040.SR", "8050.SR", "8060.SR", "8070.SR", "8100.SR",
    "8120.SR", "8150.SR", "8160.SR", "8170.SR", "8180.SR", "8190.SR",
    "8200.SR", "8210.SR", "8230.SR", "8240.SR", "8250.SR", "8260.SR",
    "8280.SR", "8300.SR", "8310.SR", "8311.SR", "8313.SR", "9510.SR",
    "9513.SR", "9514.SR", "9515.SR", "9516.SR", "9517.SR", "9521.SR",
    "9522.SR", "9523.SR", "9524.SR", "9527.SR", "9530.SR", "9532.SR",
    "9533.SR", "9535.SR", "9536.SR", "9537.SR", "9538.SR", "9539.SR",
    "9540.SR", "9541.SR", "9542.SR", "9543.SR", "9544.SR", "9545.SR",
    "9546.SR", "9547.SR", "9548.SR", "9549.SR", "9550.SR", "9551.SR",
    "9552.SR", "9553.SR", "9555.SR", "9557.SR", "9558.SR", "9559.SR",
    "9560.SR", "9561.SR", "9562.SR", "9563.SR", "9564.SR", "9565.SR",
    "9566.SR", "9567.SR", "9568.SR", "9569.SR", "9570.SR", "9571.SR",
    "9572.SR", "9574.SR", "9575.SR", "9576.SR", "9577.SR", "9578.SR",
    "9579.SR", "9580.SR", "9581.SR", "9583.SR", "9584.SR", "9585.SR",
    "9586.SR", "9587.SR", "9588.SR", "9589.SR", "9590.SR", "9591.SR",
    "9592.SR", "9593.SR", "9594.SR", "9595.SR", "9596.SR", "9597.SR",
    "9598.SR", "9599.SR", "9600.SR", "9601.SR", "9602.SR", "9603.SR",
    "9604.SR", "9605.SR", "9606.SR", "9607.SR", "9608.SR", "9609.SR",
    "9610.SR", "9611.SR", "9612.SR", "9613.SR", "9614.SR", "9615.SR",
    "9616.SR", "9617.SR", "9618.SR", "9619.SR", "9620.SR", "9621.SR",
    "9622.SR", "9623.SR", "9624.SR", "9625.SR", "9626.SR", "9627.SR",
    "9628.SR", "9630.SR", "9631.SR", "9632.SR", "9633.SR", "9634.SR",
    "9635.SR", "9636.SR", "9637.SR", "9639.SR", "9640.SR", "9641.SR",
    "9642.SR", "9644.SR", "9645.SR", "9647.SR", "9648.SR", "9649.SR",
    "9650.SR", "9651.SR", "9653.SR",
])


def upgrade_schema(conn):
    """Ensure uae_fundamentals has all needed columns."""
    new_cols = {
        "forward_pe":      "REAL",
        "net_margin":      "REAL",
        "gross_margin":    "REAL",
        "roe":             "REAL",
        "debt_equity":     "REAL",
        "revenue_growth":  "REAL",
        "earnings_growth": "REAL",
        "sector":          "TEXT",
        "industry":        "TEXT",
        "net_income":      "REAL",
        "shares_out":      "REAL",
        "week_52_high":    "REAL",
        "week_52_low":     "REAL",
        "price":           "REAL",
        "company_name":    "TEXT",
        "source":          "TEXT",
    }
    existing = {row[1] for row in conn.execute("PRAGMA table_info(uae_fundamentals)")}
    for col, dtype in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE uae_fundamentals ADD COLUMN {col} {dtype}")
            logger.info(f"  Added column: {col} ({dtype})")
    conn.commit()


def fetch_yfinance(ticker: str) -> dict:
    """Fetch fundamentals via yfinance — primary source for Saudi."""
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

        result = {}
        if price:           result["price"] = round(float(price), 3)
        if mc:              result["market_cap"] = float(mc)
        if pe:              result["pe_ratio"] = round(float(pe), 2)
        if fpe:             result["forward_pe"] = round(float(fpe), 2)
        if eps:             result["eps"] = round(float(eps), 4)
        if beta:            result["beta"] = round(float(beta), 3)
        if div:
            # yfinance gives dividend yield as decimal (e.g. 0.0502 = 5.02%)
            dv = float(div)
            result["div_yield"] = round(dv if dv < 1 else dv / 100, 4)
        if rev:             result["revenue"] = float(rev)
        if ni:              result["net_income"] = float(ni)
        if sector:          result["sector"] = sector
        if industry:        result["industry"] = industry
        if name:            result["company_name"] = name
        if w52h:            result["week_52_high"] = round(float(w52h), 3)
        if w52l:            result["week_52_low"] = round(float(w52l), 3)
        if roe:             result["roe"] = round(float(roe) * 100, 2)
        if gross_m:         result["gross_margin"] = round(float(gross_m) * 100, 2)
        if net_m:           result["net_margin"] = round(float(net_m) * 100, 2)
        if de:              result["debt_equity"] = round(float(de) / 100, 3)
        if rev_growth:      result["revenue_growth"] = round(float(rev_growth) * 100, 2)
        if earn_growth:     result["earnings_growth"] = round(float(earn_growth) * 100, 2)
        if shares:          result["shares_out"] = float(shares)

        # Compute net margin from revenue + net_income if not provided
        if not result.get("net_margin") and result.get("net_income") and result.get("revenue") and result["revenue"] > 0:
            result["net_margin"] = round((result["net_income"] / result["revenue"]) * 100, 2)

        result["source"] = "yfinance (Tadawul)"
        return result

    except Exception as e:
        logger.warning(f"  [{ticker}] yfinance error: {e}")
        return {}


def fetch_stockanalysis_sa(ticker: str) -> dict:
    """Fetch additional data from StockAnalysis tadawul exchange."""
    try:
        import requests
        slug = ticker.upper().replace(".SR", "").lower()
        url = f"https://stockanalysis.com/quote/tadawul/{slug}/"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return {}
        text = r.text

        def _rx(pattern, grp=1):
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(grp).strip() if m else None

        def _pct_to_float(val):
            if not val: return None
            try:
                return float(str(val).replace('%', '').replace('+', '').strip())
            except:
                return None

        result = {}

        # Revenue growth / earnings growth (often available on SA)
        rg = _rx(r'revenueGrowth[\"\':\s]+([0-9.\-+%]+)')
        if rg: result["revenue_growth"] = _pct_to_float(rg)

        eg = _rx(r'epsGrowth[\"\':\s]+([0-9.\-+%]+)')
        if eg: result["earnings_growth"] = _pct_to_float(eg)

        # Sector / name from page title
        title = _rx(r'<title>([^<]+)</title>')
        if title and not result.get("company_name"):
            # "Saudi Aramco (2222) Stock Price, News & Analysis | StockAnalysis"
            name_m = re.match(r'^([^(]+)', title)
            if name_m:
                result["company_name"] = name_m.group(1).strip()

        return {k: v for k, v in result.items() if v is not None}

    except Exception as e:
        logger.warning(f"  [{ticker}] StockAnalysis SA error: {e}")
        return {}


def populate():
    conn = sqlite3.connect(str(DB_PATH))
    upgrade_schema(conn)

    total = len(SA_TICKERS)
    logger.info(f"🚀 Starting Saudi fundamentals population for {total} tickers")

    success = 0
    failed = 0
    skipped = 0

    for i, ticker in enumerate(SA_TICKERS, 1):
        logger.info(f"[{i}/{total}] {ticker}...")

        # Primary: yfinance (rich for Saudi)
        yf_data = fetch_yfinance(ticker)

        if not yf_data:
            logger.warning(f"  ⚠️ {ticker}: No yfinance data")
            skipped += 1
            continue

        # Secondary: StockAnalysis for supplementary fields
        sa_data = fetch_stockanalysis_sa(ticker)

        # Merge: yfinance wins, SA fills gaps
        merged = {**sa_data, **yf_data}  # yf_data has priority

        company_name = merged.get("company_name") or ticker.replace(".SR", "")

        useful_fields = sum(1 for v in [
            merged.get("market_cap"), merged.get("pe_ratio"), merged.get("beta"),
            merged.get("eps"), merged.get("div_yield"), merged.get("revenue"),
            merged.get("net_margin"), merged.get("revenue_growth")
        ] if v)

        try:
            conn.execute("""
                INSERT INTO uae_fundamentals
                (ticker, name, company_name, market_cap, pe_ratio, beta,
                 eps, div_yield, revenue, forward_pe, net_margin, gross_margin,
                 roe, debt_equity, revenue_growth, earnings_growth,
                 net_income, shares_out, sector, industry,
                 week_52_high, week_52_low, price, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name              = COALESCE(excluded.name, uae_fundamentals.name),
                    company_name      = COALESCE(excluded.company_name, uae_fundamentals.company_name),
                    market_cap        = COALESCE(excluded.market_cap, uae_fundamentals.market_cap),
                    pe_ratio          = COALESCE(excluded.pe_ratio, uae_fundamentals.pe_ratio),
                    beta              = COALESCE(excluded.beta, uae_fundamentals.beta),
                    eps               = COALESCE(excluded.eps, uae_fundamentals.eps),
                    div_yield         = COALESCE(excluded.div_yield, uae_fundamentals.div_yield),
                    revenue           = COALESCE(excluded.revenue, uae_fundamentals.revenue),
                    forward_pe        = COALESCE(excluded.forward_pe, uae_fundamentals.forward_pe),
                    net_margin        = COALESCE(excluded.net_margin, uae_fundamentals.net_margin),
                    gross_margin      = COALESCE(excluded.gross_margin, uae_fundamentals.gross_margin),
                    roe               = COALESCE(excluded.roe, uae_fundamentals.roe),
                    debt_equity       = COALESCE(excluded.debt_equity, uae_fundamentals.debt_equity),
                    revenue_growth    = COALESCE(excluded.revenue_growth, uae_fundamentals.revenue_growth),
                    earnings_growth   = COALESCE(excluded.earnings_growth, uae_fundamentals.earnings_growth),
                    net_income        = COALESCE(excluded.net_income, uae_fundamentals.net_income),
                    shares_out        = COALESCE(excluded.shares_out, uae_fundamentals.shares_out),
                    sector            = COALESCE(excluded.sector, uae_fundamentals.sector),
                    industry          = COALESCE(excluded.industry, uae_fundamentals.industry),
                    week_52_high      = COALESCE(excluded.week_52_high, uae_fundamentals.week_52_high),
                    week_52_low       = COALESCE(excluded.week_52_low, uae_fundamentals.week_52_low),
                    price             = COALESCE(excluded.price, uae_fundamentals.price),
                    source            = COALESCE(excluded.source, uae_fundamentals.source),
                    updated_at        = excluded.updated_at
            """, (
                ticker, company_name, company_name,
                merged.get("market_cap"), merged.get("pe_ratio"), merged.get("beta"),
                merged.get("eps"), merged.get("div_yield"), merged.get("revenue"),
                merged.get("forward_pe"), merged.get("net_margin"), merged.get("gross_margin"),
                merged.get("roe"), merged.get("debt_equity"),
                merged.get("revenue_growth"), merged.get("earnings_growth"),
                merged.get("net_income"), merged.get("shares_out"),
                merged.get("sector"), merged.get("industry"),
                merged.get("week_52_high"), merged.get("week_52_low"),
                merged.get("price"), merged.get("source"),
                datetime.now().isoformat()
            ))
            conn.commit()
            success += 1
            logger.info(f"  ✅ {ticker} [{company_name}]: {useful_fields} fields | "
                        f"PE={merged.get('pe_ratio')} | Beta={merged.get('beta')} | "
                        f"Rev={merged.get('revenue')} | Sector={merged.get('sector')}")
        except Exception as e:
            logger.error(f"  ❌ {ticker}: DB insert failed: {e}")
            failed += 1

        # Rate limit
        if i < total:
            time.sleep(1.0)

    conn.close()

    logger.info(f"""
{'='*60}
✅ Saudi Fundamentals Population Complete
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
        FROM uae_fundamentals
        WHERE ticker LIKE '%.SR'
    """).fetchone()
    conn2.close()

    logger.info(f"""
📊 Saudi Coverage Summary:
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
