#!/usr/bin/env python3
"""
enrich_egx_sectors.py — Enrich EGX stock data with sector, company name, PE ratio, revenue
from StockAnalysis.com EGX pages.

Reads egx_fundamentals rows where sector IS NULL or company_name contains a comma,
fetches the StockAnalysis EGX page for each, extracts:
  - Company name (nameFull from SvelteKit data)
  - Industry (from infoTable)
  - Sector (derived from industry via mapping)
  - PE ratio (peRatio field)
  - Revenue (revenue field)
  - Additional financials: marketCap, netIncome, forwardPE, dividendYield, beta, eps

Updates egx_fundamentals with extracted data.
Sleep: 1.5s between requests.

Usage:
    python3 scripts/enrich_egx_sectors.py
"""

import sys, os, re, json, sqlite3, time, logging
sys.path.insert(0, "/home/ubuntu/investwise")
os.chdir("/home/ubuntu/investwise")

from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("enrich_egx")

DB_PATH = Path("/home/ubuntu/investwise/core/investwise.db")

# ── Industry → Sector mapping ─────────────────────────────────────────────────
# Based on standard GICS/EGX sector classifications
INDUSTRY_TO_SECTOR = {
    # Financials
    "Commercial Banks":                             "Financials",
    "State Commercial Banks-NEC":                   "Financials",
    "Savings Institution, Federally Chartered":     "Financials",
    "Savings Institutions":                         "Financials",
    "National Commercial Banks":                    "Financials",
    "Foreign Banks":                                "Financials",
    "State Commercial Banks":                       "Financials",
    "Investment Offices":                           "Financials",
    "Finance Services":                             "Financials",
    "Insurance Agents, Brokers and Service":        "Financials",
    "Life Insurance":                               "Financials",
    "Accident and Health Insurance":                "Financials",
    "Fire, Marine & Casualty Insurance":            "Financials",
    "Title Insurance":                              "Financials",
    "Security & Commodity Brokers, Dealers, Exchanges & Services": "Financials",
    "Security Brokers, Dealers, and Flotation Companies": "Financials",
    "Investment Advice":                            "Financials",
    "Investors, not elsewhere classified":          "Financials",
    "Finance & Insurance":                          "Financials",
    "Asset Management":                             "Financials",
    "Financial Services":                           "Financials",
    "Leasing":                                      "Financials",
    "Leasing & Rental":                             "Financials",
    "Al Tawfeek Leasing":                           "Financials",
    "Mortgage Bankers, Loan Correspondents":        "Financials",
    "State Chartered Banks, Federal Reserve Members": "Financials",
    "Federal-Sponsored Credit Agencies":            "Financials",
    "Short-Term Business Credit Institutions":      "Financials",
    "Holding & Other Investment Offices":           "Financials",
    "Real Estate Investment Trusts":                "Real Estate",

    # Real Estate
    "Real Estate Dealers (for their own account)":  "Real Estate",
    "Land Subdividers & Developers (No Cemeteries)": "Real Estate",
    "Operative Builders":                           "Real Estate",
    "Real Estate":                                  "Real Estate",
    "Real Estate Agents and Managers":              "Real Estate",
    "Subdividers and Developers":                   "Real Estate",
    "Real Estate Development":                      "Real Estate",
    "Hotels and Motels":                            "Real Estate",
    "Mfg-Misc. Plastics Products":                  "Industrials",

    # Energy
    "Petroleum Refining":                           "Energy",
    "Oil and Gas Field Services, NEC":              "Energy",
    "Crude Petroleum and Natural Gas":              "Energy",
    "Natural Gas Liquids":                          "Energy",
    "Coal Mining":                                  "Energy",
    "Industrial and Commercial Machinery and Computer Equipment": "Industrials",

    # Materials
    "Primary Metal Industries":                     "Materials",
    "Steel Works, Blast Furnaces":                  "Materials",
    "Iron and Steel Foundries":                     "Materials",
    "Rolling Drawing & Extruding of Nonferrous Metals": "Materials",
    "Aluminum Die-Castings":                        "Materials",
    "Nonferrous Foundries":                         "Materials",
    "Metal Mining":                                 "Materials",
    "Gold and Silver Ores":                         "Materials",
    "Phosphate Rock":                               "Materials",
    "Mining & Quarrying of Nonmetallic Minerals (No Fuels)": "Materials",
    "Cement, Hydraulic":                            "Materials",
    "Pottery and Related Products":                 "Materials",
    "Glass and Glassware, Pressed or Blown":        "Materials",
    "Agricultural Chemicals":                       "Materials",
    "Plastics Materials, Synthetic Resins & Nonvulcan. Elastomers": "Materials",
    "Industrial Chemicals and Synthetics":          "Materials",
    "Paints, Varnishes, Lacquers, Enamels, and Allied Products": "Materials",
    "Industrial Gases":                             "Materials",
    "Pharmaceutical Preparations":                  "Health Care",
    "Chemicals & Allied Products":                  "Materials",
    "Chemical & Allied Products":                   "Materials",
    "Fertilizers":                                  "Materials",
    "Nitrogenous Fertilizers":                      "Materials",
    "Paper and Allied Products":                    "Materials",
    "Paperboard Mills":                             "Materials",
    "Lumber & Wood Products (No Furniture)":        "Materials",

    # Industrials
    "Electric Services":                            "Utilities",
    "Water Supply":                                 "Utilities",
    "Natural Gas Distribution":                     "Utilities",
    "Electric, Gas & Sanitary Services":            "Utilities",
    "Electrical Industrial Apparatus":              "Industrials",
    "Electronic & Other Electrical Equipment":      "Industrials",
    "Electrical Work":                              "Industrials",
    "Wiring Devices":                               "Industrials",
    "Construction":                                 "Industrials",
    "General Building Contractors-Residential Buildings": "Industrials",
    "General Building Contractors-Industrial Buildings": "Industrials",
    "Heavy Construction, Except Building Construction": "Industrials",
    "Plumbing, Heating, Air-Conditioning":          "Industrials",
    "Engineering Services":                         "Industrials",
    "Services-Engineering, Accounting, Research": "Industrials",
    "Transportation":                               "Industrials",
    "Air Transportation":                           "Industrials",
    "Marine Transportation":                        "Industrials",
    "Trucking & Warehousing":                       "Industrials",
    "Railroads":                                    "Industrials",
    "Miscellaneous Fabricated Metal Products":      "Industrials",
    "Construction, Mining & Materials Handling Machinery & Equipment": "Industrials",
    "Special Industry Machinery, NEC":              "Industrials",
    "Ordnance & Accessories":                       "Industrials",
    "Valves and Pipe Fittings":                     "Industrials",
    "Pumps & Pumping Equipment":                    "Industrials",
    "Turbines & Turbine Generator Sets":            "Industrials",
    "Hardware":                                     "Industrials",
    "Scrap and Waste Materials":                    "Materials",

    # Consumer Discretionary
    "Retail Stores-Food Stores":                    "Consumer Staples",
    "Retail Stores":                                "Consumer Discretionary",
    "Retail-Eating & Drinking Places":              "Consumer Discretionary",
    "Hotel & Gaming":                               "Consumer Discretionary",
    "Amusement & Recreation Services":              "Consumer Discretionary",
    "Motion Picture Production and Distribution":   "Consumer Discretionary",
    "Real Estate Dealers":                          "Real Estate",
    "Auto Dealers & Service Stations":              "Consumer Discretionary",
    "Motor Vehicles & Passenger Car Bodies":        "Consumer Discretionary",
    "Auto Parts & Equipment":                       "Consumer Discretionary",
    "Retail-Auto Dealers & Service Stations":       "Consumer Discretionary",
    "Apparel & Other Finished Products":            "Consumer Discretionary",
    "Textile Mill Products":                        "Consumer Discretionary",
    "Weaving Mills, Cotton":                        "Consumer Discretionary",
    "Yarn Throwing and Winding Mills":              "Consumer Discretionary",
    "Household Furniture":                          "Consumer Discretionary",
    "Misc. Manufacturing Industries":               "Consumer Discretionary",

    # Consumer Staples
    "Food and Kindred Products":                    "Consumer Staples",
    "Bakery Products":                              "Consumer Staples",
    "Dairy Products":                               "Consumer Staples",
    "Poultry Slaughtering and Processing":          "Consumer Staples",
    "Canned, Frozen & Preserved Fruit, Veg & Food Specialties": "Consumer Staples",
    "Grain Mill Products":                          "Consumer Staples",
    "Sugar and Confectionery Products":             "Consumer Staples",
    "Beverages":                                    "Consumer Staples",
    "Cigarettes":                                   "Consumer Staples",
    "Tobacco Products":                             "Consumer Staples",
    "Perfumes, Cosmetics, and Other Toilet Preparations": "Consumer Staples",
    "Food Preparations, NEC":                       "Consumer Staples",
    "Wholesale-Groceries & Related Products":       "Consumer Staples",
    "Vegetable Oil Mills":                          "Consumer Staples",
    "Animal Feeds":                                 "Consumer Staples",

    # Health Care
    "Health Services":                              "Health Care",
    "Hospitals":                                    "Health Care",
    "Medical Laboratories":                         "Health Care",
    "Home Health Care Services":                    "Health Care",
    "Pharmaceutical Preparations":                  "Health Care",
    "Pharmaceutical":                               "Health Care",
    "Biological Products":                          "Health Care",
    "Medical Instruments & Supplies":               "Health Care",
    "Drugs":                                        "Health Care",
    "Drug Stores and Proprietary Stores":           "Health Care",
    "Services-Health Services":                     "Health Care",
    "Medical Devices":                              "Health Care",
    "Biotechnology":                                "Health Care",

    # Communication Services / Technology
    "Telephone Communications":                     "Communication Services",
    "Telephone & Telegraph Apparatus":              "Communication Services",
    "Radio & TV Broadcasting & Communications Equipment": "Communication Services",
    "Cable & Other Pay Television Services":        "Communication Services",
    "Communications":                               "Communication Services",
    "Computer Processing and Data Preparation":     "Technology",
    "Computer Integrated Systems Design":           "Technology",
    "Services-Computer Programming, Data Processing": "Technology",
    "Electronic Computers":                         "Technology",
    "Computer Software":                            "Technology",
    "Services-Prepackaged Software":                "Technology",
    "Semiconductors":                               "Technology",
    "Information Technology":                       "Technology",
    "Technology":                                   "Technology",

    # Education
    "Educational Services":                         "Consumer Discretionary",
    "Services-Educational Services":               "Consumer Discretionary",
    "Services-Schools":                             "Consumer Discretionary",

    # Default
    "Miscellaneous Business Services":              "Industrials",
    "Services-Misc. Business Services NEC":         "Industrials",
    "Printing, Publishing, and Allied Industries":  "Communication Services",
}


def get_sector_from_industry(industry: str) -> str:
    """Map industry string to sector."""
    if not industry:
        return None
    # Try exact match first
    if industry in INDUSTRY_TO_SECTOR:
        return INDUSTRY_TO_SECTOR[industry]
    # Try partial match
    industry_lower = industry.lower()
    for key, sector in INDUSTRY_TO_SECTOR.items():
        if key.lower() in industry_lower or industry_lower in key.lower():
            return sector
    # Keyword-based fallback
    kw_map = {
        "bank": "Financials",
        "financ": "Financials",
        "invest": "Financials",
        "insur": "Financials",
        "leasing": "Financials",
        "real estate": "Real Estate",
        "property": "Real Estate",
        "construction": "Industrials",
        "petroleum": "Energy",
        "oil": "Energy",
        "gas": "Energy",
        "pharma": "Health Care",
        "health": "Health Care",
        "hospital": "Health Care",
        "medical": "Health Care",
        "food": "Consumer Staples",
        "tobacco": "Consumer Staples",
        "beverage": "Consumer Staples",
        "telecom": "Communication Services",
        "communic": "Communication Services",
        "technolog": "Technology",
        "software": "Technology",
        "computer": "Technology",
        "steel": "Materials",
        "metal": "Materials",
        "cement": "Materials",
        "chemical": "Materials",
        "fertilizer": "Materials",
        "aluminum": "Materials",
        "mining": "Materials",
        "textile": "Consumer Discretionary",
        "weaving": "Consumer Discretionary",
        "cotton": "Consumer Discretionary",
        "electric": "Utilities",
        "utility": "Utilities",
        "water": "Utilities",
        "transport": "Industrials",
        "aviation": "Industrials",
        "shipping": "Industrials",
        "education": "Consumer Discretionary",
        "tourism": "Consumer Discretionary",
        "hotel": "Consumer Discretionary",
    }
    for kw, sector in kw_map.items():
        if kw in industry_lower:
            return sector
    return None


def parse_size_to_float(s: str) -> float:
    """Convert '128.54B' or '438.12B' to float."""
    if not s or s in ("N/A", "—", ""):
        return None
    s = str(s).strip()
    try:
        if s.endswith("T"):
            return float(s[:-1]) * 1e12
        elif s.endswith("B"):
            return float(s[:-1]) * 1e9
        elif s.endswith("M"):
            return float(s[:-1]) * 1e6
        elif s.endswith("K"):
            return float(s[:-1]) * 1e3
        return float(s.replace(",", ""))
    except Exception:
        return None


def fetch_stockanalysis_egx(ticker: str) -> dict:
    """
    Fetch StockAnalysis EGX page for a ticker and extract financial data.

    Extracts from the embedded SvelteKit JSON data block:
      - nameFull: company name
      - peRatio: trailing PE
      - forwardPE: forward PE
      - revenue: TTM revenue (string like '128.54B')
      - marketCap: market cap (string like '438.12B')
      - netIncome: net income (string)
      - beta: beta
      - dividendYield: dividend yield
      - eps: EPS
      - infoTable: contains Industry field
    """
    try:
        import requests
        slug = ticker.upper().replace(".CA", "").lower()
        url = f"https://stockanalysis.com/quote/egx/{slug}/"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            logger.debug(f"  [{ticker}] HTTP {r.status_code}")
            return {}

        text = r.text

        result = {}

        # ── Extract from SvelteKit JS data block ─────────────────────────────
        # StockAnalysis uses SvelteKit SSR with unquoted JS object keys:
        # e.g.  nameFull:"Commercial International Bank...",peRatio:"7.19",...

        # nameFull (company name)
        name_m = re.search(r'nameFull\s*:\s*"([^"]+)"', text)
        if name_m:
            nm = name_m.group(1).strip()
            # Filter out garbage names like "TICKER,0P0000XXXX,123456"
            if len(nm) > 3 and ',' not in nm and not re.match(r'^[A-Z]+\.CA,', nm):
                result["company_name"] = nm

        # peRatio (trailing PE)
        pe_m = re.search(r'peRatio\s*:\s*"([^"]+)"', text)
        if pe_m:
            try:
                result["pe_ratio"] = round(float(pe_m.group(1).replace(",", "")), 2)
            except Exception:
                pass

        # forwardPE
        fpe_m = re.search(r'forwardPE\s*:\s*"([^"]+)"', text)
        if fpe_m:
            try:
                result["forward_pe"] = round(float(fpe_m.group(1).replace(",", "")), 2)
            except Exception:
                pass

        # revenue (e.g. "128.54B")
        rev_m = re.search(r'(?<![a-z])revenue\s*:\s*"([^"]+)"', text)
        if rev_m:
            rv = parse_size_to_float(rev_m.group(1))
            if rv:
                result["revenue_raw"] = rv  # store raw float for later formatting

        # marketCap (e.g. "438.12B")
        mc_m = re.search(r'marketCap\s*:\s*"([^"]+)"', text)
        if mc_m:
            mc = parse_size_to_float(mc_m.group(1))
            if mc:
                result["market_cap_raw"] = mc

        # netIncome
        ni_m = re.search(r'netIncome\s*:\s*"([^"]+)"', text)
        if ni_m:
            ni = parse_size_to_float(ni_m.group(1))
            if ni:
                result["net_income"] = ni

        # beta
        beta_m = re.search(r'(?<![a-z])beta\s*:\s*"([^"]+)"', text)
        if beta_m:
            try:
                result["beta"] = round(float(beta_m.group(1).replace(",", "")), 3)
            except Exception:
                pass

        # eps
        eps_m = re.search(r'(?<![a-z])eps\s*:\s*"([^"]+)"', text)
        if eps_m:
            try:
                result["eps"] = round(float(eps_m.group(1).replace(",", "")), 4)
            except Exception:
                pass

        # dividendYield (e.g. "5.02%")
        dy_m = re.search(r'dividendYield\s*:\s*"([^"%]+)%"', text)
        if dy_m:
            try:
                result["div_yield"] = round(float(dy_m.group(1)) / 100, 4)
            except Exception:
                pass

        # ── Extract Industry from infoTable ──────────────────────────────────
        # SvelteKit JS pattern: {t:"Industry",v:"Commercial Banks",u:null}
        industry_m = re.search(r't\s*:\s*"Industry"\s*,\s*v\s*:\s*"([^"]+)"', text)
        if not industry_m:
            # Try quoted JSON pattern
            industry_m = re.search(r'"t"\s*:\s*"Industry"\s*,\s*"v"\s*:\s*"([^"]+)"', text)

        if industry_m:
            industry = industry_m.group(1).strip()
            result["industry"] = industry
            sector = get_sector_from_industry(industry)
            if sector:
                result["sector"] = sector

        return {k: v for k, v in result.items() if v is not None}

    except Exception as e:
        logger.warning(f"  [{ticker}] StockAnalysis EGX error: {e}")
        return {}


def format_display(val: float, suffix: str = " EGP") -> str:
    """Format large number as human-readable string."""
    if not val:
        return None
    if val >= 1e12:
        return f"{val/1e12:.2f}T{suffix}"
    elif val >= 1e9:
        return f"{val/1e9:.1f}B{suffix}"
    elif val >= 1e6:
        return f"{val/1e6:.0f}M{suffix}"
    return str(val)


def enrich():
    conn = sqlite3.connect(str(DB_PATH))

    # Get tickers needing enrichment: sector IS NULL OR company_name contains comma
    rows = conn.execute("""
        SELECT ticker, company_name, sector, pe_ratio, revenue
        FROM egx_fundamentals
        WHERE sector IS NULL
           OR company_name LIKE '%,%'
           OR company_name IS NULL
        ORDER BY ticker
    """).fetchall()

    total = len(rows)
    logger.info(f"Found {total} EGX tickers needing sector/name enrichment")

    enriched = 0
    sector_filled = 0
    name_fixed = 0
    pe_filled = 0
    rev_filled = 0
    failed = 0

    for i, (ticker, old_name, old_sector, old_pe, old_rev) in enumerate(rows, 1):
        logger.info(f"[{i}/{total}] {ticker} | name={old_name!r:.40} | sector={old_sector}")

        data = fetch_stockanalysis_egx(ticker)

        if not data:
            logger.warning(f"  [{ticker}] No data returned")
            failed += 1
            if i < total:
                time.sleep(1.5)
            continue

        # Build update values
        new_name    = data.get("company_name")
        new_sector  = data.get("sector")
        new_industry= data.get("industry")
        new_pe      = data.get("pe_ratio")
        new_fpe     = data.get("forward_pe")
        new_beta    = data.get("beta")
        new_eps     = data.get("eps")
        new_div     = data.get("div_yield")
        new_ni      = data.get("net_income")

        # Format revenue and market_cap as TEXT strings
        new_rev_display = format_display(data.get("revenue_raw"))
        new_mc_display  = format_display(data.get("market_cap_raw"))

        # Track what's being filled
        if new_sector and not old_sector:
            sector_filled += 1
        if new_name and (not old_name or ',' in (old_name or '')):
            name_fixed += 1
        if new_pe and not old_pe:
            pe_filled += 1
        if new_rev_display and not old_rev:
            rev_filled += 1

        try:
            conn.execute("""
                UPDATE egx_fundamentals SET
                    company_name  = COALESCE(
                        CASE WHEN ? IS NOT NULL AND (company_name IS NULL OR company_name LIKE '%,%') THEN ? ELSE company_name END,
                        company_name
                    ),
                    name          = COALESCE(
                        CASE WHEN ? IS NOT NULL AND (name IS NULL OR name LIKE '%,%') THEN ? ELSE name END,
                        name
                    ),
                    sector        = COALESCE(sector, ?),
                    industry      = COALESCE(industry, ?),
                    pe_ratio      = COALESCE(pe_ratio, ?),
                    forward_pe    = COALESCE(forward_pe, ?),
                    beta          = COALESCE(beta, ?),
                    eps           = COALESCE(eps, ?),
                    div_yield     = COALESCE(div_yield, ?),
                    net_income    = COALESCE(net_income, ?),
                    revenue       = COALESCE(revenue, ?),
                    market_cap    = COALESCE(market_cap, ?),
                    source        = COALESCE(source, 'StockAnalysis EGX'),
                    updated_at    = ?
                WHERE ticker = ?
            """, (
                new_name, new_name,
                new_name, new_name,
                new_sector,
                new_industry,
                new_pe,
                new_fpe,
                new_beta,
                new_eps,
                new_div,
                new_ni,
                new_rev_display,
                new_mc_display,
                datetime.now().isoformat(),
                ticker,
            ))
            conn.commit()
            enriched += 1

            logger.info(
                f"  OK [{new_name or old_name or ticker}] "
                f"sector={new_sector} | industry={new_industry} | "
                f"PE={new_pe} | rev={new_rev_display} | mc={new_mc_display}"
            )
        except Exception as e:
            logger.error(f"  [{ticker}] DB update failed: {e}")
            failed += 1

        if i < total:
            time.sleep(1.5)

    conn.close()

    logger.info(f"""
{'='*60}
EGX Sector Enrichment Complete
{'='*60}
Total processed:  {total}
Enriched:         {enriched}
  Sectors filled: {sector_filled}
  Names fixed:    {name_fixed}
  PEs filled:     {pe_filled}
  Revenue filled: {rev_filled}
Failed:           {failed}
{'='*60}
""")

    # Final coverage stats
    conn2 = sqlite3.connect(str(DB_PATH))
    stats = conn2.execute("""
        SELECT
            COUNT(*)                                                          AS total,
            SUM(CASE WHEN sector       IS NOT NULL THEN 1 ELSE 0 END)        AS has_sector,
            SUM(CASE WHEN sector       IS NULL     THEN 1 ELSE 0 END)        AS no_sector,
            SUM(CASE WHEN industry     IS NOT NULL THEN 1 ELSE 0 END)        AS has_industry,
            SUM(CASE WHEN pe_ratio     IS NOT NULL THEN 1 ELSE 0 END)        AS has_pe,
            SUM(CASE WHEN revenue      IS NOT NULL THEN 1 ELSE 0 END)        AS has_rev,
            SUM(CASE WHEN company_name IS NOT NULL AND company_name NOT LIKE '%,%' THEN 1 ELSE 0 END) AS good_name,
            SUM(CASE WHEN company_name LIKE '%,%' THEN 1 ELSE 0 END)         AS bad_name
        FROM egx_fundamentals
    """).fetchone()
    conn2.close()

    print(f"""
Coverage after enrichment:
  Total rows:      {stats[0]}
  Has sector:      {stats[1]}  (missing: {stats[2]})
  Has industry:    {stats[3]}
  Has PE:          {stats[4]}
  Has revenue:     {stats[5]}
  Good name:       {stats[6]}  (bad/comma: {stats[7]})
""")

    return sector_filled


if __name__ == "__main__":
    enrich()
