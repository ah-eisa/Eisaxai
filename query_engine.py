"""
EisaX Query Engine  v1.0
=========================
Reads from the institutional Data Layer instantly.
Never scrapes on-demand — snapshot-first always.

Usage:
    from query_engine import QueryEngine
    from pipeline import cache, fetcher

    qe = QueryEngine(cache, fetcher)

    stock  = qe.get_stock("EMAAR", "uae")
    movers = qe.top_movers("ksa")
    peers  = qe.peer_comparison("2222.SR", "ksa")
    gulf   = qe.cross_market("Energy")
    port   = qe.portfolio_analyze({"EMAAR": 500, "2222.SR": 200})
"""

import logging
import re
from datetime import datetime

import pandas as pd

from core.data_layer import market_cache_adapter as _mca

log = logging.getLogger("eisax.query")

# Gulf markets used for regional peer comparisons
GULF_MARKETS = ["uae", "ksa", "qatar", "kuwait", "bahrain"]

# ── Sector aliases ─────────────────────────────────────────────────────────────
# TradingView doesn't have a "Real Estate" sector — it classifies property
# companies under "Finance". Map yfinance/report sector names → TV sector +
# optional name-based filter keywords to separate RE from banks.
_SECTOR_ALIASES: dict[str, dict] = {
    "real estate": {
        "tv_sector": "Finance",
        "name_keywords": [
            "emaar", "aldar", "damac", "nakheel", "deyaar", "properties",
            "property", "real estate", "realty", "dar", "development",
            "dev", "mall", "residential", "estithmaar", "mazaya",
            "jabal omar", "dar al arkan", "knowledge", "city",
        ],
    },
}


class QueryEngine:

    def __init__(self, cache=None, fetcher=None):
        """
        cache   : retained for signature back-compat; reads now route through
                  core.data_layer.market_cache_adapter regardless of value.
        fetcher : MarketFetcher instance (optional — used to refresh writers
                  when callers detect a stale snapshot externally).
        """
        self.cache   = cache
        self.fetcher = fetcher

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load(self, market: str) -> pd.DataFrame | None:
        """Latest snapshot for a market via the institutional Data Layer."""
        df = _mca.get_latest_snapshot(market)
        if df is None:
            log.warning(f"⚠️  No snapshot available for [{market}]")
            return None
        ts = _mca.snapshot_timestamp(market)
        log.debug(f"📂 Loaded [{market}]: {len(df)} rows, snapshot={ts}")
        return df

    def _find_ticker(self, df: pd.DataFrame, ticker: str) -> pd.Series | None:
        """
        Find a stock row by ticker or name.
        Handles formats: "2222.SR", "EMAAR", "DFM:EMAAR", "TADAWUL:2222"
        Returns the first matching row as a Series, or None.
        """
        ticker_upper = ticker.upper()

        # strip exchange suffixes (.SR .AE .DU .CA .KW .QA) to get bare code
        bare = ticker_upper
        for sfx in (".SR", ".AE", ".DU", ".CA", ".KW", ".QA", ".BH"):
            if bare.endswith(sfx):
                bare = bare[: -len(sfx)]
                break

        if "ticker" in df.columns:
            tv = df["ticker"].str.upper()

            # 1. exact full match
            if (tv == ticker_upper).any():
                return df[tv == ticker_upper].iloc[0]

            # 2. exact bare match  (e.g. "2222" == "2222")
            if bare != ticker_upper and (tv == bare).any():
                return df[tv == bare].iloc[0]

            # 3. partial — bare code inside TradingView ticker  (e.g. "TADAWUL:2222")
            mask = tv.str.contains(re.escape(bare), na=False)
            if mask.any():
                return df[mask].iloc[0]

            # 4. partial — full ticker inside TradingView ticker  (e.g. "DFM:EMAAR")
            mask = tv.str.contains(re.escape(ticker_upper), na=False)
            if mask.any():
                return df[mask].iloc[0]

        # 5. name match (last resort)
        if "name" in df.columns:
            mask = df["name"].str.upper().str.contains(re.escape(bare), na=False)
            if mask.any():
                return df[mask].iloc[0]

        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_stock(self, ticker: str, market: str) -> dict | None:
        """
        Single stock lookup.
        Returns a plain dict with all available fields, or None.
        """
        df = self._load(market)
        if df is None:
            return None

        row = self._find_ticker(df, ticker)
        if row is None:
            log.warning(f"🔍 '{ticker}' not found in [{market}]")
            return None

        result = row.dropna().to_dict()
        result["_market"] = market
        result["_queried_at"] = datetime.now().isoformat()
        return result

    def screen(
        self,
        market: str,
        filters: dict | None = None,
        sort_by: str = "market_cap_basic",
        ascending: bool = False,
        limit: int = 50,
    ) -> pd.DataFrame | None:
        """
        Filter and sort a market DataFrame.

        filters examples:
            {"RSI": ("<", 30)}          ← oversold
            {"sector": ("==", "Energy")}
            {"change": (">", 2.0)}      ← up >2% today
            {"price_earnings_ttm": ("<", 15)}  ← cheap P/E

        Multiple filters are ANDed together.
        """
        df = self._load(market)
        if df is None:
            return None

        if filters:
            for col, (op, val) in filters.items():
                if col not in df.columns:
                    continue
                if op == "<":
                    df = df[df[col] < val]
                elif op == ">":
                    df = df[df[col] > val]
                elif op == "<=":
                    df = df[df[col] <= val]
                elif op == ">=":
                    df = df[df[col] >= val]
                elif op == "==":
                    df = df[df[col] == val]
                elif op == "!=":
                    df = df[df[col] != val]
                elif op == "contains":
                    df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        return df.head(limit).reset_index(drop=True)

    def top_movers(self, market: str, n: int = 10) -> dict:
        """
        Top gainers and losers for a market.
        Returns {"gainers": DataFrame, "losers": DataFrame}.
        """
        df = self._load(market)
        if df is None or "change" not in df.columns:
            return {"gainers": pd.DataFrame(), "losers": pd.DataFrame()}

        df = df.dropna(subset=["change"])
        cols = [c for c in ["ticker", "name", "close", "change", "volume", "RSI", "sector"]
                if c in df.columns]

        gainers = df.nlargest(n, "change")[cols].reset_index(drop=True)
        losers  = df.nsmallest(n, "change")[cols].reset_index(drop=True)

        return {"gainers": gainers, "losers": losers}

    def sector_summary(self, market: str) -> pd.DataFrame | None:
        """
        Aggregate statistics by sector for a market.
        Returns DataFrame with one row per sector.
        """
        df = self._load(market)
        if df is None or "sector" not in df.columns:
            return None

        df = df.dropna(subset=["sector"])

        agg = {
            "name": "count",
            "change": "mean",
            "RSI": "mean",
        }
        if "market_cap_basic" in df.columns:
            agg["market_cap_basic"] = "sum"
        if "price_earnings_ttm" in df.columns:
            agg["price_earnings_ttm"] = "median"
        if "dividend_yield_recent" in df.columns:
            agg["dividend_yield_recent"] = "mean"

        summary = df.groupby("sector").agg(agg).round(2)
        summary.rename(columns={"name": "stock_count"}, inplace=True)
        summary.sort_values("market_cap_basic" if "market_cap_basic" in summary.columns
                            else "stock_count", ascending=False, inplace=True)
        return summary.reset_index()

    def peer_comparison(self, ticker: str, market: str, max_peers: int = 10) -> pd.DataFrame | None:
        """
        Compare a stock vs its sector peers in the same market.
        Returns a DataFrame with the stock highlighted at top, followed by peers.
        """
        df = self._load(market)
        if df is None:
            return None

        row = self._find_ticker(df, ticker)
        if row is None:
            log.warning(f"🔍 '{ticker}' not found in [{market}] for peer comparison")
            return None

        sector = row.get("sector")
        if not sector or pd.isna(sector):
            log.warning(f"⚠️  No sector data for {ticker}")
            return None

        # all stocks in same sector
        peers = df[df["sector"] == sector].copy()

        cols = [c for c in [
            "ticker", "name", "close", "change",
            "price_earnings_ttm", "dividend_yield_recent",
            "RSI", "SMA50", "SMA200", "market_cap_basic", "sector"
        ] if c in peers.columns]

        peers = peers[cols].dropna(subset=["close"])

        # rank by market cap
        if "market_cap_basic" in peers.columns:
            peers = peers.sort_values("market_cap_basic", ascending=False)

        peers = peers.head(max_peers).reset_index(drop=True)

        # add rank column
        peers.insert(0, "rank", range(1, len(peers) + 1))

        # mark the queried stock
        if "ticker" in peers.columns:
            ticker_upper = ticker.upper()
            peers["is_target"] = peers["ticker"].str.upper().str.contains(ticker_upper, na=False)
        else:
            peers["is_target"] = False

        return peers

    def cross_market(self, sector: str, markets: list | None = None) -> pd.DataFrame | None:
        """
        Same sector across multiple markets (default: all Gulf markets).
        Returns a combined DataFrame with a '_query_market' column.
        Handles TradingView sector misclassifications via _SECTOR_ALIASES.
        """
        if markets is None:
            markets = GULF_MARKETS

        # Check for sector alias (e.g. "Real Estate" → Finance + name filter)
        _alias = _SECTOR_ALIASES.get(sector.lower())
        _tv_sector  = _alias["tv_sector"]    if _alias else sector
        _name_keys  = _alias["name_keywords"] if _alias else None

        frames = []
        for market in markets:
            df = self._load(market)
            if df is None or "sector" not in df.columns:
                continue

            mask = df["sector"].str.contains(_tv_sector, case=False, na=False)
            subset = df[mask].copy()

            # If alias has name keywords, filter to matching company names only
            if _name_keys and "name" in subset.columns and len(subset) > 0:
                name_mask = subset["name"].str.lower().str.contains(
                    "|".join(_name_keys), na=False
                )
                # Also check ticker for common RE patterns
                if "ticker" in subset.columns:
                    ticker_mask = subset["ticker"].str.lower().str.contains(
                        "|".join(_name_keys), na=False
                    )
                    name_mask = name_mask | ticker_mask
                subset = subset[name_mask]

            if len(subset) == 0:
                continue

            subset["_query_market"] = market
            frames.append(subset)

        if not frames:
            log.warning(f"No data found for sector '{sector}' across {markets}")
            return None

        combined = pd.concat(frames, ignore_index=True)

        cols = [c for c in [
            "_query_market", "ticker", "name", "close", "change",
            "price_earnings_ttm", "dividend_yield_recent",
            "RSI", "market_cap_basic", "sector"
        ] if c in combined.columns]

        return combined[cols].sort_values("market_cap_basic",
                                          ascending=False).reset_index(drop=True)

    def portfolio_analyze(self, holdings: dict) -> dict:
        """
        Analyze a portfolio of holdings.

        holdings: {ticker_with_market: quantity}
            e.g. {"uae:EMAAR": 1000, "ksa:2222.SR": 500, "egypt:COMI": 200}
            or   {"EMAAR": 1000}   ← will search all Gulf markets

        Returns:
            {
              "positions":       DataFrame,
              "total_value":     float,
              "sector_weights":  dict,
              "market_weights":  dict,
              "avg_rsi":         float,
              "avg_pe":          float,
              "avg_div_yield":   float,
              "risk_score":      float,   (0–100, higher = riskier)
              "concentration":   str,     ("Diversified" / "Concentrated")
            }
        """
        rows = []

        for raw_ticker, qty in holdings.items():
            # parse "market:ticker" or just "ticker"
            if ":" in raw_ticker and raw_ticker.split(":")[0] in [*list(GULF_MARKETS), "egypt", "morocco", "america", "crypto"]:
                market, ticker = raw_ticker.split(":", 1)
                search_markets = [market]
            else:
                ticker = raw_ticker
                search_markets = GULF_MARKETS + ["egypt", "america"]

            stock = None
            found_market = None
            for m in search_markets:
                stock = self.get_stock(ticker, m)
                if stock:
                    found_market = m
                    break

            if stock is None:
                log.warning(f"⚠️  Portfolio: could not find '{raw_ticker}'")
                rows.append({
                    "ticker": raw_ticker, "market": "unknown",
                    "name": raw_ticker, "quantity": qty,
                    "price": None, "value": None,
                    "sector": "Unknown", "change": None,
                    "RSI": None, "price_earnings_ttm": None,
                    "dividend_yield_recent": None,
                })
                continue

            price = stock.get("close")
            value = price * qty if price else None
            rows.append({
                "ticker":                raw_ticker,
                "market":                found_market,
                "name":                  stock.get("name", raw_ticker),
                "quantity":              qty,
                "price":                 price,
                "value":                 value,
                "sector":                stock.get("sector", "Unknown"),
                "change":                stock.get("change"),
                "RSI":                   stock.get("RSI"),
                "price_earnings_ttm":    stock.get("price_earnings_ttm"),
                "dividend_yield_recent": stock.get("dividend_yield_recent"),
                "SMA50":                 stock.get("SMA50"),
                "SMA200":                stock.get("SMA200"),
            })

        positions = pd.DataFrame(rows)

        # ── Aggregates ────────────────────────────────────────────────────────
        total_value = positions["value"].sum() if "value" in positions.columns else 0

        # sector weights
        sector_weights = {}
        if total_value and "sector" in positions.columns:
            for sector, grp in positions.groupby("sector"):
                w = grp["value"].sum() / total_value * 100
                sector_weights[sector] = round(w, 1)

        # market weights
        market_weights = {}
        if total_value and "market" in positions.columns:
            for mkt, grp in positions.groupby("market"):
                w = grp["value"].sum() / total_value * 100
                market_weights[mkt] = round(w, 1)

        # averages (weighted by value)
        def weighted_avg(col):
            valid = positions.dropna(subset=[col, "value"])
            if valid.empty or valid["value"].sum() == 0:
                return None
            return round((valid[col] * valid["value"]).sum() / valid["value"].sum(), 2)

        avg_rsi       = weighted_avg("RSI")
        avg_pe        = weighted_avg("price_earnings_ttm")
        avg_div_yield = weighted_avg("dividend_yield_recent")

        # ── Risk Score (0–100) ────────────────────────────────────────────────
        # Higher = more risk
        risk = 50.0  # base

        # sector concentration penalty
        if sector_weights:
            top_sector_pct = max(sector_weights.values())
            if top_sector_pct > 60:
                risk += 20
            elif top_sector_pct > 40:
                risk += 10

        # RSI risk (overbought portfolio)
        if avg_rsi and avg_rsi > 70:
            risk += 15
        elif avg_rsi and avg_rsi < 30:
            risk -= 10  # oversold = potential opportunity

        # single-market concentration
        if market_weights:
            top_market_pct = max(market_weights.values())
            if top_market_pct > 80:
                risk += 15
            elif top_market_pct > 60:
                risk += 7

        risk = max(0, min(100, risk))

        # ── Concentration label ───────────────────────────────────────────────
        n_sectors = len(sector_weights)
        n_markets = len(market_weights)
        if n_sectors >= 4 and n_markets >= 3:
            concentration = "Well Diversified"
        elif n_sectors >= 2 or n_markets >= 2:
            concentration = "Moderately Diversified"
        else:
            concentration = "Concentrated"

        return {
            "positions":      positions,
            "total_value":    round(total_value, 2) if total_value else None,
            "sector_weights": sector_weights,
            "market_weights": market_weights,
            "avg_rsi":        avg_rsi,
            "avg_pe":         avg_pe,
            "avg_div_yield":  avg_div_yield,
            "risk_score":     round(risk, 1),
            "concentration":  concentration,
        }

    def cache_status(self) -> dict:
        """Snapshot timestamps for every market exposed by the data layer."""
        return {m: _mca.snapshot_timestamp(m) for m in _mca.list_markets()}
