"""
EisaX Report Enhancer  v1.0
============================
Injects four new data-driven sections into the assembled report markdown,
using the QueryEngine (cache-first, no on-demand scraping).

New sections injected between the News block and the Positioning Guide:

  ── Market Context         ← RSI vs sector, outperformance rank
  ── Regional Peers         ← same sector across Gulf markets table
  ── Cross-Market Intel     ← oil correlation, sector rotation signal
  ── Portfolio Builder      ← user can call separately

Public API:
    from report_enhancer import ReportEnhancer
    enhancer = ReportEnhancer(qe)
    enriched_report = enhancer.enhance(report_md, ticker="2222.SR")
"""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger("eisax.enhancer")

# ── Market label mapping (for peer table) ─────────────────────────────────────
_MARKET_TO_LABEL = {
    "ksa":     "🇸🇦 KSA",
    "uae":     "🇦🇪 UAE",
    "qatar":   "🇶🇦 Qatar",
    "kuwait":  "🇰🇼 Kuwait",
    "bahrain": "🇧🇭 Bahrain",
    "egypt":   "🇪🇬 Egypt",
    "morocco": "🇲🇦 Morocco",
    "oman":    "🇴🇲 Oman",
    "america": "🇺🇸 USA",
    "crypto":  "₿ Crypto",
}

# ── Market code resolver ───────────────────────────────────────────────────────
_SUFFIX_TO_MARKET = {
    ".sr":  "ksa",
    ".ae":  "uae",
    ".du":  "uae",
    ".ca":  "egypt",
    ".kw":  "kuwait",
    ".qa":  "qatar",
    ".bh":  "bahrain",
}

_ENERGY_KEYWORDS = [
    "energy", "oil", "gas", "petroleum", "refin", "aramco",
    "adnoc", "taqa", "oman oil",
]

_INJECTION_ANCHOR = "📊 **Positioning Guide**"


def _resolve_market(ticker: str) -> str | None:
    """Guess market from ticker suffix."""
    t = ticker.lower()
    for suffix, market in _SUFFIX_TO_MARKET.items():
        if t.endswith(suffix):
            return market
    # Bare tickers without suffix — try UAE first (DFM:EMAAR style)
    return None


def _fmt_num(val, decimals: int = 2, suffix: str = "") -> str:
    try:
        import math
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return "—"
        return f"{f:.{decimals}f}{suffix}"
    except Exception:
        return "—"


def _fmt_pct(val) -> str:
    return _fmt_num(val, 2, "%")


def _fmt_x(val) -> str:
    return _fmt_num(val, 1, "x")


def _rsi_label(rsi) -> str:
    try:
        r = float(rsi)
        if r >= 70: return f"{r:.1f} 🔴 Overbought"
        if r <= 30: return f"{r:.1f} 🟢 Oversold"
        return f"{r:.1f} 🟡 Neutral"
    except Exception:
        return "—"


def _change_arrow(val) -> str:
    try:
        v = float(val)
        return f"+{v:.2f}% ▲" if v >= 0 else f"{v:.2f}% ▼"
    except Exception:
        return "—"


def _extract_report_sector(report_md: str) -> str | None:
    """
    Parse the sector from the EisaX report header line, e.g.:
      **Sector:** Real Estate |
    TradingView sometimes misclassifies stocks (e.g. EMAAR as "Finance"),
    so we prefer the fundamental sector from the report header.
    """
    import re
    m = re.search(r'\*\*Sector:\*\*\s+([^|*\n]+)', report_md)
    if m:
        return m.group(1).strip()
    return None


# ── TradingView sector → canonical name used by cross_market() ────────────────
# Maps yfinance/fundamental sector names to TradingView sector names so we can
# query the cache with the right sector label.
_SECTOR_YFINANCE_TO_TV = {
    "real estate":          "Real Estate",
    "realty":               "Real Estate",
    "financials":           "Finance",
    "financial services":   "Finance",
    "information technology": "Electronic Technology",
    "technology":           "Electronic Technology",
    "consumer discretionary": "Consumer Durables",
    "consumer staples":     "Consumer Non-Durables",
    "health care":          "Health Technology",
    "healthcare":           "Health Technology",
    "industrials":          "Industrial Services",
    "utilities":            "Utilities",
    "materials":            "Non-Energy Minerals",
    "communication services": "Communications",
    "energy":               "Energy Minerals",
}


# ══════════════════════════════════════════════════════════════════════════════
# ReportEnhancer
# ══════════════════════════════════════════════════════════════════════════════

class ReportEnhancer:
    """
    Enhances an assembled EisaX report markdown string with live cache data.

    Usage:
        enhancer = ReportEnhancer(query_engine)
        md = enhancer.enhance(report_md, ticker="2222.SR")
    """

    def __init__(self, query_engine):
        self.qe = query_engine

    # ── Main entry ─────────────────────────────────────────────────────────────

    def enhance(self, report_md: str, ticker: str, market: str | None = None) -> str:
        """
        Inject new sections into the report markdown.
        Finds _INJECTION_ANCHOR and inserts sections just before it.
        Falls back gracefully if cache is unavailable.
        """
        if market is None:
            market = _resolve_market(ticker)

        if market is None:
            log.warning(f"[Enhancer] Cannot resolve market for '{ticker}' — skipping enhancement")
            return report_md

        # Fetch stock data from cache
        stock = self.qe.get_stock(ticker, market)
        if stock is None:
            log.warning(f"[Enhancer] '{ticker}' not found in [{market}] cache — skipping")
            return report_md

        # Build each enhancement block
        blocks = []

        ctx = _build_context(stock, ticker, market)

        # ── Sector override: prefer fundamental sector from report header ──────
        # TradingView misclassifies some stocks (e.g. EMAAR → "Finance" instead
        # of "Real Estate"). Parse the sector from the report text and use it
        # for cross_market() queries — the query_engine handles the TV mapping.
        _report_sector = _extract_report_sector(report_md)
        if _report_sector and _report_sector != ctx.get("sector"):
            log.info(f"[Enhancer] Sector override for {ticker}: "
                     f"TV='{ctx.get('sector')}' → report='{_report_sector}'")
            ctx["sector"] = _report_sector

        market_ctx  = self._build_market_context(stock, market, ctx)
        regional    = self._build_regional_peers(stock, ticker, market, ctx)
        cross_mkt   = self._build_cross_market_intel(stock, market, ctx)

        if market_ctx:  blocks.append(market_ctx)
        if regional:    blocks.append(regional)
        if cross_mkt:   blocks.append(cross_mkt)

        if not blocks:
            return report_md

        injection = "\n\n".join(blocks) + "\n\n"

        # Inject before Positioning Guide
        if _INJECTION_ANCHOR in report_md:
            return report_md.replace(
                _INJECTION_ANCHOR,
                injection + _INJECTION_ANCHOR,
                1
            )
        else:
            # Fallback: append before scorecard or at end
            for fallback in ["## 🎯 EisaX", "---\n> ⚠️ **Disclaimer"]:
                if fallback in report_md:
                    return report_md.replace(fallback, injection + fallback, 1)
            return report_md + "\n\n" + injection

    # ── Section A: Market Context ──────────────────────────────────────────────

    def _build_market_context(self, stock: dict, market: str, ctx: dict) -> str | None:
        """
        Shows where the stock stands within its market and sector:
        - RSI vs sector average
        - Today's change vs sector average change
        - Percentile rank within sector
        """
        try:
            df, _ = self.qe.cache.get_latest(market)
            if df is None or df.empty:
                return None

            sector = ctx.get("sector")
            rsi    = ctx.get("rsi")
            change = ctx.get("change")
            name   = ctx.get("name", ticker if (ticker := stock.get("ticker", "")) else "")

            if not sector:
                return None

            # Resolve TV sector for cache lookup (handles misclassifications)
            from query_engine import _SECTOR_ALIASES
            _alias = _SECTOR_ALIASES.get(sector.lower())
            _tv_sector_search = _alias["tv_sector"] if _alias else sector
            _name_keys = _alias["name_keywords"] if _alias else None

            if "sector" in df.columns:
                sector_df = df[df["sector"].str.contains(_tv_sector_search, case=False, na=False)].copy()
                # For aliased sectors (e.g. Real Estate inside Finance), filter by name
                if _name_keys and len(sector_df) > 0 and "name" in sector_df.columns:
                    nm_mask = sector_df["name"].str.lower().str.contains("|".join(_name_keys), na=False)
                    if "ticker" in sector_df.columns:
                        nm_mask = nm_mask | sector_df["ticker"].str.lower().str.contains("|".join(_name_keys), na=False)
                    sector_df = sector_df[nm_mask]
            else:
                sector_df = pd.DataFrame()
            market_df = df

            # sector averages
            sec_rsi_avg    = sector_df["RSI"].dropna().mean()    if "RSI"    in sector_df.columns and not sector_df.empty else None
            sec_change_avg = sector_df["change"].dropna().mean() if "change" in sector_df.columns and not sector_df.empty else None
            mkt_rsi_avg    = market_df["RSI"].dropna().mean()    if "RSI"    in market_df.columns else None
            mkt_change_avg = market_df["change"].dropna().mean() if "change" in market_df.columns else None

            # percentile rank in sector by today's change
            peer_rank_txt = ""
            if "change" in sector_df.columns and change is not None:
                ranked = sector_df["change"].dropna().sort_values(ascending=False)
                n_total = len(ranked)
                if n_total > 1:
                    n_beating = (ranked > change).sum()
                    pct_outperforming = round((1 - n_beating / n_total) * 100)
                    peer_rank_txt = (
                        f"outperforming **{pct_outperforming}%** of its {sector} sector peers today"
                        if pct_outperforming >= 50
                        else f"underperforming **{100 - pct_outperforming}%** of its {sector} sector peers today"
                    )

            # RSI interpretation vs sector
            rsi_vs_sector = ""
            if rsi is not None and sec_rsi_avg is not None:
                diff = rsi - sec_rsi_avg
                direction = "above" if diff >= 0 else "below"
                rsi_vs_sector = f"RSI **{rsi:.1f}** *(cache snapshot)* — {abs(diff):.1f} pts {direction} sector avg ({sec_rsi_avg:.1f})"

            # change vs market
            change_vs_mkt = ""
            if change is not None and mkt_change_avg is not None:
                diff = change - mkt_change_avg
                direction = "above" if diff >= 0 else "below"
                change_vs_mkt = f"Today **{change:+.2f}%** — {abs(diff):.2f}% {direction} market avg ({mkt_change_avg:+.2f}%)"

            lines = ["---", "### 🌡️ Market Context *(EisaX Cache Data)*", ""]
            if rsi_vs_sector:
                lines.append(f"- **Momentum:** {rsi_vs_sector}")
            if change_vs_mkt:
                lines.append(f"- **Today:** {change_vs_mkt}")
            if peer_rank_txt:
                lines.append(f"- **Sector Standing:** {stock.get('name', '')} is {peer_rank_txt}")

            # sector heatmap row
            if sec_rsi_avg is not None and sec_change_avg is not None:
                mkt_rsi_str    = f"{mkt_rsi_avg:.1f}"    if mkt_rsi_avg    else "—"
                mkt_change_str = f"{mkt_change_avg:+.2f}%" if mkt_change_avg else "—"
                lines += [
                    "",
                    f"| | This Stock | Sector Avg ({sector}) | Market Avg |",
                    "|---|---|---|---|",
                    f"| RSI | {_rsi_label(rsi)} | {sec_rsi_avg:.1f} | {mkt_rsi_str} |",
                    f"| Change Today | {_change_arrow(change)} | {sec_change_avg:+.2f}% | {mkt_change_str} |",
                ]

            return "\n".join(lines)

        except Exception as e:
            log.error(f"[Enhancer] Market Context failed: {e}")
            return None

    # ── Section B: Regional Peer Comparison ───────────────────────────────────

    def _build_regional_peers(self, stock: dict, ticker: str, market: str, ctx: dict) -> str | None:
        """
        Same sector across Gulf markets — ranked comparison table.
        """
        try:
            sector = ctx.get("sector")
            if not sector:
                return None

            cross_df = self.qe.cross_market(sector)
            if cross_df is None or cross_df.empty:
                return None

            # limit to 12 best peers by market cap
            if "market_cap_basic" in cross_df.columns:
                cross_df = cross_df.dropna(subset=["market_cap_basic"])
                cross_df = cross_df.nlargest(12, "market_cap_basic")

            # find rank of target stock
            target_rank = None
            ticker_upper = ticker.upper()
            if "ticker" in cross_df.columns:
                matches = cross_df[cross_df["ticker"].str.upper().str.contains(ticker_upper, na=False)]
                if not matches.empty:
                    target_rank = cross_df.index.get_loc(matches.index[0]) + 1

            # rename _query_market so itertuples() can access it (leading _ breaks namedtuple)
            cross_df = cross_df.rename(columns={"_query_market": "qmarket"})

            lines = [
                "---",
                f"### 🌍 Regional Peer Comparison — *{sector} Sector Across Gulf Markets*",
                "",
                "| # | Stock | Market | Price | Change | P/E | RSI | Div Yield | Mkt Cap |",
                "|---|-------|--------|-------|--------|-----|-----|-----------|---------|",
            ]

            for i, row in enumerate(cross_df.itertuples(), start=1):
                t          = getattr(row, "ticker", "—")
                nm         = getattr(row, "name", "—")
                _qmkt      = getattr(row, "qmarket", "")
                mkt        = _MARKET_TO_LABEL.get(str(_qmkt).lower(), str(_qmkt).upper() or "—")
                price      = _fmt_num(getattr(row, "close", None))
                chg        = _change_arrow(getattr(row, "change", None))
                pe         = _fmt_x(getattr(row, "price_earnings_ttm", None))
                rsi        = _fmt_num(getattr(row, "RSI", None), 1)
                div        = _fmt_pct(getattr(row, "dividend_yield_recent", None))
                mcap_raw   = getattr(row, "market_cap_basic", None)
                try:
                    mcap = f"{float(mcap_raw)/1e9:.1f}B" if mcap_raw else "—"
                except Exception:
                    mcap = "—"

                # highlight target stock
                is_target = ticker_upper in str(t).upper()
                marker = " ← **YOU**" if is_target else ""

                lines.append(
                    f"| {i} | **{nm}** `{t}`{marker} | {mkt} | {price} | {chg} | {pe} | {rsi} | {div} | {mcap} |"
                )

            if target_rank:
                n = len(cross_df)
                lines += ["", f"> 📍 **{stock.get('name', ticker)}** ranks **#{target_rank} of {n}** in the {sector} sector across Gulf markets by market cap."]

            return "\n".join(lines)

        except Exception as e:
            log.error(f"[Enhancer] Regional Peers failed: {e}")
            return None

    # ── Section C: Cross-Market Intelligence ──────────────────────────────────

    def _build_cross_market_intel(self, stock: dict, market: str, ctx: dict) -> str | None:
        """
        - Oil correlation note for energy stocks
        - Sector rotation signal (sector RSI vs market RSI)
        - Market breadth: % of market up today
        """
        try:
            sector  = ctx.get("sector", "")
            rsi     = ctx.get("rsi")
            is_energy = ctx.get("is_energy", False)

            df, _ = self.qe.cache.get_latest(market)
            if df is None or df.empty:
                return None

            lines = ["---", "### 🔀 Cross-Market Intelligence", ""]

            # ── Sector rotation signal ────────────────────────────────────────
            summary = self.qe.sector_summary(market)
            if summary is not None and not summary.empty and "RSI" in summary.columns:
                market_rsi_avg = df["RSI"].dropna().mean() if "RSI" in df.columns else None
                if sector and market_rsi_avg:
                    sec_row = summary[summary["sector"].str.contains(sector, case=False, na=False)]
                    if not sec_row.empty:
                        sec_rsi = float(sec_row["RSI"].iloc[0])
                        diff    = sec_rsi - market_rsi_avg
                        if diff > 5:
                            signal = f"🔥 **{sector}** sector is showing momentum (RSI {sec_rsi:.1f} vs market avg {market_rsi_avg:.1f}) — potential overbought rotation risk"
                        elif diff < -5:
                            signal = f"💡 **{sector}** sector is lagging (RSI {sec_rsi:.1f} vs market avg {market_rsi_avg:.1f}) — potential mean-reversion opportunity"
                        else:
                            signal = f"⚖️ **{sector}** sector RSI ({sec_rsi:.1f}) is in line with the market ({market_rsi_avg:.1f}) — no strong rotation signal"
                        lines.append(f"**Sector Rotation:** {signal}")

            # ── Market breadth ────────────────────────────────────────────────
            if "change" in df.columns:
                total   = len(df.dropna(subset=["change"]))
                gainers = (df["change"] > 0).sum()
                losers  = (df["change"] < 0).sum()
                if total > 0:
                    pct_up = gainers / total * 100
                    breadth_emoji = "🟢" if pct_up > 55 else "🔴" if pct_up < 45 else "🟡"
                    lines.append(
                        f"**Market Breadth ({market.upper()}):** {breadth_emoji} "
                        f"{gainers} advancing / {losers} declining "
                        f"({pct_up:.0f}% of stocks up today)"
                    )

            # ── Oil correlation note for energy stocks ────────────────────────
            if is_energy:
                # load commodities cache
                comm_df, comm_ts = self.qe.cache.get_latest("commodities")
                if comm_df is not None:
                    oil_row = comm_df[comm_df["ticker"].isin(["BZ=F", "CL=F"])].iloc[0] if not comm_df[comm_df["ticker"].isin(["BZ=F", "CL=F"])].empty else None
                    if oil_row is not None:
                        oil_price  = oil_row.get("close")
                        oil_change = oil_row.get("change")
                        oil_name   = oil_row.get("name", "Oil")
                        if oil_price:
                            lines.append(
                                f"**Oil Correlation:** {oil_name} at **${oil_price:.2f}** "
                                f"({_change_arrow(oil_change)}) — "
                                f"Energy stocks typically move with oil; monitor Brent for directional cues."
                            )

            # ── Cache freshness note ──────────────────────────────────────────
            age = self.qe.cache.cache_age_minutes(market)
            if age is not None:
                lines.append(f"\n> 🕐 *Cache data is **{age:.0f} min** old — refreshed every 15 min by EisaX Pipeline*")

            return "\n".join(lines) if len(lines) > 3 else None

        except Exception as e:
            log.error(f"[Enhancer] Cross-Market Intel failed: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio Builder section  (standalone — called separately from Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

def build_portfolio_section(qe, holdings: dict) -> str:
    """
    Builds a standalone markdown Portfolio Analysis section.

    holdings: {"uae:EMAAR": 1000, "ksa:2222.SR": 500}

    Returns a markdown string ready to append to any report or display standalone.
    """
    result = qe.portfolio_analyze(holdings)
    if not result or result.get("total_value") is None:
        return "> ⚠️ Portfolio analysis unavailable — check tickers and cache."

    pos       = result["positions"]
    total_val = result["total_value"]
    sectors   = result["sector_weights"]
    markets   = result["market_weights"]
    risk      = result["risk_score"]
    conc      = result["concentration"]
    avg_rsi   = result["avg_rsi"]
    avg_pe    = result["avg_pe"]
    avg_div   = result["avg_div_yield"]

    # risk colour
    risk_emoji = "🟢" if risk < 40 else "🔴" if risk > 70 else "🟡"

    lines = [
        "---",
        "### 💼 Portfolio Analysis *(EisaX Cache Data)*",
        "",
        f"**Total Estimated Value:** {total_val:,.0f}  |  "
        f"**Risk Score:** {risk_emoji} {risk}/100  |  "
        f"**Diversification:** {conc}",
        "",
        "#### Holdings",
        "| Stock | Market | Qty | Price | Value | Sector | RSI | P/E |",
        "|-------|--------|-----|-------|-------|--------|-----|-----|",
    ]

    for _, row in pos.iterrows():
        val = f"{row['value']:,.0f}" if pd.notna(row.get('value')) else "—"
        lines.append(
            f"| **{row.get('name','—')}** | {row.get('market','—').upper()} | "
            f"{int(row['quantity'])} | {_fmt_num(row.get('price'))} | {val} | "
            f"{row.get('sector','—')} | {_fmt_num(row.get('RSI'),1)} | {_fmt_x(row.get('price_earnings_ttm'))} |"
        )

    # sector weights bar
    if sectors:
        lines += ["", "#### Sector Allocation"]
        for sec, weight in sorted(sectors.items(), key=lambda x: -x[1]):
            bar_len = int(weight / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"- **{sec}** `{bar}` {weight}%")

    # market weights
    if markets:
        lines += ["", "#### Geographic Allocation"]
        for mkt, weight in sorted(markets.items(), key=lambda x: -x[1]):
            lines.append(f"- **{mkt.upper()}** — {weight}%")

    # averages
    lines += [
        "",
        "#### Portfolio Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Weighted Avg RSI | {_rsi_label(avg_rsi)} |",
        f"| Weighted Avg P/E | {_fmt_x(avg_pe)} |",
        f"| Weighted Avg Dividend Yield | {_fmt_pct(avg_div)} |",
        f"| Risk Score | {risk_emoji} {risk}/100 |",
        f"| Diversification | {conc} |",
    ]

    return "\n".join(lines)


# ── Internal helper ────────────────────────────────────────────────────────────

def _build_context(stock: dict, ticker: str, market: str) -> dict:
    """Extract common fields from stock dict."""
    sector = stock.get("sector", "")
    return {
        "sector":    sector,
        "rsi":       stock.get("RSI"),
        "change":    stock.get("change"),
        "name":      stock.get("name", ticker),
        "is_energy": any(kw in sector.lower() for kw in _ENERGY_KEYWORDS),
    }
