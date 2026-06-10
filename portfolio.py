"""
EisaX Portfolio  v1.0
======================
Cache-first portfolio analysis, complement suggestions, stress testing,
and rebalancing — all from the 15-min institutional Data Layer snapshots.

Distinct from core/portfolio_manager.py (which uses MPT + yfinance history).
This module is instant: reads parquet, returns results in milliseconds.

Usage:
    from portfolio import Portfolio
    from pipeline import cache, fetcher
    from query_engine import QueryEngine

    qe = QueryEngine(cache, fetcher)
    p  = Portfolio(qe)

    p.add("EMAAR",   market="uae",  qty=1000, cost_basis=14.50)
    p.add("2222.SR", market="ksa",  qty=500,  cost_basis=30.00)
    p.add("COMI",    market="egypt",qty=200,  cost_basis=70.00)

    summary  = p.summary()
    comps    = p.suggest_complements(n=5)
    stress   = p.stress_test()
    rebal    = p.rebalance_to({"Finance": 40, "Energy": 30, "Technology": 30})
    report   = p.to_markdown()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.data_layer import market_cache_adapter as _mca

log = logging.getLogger("eisax.portfolio")

# ── Scenario definitions for stress test ──────────────────────────────────────
SCENARIOS = [
    {
        "name":   "🛢️ Oil Crash (Brent → $50)",
        "shocks": {"Energy Minerals": -0.30, "Energy": -0.25, "Finance": -0.10},
        "default": -0.12,
    },
    {
        "name":   "📉 Gulf Market Selloff (−20%)",
        "shocks": {},
        "default": -0.20,
    },
    {
        "name":   "🏦 Fed Rate Shock (+200bps)",
        "shocks": {"Finance": +0.05, "Real Estate": -0.18, "Technology Services": -0.15},
        "default": -0.08,
    },
    {
        "name":   "🚀 Oil Spike (Brent → $150)",
        "shocks": {"Energy Minerals": +0.25, "Energy": +0.20, "Finance": +0.05},
        "default": +0.04,
    },
    {
        "name":   "🌱 ESG Rotation (10yr horizon)",
        "shocks": {"Energy Minerals": -0.35, "Energy": -0.30, "Technology Services": +0.20},
        "default": -0.05,
    },
]


# ── Position dataclass ─────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker:     str
    market:     str
    qty:        float
    cost_basis: Optional[float] = None   # price you paid per share
    # filled from cache
    name:       str = ""
    price:      Optional[float] = None
    sector:     str = "Unknown"
    rsi:        Optional[float] = None
    pe:         Optional[float] = None
    div_yield:  Optional[float] = None
    sma50:      Optional[float] = None
    sma200:     Optional[float] = None
    change:     Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio
# ══════════════════════════════════════════════════════════════════════════════

class Portfolio:
    """
    Cache-first portfolio: instant analysis, suggestions, stress test.
    """

    def __init__(self, query_engine):
        self.qe        = query_engine
        self._positions: list[Position] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def add(self, ticker: str, market: str, qty: float,
            cost_basis: Optional[float] = None) -> "Portfolio":
        """Add a position. Chainable."""
        pos = Position(ticker=ticker, market=market, qty=qty, cost_basis=cost_basis)
        self._positions.append(pos)
        return self

    def clear(self):
        self._positions.clear()

    @classmethod
    def from_dict(cls, qe, holdings: dict) -> "Portfolio":
        """
        Build from a dict.

        Format A: {"uae:EMAAR": {"qty": 1000, "cost": 14.5}}
        Format B: {"uae:EMAAR": 1000}            ← qty only
        Format C: {"EMAAR": 1000}                ← no market prefix
        """
        p = cls(qe)
        for raw_key, val in holdings.items():
            if isinstance(val, dict):
                qty  = float(val.get("qty", val.get("quantity", 1)))
                cost = val.get("cost", val.get("cost_basis"))
            else:
                qty  = float(val)
                cost = None

            if ":" in raw_key:
                market, ticker = raw_key.split(":", 1)
            else:
                market = None
                ticker = raw_key

            if market is None:
                # try to guess from ticker suffix
                from report_enhancer import _resolve_market
                market = _resolve_market(ticker) or "uae"

            p.add(ticker, market, qty, cost)
        return p

    # ── Enrich from cache ──────────────────────────────────────────────────────

    def _enrich(self) -> list[Position]:
        """Fill price/sector/technicals from cache for all positions."""
        enriched = []
        for pos in self._positions:
            stock = self.qe.get_stock(pos.ticker, pos.market)
            if stock:
                pos.name      = stock.get("name", pos.ticker)
                pos.price     = stock.get("close")
                pos.sector    = stock.get("sector", "Unknown") or "Unknown"
                pos.rsi       = stock.get("RSI")
                pos.pe        = stock.get("price_earnings_ttm")
                pos.div_yield = stock.get("dividend_yield_recent")
                pos.sma50     = stock.get("SMA50")
                pos.sma200    = stock.get("SMA200")
                pos.change    = stock.get("change")
            else:
                log.warning(f"[Portfolio] Not found in cache: {pos.ticker} [{pos.market}]")
                pos.name = pos.ticker
            enriched.append(pos)
        return enriched

    # ── Core summary ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Full portfolio summary.
        Returns a rich dict with positions DataFrame + aggregates.
        """
        positions = self._enrich()

        rows = []
        for p in positions:
            value    = (p.price * p.qty)        if p.price        else None
            pnl      = ((p.price - p.cost_basis) * p.qty) if (p.price and p.cost_basis) else None
            pnl_pct  = ((p.price / p.cost_basis - 1) * 100) if (p.price and p.cost_basis) else None
            rows.append({
                "ticker":     p.ticker,
                "name":       p.name,
                "market":     p.market.upper(),
                "qty":        p.qty,
                "price":      p.price,
                "cost_basis": p.cost_basis,
                "value":      value,
                "pnl":        pnl,
                "pnl_pct":    pnl_pct,
                "sector":     p.sector,
                "change_today": p.change,
                "RSI":        p.rsi,
                "PE":         p.pe,
                "div_yield":  p.div_yield,
                "SMA50":      p.sma50,
                "SMA200":     p.sma200,
            })

        df = pd.DataFrame(rows)
        total_value = df["value"].sum() if "value" in df.columns else 0
        total_pnl   = df["pnl"].sum()   if "pnl" in df.columns  else None

        def _weighted_avg(col):
            valid = df.dropna(subset=[col, "value"])
            if valid.empty or valid["value"].sum() == 0:
                return None
            return round((valid[col] * valid["value"]).sum() / valid["value"].sum(), 2)

        # weights
        def _weights(col):
            if not total_value:
                return {}
            grp = df.dropna(subset=[col, "value"]).groupby(col)["value"].sum()
            return {k: round(v / total_value * 100, 1) for k, v in grp.items()}

        sector_weights = _weights("sector")
        market_weights = _weights("market")

        # risk score
        risk = _compute_risk(sector_weights, market_weights, _weighted_avg("RSI"))

        # ── Institutional diversification metrics ──────────────────────────
        from core.services.portfolio_analytics import (
            compute_effective_n,
            diversification_label,
            diversification_emoji,
            compute_economic_buckets,
            bucket_concentration_warning,
            sharpe_context_note,
            diversification_soft_suggestion,
            consistent_diversification_phrase,
        )
        _pos_weights = [
            (row.get("value") or 0) / total_value
            for _, row in df.iterrows()
            if total_value and (row.get("value") or 0) > 0
        ]
        effective_n = compute_effective_n(_pos_weights)
        eff_n_label_text = diversification_label(effective_n)
        eff_n_icon = diversification_emoji(effective_n)

        # Economic buckets (correlation-aware)
        _pos_list = []
        for _, row in df.iterrows():
            val = row.get("value") or 0
            if not total_value or val <= 0:
                continue
            _pos_list.append({
                "ticker": row.get("ticker", ""),
                "sector": row.get("sector", ""),
                "market": row.get("market", ""),
                "weight": val / total_value,
            })
        economic_buckets = compute_economic_buckets(_pos_list)
        bucket_warning = bucket_concentration_warning(economic_buckets, threshold=50.0)
        soft_diversif_suggestion = diversification_soft_suggestion(
            effective_n, economic_buckets,
        )

        # Legacy label — but kept consistent with Effective N verdict
        n_sec = len(sector_weights)
        n_mkt = len(market_weights)
        if n_sec >= 4 and n_mkt >= 3:
            _raw_div = "Well Diversified 🟢"
        elif n_sec >= 2 or n_mkt >= 2:
            _raw_div = "Moderately Diversified 🟡"
        else:
            _raw_div = "Concentrated 🔴"
        # If Effective N says concentrated, the legacy "well diversified"
        # label must NOT survive — keeps report text and metrics aligned.
        diversification = consistent_diversification_phrase(effective_n, _raw_div)
        if diversification == _raw_div and effective_n >= 5 and eff_n_label_text == "Well diversified":
            # keep legacy wording when it agrees
            pass

        # today's portfolio P&L
        today_pnl_pct = _weighted_avg("change_today")

        return {
            "positions":        df,
            "total_value":      round(total_value, 2) if total_value else None,
            "total_pnl":        round(total_pnl,   2) if total_pnl   else None,
            "sector_weights":   sector_weights,
            "market_weights":   market_weights,
            "avg_rsi":          _weighted_avg("RSI"),
            "avg_pe":           _weighted_avg("PE"),
            "avg_div_yield":    _weighted_avg("div_yield"),
            "today_change":     today_pnl_pct,
            "risk_score":       risk,
            "diversification":  diversification,
            "n_positions":      len(df),
            "effective_n":      round(effective_n, 2),
            "effective_n_label": eff_n_label_text,
            "effective_n_icon": eff_n_icon,
            "economic_buckets": economic_buckets,
            "bucket_warning":   bucket_warning,
            "diversif_suggestion": soft_diversif_suggestion,
        }

    # ── Complement suggestions ─────────────────────────────────────────────────

    def suggest_complements(self, n: int = 5) -> pd.DataFrame:
        """
        Suggests stocks from the cache that would improve diversification.

        Logic:
        1. Find under-represented sectors in the portfolio
        2. Find under-represented markets
        3. Screen those sectors/markets for quality stocks (RSI 35-60, low P/E, div yield > 0)
        4. Exclude stocks already held
        5. Return top N by market cap
        """
        summ = self.summary()
        held_names = set(
            summ["positions"]["name"].str.upper().tolist() +
            summ["positions"]["ticker"].str.upper().tolist()
        )
        sector_weights = summ["sector_weights"]
        market_weights = summ["market_weights"]

        # markets to search — prefer under-represented ones
        all_markets = ["uae", "ksa", "egypt", "kuwait", "qatar"]
        held_markets = {m.lower() for m in market_weights.keys()}
        search_markets = sorted(all_markets, key=lambda m: market_weights.get(m.upper(), 0))

        # sectors to target — under-represented
        held_sectors = set(sector_weights.keys())

        candidates = []
        for market in search_markets:
            df = _mca.get_latest_snapshot(market)
            if df is None or df.empty:
                continue

            # filter: healthy RSI, positive price, some market cap
            mask = pd.Series([True] * len(df))
            if "RSI"    in df.columns: mask &= df["RSI"].between(30, 65)
            if "close"  in df.columns: mask &= df["close"] > 0
            if "change" in df.columns: mask &= df["change"].notna()
            subset = df[mask].copy()

            if subset.empty:
                continue

            subset["_market"] = market

            # prefer under-represented sectors
            if "sector" in subset.columns and held_sectors:
                subset["_sector_weight"] = subset["sector"].map(
                    lambda s: sector_weights.get(s, 0)
                )
                subset = subset.sort_values("_sector_weight", ascending=True)

            candidates.append(subset)

        if not candidates:
            return pd.DataFrame()

        all_candidates = pd.concat(candidates, ignore_index=True)

        # exclude already held
        if "name" in all_candidates.columns:
            all_candidates = all_candidates[
                ~all_candidates["name"].str.upper().isin(held_names)
            ]
        if "ticker" in all_candidates.columns:
            all_candidates = all_candidates[
                ~all_candidates["ticker"].str.upper().isin(held_names)
            ]

        # sort by market cap descending
        if "market_cap_basic" in all_candidates.columns:
            all_candidates = all_candidates.sort_values("market_cap_basic", ascending=False)

        cols = [c for c in [
            "_market", "ticker", "name", "close", "change",
            "sector", "RSI", "price_earnings_ttm",
            "dividend_yield_recent", "market_cap_basic"
        ] if c in all_candidates.columns]

        result = all_candidates[cols].head(n).reset_index(drop=True)
        result.rename(columns={"_market": "market"}, inplace=True)
        result.insert(0, "#", range(1, len(result) + 1))
        return result

    # ── Stress test ────────────────────────────────────────────────────────────

    def stress_test(self) -> pd.DataFrame:
        """
        Apply macro shock scenarios to the portfolio.
        Returns a DataFrame with one row per scenario.
        """
        summ = self.summary()
        df   = summ["positions"]
        total_value = summ["total_value"] or 0

        if total_value == 0 or df.empty:
            return pd.DataFrame()

        results = []
        for scenario in SCENARIOS:
            shocks  = scenario["shocks"]
            default = scenario["default"]

            portfolio_impact = 0.0
            for _, row in df.iterrows():
                val    = row.get("value")
                sector = row.get("sector", "")
                if not val:
                    continue
                shock = shocks.get(sector, default)
                portfolio_impact += val * shock

            pct_impact   = (portfolio_impact / total_value * 100) if total_value else 0
            new_value    = total_value + portfolio_impact
            emoji        = "📈" if pct_impact > 0 else "📉"

            results.append({
                "Scenario":          scenario["name"],
                "Portfolio Impact":  f"{pct_impact:+.1f}%",
                "Value Change":      f"{portfolio_impact:+,.0f}",
                "New Portfolio Value": f"{new_value:,.0f}",
                "Signal":            emoji,
            })

        return pd.DataFrame(results)

    # ── Rebalancing suggestions ────────────────────────────────────────────────

    def rebalance_to(self, target_sector_weights: dict) -> pd.DataFrame:
        """
        Calculate buy/sell actions to reach target sector allocation.

        target_sector_weights: {"Finance": 40, "Energy": 30, "Technology": 30}
        Weights should sum to ~100 (%).

        Returns a DataFrame with suggested actions.
        """
        summ        = self.summary()
        total_value = summ["total_value"] or 0
        current_sw  = summ["sector_weights"]

        if total_value == 0:
            return pd.DataFrame()

        rows = []
        all_sectors = set(list(current_sw.keys()) + list(target_sector_weights.keys()))

        for sector in sorted(all_sectors):
            current_pct = current_sw.get(sector, 0.0)
            target_pct  = target_sector_weights.get(sector, 0.0)
            diff_pct    = target_pct - current_pct
            diff_value  = diff_pct / 100 * total_value

            if abs(diff_pct) < 1.0:
                action = "✅ On Target"
            elif diff_pct > 0:
                action = f"🟢 BUY +{diff_pct:.1f}% (≈ {diff_value:+,.0f})"
            else:
                action = f"🔴 SELL {diff_pct:.1f}% (≈ {diff_value:+,.0f})"

            rows.append({
                "Sector":        sector,
                "Current Weight": f"{current_pct:.1f}%",
                "Target Weight":  f"{target_pct:.1f}%",
                "Gap":            f"{diff_pct:+.1f}%",
                "Action":         action,
            })

        return pd.DataFrame(rows)

    # ── Markdown report ────────────────────────────────────────────────────────

    def to_markdown(
        self,
        target_weights:   dict | None = None,
        cash:             float = 0.0,
        final_decision:   dict | None = None,
        execution_plan:   list | None = None,
        benchmark:        dict | None = None,
        drawdown_target:  float = 25.0,    # client mandate drawdown floor (%)
    ) -> str:
        """
        Institutional-grade portfolio report.

        Section order (command-first):
          1. Decision
          2. Execution Plan
          3. Benchmark / Alpha
          4. Summary
          5. Holdings
          6. Allocation (Sector + Geographic + Cash)
          7. Risk Breakdown
          8. Portfolio Metrics
          9. Stress Test
          10. Suggested Additions
          11. Rebalancing (if target_weights supplied)
        """
        from datetime import date as _date

        summ        = self.summary()
        df          = summ["positions"]
        total_value = summ["total_value"] or 0
        total_pnl   = summ["total_pnl"]
        risk        = summ["risk_score"]
        diversif    = summ["diversification"]
        today_chg   = summ["today_change"]
        avg_rsi     = summ["avg_rsi"]
        avg_pe      = summ["avg_pe"]
        avg_div     = summ["avg_div_yield"]
        sectors     = summ["sector_weights"]
        markets     = summ["market_weights"]
        n_pos       = summ["n_positions"]
        eff_n       = summ.get("effective_n", 0.0)
        eff_n_lbl   = summ.get("effective_n_label", "Unknown")
        eff_n_ic    = summ.get("effective_n_icon", "⚪")
        econ_bkts   = summ.get("economic_buckets", {}) or {}
        bkt_warn    = summ.get("bucket_warning")
        soft_sugg   = summ.get("diversif_suggestion")

        risk_emoji = "🟢" if risk < 40 else "🔴" if risk > 70 else "🟡"
        today_str  = f"{today_chg:+.2f}%" if today_chg is not None else "—"
        total_with_cash = total_value + cash

        lines = [
            "---",
            f"# 💼 EisaX Portfolio Report",
            f"*Generated {_date.today().isoformat()} — data from 15-min pipeline cache*",
            "",
        ]

        # ══════════════════════════════════════════════════════════════════════
        # 1. PORTFOLIO DECISION
        # ══════════════════════════════════════════════════════════════════════
        if final_decision:
            _icon  = final_decision.get("icon", "📊")
            _dec   = final_decision.get("decision", "Hold")
            _conf  = final_decision.get("confidence", 70)
            _rsns  = final_decision.get("reasons", [])
            lines += [
                "## 📊 Portfolio Decision",
                "",
                f"### {_icon} {_dec}",
                f"**Confidence:** {_conf}%",
                "",
            ]
            if _rsns:
                lines.append("**Why:**")
                for r in _rsns:
                    lines.append(f"- {r}")
            lines.append("")
        else:
            # Auto-generate decision from summary data
            _auto_dec, _auto_icon, _auto_conf, _auto_rsns = "Hold Portfolio", "✅", 75, []
            for sec, w in sectors.items():
                if w > 50:
                    _auto_dec, _auto_icon, _auto_conf = "Rebalance", "💛", 80
                    _auto_rsns.append(f"Sector concentration: {sec} = {w:.0f}%")
            for mkt, w in markets.items():
                if w > 60:
                    _auto_dec, _auto_icon, _auto_conf = "Reduce Exposure", "🔴", 85
                    _auto_rsns.append(f"Market over-exposure: {mkt.upper()} = {w:.0f}%")
            if avg_rsi and float(avg_rsi) > 70:
                _auto_rsns.append(f"Portfolio overbought: avg RSI = {float(avg_rsi):.0f}")
            if n_pos < 3:
                _auto_rsns.append("Low diversification: fewer than 3 positions")
            if risk > 70 and _auto_dec == "Hold Portfolio":
                _auto_dec, _auto_icon, _auto_conf = "Reduce Exposure", "🔴", 82
            if not _auto_rsns:
                _auto_rsns = ["Portfolio within normal parameters"]
            lines += [
                "## 📊 Portfolio Decision",
                "",
                f"### {_auto_icon} {_auto_dec}",
                f"**Confidence:** {_auto_conf}%",
                "",
                "**Why:**",
            ]
            for r in _auto_rsns:
                lines.append(f"- {r}")
            lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # 2. EXECUTION PLAN
        # ══════════════════════════════════════════════════════════════════════
        lines += ["## ⚡ Execution Plan", ""]
        if execution_plan:
            for ea in execution_plan:
                lines.append(f"{ea.get('icon','•')} **{ea.get('action','')}**")
                if ea.get("detail"):
                    lines.append(f"  → {ea['detail']}")
            lines.append("")
        else:
            # Auto-generate from positions
            _TARGET_W = 20.0
            _exec_auto = []
            for _, row in df.iterrows():
                _tk  = row["ticker"]
                _w   = round((row.get("value") or 0) / max(total_value, 1) * 100, 1)
                _rsi = row.get("RSI")
                _pr  = row.get("price")
                _qty = row.get("qty")
                if _w > _TARGET_W + 5 and _pr and _qty:
                    _sell_val = (_w - _TARGET_W) / 100 * total_value
                    _sell_sh  = int(_sell_val / float(_pr))
                    if _sell_sh > 0:
                        _exec_auto.append(
                            f"🔴 **Sell {_sell_sh:,} shares of {_tk}** "
                            f"→ Reduce from {_w:.0f}% → {_TARGET_W:.0f}% (free ~{_sell_val:,.0f})"
                        )
                elif _rsi and float(_rsi) > 70 and _qty:
                    _sell_sh = max(1, int(float(_qty) * 0.25))
                    _exec_auto.append(
                        f"🟡 **Sell 25% of {_tk} ({_sell_sh:,} shares)** "
                        f"→ RSI overbought at {float(_rsi):.0f} — partial profit"
                    )
                elif _rsi and float(_rsi) < 30:
                    _exec_auto.append(
                        f"🟢 **Add to {_tk}** → RSI oversold at {float(_rsi):.0f} — entry opportunity"
                    )
            if cash > 0:
                _cash_pct = cash / total_with_cash * 100
                _under = [s for s, w in sectors.items() if w < 10]
                if _cash_pct > 30:
                    _exec_auto.append(
                        f"💵 **Deploy cash ({cash:,.0f})** "
                        f"→ {_cash_pct:.0f}% idle — consider: {', '.join(_under[:3]) or 'diversify'}"
                    )
                else:
                    _exec_auto.append(
                        f"✅ **Maintain cash reserve ({cash:,.0f} / {_cash_pct:.0f}%)** → adequate buffer"
                    )
            if len(markets) < 2:
                _exec_auto.append(
                    "🌍 **Add geographic diversification** → single market exposure — consider UAE / KSA / Egypt"
                )
            if not _exec_auto:
                _exec_auto = ["✅ **No immediate action required** → monitor RSI weekly"]
            for ea in _exec_auto:
                lines.append(ea)
            lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # 3. BENCHMARK / ALPHA
        # ══════════════════════════════════════════════════════════════════════
        if benchmark:
            _bn   = benchmark.get("name", "Benchmark")
            _pr   = benchmark.get("ptf_ret", 0)
            _br   = benchmark.get("bench_ret", 0)
            _al   = benchmark.get("alpha", _pr - _br)
            _per  = benchmark.get("period_start", "—")
            _al_e = "🟢" if _al >= 0 else "🔴"
            lines += [
                "## 📊 Benchmark Comparison",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Your Portfolio Return | {_pr:+.1f}% |",
                f"| {_bn} Return | {_br:+.1f}% |",
                f"| **Alpha** | {_al_e} **{_al:+.1f}%** |",
                f"| Period | {_per} → today |",
                "",
            ]

        # ══════════════════════════════════════════════════════════════════════
        # 4. SUMMARY
        # ══════════════════════════════════════════════════════════════════════
        pnl_emoji = "🟢" if (total_pnl or 0) >= 0 else "🔴"
        lines += [
            "## 💰 Portfolio Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Invested (Cost Basis) | {sum((row.get('qty',0) or 0) * (row.get('cost_basis',0) or 0) for _, row in df.iterrows()):,.0f} |",
            f"| Market Value | {total_value:,.0f} |",
            f"| Unrealized P&L | {pnl_emoji} {total_pnl:+,.0f} |" if total_pnl is not None else "| Unrealized P&L | — |",
            f"| Cash | {cash:,.0f} ({cash/total_with_cash*100:.0f}%) |" if total_with_cash else "| Cash | 0 |",
            f"| Total Portfolio (incl. cash) | {total_with_cash:,.0f} |",
            f"| Today's Move | {today_str} |",
            f"| Positions | {n_pos} |",
            "",
        ]

        # ══════════════════════════════════════════════════════════════════════
        # 5. HOLDINGS TABLE
        # ══════════════════════════════════════════════════════════════════════
        lines += [
            "## 📋 Holdings",
            "",
            "| # | Stock | Mkt | Weight | Qty | Price | Value | P&L | RSI | P/E |",
            "|---|-------|-----|--------|-----|-------|-------|-----|-----|-----|",
        ]
        for i, row in df.iterrows():
            _w   = round((row.get("value") or 0) / max(total_value, 1) * 100, 1)
            val  = f"{row['value']:,.0f}"      if pd.notna(row.get('value'))   else "—"
            pnl_s  = f"{row['pnl']:+,.0f}"    if pd.notna(row.get('pnl'))     else "—"
            pnl_pct = f"({row['pnl_pct']:+.1f}%)" if pd.notna(row.get('pnl_pct')) else ""
            rsi  = f"{row['RSI']:.1f}"         if pd.notna(row.get('RSI'))     else "—"
            pe   = f"{row['PE']:.1f}x"         if pd.notna(row.get('PE'))      else "—"
            price = f"{row['price']:,.3f}"     if pd.notna(row.get('price'))   else "—"
            lines.append(
                f"| {i+1} | **{row.get('name','—')}** | {row.get('market','—').upper()} | "
                f"{_w:.1f}% | {int(row['qty'])} | {price} | {val} | {pnl_s} {pnl_pct} | {rsi} | {pe} |"
            )
        lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # 6. ALLOCATION
        # ══════════════════════════════════════════════════════════════════════
        if sectors:
            lines += ["## 🏭 Sector Allocation", ""]
            for sec, w in sorted(sectors.items(), key=lambda x: -x[1]):
                bar   = "█" * int(w / 5) + "░" * (20 - int(w / 5))
                flag  = "⚠️ " if w > 50 else ""
                lines.append(f"- {flag}**{sec}** `{bar}` {w:.0f}%")
            lines.append("")

        if markets:
            lines += ["## 🌍 Geographic Allocation", ""]
            for mkt, w in sorted(markets.items(), key=lambda x: -x[1]):
                flag = "⚠️ " if w > 60 else ""
                lines.append(f"- {flag}**{mkt.upper()}** — {w:.0f}%")
            if total_with_cash and cash > 0:
                _cash_alloc = round(cash / total_with_cash * 100, 1)
                lines.append(f"- 💵 **CASH** — {_cash_alloc:.0f}%")
            lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # 7. RISK BREAKDOWN
        # ══════════════════════════════════════════════════════════════════════
        lines += ["## 🚨 Risk Breakdown", ""]
        # Concentration risk
        _max_pos = max(((row.get("value") or 0) / max(total_value, 1) * 100 for _, row in df.iterrows()), default=0)
        _conc_risk = "🔴 High" if _max_pos > 40 else "🟡 Medium" if _max_pos > 25 else "🟢 Low"
        # Sector risk
        _max_sec = max(sectors.values(), default=0)
        _sec_risk = "🔴 High" if _max_sec > 60 else "🟡 Medium" if _max_sec > 40 else "🟢 Low"
        # Market risk
        _max_mkt = max(markets.values(), default=0)
        _mkt_risk = "🔴 High" if _max_mkt > 70 else "🟡 Medium" if _max_mkt > 50 else "🟢 Low"
        # RSI / Momentum risk
        _rsi_risk = "🔴 Overbought" if (avg_rsi or 0) > 65 else "🟢 Neutral" if (avg_rsi or 0) < 40 else "🟡 Moderate"

        lines += [
            f"| Risk Factor | Rating | Detail |",
            f"|-------------|--------|--------|",
            f"| Overall | {risk_emoji} {risk}/100 | {diversif} |",
            f"| Position Concentration | {_conc_risk} | Largest position = {_max_pos:.0f}% |",
            f"| Sector Concentration | {_sec_risk} | Largest sector = {_max_sec:.0f}% |",
            f"| Market Concentration | {_mkt_risk} | Largest market = {_max_mkt:.0f}% |",
            f"| Momentum (RSI) | {_rsi_risk} | Avg RSI = {avg_rsi:.0f} |" if avg_rsi else f"| Momentum (RSI) | — | — |",
            "",
        ]

        # ══════════════════════════════════════════════════════════════════════
        # 7b. EFFECTIVE DIVERSIFICATION & ECONOMIC EXPOSURE
        # ══════════════════════════════════════════════════════════════════════
        lines += [
            "## 🧭 Effective Diversification & Economic Exposure",
            "",
            "| Metric | Value | Interpretation |",
            "|--------|-------|----------------|",
            f"| Effective Diversification (N) | {eff_n:.1f} | {eff_n_ic} {eff_n_lbl} |",
            "",
        ]
        if econ_bkts:
            lines += [
                "**Economic Exposure Breakdown** (correlation-aware — sectors that move together are grouped):",
                "",
                "| Bucket | Exposure |",
                "|--------|----------|",
            ]
            for name, pct in sorted(econ_bkts.items(), key=lambda kv: -kv[1]):
                flag = "⚠️ " if pct > 50 else ""
                lines.append(f"| {flag}{name} | {pct:.1f}% |")
            lines.append("")
        if bkt_warn:
            lines += [f"> ⚠️ {bkt_warn}", ""]
        if soft_sugg:
            lines += [f"> 💡 {soft_sugg}", ""]

        # ══════════════════════════════════════════════════════════════════════
        # 8. PORTFOLIO METRICS
        # ══════════════════════════════════════════════════════════════════════
        rsi_lbl = _rsi_label(avg_rsi)
        lines += [
            "## 📈 Portfolio Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Weighted Avg RSI | {rsi_lbl} |",
            f"| Weighted Avg P/E | {f'{avg_pe:.1f}x' if avg_pe else '—'} |",
            f"| Weighted Avg Div Yield | {f'{avg_div:.2f}%' if avg_div else '—'} |",
            "",
        ]

        # ══════════════════════════════════════════════════════════════════════
        # 9. STRESS TEST
        # ══════════════════════════════════════════════════════════════════════
        stress_df = self.stress_test()
        if not stress_df.empty:
            # Worst-case extraction (min over all scenario Portfolio Impact %)
            from core.services.portfolio_analytics import (
                compute_worst_case_drawdown,
                readiness_with_drawdown,
            )
            _scenario_pcts: list[float] = []
            for _, row in stress_df.iterrows():
                raw = str(row.get("Portfolio Impact", "")).replace("%", "").strip()
                try:
                    _scenario_pcts.append(float(raw))
                except ValueError:
                    continue
            worst_pct = compute_worst_case_drawdown(_scenario_pcts)
            target_pct = float(drawdown_target)
            lines += [
                "## 🧪 Stress Test Scenarios",
                "",
                "| Scenario | Impact | Value Change | New Value |",
                "|----------|--------|--------------|-----------|",
            ]
            for _, row in stress_df.iterrows():
                lines.append(
                    f"| {row['Scenario']} | {row['Signal']} {row['Portfolio Impact']} "
                    f"| {row['Value Change']} | {row['New Portfolio Value']} |"
                )
            lines.append("")
            if worst_pct is not None:
                _verdict = readiness_with_drawdown(
                    base_status="✅ APPROVED",
                    worst_case=worst_pct / 100.0,
                    target_drawdown=target_pct / 100.0,
                )
                lines += [
                    "| Drawdown Metric | Value |",
                    "|-----------------|-------|",
                    f"| Target Drawdown | {target_pct:.0f}% (mandate floor) |",
                    f"| Estimated Worst-Case Drawdown | {worst_pct:+.1f}% |",
                    f"| Readiness | {_verdict.status} |",
                    "",
                ]
                if _verdict.note:
                    lines += [f"> ⚠️ {_verdict.note}", ""]

        # ══════════════════════════════════════════════════════════════════════
        # 10. SUGGESTED ADDITIONS
        # ══════════════════════════════════════════════════════════════════════
        comps = self.suggest_complements(n=5)
        if not comps.empty:
            lines += [
                "## 💡 Suggested Additions (Diversification)",
                "",
                "| # | Stock | Market | Sector | Price | RSI | P/E | Div% |",
                "|---|-------|--------|--------|-------|-----|-----|------|",
            ]
            for _, row in comps.iterrows():
                try:
                    rsi_v = f"{float(row.get('RSI', 0)):.1f}"
                    pe_v  = f"{float(row.get('price_earnings_ttm', 0)):.1f}x"
                    div_v = f"{float(row.get('dividend_yield_recent', 0)):.2f}%"
                except Exception:
                    rsi_v, pe_v, div_v = "—", "—", "—"
                lines.append(
                    f"| {row.get('#','—')} | **{row.get('name','—')}** | "
                    f"{str(row.get('market','—')).upper()} | {row.get('sector','—')} | "
                    f"{row.get('close','—')} | {rsi_v} | {pe_v} | {div_v} |"
                )
            lines.append("")

        # ══════════════════════════════════════════════════════════════════════
        # 11. REBALANCING (optional)
        # ══════════════════════════════════════════════════════════════════════
        if target_weights:
            rebal_df = self.rebalance_to(target_weights)
            if not rebal_df.empty:
                lines += [
                    "## ⚖️ Sector Rebalancing Plan",
                    "",
                    "| Sector | Current | Target | Gap | Action |",
                    "|--------|---------|--------|-----|--------|",
                ]
                for _, row in rebal_df.iterrows():
                    lines.append(
                        f"| {row['Sector']} | {row['Current Weight']} | "
                        f"{row['Target Weight']} | {row['Gap']} | {row['Action']} |"
                    )
                lines.append("")

        lines += ["---", "> 🕐 *EisaX 15-min pipeline cache — not real-time. For informational purposes only.*"]
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rsi_label(rsi) -> str:
    try:
        r = float(rsi)
        if r >= 70: return f"{r:.1f} 🔴 Overbought"
        if r <= 30: return f"{r:.1f} 🟢 Oversold"
        return f"{r:.1f} 🟡 Neutral"
    except Exception:
        return "—"


def _compute_risk(sector_weights: dict, market_weights: dict,
                  avg_rsi: Optional[float]) -> float:
    risk = 50.0

    if sector_weights:
        top = max(sector_weights.values())
        if top > 60: risk += 20
        elif top > 40: risk += 10

    if market_weights:
        top = max(market_weights.values())
        if top > 80: risk += 15
        elif top > 60: risk += 7

    if avg_rsi is not None:
        if avg_rsi > 70: risk += 15
        elif avg_rsi < 30: risk -= 10

    return round(max(0, min(100, risk)), 1)
