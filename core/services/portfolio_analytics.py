"""
portfolio_analytics.py — Institutional portfolio diagnostics.

Pure, side-effect-free helpers that turn a portfolio (weights + sectors +
buckets) into the metrics needed for correlation-aware risk reporting:

  • Effective N (inverse Herfindahl) — true diversification count
  • Economic-bucket exposure (Growth Beta / Commodity Cycle / Defensive /
    Regional Beta) — surfaces hidden correlation overlap
  • Worst-case drawdown extraction from stress scenarios
  • Sharpe-context note for low-conviction setups
  • Conditional-approval gate when worst-case > target drawdown

No network I/O, no DataFrame mutation, no state.  Every function is cheap
enough to call inside a report render.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Economic-bucket taxonomy
# ─────────────────────────────────────────────────────────────────────────────
#
# A position's "economic bucket" captures what it responds to in a macro
# regime — not its sector label.  Two stocks in different sectors can still
# share a bucket (e.g. NVDA and Cloudflare both live in Growth Beta).

GROWTH_BETA_TICKERS: frozenset[str] = frozenset({
    # Mega-cap US tech
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "AMD", "CRM", "ADBE", "ORCL", "NFLX", "NET", "SNOW", "CRWD",
    "PLTR", "SHOP", "UBER", "MU",
    # Crypto proxies
    "BTC-USD", "ETH-USD", "SOL-USD", "COIN", "MSTR", "MARA", "RIOT",
})

GROWTH_BETA_SECTORS: frozenset[str] = frozenset({
    "Technology", "Technology Services", "Electronic Technology",
    "Communications", "Internet Software/Services",
    "Software", "Semiconductors", "Crypto",
})

COMMODITY_CYCLE_TICKERS: frozenset[str] = frozenset({
    "GSG", "PDBC", "DBC", "USO", "UNG", "BNO",
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "EOG",
    "2222.SR",  # Aramco — explicit oil beta
})

COMMODITY_CYCLE_SECTORS: frozenset[str] = frozenset({
    "Energy", "Energy Minerals", "Oil & Gas", "Non-Energy Minerals",
    "Industrial Metals", "Commodities",
})

DEFENSIVE_TICKERS: frozenset[str] = frozenset({
    "GLD", "IAU", "SGOL", "GOLD",
    "TLT", "IEF", "SHY", "BND", "AGG", "LQD",
    "VNQ", "VPU",
    "KO", "PEP", "PG", "JNJ", "WMT", "COST", "MCD",
})

DEFENSIVE_SECTORS: frozenset[str] = frozenset({
    "Bonds", "Fixed Income", "Utilities",
    "Consumer Non-Durables", "Consumer Staples",
    "Health Services", "Health Technology", "Gold",
})

REGIONAL_BETA_SUFFIXES: tuple[str, ...] = (
    ".SR", ".AE", ".DU", ".QA", ".KW", ".BH", ".OM", ".EG", ".CA",
)

REGIONAL_BETA_MARKETS: frozenset[str] = frozenset({
    "ksa", "uae", "egypt", "kuwait", "qatar", "bahrain", "oman",
    "KSA", "UAE", "EGYPT", "KUWAIT", "QATAR", "BAHRAIN", "OMAN",
    "TADAWUL", "ADX", "DFM", "QSE", "EGX",
})

BUCKET_ORDER: tuple[str, ...] = (
    "Growth Beta",
    "Commodity Cycle",
    "Defensive",
    "Regional Beta",
    "Other",
)


# ─────────────────────────────────────────────────────────────────────────────
# Core diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_effective_n(weights: Iterable[float]) -> float:
    """
    Effective number of independent positions (inverse Herfindahl).

        N_eff = 1 / Σ wᵢ²    where Σ wᵢ = 1

    Returns 0.0 for empty / all-zero inputs.  Caller passes weights as
    fractions (0.25) or percents (25) — this helper self-normalises.
    """
    clean = [float(w) for w in weights if w is not None and float(w) > 0]
    if not clean:
        return 0.0
    total = sum(clean)
    if total <= 0:
        return 0.0
    norm = [w / total for w in clean]
    hhi = sum(w * w for w in norm)
    if hhi <= 0:
        return 0.0
    return 1.0 / hhi


def diversification_label(effective_n: float) -> str:
    """Map Effective N → plain-English diversification verdict."""
    try:
        n = float(effective_n)
    except (TypeError, ValueError):
        return "Unknown"
    if n < 5:
        return "Highly concentrated"
    if n <= 10:
        return "Moderately diversified"
    return "Well diversified"


def diversification_emoji(effective_n: float) -> str:
    try:
        n = float(effective_n)
    except (TypeError, ValueError):
        return "⚪"
    if n < 5:
        return "🔴"
    if n <= 10:
        return "🟡"
    return "🟢"


# ─────────────────────────────────────────────────────────────────────────────
# Economic-bucket classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_bucket(
    ticker: str | None,
    sector: str | None = None,
    market: str | None = None,
    bucket_hint: str | None = None,
) -> str:
    """
    Classify a single position into one of BUCKET_ORDER.

    Priority (most-specific first):
      1. Explicit ticker match (crypto, gold, oil majors, 2222.SR…)
      2. Regional suffix / market — catches GCC names even before sector
      3. Sector lookup
      4. `bucket_hint` from upstream allocator (e.g. "us_equity" → Growth Beta)
      5. Fallback: "Other"
    """
    t = (ticker or "").strip().upper()
    sec = (sector or "").strip()
    mkt = (market or "").strip()
    hint = (bucket_hint or "").strip().lower()

    if t:
        if t in GROWTH_BETA_TICKERS:
            return "Growth Beta"
        if t in COMMODITY_CYCLE_TICKERS:
            return "Commodity Cycle"
        if t in DEFENSIVE_TICKERS:
            return "Defensive"
        for sfx in REGIONAL_BETA_SUFFIXES:
            if t.endswith(sfx):
                return "Regional Beta"

    if mkt and mkt in REGIONAL_BETA_MARKETS:
        return "Regional Beta"

    if sec:
        if sec in GROWTH_BETA_SECTORS:
            return "Growth Beta"
        if sec in COMMODITY_CYCLE_SECTORS:
            return "Commodity Cycle"
        if sec in DEFENSIVE_SECTORS:
            return "Defensive"

    if hint:
        if hint in {"us_equity", "crypto"}:
            return "Growth Beta"
        if hint in {"commodities"}:
            return "Commodity Cycle"
        if hint in {"bonds", "gold", "cash"}:
            return "Defensive"
        if hint in {"gcc_equity", "egypt_equity"}:
            return "Regional Beta"

    return "Other"


def compute_economic_buckets(
    positions: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """
    Aggregate position weights into economic buckets.

    Each position is a mapping with at least one of:
      weight | ticker | sector | market | bucket

    Weights are normalised to sum to 100 (percent).  Buckets that end up
    with 0% are dropped.
    """
    raw: dict[str, float] = {k: 0.0 for k in BUCKET_ORDER}
    total = 0.0
    for p in positions:
        w_raw = p.get("weight")
        try:
            w = float(w_raw) if w_raw is not None else 0.0
        except (TypeError, ValueError):
            w = 0.0
        if w <= 0:
            continue
        bucket = classify_bucket(
            ticker=str(p.get("ticker") or "") or None,
            sector=str(p.get("sector") or "") or None,
            market=str(p.get("market") or "") or None,
            bucket_hint=str(p.get("bucket") or "") or None,
        )
        raw[bucket] += w
        total += w

    if total <= 0:
        return {}

    # If weights came in as fractions (sum ≈ 1), scale to percent.
    scale = 100.0 / total
    out = {k: round(v * scale, 1) for k, v in raw.items() if v > 0}
    return out


def bucket_concentration_warning(
    buckets: Mapping[str, float],
    threshold: float = 50.0,
) -> str | None:
    """
    Return a single-sentence warning if any bucket exceeds `threshold` (%).
    Returns None if portfolio is balanced.
    """
    if not buckets:
        return None
    offender = max(buckets.items(), key=lambda kv: kv[1])
    name, pct = offender
    if pct > threshold:
        return (
            f"{pct:.0f}% of the portfolio is concentrated in **{name}** — "
            f"these holdings tend to move together in a macro regime and "
            f"reduce the effective diversification benefit."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_worst_case_drawdown(
    scenario_returns: Iterable[float],
) -> float | None:
    """
    Return the minimum (most-negative) scenario return from a stress run.

    Input: an iterable of percentage returns (e.g. [-0.30, -0.25, -0.40]).
    Output: the most-negative value, or None if no valid returns.
    """
    vals: list[float] = []
    for r in scenario_returns:
        if r is None:
            continue
        try:
            vals.append(float(r))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return min(vals)


def estimate_worst_case_from_vol(
    volatility: float,
    z: float = 2.0,
) -> float:
    """
    Fallback worst-case estimator when no stress table is available.

    Uses a 2σ lower bound on annual return: `worst ≈ -z * vol`.
    This is rough but directionally honest — it's always more conservative
    than the Sharpe/vol-only framing.

    `volatility` is annualised (0.20 = 20 %).  Result is a negative decimal.
    """
    try:
        v = float(volatility)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return 0.0
    return -abs(z) * v


@dataclass
class ReadinessVerdict:
    status: str              # "✅ APPROVED" / "⚠️ CONDITIONAL APPROVAL" / ...
    breaches_drawdown: bool
    note: str | None = None


def readiness_with_drawdown(
    base_status: str,
    worst_case: float | None,
    target_drawdown: float,
) -> ReadinessVerdict:
    """
    Apply the drawdown gate on top of a pre-computed readiness status.

    Rules:
      • target_drawdown is positive (e.g. 0.25 for a −25 % mandate floor).
      • worst_case is negative (e.g. −0.40 for a −40 % estimate).
      • If |worst_case| > target_drawdown → force "⚠️ CONDITIONAL APPROVAL".
      • Otherwise keep base_status.
    """
    if worst_case is None:
        return ReadinessVerdict(status=base_status, breaches_drawdown=False)
    try:
        wc = float(worst_case)
        tgt = abs(float(target_drawdown))
    except (TypeError, ValueError):
        return ReadinessVerdict(status=base_status, breaches_drawdown=False)

    breaches = abs(wc) > tgt
    if breaches:
        note = (
            f"Estimated worst-case drawdown ({wc*100:.0f}%) exceeds the "
            f"client's target drawdown ({tgt*100:.0f}%). Strategy flagged "
            f"for conditional approval — trim highest-beta sleeves or "
            f"reduce concentration before final sign-off."
        )
        return ReadinessVerdict(
            status="⚠️ CONDITIONAL APPROVAL",
            breaches_drawdown=True,
            note=note,
        )
    return ReadinessVerdict(status=base_status, breaches_drawdown=False)


# ─────────────────────────────────────────────────────────────────────────────
# Narrative helpers
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_context_note(
    sharpe: float | None,
    volatility: float | None,
) -> str | None:
    """
    Emit a context paragraph when a portfolio shows a weak risk-adjusted
    return (low Sharpe) combined with high volatility — i.e. the portfolio
    is taking real risk without being paid for it.
    """
    try:
        s = float(sharpe) if sharpe is not None else None
        v = float(volatility) if volatility is not None else None
    except (TypeError, ValueError):
        return None
    if s is None or v is None:
        return None
    if s < 0.5 and v > 0.20:
        return (
            f"Sharpe ratio ({s:.2f}) is below the 0.5 quality threshold while "
            f"annualised volatility ({v*100:.0f}%) is elevated — the portfolio "
            f"is taking meaningful risk without adequate compensation. "
            f"Consider trimming higher-volatility sleeves or adding a "
            f"defensive anchor to improve risk-adjusted return."
        )
    return None


def diversification_soft_suggestion(
    effective_n: float,
    buckets: Mapping[str, float],
    bucket_threshold: float = 50.0,
) -> str | None:
    """
    Soft, advisory text for when diversification is weak — never auto-acts.

    Triggers when either Effective N < 5 OR any bucket > threshold.
    """
    try:
        n = float(effective_n)
    except (TypeError, ValueError):
        n = 0.0

    bucket_hot = any(v > bucket_threshold for v in (buckets or {}).values())
    if n >= 5 and not bucket_hot:
        return None

    reasons: list[str] = []
    if n < 5:
        reasons.append(f"Effective N = {n:.1f} (below 5)")
    if bucket_hot and buckets:
        top = max(buckets.items(), key=lambda kv: kv[1])
        reasons.append(f"{top[1]:.0f}% in {top[0]}")

    why = "; ".join(reasons) if reasons else "concentration detected"
    return (
        "Portfolio diversification can be improved by introducing "
        "lower-correlation assets such as defensive bonds, gold, or "
        f"regional equities with distinct macro drivers ({why}). "
        "This is a suggestion — no automatic rebalancing is applied."
    )


def consistent_diversification_phrase(
    effective_n: float,
    original_phrase: str,
) -> str:
    """
    Guard against the report text saying 'well diversified' when the
    underlying Effective N says otherwise.  Used by the post-render
    cleanup / interpretation guard layer.
    """
    try:
        n = float(effective_n)
    except (TypeError, ValueError):
        return original_phrase
    if n < 5 and "well diversified" in (original_phrase or "").lower():
        return "Concentrated high-conviction structure"
    return original_phrase
