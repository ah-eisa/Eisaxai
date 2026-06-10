"""
core/services/institutional_stock_wrapper.py
────────────────────────────────────────────
Phase G — Institutional Single-Asset Research Parity wrapper.

Non-invasive layer that augments LLM-generated stock/crypto reports with:
  • Section A — Institutional Metadata Layer (Confidence + Implementation + Adaptive Disclaimers)
  • Section G — Audit Appendix (Snapshot ID, Model Constraints, Reproducibility)
  • Bilingual (EN / AR) headers
  • Institutional terminology only — no retail wording
  • Adaptive content based on asset type, region, and data coverage

Public entry point:
    wrap_stock_report(report_text, *, symbol, market, asset_type, language, ...) -> str

Designed to be called from the stock analysis pipeline at the point where
the raw LLM output is assembled and before the response is shipped.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# Asset-type classification helpers
# ──────────────────────────────────────────────────────────────────────────
def _classify_reliability_tier(
    *, asset_type: str, market: str, symbol: str, fundamentals_available: bool
) -> tuple[str, str, str, int]:
    """
    Returns (reliability_tier, evidence_breadth, coverage_quality, confidence_pct).
    """
    asset_type = (asset_type or "").lower()
    market     = (market or "").upper()
    symbol     = (symbol or "").upper()

    # Crypto → Indicative
    if asset_type in ("crypto", "cryptocurrency") or symbol.endswith("-USD") or symbol.endswith("USDT"):
        return "Indicative", "Limited", "Sparse", 55
    # Commodities futures
    if asset_type in ("commodity", "commodities") or symbol.endswith("=F"):
        return "Institutional-Lite", "Moderate", "Partial", 70
    # Frontier markets — Egypt
    if market in ("EGY", "EGYPT"):
        return "Institutional-Lite", "Moderate", "Partial", 65
    # GCC — most names liquid (US-listed ADRs vary)
    if market in ("SAU", "KSA", "UAE", "QAT", "BHR", "GCC"):
        if fundamentals_available:
            return "Institutional-Lite", "Moderate", "Partial", 75
        return "Indicative", "Limited", "Sparse", 60
    # US — default Institutional
    if market in ("US", "USA"):
        if fundamentals_available:
            return "Institutional", "Broad", "Full", 85
        return "Institutional-Lite", "Moderate", "Partial", 72
    # Default
    return "Institutional-Lite", "Moderate", "Partial", 70


def _classify_volatility_regime(realized_vol_pct: Optional[float]) -> str:
    if realized_vol_pct is None:
        return "Unknown"
    if realized_vol_pct < 20:
        return "Low (sub-20% annualized)"
    if realized_vol_pct < 35:
        return "Moderate (20–35% annualized)"
    if realized_vol_pct < 60:
        return "Elevated (35–60% annualized)"
    return "Extreme (60%+ annualized — satellite sizing only)"


def _liquidity_profile(*, market: str, asset_type: str, avg_daily_volume: Optional[float]) -> str:
    asset_type = (asset_type or "").lower()
    market     = (market or "").upper()
    if asset_type in ("crypto",) or (market in ("US",) and (avg_daily_volume or 0) > 5_000_000):
        return "Deep · institutional-grade execution"
    if market in ("EGY", "EGYPT"):
        return "Thin · execution complexity elevated"
    if market in ("SAU", "KSA", "UAE", "QAT", "BHR") and (avg_daily_volume or 0) < 500_000:
        return "Moderate · regional broker required, wider spreads possible"
    if (avg_daily_volume or 0) > 1_000_000:
        return "Deep · standard institutional execution"
    return "Moderate · monitor execution slippage"


def _portfolio_role_suitability(
    *, asset_type: str, market: str, realized_vol_pct: Optional[float], beta: Optional[float]
) -> str:
    asset_type = (asset_type or "").lower()
    if asset_type in ("crypto",):
        return "Opportunistic Satellite (≤5% — asymmetric high-volatility exposure, not a hedge)"
    if asset_type in ("commodity", "commodities") and (market or "").upper() == "GLOBAL":
        return "Real-Asset Sleeve · Inflation Hedge (≤10–15%)"
    if (realized_vol_pct or 0) > 45:
        return "Tactical / Satellite (elevated volatility; concentration suitability LIMITED)"
    if beta is not None and beta > 1.30:
        return "Tactical Allocation (above-benchmark beta — size proportionally to risk budget)"
    if beta is not None and beta < 0.70:
        return "Strategic Core / Defensive Sleeve (sub-benchmark beta — diversifying anchor)"
    return "Strategic Core or Tactical Allocation (size based on conviction and risk budget)"


def _position_sizing_guidance(*, reliability_tier: str, realized_vol_pct: Optional[float]) -> str:
    rv = realized_vol_pct or 0
    if reliability_tier == "Indicative" or rv > 45:
        return "≤5% of portfolio (satellite sizing only)"
    if reliability_tier == "Institutional-Lite" or rv > 30:
        return "≤10% of portfolio (tactical sizing)"
    return "Up to 15–20% of portfolio (core allocation eligible, subject to mandate)"


# ──────────────────────────────────────────────────────────────────────────
# Adaptive disclaimers — conditional on asset characteristics
# ──────────────────────────────────────────────────────────────────────────
def _build_adaptive_disclaimers(
    *,
    asset_type: str,
    market: str,
    symbol: str,
    realized_vol_pct: Optional[float],
    beta: Optional[float],
    fundamentals_available: bool,
) -> list[dict]:
    out: list[dict] = []
    asset_type = (asset_type or "").lower()
    market     = (market or "").upper()

    if asset_type in ("crypto",) or symbol.upper().endswith("-USD"):
        out.append({
            "severity": "HIGH",
            "topic":    "Crypto Analytical Framework",
            "note":     ("Crypto positions evaluated on network activity, ETF flows, liquidity regime, "
                         "and cycle positioning — not equity-style valuation multiples or earnings durability. "
                         "24/7 trading and regulatory regime shifts apply."),
        })
    if asset_type in ("commodity", "commodities"):
        out.append({
            "severity": "MODERATE",
            "topic":    "Commodity Pricing Framework",
            "note":     ("Commodity contracts evaluated on supply/demand balance, USD direction, real-yield "
                         "sensitivity, and central-bank policy — not corporate fundamentals. Contango/backwardation "
                         "impacts roll yield in ETF wrappers."),
        })
    if market in ("EGY", "EGYPT"):
        out.append({
            "severity": "HIGH",
            "topic":    "Frontier-Market Exposure",
            "note":     ("Egyptian holdings subject to currency-floating risk, political volatility, and reduced "
                         "liquidity. Trading-volume and bid-ask spread profile differ materially from developed markets."),
        })
    if market in ("SAU", "KSA", "UAE", "QAT", "BHR", "GCC"):
        out.append({
            "severity": "MODERATE",
            "topic":    "GCC Regional Exposure",
            "note":     ("GCC equities co-move with oil-price cycle and USD direction (peg-driven). "
                         "Regional dividend distributions and reporting cadence differ from US benchmark standards."),
        })
    if (realized_vol_pct or 0) > 45:
        out.append({
            "severity": "HIGH",
            "topic":    "Elevated Volatility Regime",
            "note":     (f"Realized volatility (~{realized_vol_pct:.0f}%) materially above market average. "
                         "Position sizing should reflect tail-risk asymmetry and avoid core-allocation status."),
        })
    if beta is not None and beta > 1.50:
        out.append({
            "severity": "HIGH",
            "topic":    "Elevated Market Sensitivity",
            "note":     (f"Beta ({beta:.2f}) significantly above 1.0 — amplifies market drawdowns. "
                         f"Loss expectation in a 20% market correction: approximately {beta * 20:.0f}%."),
        })
    if not fundamentals_available:
        out.append({
            "severity": "MODERATE",
            "topic":    "Fundamental Data Coverage",
            "note":     ("Earnings, valuation, and balance-sheet metrics are limited or unavailable. "
                         "Analysis relies primarily on price behavior, factor exposure, and macro positioning. "
                         "Verify coverage in proprietary data systems before institutional deployment."),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────
# Section A — Institutional Metadata Layer (prepended)
# ──────────────────────────────────────────────────────────────────────────
def _build_section_a(
    *,
    symbol: str,
    market: str,
    asset_type: str,
    reliability_tier: str,
    evidence_breadth: str,
    coverage_quality: str,
    confidence_pct: int,
    volatility_regime: str,
    liquidity_profile: str,
    role_suitability: str,
    position_sizing: str,
    benchmark_label: str,
    snapshot_id: str,
    generated_at: str,
    language: str,
) -> str:
    tier_tag = "[STRONG]" if reliability_tier == "Institutional" else ("[MODERATE]" if reliability_tier == "Institutional-Lite" else "[LOW]")
    if language == "ar":
        return (
            f"## A. طبقة بيانات التقرير المؤسسية\n\n"
            f"**Snapshot ID:** `{snapshot_id}`  |  **Generated:** {generated_at}  |  **Benchmark:** {benchmark_label}\n\n"
            f"**Confidence Calibration** · Score: **{confidence_pct}%** · Evidence Breadth: **{evidence_breadth}** · "
            f"Coverage Quality: **{coverage_quality}** · Reliability Tier: **{reliability_tier}** {tier_tag}\n\n"
            f"**Volatility Regime:** {volatility_regime}  |  **Liquidity Profile:** {liquidity_profile}\n\n"
            f"**Portfolio Role Suitability:** {role_suitability}\n\n"
            f"**Position Sizing Guidance:** {position_sizing}\n\n"
            f"> *الأقسام التحليلية أدناه قائمة على البيانات التاريخية والنمذجة الكميّة، مع طبقة تعليق بالذكاء الاصطناعي. التوصيات تكميلية ولا تُعدّ بديلاً عن المشورة الاستثمارية الموثّقة.*\n\n"
            f"---\n\n"
        )
    return (
        f"## A. Institutional Report Metadata\n\n"
        f"**Snapshot ID:** `{snapshot_id}`  |  **Generated:** {generated_at}  |  **Benchmark:** {benchmark_label}\n\n"
        f"**Confidence Calibration** · Score: **{confidence_pct}%** · Evidence Breadth: **{evidence_breadth}** · "
        f"Coverage Quality: **{coverage_quality}** · Reliability Tier: **{reliability_tier}** {tier_tag}\n\n"
        f"**Volatility Regime:** {volatility_regime}  |  **Liquidity Profile:** {liquidity_profile}\n\n"
        f"**Portfolio Role Suitability:** {role_suitability}\n\n"
        f"**Position Sizing Guidance:** {position_sizing}\n\n"
        f"> *Analytical sections below combine deterministic historical/factor metrics with an AI commentary layer. "
        f"Recommendations are supplementary and do not replace formal investment advice.*\n\n"
        f"---\n\n"
    )


# ──────────────────────────────────────────────────────────────────────────
# Adaptive Disclaimers block (injected between A and the LLM body if any)
# ──────────────────────────────────────────────────────────────────────────
def _build_disclaimers_block(disclaimers: list[dict], language: str) -> str:
    if not disclaimers:
        return ""
    if language == "ar":
        out = ["### الإفصاحات التكيّفية", "", "*تُعرض فقط المخاطر التي تنطبق ماديًّا على هذا الأصل.*", "",
               "| الحدة | الموضوع | الملاحظة |", "|------|----------|----------|"]
    else:
        out = ["### Adaptive Disclosures", "", "*Only risks that materially apply to this asset are surfaced.*", "",
               "| Severity | Topic | Note |", "|----------|-------|------|"]
    for d in disclaimers:
        out.append(f"| [{d['severity']}] | {d['topic']} | {d['note']} |")
    out.append("")
    return "\n".join(out) + "\n"


# ──────────────────────────────────────────────────────────────────────────
# Section G — Audit Appendix (appended)
# ──────────────────────────────────────────────────────────────────────────
def _build_section_g(
    *,
    symbol: str,
    market: str,
    asset_type: str,
    snapshot_id: str,
    generated_at: str,
    benchmark_label: str,
    confidence_pct: int,
    reliability_tier: str,
    methodology_version: str,
    language: str,
) -> str:
    model_constraints = [
        ("استخدام المحاكاة التاريخية بنافذة 252 يومًا؛ التحولات الهيكلية خارج النافذة غير محتسبة."
         if language == "ar" else
         "Historical simulation uses 252-day trailing window; structural breaks beyond that window are not captured."),
        ("مصفوفة الارتباط لحظية؛ الارتباطات الزوجية ترتفع نحو 1.0 خلال أحداث السيولة."
         if language == "ar" else
         "Correlation matrix is point-in-time; pairwise correlations rise toward 1.0 during liquidity events."),
        ("التقلب غير ثابت؛ التقلب المحقق قد يختلف ماديًّا عن تقديرات النموذج خلال تحولات النظام."
         if language == "ar" else
         "Volatility is non-stationary; realized vol can diverge materially from in-sample estimates during regime shifts."),
        ("تقديرات Beta تفترض حساسية سوقية خطية؛ السلوك المحدّب (gamma) غير محتسب."
         if language == "ar" else
         "Beta estimates assume linear market sensitivity; convex behavior (gamma) ignored."),
        ("تقديرات الأرباح والتوقعات تخضع لمخاطر مراجعة وإعلان."
         if language == "ar" else
         "Earnings estimates and consensus forecasts subject to revision and announcement risk."),
        ("المؤشرات الفنية أدوات سياق نظام — وليست إشارات تداول مستقلة."
         if language == "ar" else
         "Technical indicators are regime-context tools — not standalone trading signals."),
    ]
    if (asset_type or "").lower() in ("crypto",):
        model_constraints.append(
            "بيانات سوق الكريبتو متاحة 24/7 ولكن العمق يختلف عبر البورصات؛ تقدير الانزلاق تقريبي."
            if language == "ar" else
            "Crypto market data is 24/7 but venue depth varies; slippage estimates are approximate."
        )

    if language == "ar":
        body = (
            "## G. ملحق المراجعة\n\n"
            "| الحقل | القيمة |\n"
            "|-------|--------|\n"
            f"| Snapshot ID | `{snapshot_id}` |\n"
            f"| الرمز | {symbol} |\n"
            f"| السوق | {market or '—'} |\n"
            f"| نوع الأصل | {asset_type or '—'} |\n"
            f"| المؤشر المرجعي | {benchmark_label} |\n"
            f"| تاريخ الإنشاء | {generated_at} |\n"
            f"| نسخة المنهجية | {methodology_version} |\n"
            f"| درجة الثقة | {confidence_pct}% |\n"
            f"| فئة الموثوقية | {reliability_tier} |\n\n"
            "> *قابل للتكرار: نفس المدخلات + نفس نافذة البيانات → نفس Snapshot ID → نفس النتيجة.*\n\n"
            "### قيود النموذج — الحدود الهيكلية\n\n"
        )
        for mc in model_constraints:
            body += f"- {mc}\n"
        body += "\n> *ملاحظة شفافية: القيود أعلاه متأصلة في منهجية التحليل الكمّي للأصول الفردية.*\n"
        return body

    body = (
        "## G. Audit Appendix\n\n"
        "| Field | Value |\n"
        "|-------|-------|\n"
        f"| Snapshot ID | `{snapshot_id}` |\n"
        f"| Symbol | {symbol} |\n"
        f"| Market | {market or '—'} |\n"
        f"| Asset Type | {asset_type or '—'} |\n"
        f"| Reference Benchmark | {benchmark_label} |\n"
        f"| Generated | {generated_at} |\n"
        f"| Methodology Version | {methodology_version} |\n"
        f"| Confidence Score | {confidence_pct}% |\n"
        f"| Reliability Tier | {reliability_tier} |\n\n"
        "> *Reproducible: identical inputs + identical data window → identical Snapshot ID → identical output.*\n\n"
        "### Model Constraints — Structural Limitations\n\n"
    )
    for mc in model_constraints:
        body += f"- {mc}\n"
    body += "\n> *Transparency note: the constraints above are inherent to single-asset quantitative analytics.*\n"
    return body


# ──────────────────────────────────────────────────────────────────────────
# Benchmark label inference
# ──────────────────────────────────────────────────────────────────────────
def _infer_benchmark_label(*, market: str, asset_type: str, symbol: str) -> str:
    a = (asset_type or "").lower()
    m = (market or "").upper()
    if a in ("crypto",) or symbol.upper().endswith("-USD"):
        return "BTC-USD (crypto reference)"
    if a in ("commodity", "commodities"):
        return "Bloomberg Commodity Index (BCOM)"
    if m in ("US", "USA"):
        return "S&P 500 Total Return"
    if m in ("SAU", "KSA"):
        return "Tadawul All Share Index (TASI)"
    if m in ("UAE",):
        return "ADX General Index"
    if m in ("EGY", "EGYPT"):
        return "EGX 30"
    if m in ("GCC",):
        return "S&P GCC Composite"
    return "MSCI World"


# ──────────────────────────────────────────────────────────────────────────
# Light retail-tone cleanup of existing headers in the LLM body
# Replaces emoji-decorated section markers with cleaner institutional names.
# ──────────────────────────────────────────────────────────────────────────
_HEADER_CLEANUPS = [
    ("## ⚡ Quick View",                "## Quick View"),
    ("## 📋 Full Report",               "## Full Report"),
    ("### ⚔️ Peer Comparison",          "### Peer Comparison"),
    ("### ⏰ Why Now?",                 "### Catalyst & Entry Conditions"),
    ("### 🌍 Advanced Scenario Analysis", "### Scenario Analysis"),
    ("## Decision Framework (Advisory Layer)", "## Decision Framework"),
    ("## 🎯 EisaX Proprietary Score Card", "## EisaX Proprietary Score Card"),
]


def _clean_retail_headers(text: str) -> str:
    out = text
    for old, new in _HEADER_CLEANUPS:
        out = out.replace(old, new)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
def wrap_stock_report(
    report_text: str,
    *,
    symbol: str,
    market: str = "",
    asset_type: str = "",
    language: str = "en",
    realized_vol_pct: Optional[float] = None,
    beta: Optional[float] = None,
    fundamentals_available: bool = True,
    avg_daily_volume: Optional[float] = None,
    generated_at: Optional[str] = None,
    methodology_version: str = "EisaX-Stock-2.1",
) -> str:
    """
    Wrap a raw LLM-generated stock report with institutional metadata
    (Section A prepended) and audit appendix (Section G appended).

    The original report body is preserved; only header polish is applied.
    """
    if not report_text or not report_text.strip():
        return report_text

    symbol = symbol or "—"
    market = market or ""
    asset_type = asset_type or "equity"
    language = (language or "en").lower()
    if language not in ("en", "ar"):
        language = "en"

    if not generated_at:
        generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    # Confidence calibration
    rt, eb, cq, cpct = _classify_reliability_tier(
        asset_type=asset_type, market=market, symbol=symbol,
        fundamentals_available=fundamentals_available,
    )

    # Implementation feasibility
    vol_regime = _classify_volatility_regime(realized_vol_pct)
    liq_profile = _liquidity_profile(market=market, asset_type=asset_type, avg_daily_volume=avg_daily_volume)
    role = _portfolio_role_suitability(asset_type=asset_type, market=market, realized_vol_pct=realized_vol_pct, beta=beta)
    sizing = _position_sizing_guidance(reliability_tier=rt, realized_vol_pct=realized_vol_pct)

    # Adaptive disclaimers
    disclaimers = _build_adaptive_disclaimers(
        asset_type=asset_type, market=market, symbol=symbol,
        realized_vol_pct=realized_vol_pct, beta=beta,
        fundamentals_available=fundamentals_available,
    )

    # Benchmark label
    benchmark_label = _infer_benchmark_label(market=market, asset_type=asset_type, symbol=symbol)

    # Snapshot ID — derived from symbol + market + body hash + date stamp
    snapshot_seed = f"{symbol}|{market}|{asset_type}|{generated_at[:10]}|{hashlib.sha256(report_text.encode()).hexdigest()[:8]}"
    snapshot_id = hashlib.sha256(snapshot_seed.encode()).hexdigest()[:12]

    # Build sections A + adaptive disclaimers + G
    section_a = _build_section_a(
        symbol=symbol, market=market, asset_type=asset_type,
        reliability_tier=rt, evidence_breadth=eb, coverage_quality=cq, confidence_pct=cpct,
        volatility_regime=vol_regime, liquidity_profile=liq_profile,
        role_suitability=role, position_sizing=sizing,
        benchmark_label=benchmark_label, snapshot_id=snapshot_id,
        generated_at=generated_at, language=language,
    )
    disclaimers_block = _build_disclaimers_block(disclaimers, language)
    section_g = _build_section_g(
        symbol=symbol, market=market, asset_type=asset_type,
        snapshot_id=snapshot_id, generated_at=generated_at,
        benchmark_label=benchmark_label, confidence_pct=cpct,
        reliability_tier=rt, methodology_version=methodology_version,
        language=language,
    )

    body = _clean_retail_headers(report_text.strip())

    # ── Phase H — single-asset relevant subset ────────────────────────────
    # H1 (benchmark relative) + H4 (factor decomposition) apply to a
    # single-asset wrapper; H2/H3/H5 are portfolio-level and skipped here.
    # Bench picked from existing _infer_benchmark_label / asset_type.
    phase_h_md = ""
    try:
        from phase_h import benchmarks as _ph_b
        from phase_h import factor_model as _ph_f
        from phase_h.report_helpers import finalize as _ph_finalize
        from phase_h.feature_flags import (
            PHASE_H_ENABLED,
            PHASE_H_BENCHMARK,
            PHASE_H_FACTOR_MODEL,
        )

        if PHASE_H_ENABLED:
            single = {symbol: 1.0}
            asset_kind = "crypto" if asset_type.lower() == "crypto" else (
                "commodities" if asset_type.lower() == "commodities" else "equity"
            )
            blocks = []
            if PHASE_H_BENCHMARK:
                bench_payload = _ph_b.compute_benchmark_relative(
                    weights=single,
                    asset_kind=asset_kind,
                    region_tilt=("KSA" if market.upper() in {"KSA","TADAWUL"} else
                                 "US" if market.upper() in {"US","NYSE","NASDAQ"} else None),
                    language=language,
                )
                bench_md = _ph_finalize(_ph_b.render_benchmark_relative_md(bench_payload, language))
                if bench_md.strip():
                    blocks.append(bench_md)
            if PHASE_H_FACTOR_MODEL and asset_kind == "equity":
                fac_payload = _ph_f.compute_factor_decomposition(weights=single, language=language)
                fac_md = _ph_finalize(_ph_f.render_factor_decomposition_md(fac_payload, language))
                if fac_md.strip():
                    blocks.append(fac_md)
            if blocks:
                phase_h_md = "\n\n" + "\n\n".join(blocks)
    except Exception:
        # Never break the wrapper on Phase H failure
        phase_h_md = ""

    # Find natural insertion point for disclaimers: after the existing header line
    # (first H1 line + blank) so the metadata still appears at the top.
    parts = body.split("\n", 1)
    if parts and parts[0].startswith("# "):
        # Place A immediately after H1, then disclaimers, then body, then H additions, then G
        h1 = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        assembled = f"{h1}\n\n{section_a}{disclaimers_block}{rest}{phase_h_md}\n\n---\n\n{section_g}"
    else:
        assembled = f"{section_a}{disclaimers_block}{body}{phase_h_md}\n\n---\n\n{section_g}"
    return assembled
