"""
Phase H1 — Native Benchmark Analytics Engine.

Computes benchmark-relative portfolio diagnostics from either an explicit
returns panel or the institutional data layer (which wraps the read-only
snapshot store). Sparse or missing market data degrades to an indicative
payload; it never raises through the public API.

Data-access rule: this engine MUST NOT read the snapshot store directly.
All cache reads go through `core.data_layer.market_cache_adapter`.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_flags import PHASE_H_BENCHMARK
from .report_helpers import (
    L,
    LABELS,
    fmt_num,
    fmt_pct,
    md_table,
    section_heading,
    severity_tag,
)
from .schemas import BenchmarkRelative

ENGINE_VERSION = "0.2.0"


# Benchmark catalog. Each entry: ticker -> {label, region_focus, asset_class}
BENCHMARK_CATALOG: Dict[str, Dict[str, str]] = {
    "SPY":     {"label": "S&P 500 (SPY)",              "region": "US",  "kind": "equity"},
    "URTH":    {"label": "MSCI World (URTH)",          "region": "DM",  "kind": "equity"},
    "AOR":     {"label": "60/40 Balanced (AOR)",       "region": "GLB", "kind": "balanced"},
    "^TASI":   {"label": "Tadawul (^TASI)",            "region": "KSA", "kind": "equity"},
    "BTC-USD": {"label": "Bitcoin (BTC-USD)",          "region": "GLB", "kind": "crypto"},
    "^BCOM":   {"label": "Bloomberg Commodity (BCOM)", "region": "GLB", "kind": "commodities"},
}

_FALLBACKS: Dict[str, List[str]] = {
    "SPY": ["VOO", "^GSPC"],
    "URTH": ["ACWI", "VT"],
    "AOR": ["AOR", "SYNTH_AOR"],
    "^TASI": ["KSA"],
    "BTC-USD": ["BTC", "IBIT"],
    "^BCOM": ["DBC", "GSG", "USO"],
}

_LABEL_PATCH = {
    "benchmark": {"en": "Benchmark", "ar": "المؤشر"},
    "window": {"en": "Window", "ar": "النافذة"},
    "months_suffix": {"en": "m", "ar": "شهر"},
    "excess_return_decomposition": {
        "en": "Excess Return Decomposition",
        "ar": "تحليل العائد الفائض",
    },
    "benchmark_relative_regime_behavior": {
        "en": "Benchmark-Relative Regime Behavior",
        "ar": "سلوك المحفظة النسبي حسب النظام",
    },
    "component": {"en": "Component", "ar": "المكون"},
    "contribution_pp": {"en": "Contribution (pp)", "ar": "المساهمة (نقطة مئوية)"},
    "allocation_effect": {"en": "Allocation Effect", "ar": "أثر التخصيص"},
    "selection_effect": {"en": "Selection Effect", "ar": "أثر الاختيار"},
    "factor_effect": {"en": "Factor Effect", "ar": "أثر العوامل"},
    "concentration_effect": {"en": "Concentration Effect", "ar": "أثر التركيز"},
    "outperform_envs": {
        "en": "Environments where portfolio likely outperforms",
        "ar": "البيئات التي يرجح أن تتفوق فيها المحفظة",
    },
    "lag_envs": {
        "en": "Environments where portfolio likely lags",
        "ar": "البيئات التي يرجح أن تتأخر فيها المحفظة",
    },
    "insufficient_regime_history": {
        "en": "insufficient regime history",
        "ar": "سجل الأنظمة غير كاف",
    },
    "commentary_elevated": {
        "en": (
            "Tracking error is elevated relative to benchmark composition; active share above "
            "60% indicates material structural deviation that may amplify factor-driven "
            "dispersion in stress regimes."
        ),
        "ar": (
            "خطأ التتبع مرتفع قياسا بتركيب المؤشر؛ وتشير الحصة النشطة فوق 60% إلى "
            "انحراف هيكلي ملموس قد يرفع تشتت الأداء المدفوع بالعوامل في أنظمة الضغط."
        ),
    },
    "commentary_moderate": {
        "en": (
            "Benchmark-relative risk is moderate; portfolio outcomes should remain sensitive "
            "to allocation differences and beta dispersion versus the selected benchmark."
        ),
        "ar": (
            "المخاطر النسبية مقابل المؤشر متوسطة؛ وتبقى نتائج المحفظة حساسة لفروق "
            "التخصيص وتشتت بيتا مقابل المؤشر المختار."
        ),
    },
    "commentary_low": {
        "en": (
            "Benchmark-relative dispersion is contained; active outcomes are likely to be driven "
            "more by incremental allocation differences than by structural benchmark departure."
        ),
        "ar": (
            "تشتت الأداء النسبي محدود؛ ومن المرجح أن تقود فروق التخصيص التدريجية النتائج "
            "النشطة أكثر من الابتعاد الهيكلي عن المؤشر."
        ),
    },
    "risk_on_equity_rally": {"en": "risk-on equity rally", "ar": "ارتفاع الأسهم في بيئة تقبل المخاطر"},
    "defensive_drawdown": {"en": "defensive drawdown", "ar": "انخفاض دفاعي"},
    "sideways_market": {"en": "sideways market", "ar": "سوق عرضي"},
    "high_volatility": {"en": "high-volatility regime", "ar": "نظام عالي التذبذب"},
}
LABELS.update({k: v for k, v in _LABEL_PATCH.items() if k not in LABELS})


_ASSET_PROXY: Dict[str, str] = {
    "USA": "SPY",
    "US": "SPY",
    "GCC": "KSA",
    "KSA": "KSA",
    "EGY": "EFID.CA",
    "EGYPT": "EFID.CA",
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHEREUM": "ETH-USD",
    "GLD": "GLD",
    "GOLD": "GLD",
    "TLT": "TLT",
    "EMB": "EMB",
    "CASH": "BIL",
    "BIL": "BIL",
    "OIL": "USO",
    "USO": "USO",
    "SLV": "SLV",
    "SILVER": "SLV",
    "COPR": "CPER",
    "COPPER": "CPER",
    "CPER": "CPER",
    "US LARGE CAP TECH": "QQQ",
    "US S&P 500 BROAD": "SPY",
    "US MID-CAP EQUITY": "MDY",
    "US DIVIDEND/VALUE": "VIG",
    "SAUDI EQUITIES ETF": "KSA",
    "UAE REAL ESTATE": "EMAAR.DU",
    "GCC BANKS/FINANCIALS": "QETF.QA",
    "EGYPT EQUITIES": "EFID.CA",
    "US TREASURIES (LT)": "TLT",
    "EM BONDS": "EMB",
    "CASH / T-BILLS": "BIL",
    "US HEALTHCARE": "XLV",
    "US UTILITIES": "XLU",
    "SHORT-DURATION BONDS": "SHY",
    "CRUDE OIL": "USO",
    "SILVER": "SLV",
    "COPPER": "CPER",
}

_PROXY_REGION: Dict[str, str] = {
    "QQQ": "US",
    "SPY": "US",
    "VOO": "US",
    "^GSPC": "US",
    "MDY": "US",
    "VIG": "US",
    "XLV": "US",
    "XLU": "US",
    "KSA": "GCC",
    "^TASI": "GCC",
    "EMAAR.DU": "GCC",
    "QETF.QA": "GCC",
    "EFID.CA": "Egypt",
    "BTC-USD": "Crypto",
    "BTC": "Crypto",
    "IBIT": "Crypto",
    "ETH-USD": "Crypto",
    "GLD": "Gold",
    "TLT": "Bonds",
    "AGG": "Bonds",
    "EMB": "Bonds",
    "SHY": "Bonds",
    "BIL": "Cash",
    "USO": "Commodities",
    "DBC": "Commodities",
    "GSG": "Commodities",
    "^BCOM": "Commodities",
    "SLV": "Commodities",
    "CPER": "Commodities",
    "URTH": "DM",
    "ACWI": "DM",
    "VT": "DM",
    "AOR": "Balanced",
}

_REGION_PROXY = {
    "US": "SPY",
    "DM": "ACWI",
    "GLB": "ACWI",
    "World": "ACWI",
    "GCC": "KSA",
    "KSA": "KSA",
    "Egypt": "EFID.CA",
    "Crypto": "BTC-USD",
    "Gold": "GLD",
    "Bonds": "TLT",
    "Cash": "BIL",
    "Commodities": "USO",
    "Balanced": "AOR",
}

_BENCHMARK_COMPOSITION = {
    "SPY": {"US": 1.0},
    "VOO": {"US": 1.0},
    "^GSPC": {"US": 1.0},
    "URTH": {"DM": 1.0},
    "ACWI": {"DM": 1.0},
    "VT": {"DM": 1.0},
    "AOR": {"DM": 0.60, "Bonds": 0.40},
    "SYNTH_AOR": {"DM": 0.60, "Bonds": 0.40},
    "^TASI": {"GCC": 1.0},
    "KSA": {"GCC": 1.0},
    "BTC-USD": {"Crypto": 1.0},
    "BTC": {"Crypto": 1.0},
    "IBIT": {"Crypto": 1.0},
    "^BCOM": {"Commodities": 1.0},
    "DBC": {"Commodities": 1.0},
    "GSG": {"Commodities": 1.0},
    "USO": {"Commodities": 1.0},
}


def pick_benchmark(region_tilt: Optional[str], asset_kind: Optional[str]) -> str:
    """Fallback-aware benchmark selector following the H1 policy."""
    kind = str(asset_kind or "").strip().lower()
    if kind == "crypto":
        return "BTC-USD"
    if kind in {"commodities", "commodity"}:
        return "^BCOM"
    if kind == "balanced":
        return "AOR"

    if isinstance(region_tilt, Mapping):
        region_weights = _normalise_region_weights(region_tilt)
        if region_weights:
            region, weight = max(region_weights.items(), key=lambda kv: kv[1])
            if weight <= 0.40:
                return "URTH"
            return _benchmark_for_region(region)

    region = str(region_tilt or "").strip()
    return _benchmark_for_region(region)


def compute_benchmark_relative(
    weights: Dict[str, float],
    returns_panel: Optional[Any] = None,
    benchmark_ticker: Optional[str] = None,
    region_tilt: Optional[str] = None,
    asset_kind: Optional[str] = None,
    language: str = "en",
) -> BenchmarkRelative:
    """Compute the H1 BenchmarkRelative payload."""
    if not PHASE_H_BENCHMARK:
        return {}

    notes: List[str] = []
    try:
        norm_weights = _normalise_weights(weights or {})
        portfolio_tickers = _map_weights_to_proxies(norm_weights)
        region_weights = _portfolio_region_weights(portfolio_tickers)
        selected = benchmark_ticker or pick_benchmark(region_tilt or region_weights, asset_kind)
        panel = _prepare_returns_panel(returns_panel, portfolio_tickers, selected)
        bench_ticker, bench_returns, bench_notes = _resolve_benchmark_returns(selected, panel)
        notes.extend(bench_notes)

        if bench_returns is None or bench_returns.empty:
            notes.append("benchmark data unavailable — defaulted to URTH")
            bench_ticker = "URTH"
            bench_returns = pd.Series(dtype=float)

        port_returns = _portfolio_returns(portfolio_tickers, panel)
        if port_returns.empty or bench_returns.empty:
            return _degenerate_payload(bench_ticker, notes + ["insufficient return history"])

        aligned = pd.concat(
            [port_returns.rename("portfolio"), bench_returns.rename("benchmark")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) > 36:
            aligned = aligned.tail(36)
        if aligned.empty:
            return _degenerate_payload(bench_ticker, notes + ["insufficient overlapping return history"])

        p = aligned["portfolio"].astype(float)
        b = aligned["benchmark"].astype(float)
        active = p - b
        n_months = int(len(aligned))

        p_ann = _annualised_return(p)
        b_ann = _annualised_return(b)
        active_return_pct = (p_ann - b_ann) * 100.0
        tracking_error_pct = float(active.std(ddof=1) * math.sqrt(12) * 100.0) if n_months > 1 else 0.0
        information_ratio = active_return_pct / tracking_error_pct if tracking_error_pct > 1e-12 else 0.0
        alpha_pct, beta, r_squared = _capm_stats(p.tail(12), b.tail(12))
        relative_drawdown_pct = _relative_drawdown(active)
        upside_capture = _capture_ratio(p, b, upside=True)
        downside_capture = _capture_ratio(p, b, upside=False)
        b_vol = float(b.std(ddof=1))
        relative_volatility = float(p.std(ddof=1) / b_vol) if b_vol > 1e-12 and n_months > 1 else 0.0

        active_share_pct, style_drift = _active_share_and_style(portfolio_tickers, bench_ticker)
        decomp = _excess_decomposition(
            portfolio_tickers=portfolio_tickers,
            panel=panel,
            bench_ticker=bench_ticker,
            beta=beta,
            active_return_pct=active_return_pct,
            benchmark_ann_return=b_ann,
        )
        regimes = _regime_behavior(p, b, language, notes)
        tier = _reliability_tier(n_months, r_squared)
        if n_months < 12:
            notes.append("fewer than 12 months of overlapping history")
        elif n_months < 36:
            notes.append("shorter than 36-month institutional window")

        return BenchmarkRelative(
            benchmark_ticker=bench_ticker,
            benchmark_label=_benchmark_label(bench_ticker),
            active_return_pct=round(float(active_return_pct), 4),
            tracking_error_pct=round(float(tracking_error_pct), 4),
            information_ratio=round(float(information_ratio), 4),
            rolling_alpha_12m_pct=round(float(alpha_pct), 4),
            rolling_beta_12m=round(float(beta), 4),
            relative_drawdown_pct=round(float(relative_drawdown_pct), 4),
            upside_capture=round(float(upside_capture), 4),
            downside_capture=round(float(downside_capture), 4),
            relative_volatility=round(float(relative_volatility), 4),
            active_share_pct=round(float(active_share_pct), 4),
            style_drift=style_drift,
            excess_decomp=decomp,
            regime_behavior=regimes,
            reliability_tier=tier,
            window_months=n_months,
            r_squared=round(float(r_squared), 4),
            notes=notes,
        )
    except Exception as exc:  # noqa: BLE001 - public engine must degrade
        selected = benchmark_ticker or pick_benchmark(region_tilt, asset_kind)
        return _degenerate_payload(selected, [f"benchmark engine degraded: {exc!r}"])


def render_benchmark_relative_md(
    payload: BenchmarkRelative, language: str = "en"
) -> str:
    """Render the H1 institutional benchmark-relative markdown block."""
    if not PHASE_H_BENCHMARK or not payload:
        return ""

    heading = section_heading(3, "benchmark_relative_diagnostics", language)
    window = payload.get("window_months", 0)
    bench_line = (
        f"*{L('benchmark', language)}: {payload.get('benchmark_label', '—')} · "
        f"{L('reliability', language)}: {payload.get('reliability_tier', '—')} · "
        f"{L('window', language)}: {window}{L('months_suffix', language)}*"
    )

    rows: List[List[str]] = [
        [L("active_return", language), _fmt_signed_pct(payload.get("active_return_pct")), severity_tag("moderate", language)],
        [L("tracking_error", language), fmt_pct(payload.get("tracking_error_pct")), _tracking_error_tag(payload, language)],
        [L("information_ratio", language), fmt_num(payload.get("information_ratio")), _information_ratio_tag(payload, language)],
        [L("rolling_alpha", language), _fmt_signed_pct(payload.get("rolling_alpha_12m_pct")), "—"],
        [L("rolling_beta", language), fmt_num(payload.get("rolling_beta_12m")), "—"],
        [L("relative_drawdown", language), fmt_pct(payload.get("relative_drawdown_pct")), _drawdown_tag(payload, language)],
        [L("upside_capture", language), fmt_num(payload.get("upside_capture")), _capture_tag(payload.get("upside_capture"), language)],
        [L("downside_capture", language), fmt_num(payload.get("downside_capture")), _capture_tag(payload.get("downside_capture"), language)],
        [L("relative_volatility", language), fmt_num(payload.get("relative_volatility")), "—"],
        [L("active_share", language), fmt_pct(payload.get("active_share_pct")), _active_share_tag(payload, language)],
        [L("style_drift", language), str(payload.get("style_drift", "—")), "—"],
    ]

    decomp = payload.get("excess_decomp") or {}
    decomp_rows = [
        [L("allocation_effect", language), _fmt_signed_num(decomp.get("allocation"))],
        [L("selection_effect", language), _fmt_signed_num(decomp.get("selection"))],
        [L("factor_effect", language), _fmt_signed_num(decomp.get("factor"))],
        [L("concentration_effect", language), _fmt_signed_num(decomp.get("concentration"))],
    ]

    regimes = payload.get("regime_behavior") or {}
    outperform = _join_regimes(regimes.get("outperform_envs"), language)
    lag = _join_regimes(regimes.get("lag_envs"), language)
    commentary = _commentary(payload, language)

    return "\n\n".join(
        [
            heading,
            bench_line,
            md_table([L("metric", language), L("value", language), L("tag", language)], rows),
            f"**{L('excess_return_decomposition', language)}**\n\n"
            + md_table([L("component", language), L("contribution_pp", language)], decomp_rows),
            f"**{L('benchmark_relative_regime_behavior', language)}**\n\n"
            f"- {L('outperform_envs', language)}: {outperform}\n"
            f"- {L('lag_envs', language)}: {lag}",
            commentary,
        ]
    ).strip() + "\n"


def _benchmark_for_region(region: str) -> str:
    r = str(region or "").strip().upper()
    if r in {"US", "USA", "AMERICA"}:
        return "SPY"
    if r in {"KSA", "GCC"}:
        return "^TASI"
    if r in {"DM", "WORLD", "GLB", "GLOBAL"}:
        return "URTH"
    return "URTH"


def _normalise_region_weights(region_weights: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = 0.0
    for k, v in region_weights.items():
        try:
            w = max(0.0, float(v))
        except (TypeError, ValueError):
            continue
        key = str(k)
        if key.upper() in {"USA", "AMERICA"}:
            key = "US"
        elif key.upper() in {"KSA", "GCC"}:
            key = "GCC"
        elif key.upper() in {"EGY", "EGYPT"}:
            key = "Egypt"
        out[key] = out.get(key, 0.0) + w
        total += w
    if total > 1.5:
        out = {k: v / 100.0 for k, v in out.items()}
    return out


def _normalise_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for k, v in weights.items():
        try:
            w = float(v)
        except (TypeError, ValueError):
            continue
        if w > 0:
            parsed[str(k)] = w
    total = sum(parsed.values())
    if total <= 0:
        return {}
    if total > 1.5:
        parsed = {k: v / 100.0 for k, v in parsed.items()}
        total = sum(parsed.values())
    return {k: v / total for k, v in parsed.items()}


def _map_weights_to_proxies(weights: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw_key, w in weights.items():
        ticker = _ASSET_PROXY.get(raw_key.upper(), raw_key.upper())
        ticker = _canonical_ticker(ticker)
        out[ticker] = out.get(ticker, 0.0) + float(w)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else {}


def _canonical_ticker(ticker: str) -> str:
    t = str(ticker or "").strip()
    upper = t.upper()
    if upper in {"BTCUSD", "BTC-USD"}:
        return "BTC-USD"
    if upper in {"ETHUSD", "ETH-USD"}:
        return "ETH-USD"
    return upper if t.startswith("^") else t.upper()


def _portfolio_region_weights(ticker_weights: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ticker, weight in ticker_weights.items():
        region = _PROXY_REGION.get(ticker, "US")
        out[region] = out.get(region, 0.0) + float(weight)
    return out


def _prepare_returns_panel(
    returns_panel: Optional[Any],
    portfolio_tickers: Mapping[str, float],
    selected_benchmark: str,
) -> pd.DataFrame:
    if returns_panel is not None:
        panel = _coerce_panel_to_returns(returns_panel)
    else:
        tickers = set(portfolio_tickers)
        tickers.add(selected_benchmark)
        tickers.update(_FALLBACKS.get(selected_benchmark, []))
        tickers.update({"ACWI", "VT", "AGG", "SPY", "KSA", "GLD", "TLT", "BIL", "USO"})
        panel = _load_cached_returns(tickers)
    return _add_synthetic_columns(panel)


def _coerce_panel_to_returns(panel: Any) -> pd.DataFrame:
    if isinstance(panel, pd.Series):
        df = panel.to_frame()
    else:
        df = pd.DataFrame(panel).copy()
    if df.empty:
        return df
    df.columns = [str(c).upper() if not str(c).startswith("^") else str(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_abs = float(df.abs().max().max()) if not df.dropna(how="all").empty else 0.0
    if max_abs > 2.0:
        df = df.pct_change()
    return df.dropna(how="all")


def _load_cached_returns(tickers: Iterable[str]) -> pd.DataFrame:
    """Delegate to the institutional data layer — no direct cache access."""
    try:
        from core.data_layer.market_cache_adapter import get_returns_panel
    except Exception:
        return pd.DataFrame()
    filtered = [t for t in tickers if t and not str(t).startswith("SYNTH_")]
    panel = get_returns_panel(filtered)
    if panel is None or panel.empty:
        return pd.DataFrame()
    return panel


def _add_synthetic_columns(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame()
    out = panel.copy()
    for col in list(out.columns):
        if col.startswith("^"):
            continue
        out[_canonical_ticker(col)] = out[col]
    if "URTH" not in out:
        for fallback in ("ACWI", "VT", "SPY"):
            if fallback in out:
                out["URTH"] = out[fallback]
                break
    if "BTC-USD" not in out:
        for fallback in ("BTC", "IBIT"):
            if fallback in out:
                out["BTC-USD"] = out[fallback]
                break
    if "^TASI" not in out and "KSA" in out:
        out["^TASI"] = out["KSA"]
    if "^BCOM" not in out:
        for fallback in ("DBC", "GSG", "USO"):
            if fallback in out:
                out["^BCOM"] = out[fallback]
                break
    if "AOR" not in out:
        eq_col = next((c for c in ("URTH", "ACWI", "VT", "SPY") if c in out), None)
        bond_col = "AGG" if "AGG" in out else ("TLT" if "TLT" in out else None)
        if eq_col and bond_col:
            out["AOR"] = 0.60 * out[eq_col] + 0.40 * out[bond_col]
            out["SYNTH_AOR"] = out["AOR"]
    return out


def _resolve_benchmark_returns(
    selected: str,
    panel: pd.DataFrame,
) -> Tuple[str, Optional[pd.Series], List[str]]:
    notes: List[str] = []
    chain = [selected] + [t for t in _FALLBACKS.get(selected, []) if t != selected]
    if selected == "AOR" and "SYNTH_AOR" not in chain:
        chain.append("SYNTH_AOR")
    for ticker in chain:
        if ticker in panel and panel[ticker].dropna().shape[0] > 0:
            if ticker != selected and ticker != "SYNTH_AOR":
                notes.append(f"{selected} benchmark data unavailable — used {ticker} fallback")
            if ticker == "SYNTH_AOR":
                notes.append("AOR benchmark data unavailable — used synthetic 60/40 URTH/AGG fallback")
                return "AOR", panel[ticker].dropna(), notes
            return ticker, panel[ticker].dropna(), notes
    if selected != "URTH" and "URTH" in panel and panel["URTH"].dropna().shape[0] > 0:
        notes.append("benchmark data unavailable — defaulted to URTH")
        return "URTH", panel["URTH"].dropna(), notes
    return selected, None, notes


def _portfolio_returns(ticker_weights: Mapping[str, float], panel: pd.DataFrame) -> pd.Series:
    pieces: List[pd.Series] = []
    for ticker, weight in ticker_weights.items():
        col = ticker if ticker in panel else _fallback_column_for_ticker(ticker, panel)
        if col is None:
            continue
        pieces.append(panel[col].astype(float) * float(weight))
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces, axis=1).sum(axis=1).dropna()


def _fallback_column_for_ticker(ticker: str, panel: pd.DataFrame) -> Optional[str]:
    region = _PROXY_REGION.get(ticker)
    candidates = [ticker]
    if region:
        candidates.append(_REGION_PROXY.get(region, ""))
    candidates.extend(_FALLBACKS.get(ticker, []))
    for cand in candidates:
        if cand in panel:
            return cand
    return None


def _annualised_return(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    compounded = float((1.0 + s).prod())
    if compounded <= 0:
        return -1.0
    return compounded ** (12.0 / len(s)) - 1.0


def _capm_stats(port: pd.Series, bench: pd.Series) -> Tuple[float, float, float]:
    aligned = pd.concat([port.rename("p"), bench.rename("b")], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 1.0, 0.0
    x = aligned["b"].to_numpy(dtype=float)
    y = aligned["p"].to_numpy(dtype=float)
    var_x = float(np.var(x, ddof=1))
    if var_x <= 1e-12:
        return 0.0, 1.0, 0.0
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    alpha_month = float(np.mean(y) - beta * np.mean(x))
    fitted = alpha_month + beta * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    alpha_ann = ((1.0 + alpha_month) ** 12 - 1.0) * 100.0 if alpha_month > -1 else -100.0
    return alpha_ann, beta, max(0.0, min(1.0, r2))


def _relative_drawdown(active: pd.Series) -> float:
    if active.empty:
        return 0.0
    curve = (1.0 + active).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def _capture_ratio(port: pd.Series, bench: pd.Series, *, upside: bool) -> float:
    mask = bench > 0 if upside else bench < 0
    if not mask.any():
        return 0.0
    denom = float(bench[mask].mean())
    if abs(denom) <= 1e-12:
        return 0.0
    return float(port[mask].mean() / denom)


def _active_share_and_style(
    portfolio_tickers: Mapping[str, float],
    bench_ticker: str,
) -> Tuple[float, str]:
    port_regions = _portfolio_region_weights(portfolio_tickers)
    bench_regions = _BENCHMARK_COMPOSITION.get(bench_ticker, _BENCHMARK_COMPOSITION.get("URTH", {"DM": 1.0}))
    all_regions = sorted(set(port_regions) | set(bench_regions))
    active_share = 0.5 * sum(abs(port_regions.get(r, 0.0) - bench_regions.get(r, 0.0)) for r in all_regions) * 100.0
    dist = math.sqrt(sum((port_regions.get(r, 0.0) - bench_regions.get(r, 0.0)) ** 2 for r in all_regions))
    if dist < 0.05:
        drift = "aligned"
    elif dist < 0.15:
        drift = "mild"
    elif dist < 0.30:
        drift = "material"
    else:
        drift = "severe"
    return active_share, drift


def _excess_decomposition(
    *,
    portfolio_tickers: Mapping[str, float],
    panel: pd.DataFrame,
    bench_ticker: str,
    beta: float,
    active_return_pct: float,
    benchmark_ann_return: float,
) -> Dict[str, float]:
    port_regions = _portfolio_region_weights(portfolio_tickers)
    bench_regions = _BENCHMARK_COMPOSITION.get(bench_ticker, {"DM": 1.0})
    all_regions = sorted(set(port_regions) | set(bench_regions))
    region_returns: Dict[str, float] = {}
    for region in all_regions:
        region_returns[region] = _region_annual_return(region, panel)
    allocation = sum(
        (port_regions.get(r, 0.0) - bench_regions.get(r, 0.0))
        * (region_returns.get(r, benchmark_ann_return) - benchmark_ann_return)
        for r in all_regions
    ) * 100.0
    port_region_returns = _portfolio_region_returns(portfolio_tickers, panel, region_returns)
    selection = sum(
        bench_regions.get(r, 0.0) * (port_region_returns.get(r, region_returns.get(r, 0.0)) - region_returns.get(r, 0.0))
        for r in all_regions
    ) * 100.0
    factor = (float(beta) - 1.0) * float(benchmark_ann_return) * 100.0
    concentration = float(active_return_pct) - allocation - selection - factor
    return {
        "allocation": round(float(allocation), 4),
        "selection": round(float(selection), 4),
        "factor": round(float(factor), 4),
        "concentration": round(float(concentration), 4),
    }


def _region_annual_return(region: str, panel: pd.DataFrame) -> float:
    candidates = [_REGION_PROXY.get(region, "")]
    if region == "DM":
        candidates = ["URTH", "ACWI", "VT", "SPY"]
    for col in candidates:
        if col and col in panel:
            return _annualised_return(panel[col])
    return 0.0


def _portfolio_region_returns(
    portfolio_tickers: Mapping[str, float],
    panel: pd.DataFrame,
    fallback_region_returns: Mapping[str, float],
) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for ticker, weight in portfolio_tickers.items():
        region = _PROXY_REGION.get(ticker, "US")
        grouped.setdefault(region, []).append((ticker, weight))
    out: Dict[str, float] = {}
    for region, items in grouped.items():
        total = sum(w for _, w in items)
        if total <= 0:
            continue
        returns = 0.0
        used = False
        for ticker, weight in items:
            col = ticker if ticker in panel else _fallback_column_for_ticker(ticker, panel)
            if col and col in panel:
                returns += (weight / total) * _annualised_return(panel[col])
                used = True
        out[region] = returns if used else fallback_region_returns.get(region, 0.0)
    return out


def _regime_behavior(port: pd.Series, bench: pd.Series, language: str, notes: List[str]) -> Dict[str, List[str]]:
    if len(port) < 12 or len(bench) < 12:
        notes.append("insufficient regime history")
        return {"outperform_envs": [], "lag_envs": []}
    active_6m = (1.0 + (port - bench)).rolling(6).apply(np.prod, raw=True) - 1.0
    bench_6m = (1.0 + bench).rolling(6).apply(np.prod, raw=True) - 1.0
    vol_6m = bench.rolling(6).std()
    buckets: Dict[str, List[float]] = {}
    for idx in active_6m.dropna().index:
        label = _regime_label(float(bench_6m.loc[idx]), float(vol_6m.loc[idx]))
        buckets.setdefault(label, []).append(float(active_6m.loc[idx]))
    outperform = [L(k, language) for k, vals in buckets.items() if vals and float(np.median(vals)) > 0]
    lag = [L(k, language) for k, vals in buckets.items() if vals and float(np.median(vals)) < 0]
    return {"outperform_envs": outperform, "lag_envs": lag}


def _regime_label(bench_6m: float, vol_6m: float) -> str:
    if vol_6m > 0.065:
        return "high_volatility"
    if bench_6m > 0.06:
        return "risk_on_equity_rally"
    if bench_6m < -0.04:
        return "defensive_drawdown"
    return "sideways_market"


def _reliability_tier(n_months: int, r_squared: float) -> str:
    if n_months >= 36 and r_squared >= 0.5:
        return "Institutional"
    if 18 <= n_months <= 35 or 0.3 <= r_squared < 0.5:
        return "Institutional-Lite"
    return "Indicative"


def _degenerate_payload(ticker: str, notes: List[str]) -> BenchmarkRelative:
    return BenchmarkRelative(
        benchmark_ticker=ticker,
        benchmark_label=_benchmark_label(ticker),
        active_return_pct=0.0,
        tracking_error_pct=0.0,
        information_ratio=0.0,
        rolling_alpha_12m_pct=0.0,
        rolling_beta_12m=1.0,
        relative_drawdown_pct=0.0,
        upside_capture=0.0,
        downside_capture=0.0,
        relative_volatility=0.0,
        active_share_pct=0.0,
        style_drift="aligned",
        excess_decomp={"allocation": 0.0, "selection": 0.0, "factor": 0.0, "concentration": 0.0},
        regime_behavior={"outperform_envs": [], "lag_envs": []},
        reliability_tier="Indicative",
        window_months=0,
        r_squared=0.0,
        notes=notes,
    )


def _benchmark_label(ticker: str) -> str:
    if ticker in BENCHMARK_CATALOG:
        return BENCHMARK_CATALOG[ticker]["label"]
    for primary, fallbacks in _FALLBACKS.items():
        if ticker in fallbacks and primary in BENCHMARK_CATALOG:
            return BENCHMARK_CATALOG[primary]["label"]
    return ticker


def _fmt_signed_pct(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if f > 0 else ""
    return sign + fmt_pct(f)


def _fmt_signed_num(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if f > 0 else ""
    return sign + fmt_num(f)


def _tracking_error_tag(payload: BenchmarkRelative, language: str) -> str:
    v = float(payload.get("tracking_error_pct", 0.0) or 0.0)
    return severity_tag("low" if v < 3 else "moderate" if v <= 6 else "elevated", language)


def _information_ratio_tag(payload: BenchmarkRelative, language: str) -> str:
    v = float(payload.get("information_ratio", 0.0) or 0.0)
    return severity_tag("low" if v < 0.2 else "moderate" if v <= 0.5 else "elevated", language)


def _active_share_tag(payload: BenchmarkRelative, language: str) -> str:
    v = float(payload.get("active_share_pct", 0.0) or 0.0)
    return severity_tag("low" if v < 30 else "moderate" if v <= 60 else "elevated", language)


def _capture_tag(value: Any, language: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    return severity_tag("low" if v < 0.8 else "moderate" if v <= 1.2 else "elevated", language)


def _drawdown_tag(payload: BenchmarkRelative, language: str) -> str:
    v = abs(float(payload.get("relative_drawdown_pct", 0.0) or 0.0))
    return severity_tag("low" if v < 5 else "moderate" if v <= 15 else "elevated", language)


def _join_regimes(values: Any, language: str) -> str:
    vals = [str(v) for v in (values or []) if str(v).strip()]
    return ", ".join(vals) if vals else L("insufficient_regime_history", language)


def _commentary(payload: BenchmarkRelative, language: str) -> str:
    te = float(payload.get("tracking_error_pct", 0.0) or 0.0)
    active_share = float(payload.get("active_share_pct", 0.0) or 0.0)
    if te > 6 or active_share > 60:
        return L("commentary_elevated", language)
    if te >= 3 or active_share >= 30:
        return L("commentary_moderate", language)
    return L("commentary_low", language)


__all__ = [
    "ENGINE_VERSION",
    "BENCHMARK_CATALOG",
    "pick_benchmark",
    "compute_benchmark_relative",
    "render_benchmark_relative_md",
]
