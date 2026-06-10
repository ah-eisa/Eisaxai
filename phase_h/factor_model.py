"""
Phase H4 - True Factor Model Engine (FF3 / Carhart / FF5).

The public API degrades to an Indicative payload instead of raising.  It keeps
the legacy direct FactorDecomp shape used by the orchestrator, and attaches the
canonical Phase H envelope under ``_envelope`` for contract/audit consumers.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
import urllib.request
import zipfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import make_envelope, unwrap
from .numerics import ValidationResult, validate_returns_panel
from .registry import FeatureRegistry
from .report_helpers import L, LABELS, fmt_num, md_table, section_heading
from .schemas import FactorDecomp

ENGINE_VERSION = "0.2.0"


SUPPORTED_MODELS = ("FF3", "Carhart", "FF5", "FF5_QMJ", "LowVol")

_DEFAULT_MODEL = "Carhart"
_ROLLING_WINDOW = 36
_MIN_INSTITUTIONAL_OBS = 24
_SHRINKAGE_TAU = 18.0
# Cache locations live behind the institutional data layer — engines
# never construct snapshot paths themselves. See data_layer/factor_premia.py.
_FACTOR_TTL_SECONDS = 24 * 60 * 60

_MODEL_FACTORS: Dict[str, Tuple[str, ...]] = {
    "FF3": ("MKT", "SMB", "HML"),
    "Carhart": ("MKT", "SMB", "HML", "MOM"),
    "FF5": ("MKT", "SMB", "HML", "RMW", "CMA"),
    "FF5_QMJ": ("MKT", "SMB", "HML", "RMW", "CMA", "QMJ"),
    "LowVol": ("LOWVOL",),
}

_KEN_FRENCH_URLS = {
    "ff3": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
}

_FACTOR_FILES = {
    "ff3": "ff3_monthly.csv",
    "mom": "carhart_mom_monthly.csv",
    "ff5": "ff5_monthly.csv",
}

_ASSET_PROXY: Dict[str, str] = {
    "USA": "SPY",
    "US": "SPY",
    "US LARGE CAP TECH": "QQQ",
    "US S&P 500 BROAD": "SPY",
    "US MID-CAP EQUITY": "MDY",
    "US DIVIDEND/VALUE": "VIG",
    "SAUDI EQUITIES ETF": "KSA",
    "SAUDI": "KSA",
    "KSA": "KSA",
    "GCC": "KSA",
    "UAE REAL ESTATE": "EMAAR.DU",
    "GCC BANKS/FINANCIALS": "QETF.QA",
    "EGYPT EQUITIES": "EFID.CA",
    "EGYPT": "EFID.CA",
    "EGY": "EFID.CA",
    "GOLD": "GLD",
    "US TREASURIES (LT)": "TLT",
    "EM BONDS": "EMB",
    "CASH / T-BILLS": "BIL",
    "SHORT-DURATION BONDS": "SHY",
    "CRUDE OIL": "USO",
    "SILVER": "SLV",
    "COPPER": "CPER",
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
}

_GCC_TICKERS = {
    "KSA",
    "^TASI",
    "EMAAR.DU",
    "QETF.QA",
    "ADCB.AD",
    "ALDAR.AD",
    "EMAAR",
}
_EGYPT_TICKERS = {"EFID.CA", "COMI.CA", "HRHO.CA", "EGX30.CA"}

_REGION_PRIORS = {
    "GCC": {"MKT": 0.55, "SMB": 0.18, "HML": 0.20, "MOM": 0.05, "RMW": 0.10, "CMA": 0.05},
    "Egypt": {"MKT": 0.40, "SMB": 0.25, "HML": 0.30, "MOM": 0.05, "RMW": 0.05, "CMA": 0.05},
}

_LABEL_PATCH = {
    "factor_decomposition_diagnostics": {
        "en": "Factor Risk Decomposition",
        "ar": "تحليل المخاطر بحسب العوامل",
    },
    "stability": {"en": "Stability", "ar": "الثبات"},
    "r_squared_label": {"en": "R²", "ar": "R²"},
    "window_label": {"en": "Window", "ar": "النافذة"},
    "factor_warnings_title": {"en": "Warnings", "ar": "تنبيهات"},
    "factor_loadings_footnote": {
        "en": "Loadings represent rolling 36-month exposures; t-statistics are Newey-West adjusted where available.",
        "ar": "تمثل التحميلات انكشافات متجددة لمدة 36 شهرا؛ وتعدل إحصاءات t بطريقة Newey-West عند توفرها.",
    },
    "months_suffix": {"en": "m", "ar": "شهر"},
    "factor_insufficient_history": {
        "en": "Insufficient overlapping factor history to compute the Carhart/FF decomposition for this asset. This module is hidden when factor loadings collapse to zero.",
        "ar": "السجل غير كاف لاحتساب نموذج العوامل (Carhart / FF). يُخفى هذا القسم عندما تنخفض جميع التحميلات إلى الصفر.",
    },
}
LABELS.update({k: v for k, v in _LABEL_PATCH.items() if k not in LABELS})


def compute_factor_decomposition(
    weights: Dict[str, float],
    returns_panel: Optional[Any] = None,
    factor_panel: Optional[Any] = None,
    model: str = "Carhart",
    rolling_window: int = 36,
    language: str = "en",
) -> FactorDecomp:
    """Compute rolling factor exposures and contribution diagnostics."""
    if not _feature_enabled():
        return {}

    selected_model = _select_model(model)
    factors = _MODEL_FACTORS[selected_model]
    fallback_used = False
    notes: List[str] = []
    validation = ValidationResult(ok=True)

    try:
        norm_weights = _normalise_weights(weights or {})
        if not norm_weights:
            return _degenerate_payload(selected_model, factors, ["empty portfolio weights"], fallback_used=True)

        proxy_weights = _map_weights_to_proxies(norm_weights)
        returns = _prepare_returns_panel(returns_panel, proxy_weights)
        validation.merge(validate_returns_panel(returns, min_observations=12, label="factor_returns"))

        if returns.empty:
            return _degenerate_payload(
                selected_model,
                factors,
                ["factor data unavailable - using zero loadings"],
                validation=validation,
                fallback_used=True,
            )

        factor_notes: List[str] = []
        factors_df, synthetic_or_missing = _prepare_factor_panel(
            factor_panel=factor_panel,
            returns_panel=returns,
            required_factors=factors,
            notes=factor_notes,
        )
        notes.extend(factor_notes)
        fallback_used = fallback_used or synthetic_or_missing

        if factors_df.empty or not set(factors).issubset(set(factors_df.columns)):
            return _degenerate_payload(
                selected_model,
                factors,
                notes + ["factor data unavailable - using zero loadings"],
                validation=validation,
                fallback_used=True,
            )

        port_returns = _portfolio_returns(proxy_weights, returns)
        if port_returns.empty:
            return _degenerate_payload(
                selected_model,
                factors,
                notes + ["insufficient return history"],
                validation=validation,
                fallback_used=True,
            )

        aligned = _align_portfolio_and_factors(port_returns, factors_df, factors)
        if aligned.empty:
            return _degenerate_payload(
                selected_model,
                factors,
                notes + ["insufficient overlapping factor history"],
                validation=validation,
                fallback_used=True,
            )

        n_available = int(len(aligned))
        effective_window = max(1, min(int(rolling_window or _ROLLING_WINDOW), n_available))
        latest = aligned.tail(effective_window)
        beta, t_stats, r_squared = _fit_factor_model(latest["portfolio"], latest, factors)
        rolling_betas = _rolling_betas(aligned, factors, int(rolling_window or _ROLLING_WINDOW))
        stability = _rolling_stability(rolling_betas, factors)

        asset_diagnostics, shrinkage_delta = _sparse_region_shrinkage(
            raw_weights=norm_weights,
            proxy_weights=proxy_weights,
            returns_panel=returns,
            factor_panel=aligned,
            factors=factors,
            base_portfolio_beta=beta,
        )
        shrinkage_applied = bool(shrinkage_delta)
        if shrinkage_applied:
            for factor, delta in shrinkage_delta.items():
                beta[factor] = float(beta.get(factor, 0.0) + delta)
            notes.append("Bayesian shrinkage applied for sparse GCC/Egypt factor history")

        contribution_return = _contribution_return(beta, latest, factors)
        contribution_vol = _contribution_vol(beta, latest, factors)
        contribution_drawdown = _contribution_drawdown(beta, latest, factors)
        warnings = _factor_warnings(beta, t_stats, stability)
        reliability = _reliability_tier(effective_window, r_squared, stability)
        if fallback_used:
            reliability = "Indicative"
        if shrinkage_applied and reliability == "Indicative":
            reliability = "Institutional-Lite"
        if n_available < _MIN_INSTITUTIONAL_OBS and not shrinkage_applied:
            notes.append("fewer than 24 months of overlapping factor history")
        elif effective_window < _ROLLING_WINDOW:
            notes.append("shorter than 36-month institutional factor window")

        payload: FactorDecomp = FactorDecomp(
            model=selected_model,
            loadings={f: round(float(beta.get(f, 0.0)), 4) for f in factors},
            t_stats={f: round(float(t_stats.get(f, 0.0)), 4) for f in factors},
            contribution_return={f: round(float(contribution_return.get(f, 0.0)), 6) for f in factors},
            contribution_vol={f: round(float(contribution_vol.get(f, 0.0)), 6) for f in factors},
            contribution_drawdown={f: round(float(contribution_drawdown.get(f, 0.0)), 6) for f in factors},
            r_squared=round(float(r_squared), 4),
            rolling_stability=round(float(stability), 4),
            warnings=warnings,
            reliability_tier=reliability,
            notes=notes,
            window_months=effective_window,
            asset_diagnostics=asset_diagnostics,
            shrinkage_applied=shrinkage_applied,
        )
        _attach_envelope(payload, validation=validation, fallback_used=fallback_used, notes=notes)
        return payload
    except Exception as exc:  # noqa: BLE001 - public engine must degrade
        return _degenerate_payload(
            selected_model,
            factors,
            [f"factor engine degraded: {exc!r}"],
            validation=validation,
            fallback_used=True,
        )


def _factor_payload_is_empty(payload: Mapping[str, Any]) -> bool:
    """Detect a degenerate factor payload — zero window, no loadings, or every
    loading == 0. Surfacing a table of zeros pretends the regression ran.
    """
    if not payload:
        return True
    if int(payload.get("window_months") or 0) <= 0:
        return True
    loadings = payload.get("loadings") or {}
    if not loadings:
        return True
    for _f, beta in loadings.items():
        try:
            if abs(float(beta)) > 1e-9:
                return False
        except (TypeError, ValueError):
            continue
    return True


def render_factor_decomposition_md(payload: FactorDecomp, language: str = "en") -> str:
    if not _feature_enabled() or not payload:
        return ""
    payload = _payload(payload)
    if not payload:
        return ""

    # Suppress degenerate payloads — empty regression looks identical to real
    # output if the renderer fills the table with zeros.
    if _factor_payload_is_empty(payload):
        heading = section_heading(3, "factor_decomposition_diagnostics", language)
        warnings = payload.get("warnings") or []
        note_line = ""
        if warnings:
            note_line = f" ({', '.join(str(w) for w in warnings if w)})"
        return (
            f"{heading}\n\n"
            f"> ℹ️ {L('factor_insufficient_history', language)}{note_line}\n"
        )

    heading = section_heading(3, "factor_decomposition_diagnostics", language)
    window = payload.get("window_months", _ROLLING_WINDOW)
    diag_line = (
        f"*Model: {payload.get('model', '-')}"
        f" · {L('r_squared_label', language)}: {fmt_num(payload.get('r_squared'), 2)}"
        f" · {L('stability', language)}: {fmt_num(payload.get('rolling_stability'), 2)}"
        f" · {L('reliability', language)}: {payload.get('reliability_tier', '-')}"
        f" · {L('window_label', language)}: {window}{L('months_suffix', language)}*"
    )
    if language == "ar":
        diag_line = (
            f"*النموذج: {payload.get('model', '-')}"
            f" · {L('r_squared_label', language)}: {fmt_num(payload.get('r_squared'), 2)}"
            f" · {L('stability', language)}: {fmt_num(payload.get('rolling_stability'), 2)}"
            f" · {L('reliability', language)}: {payload.get('reliability_tier', '-')}"
            f" · {L('window_label', language)}: {window}{L('months_suffix', language)}*"
        )

    headers = [
        L("factor", language),
        L("loading", language),
        L("t_stat", language),
        f"{L('contribution', language)} (ret)",
        f"{L('contribution', language)} (vol)",
    ]
    rows: List[List[str]] = []
    loadings = payload.get("loadings") or {}
    t_stats = payload.get("t_stats") or {}
    ret = payload.get("contribution_return") or {}
    vol = payload.get("contribution_vol") or {}
    for factor, beta in loadings.items():
        rows.append(
            [
                str(factor),
                fmt_num(beta, 2),
                fmt_num(t_stats.get(factor), 2),
                _fmt_signed_pct(ret.get(factor)),
                _fmt_abs_pct(vol.get(factor)),
            ]
        )

    parts = [heading, diag_line, md_table(headers, rows)]
    warnings = payload.get("warnings") or []
    if warnings:
        parts.append(
            f"**{L('factor_warnings_title', language)}**\n\n"
            + "\n".join(f"- {warning}" for warning in warnings)
        )
    parts.append(f"*{L('factor_loadings_footnote', language)}*")
    return "\n\n".join(parts).strip() + "\n"


def _feature_enabled() -> bool:
    try:
        return bool(FeatureRegistry.is_enabled("phase_h_factor_model"))
    except Exception:
        return True


def _select_model(requested: Optional[str]) -> str:
    configured = os.environ.get("EISAX_FACTOR_MODEL", "").strip()
    try:
        reg_model = str(FeatureRegistry.get("factor_model") or "").strip()
        configured = reg_model or configured
    except Exception:
        pass
    requested_str = str(requested or "").strip()
    candidate = requested_str if requested_str and requested_str != _DEFAULT_MODEL else (configured or _DEFAULT_MODEL)
    return candidate if candidate in SUPPORTED_MODELS else _DEFAULT_MODEL


def _degenerate_payload(
    model: str,
    factors: Sequence[str],
    notes: Sequence[str],
    *,
    validation: Optional[ValidationResult] = None,
    fallback_used: bool = True,
) -> FactorDecomp:
    payload: FactorDecomp = FactorDecomp(
        model=model,
        loadings={f: 0.0 for f in factors},
        t_stats={f: 0.0 for f in factors},
        contribution_return={f: 0.0 for f in factors},
        contribution_vol={f: 0.0 for f in factors},
        contribution_drawdown={f: 0.0 for f in factors},
        r_squared=0.0,
        rolling_stability=0.0,
        warnings=[],
        reliability_tier="Indicative",
        notes=list(notes),
        window_months=0,
        asset_diagnostics={},
        shrinkage_applied=False,
    )
    _attach_envelope(payload, validation=validation, fallback_used=fallback_used, notes=list(notes))
    return payload


def _attach_envelope(
    payload: FactorDecomp,
    *,
    validation: Optional[ValidationResult],
    fallback_used: bool,
    notes: Sequence[str],
) -> None:
    envelope_payload = {k: v for k, v in payload.items() if k != "_envelope"}
    payload["_envelope"] = make_envelope(
        "factor_decomp",
        envelope_payload,
        validation=validation,
        fallback_used=fallback_used,
        notes=list(notes),
    )


def _payload(payload_or_envelope: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload_or_envelope, Mapping):
        return {}
    if payload_or_envelope.get("version") and payload_or_envelope.get("engine") == "factor_decomp":
        return unwrap(payload_or_envelope)
    return dict(payload_or_envelope)


def _normalise_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for key, value in weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight > 0:
            parsed[str(key)] = weight
    total = sum(parsed.values())
    if total <= 0:
        return {}
    if total > 1.5:
        parsed = {k: v / 100.0 for k, v in parsed.items()}
        total = sum(parsed.values())
    return {k: v / total for k, v in parsed.items()}


def _map_weights_to_proxies(weights: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw_key, weight in weights.items():
        ticker = _canonical_ticker(_ASSET_PROXY.get(str(raw_key).upper(), str(raw_key)))
        out[ticker] = out.get(ticker, 0.0) + float(weight)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else {}


def _canonical_ticker(ticker: str) -> str:
    t = str(ticker or "").strip()
    upper = t.upper()
    if upper in {"BTCUSD", "BTC-USD"}:
        return "BTC-USD"
    if upper in {"ETHUSD", "ETH-USD"}:
        return "ETH-USD"
    return t if t.startswith("^") else upper


def _prepare_returns_panel(returns_panel: Optional[Any], proxy_weights: Mapping[str, float]) -> pd.DataFrame:
    if returns_panel is not None:
        panel = _coerce_returns_panel(returns_panel, list(proxy_weights))
    else:
        panel = _load_cached_returns(set(proxy_weights) | {"SPY", "QQQ", "MDY", "VIG", "KSA", "EFID.CA", "TLT", "GLD"})
    return _add_synthetic_return_columns(panel)


def _coerce_returns_panel(panel: Any, weight_keys: Sequence[str]) -> pd.DataFrame:
    if isinstance(panel, pd.Series):
        df = panel.to_frame(name=weight_keys[0] if weight_keys else "portfolio")
    else:
        df = pd.DataFrame(panel).copy()
    if df.empty:
        return df
    if all(isinstance(c, int) for c in df.columns) and len(weight_keys) == len(df.columns):
        df.columns = list(weight_keys)
    df.columns = [_canonical_ticker(c) for c in df.columns]
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
    panel = get_returns_panel([t for t in tickers if t])
    return panel if panel is not None else pd.DataFrame()


def _add_synthetic_return_columns(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame()
    out = panel.copy()
    for col in list(out.columns):
        out[_canonical_ticker(col)] = out[col]
    if "SPY" not in out:
        numeric = out.select_dtypes(include=[np.number])
        if not numeric.empty:
            out["SPY"] = numeric.mean(axis=1)
    if "KSA" not in out and "^TASI" in out:
        out["KSA"] = out["^TASI"]
    if "^TASI" not in out and "KSA" in out:
        out["^TASI"] = out["KSA"]
    return out


def _prepare_factor_panel(
    *,
    factor_panel: Optional[Any],
    returns_panel: pd.DataFrame,
    required_factors: Sequence[str],
    notes: List[str],
) -> Tuple[pd.DataFrame, bool]:
    if factor_panel is not None:
        df = _coerce_factor_panel(factor_panel)
        df = _ensure_factor_columns(df, returns_panel, required_factors, notes)
        return df, False

    cached = _load_factor_cache(notes)
    if not cached.empty:
        cached = _ensure_factor_columns(cached, returns_panel, required_factors, notes)
        return cached, False

    synthetic = _synthetic_factor_panel(returns_panel, required_factors)
    if synthetic.empty:
        return synthetic, True
    notes.append("Ken French factor data unavailable - using synthetic factor panel")
    return synthetic, True


def _load_factor_cache(notes: List[str]) -> pd.DataFrame:
    try:
        from core.data_layer.factor_premia import factor_panel_path
        frames: List[pd.DataFrame] = []
        if _cache_file_fresh(_FACTOR_FILES["ff3"]):
            frames.append(pd.read_csv(factor_panel_path(_FACTOR_FILES["ff3"]), index_col=0, parse_dates=True))
        else:
            ff3 = _fetch_ken_french("ff3", notes)
            if not ff3.empty:
                frames.append(ff3)
        if _cache_file_fresh(_FACTOR_FILES["mom"]):
            frames.append(pd.read_csv(factor_panel_path(_FACTOR_FILES["mom"]), index_col=0, parse_dates=True))
        else:
            mom = _fetch_ken_french("mom", notes)
            if not mom.empty:
                frames.append(mom)
        if _cache_file_fresh(_FACTOR_FILES["ff5"]):
            frames.append(pd.read_csv(factor_panel_path(_FACTOR_FILES["ff5"]), index_col=0, parse_dates=True))
        else:
            ff5 = _fetch_ken_french("ff5", notes)
            if not ff5.empty:
                frames.append(ff5)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        return _coerce_factor_panel(out)
    except Exception as exc:
        notes.append(f"factor cache unavailable: {exc!r}")
        return pd.DataFrame()


def _cache_file_fresh(filename: str) -> bool:
    from core.data_layer.factor_premia import factor_panel_age_seconds
    age = factor_panel_age_seconds(filename)
    return age is not None and age < _FACTOR_TTL_SECONDS


def _fetch_ken_french(kind: str, notes: List[str]) -> pd.DataFrame:
    try:
        from core.data_layer.factor_premia import write_factor_panel
        url = _KEN_FRENCH_URLS[kind]
        with urllib.request.urlopen(url, timeout=6) as resp:  # noqa: S310 - fixed public data URLs
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            text = zf.read(name).decode("latin-1")
        parsed = _parse_ken_french_csv(text)
        if parsed.empty:
            return parsed
        write_factor_panel(_FACTOR_FILES[kind], parsed)
        return parsed
    except Exception as exc:
        notes.append(f"Ken French {kind} fetch unavailable: {exc!r}")
        return pd.DataFrame()


def _parse_ken_french_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        first = line.split(",", 1)[0].strip()
        if first == "" and "," in line:
            header_idx = i
            break
    if header_idx < 0:
        return pd.DataFrame()
    header = ["date"] + [h.strip() for h in lines[header_idx].split(",")[1:]]
    rows: List[List[str]] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            if rows:
                break
            continue
        first = line.split(",", 1)[0].strip()
        if not (len(first) == 6 and first.isdigit()):
            if rows:
                break
            continue
        vals = [v.strip() for v in line.split(",")]
        if len(vals) >= len(header):
            rows.append(vals[: len(header)])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    idx = pd.to_datetime(df.pop("date"), format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)
    df.index = idx
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return _coerce_factor_panel(df.dropna(how="all"))


def _coerce_factor_panel(panel: Any) -> pd.DataFrame:
    df = pd.DataFrame(panel).copy()
    if df.empty:
        return df
    rename: Dict[Any, str] = {}
    for col in df.columns:
        key = str(col).strip().upper().replace("_", "-")
        if key in {"MKT-RF", "MKT", "MARKET", "MARKET-RF"}:
            rename[col] = "MKT"
        elif key == "RF":
            rename[col] = "RF"
        elif key in {"MOM", "UMD", "MOMENTUM"}:
            rename[col] = "MOM"
        elif key in {"SMB", "HML", "RMW", "CMA", "QMJ"}:
            rename[col] = key
        elif key in {"BAB", "LOWVOL", "LOW-VOL", "LOW-VOLATILITY"}:
            rename[col] = "LOWVOL"
    df = df.rename(columns=rename)
    df = df[[c for c in df.columns if str(c) in {"MKT", "SMB", "HML", "MOM", "RMW", "CMA", "QMJ", "LOWVOL", "RF"}]]
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_abs = float(df.abs().max().max()) if not df.dropna(how="all").empty else 0.0
    if max_abs > 1.0:
        df = df / 100.0
    if not isinstance(df.index, pd.DatetimeIndex):
        return df.dropna(how="all")
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df[~df.index.isna()].sort_index().dropna(how="all")


def _ensure_factor_columns(
    df: pd.DataFrame,
    returns_panel: pd.DataFrame,
    required_factors: Sequence[str],
    notes: List[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "QMJ" in required_factors and "QMJ" not in out:
        if {"RMW", "CMA"}.issubset(out.columns):
            out["QMJ"] = 0.5 * out["RMW"] - 0.5 * out["CMA"]
        else:
            out["QMJ"] = 0.0
            notes.append("QMJ factor unavailable - neutral overlay used")
    if "LOWVOL" in required_factors and "LOWVOL" not in out:
        synth = _synthetic_factor_panel(returns_panel, ("LOWVOL",))
        if "LOWVOL" in synth:
            out = _align_by_position_or_index(out, synth[["LOWVOL"]])
        else:
            out["LOWVOL"] = 0.0
            notes.append("LowVol factor unavailable - neutral overlay used")
    missing = [f for f in required_factors if f not in out]
    if missing:
        synth = _synthetic_factor_panel(returns_panel, missing)
        for factor in missing:
            out[factor] = synth[factor].values[-len(out) :] if factor in synth and len(synth) >= len(out) else 0.0
    if "RF" not in out:
        out["RF"] = 0.0
    return out


def _synthetic_factor_panel(returns_panel: pd.DataFrame, required_factors: Sequence[str]) -> pd.DataFrame:
    if returns_panel is None or returns_panel.empty:
        return pd.DataFrame()
    panel = returns_panel.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = panel.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    mkt = numeric["SPY"] if "SPY" in numeric else numeric.mean(axis=1)
    out = pd.DataFrame(index=numeric.index)
    out["MKT"] = mkt.fillna(0.0)
    cols = list(numeric.columns)
    left = numeric[cols[: max(1, len(cols) // 2)]].mean(axis=1)
    right = numeric[cols[max(1, len(cols) // 2) :]].mean(axis=1) if len(cols) > 1 else mkt * 0.0
    out["SMB"] = (left - right).fillna(0.0)
    out["HML"] = (right - mkt).fillna(0.0)
    out["MOM"] = mkt.rolling(3, min_periods=1).sum().shift(1).fillna(0.0) / 3.0
    out["RMW"] = (mkt - out["SMB"]).fillna(0.0) * 0.35
    out["CMA"] = (out["HML"] - out["SMB"]).fillna(0.0) * 0.35
    out["QMJ"] = 0.5 * out["RMW"] - 0.5 * out["CMA"]
    out["LOWVOL"] = (-mkt.abs().rolling(3, min_periods=1).mean()).fillna(0.0)
    out["RF"] = 0.0
    return out[[c for c in list(required_factors) + ["RF"] if c in out]]


def _align_by_position_or_index(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if isinstance(left.index, pd.DatetimeIndex) and isinstance(right.index, pd.DatetimeIndex):
        return left.join(right, how="left")
    out = left.copy()
    for col in right.columns:
        vals = right[col].values
        out[col] = vals[-len(out) :] if len(vals) >= len(out) else np.nan
    return out


def _portfolio_returns(weights: Mapping[str, float], returns: pd.DataFrame) -> pd.Series:
    pieces: List[pd.Series] = []
    used_weights: List[float] = []
    for ticker, weight in weights.items():
        if ticker in returns:
            pieces.append(pd.to_numeric(returns[ticker], errors="coerce") * float(weight))
            used_weights.append(float(weight))
    if not pieces:
        return pd.Series(dtype=float)
    scale = sum(used_weights)
    if scale <= 0:
        return pd.Series(dtype=float)
    return (sum(pieces) / scale).rename("portfolio").dropna()


def _align_portfolio_and_factors(portfolio: pd.Series, factor_panel: pd.DataFrame, factors: Sequence[str]) -> pd.DataFrame:
    factor_cols = list(factors) + (["RF"] if "RF" in factor_panel else [])
    f = factor_panel[factor_cols].copy()
    if isinstance(portfolio.index, pd.DatetimeIndex) and isinstance(f.index, pd.DatetimeIndex):
        out = pd.concat([portfolio.rename("portfolio"), f], axis=1)
    else:
        n = min(len(portfolio), len(f))
        if n <= 0:
            return pd.DataFrame()
        p_tail = portfolio.tail(n).reset_index(drop=True).rename("portfolio")
        f_tail = f.tail(n).reset_index(drop=True)
        out = pd.concat([p_tail, f_tail], axis=1)
    if "RF" not in out:
        out["RF"] = 0.0
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["portfolio", *factors])


def _fit_factor_model(y_returns: pd.Series, aligned: pd.DataFrame, factors: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    y = (pd.to_numeric(y_returns, errors="coerce") - pd.to_numeric(aligned.get("RF", 0.0), errors="coerce").fillna(0.0)).to_numpy(dtype=float)
    x = aligned[list(factors)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[mask]
    x = x[mask]
    if y.size == 0 or x.size == 0:
        return ({f: 0.0 for f in factors}, {f: 0.0 for f in factors}, 0.0)
    try:
        import statsmodels.api as sm  # type: ignore

        res = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        params = np.asarray(res.params[1:], dtype=float)
        tvals = np.asarray(res.tvalues[1:], dtype=float)
        r2 = float(res.rsquared) if np.isfinite(res.rsquared) else 0.0
        return (
            {f: float(params[i]) for i, f in enumerate(factors)},
            {f: float(tvals[i]) if np.isfinite(tvals[i]) else 0.0 for i, f in enumerate(factors)},
            max(0.0, min(1.0, r2)),
        )
    except Exception:
        return _fit_factor_model_numpy(y, x, factors)


def _fit_factor_model_numpy(y: np.ndarray, x: np.ndarray, factors: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    x1 = np.column_stack([np.ones(len(x)), x])
    try:
        params, *_ = np.linalg.lstsq(x1, y, rcond=None)
    except np.linalg.LinAlgError:
        return ({f: 0.0 for f in factors}, {f: 0.0 for f in factors}, 0.0)
    fitted = x1 @ params
    resid = y - fitted
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-16 else 0.0
    dof = max(0, len(y) - x1.shape[1])
    tvals = np.zeros(len(params))
    if dof > 0:
        try:
            sigma2 = sse / dof
            cov = sigma2 * np.linalg.pinv(x1.T @ x1)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
            tvals = np.divide(params, se, out=np.zeros_like(params), where=se > 1e-12)
        except np.linalg.LinAlgError:
            tvals = np.zeros(len(params))
    return (
        {f: float(params[i + 1]) for i, f in enumerate(factors)},
        {f: float(tvals[i + 1]) if np.isfinite(tvals[i + 1]) else 0.0 for i, f in enumerate(factors)},
        max(0.0, min(1.0, float(r2) if np.isfinite(r2) else 0.0)),
    )


def _rolling_betas(aligned: pd.DataFrame, factors: Sequence[str], rolling_window: int) -> List[Dict[str, float]]:
    n = len(aligned)
    if n == 0:
        return []
    window = min(max(1, rolling_window), n)
    starts = range(0, n - window + 1)
    out: List[Dict[str, float]] = []
    for start in starts:
        chunk = aligned.iloc[start : start + window]
        beta, _, _ = _fit_factor_model(chunk["portfolio"], chunk, factors)
        out.append(beta)
    if not out:
        beta, _, _ = _fit_factor_model(aligned["portfolio"], aligned, factors)
        out.append(beta)
    return out


def _rolling_stability(rolling_betas: Sequence[Mapping[str, float]], factors: Sequence[str]) -> float:
    if len(rolling_betas) <= 1:
        return 1.0 if rolling_betas else 0.0
    cvs: List[float] = []
    for factor in factors:
        vals = np.asarray([float(b.get(factor, 0.0)) for b in rolling_betas], dtype=float)
        if vals.size == 0:
            continue
        denom = max(abs(float(np.mean(vals))), 0.25)
        cvs.append(float(np.std(vals, ddof=0) / denom))
    if not cvs:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(np.mean(cvs))))


def _sparse_region_shrinkage(
    *,
    raw_weights: Mapping[str, float],
    proxy_weights: Mapping[str, float],
    returns_panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    factors: Sequence[str],
    base_portfolio_beta: Mapping[str, float],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    diagnostics: Dict[str, Dict[str, Any]] = {}
    delta = {factor: 0.0 for factor in factors}
    for raw_ticker, raw_weight in raw_weights.items():
        proxy = _canonical_ticker(_ASSET_PROXY.get(str(raw_ticker).upper(), str(raw_ticker)))
        region = _region_for_ticker(raw_ticker, proxy)
        if region not in _REGION_PRIORS:
            continue
        if proxy not in returns_panel:
            diagnostics[proxy] = {"region": region, "observations": 0, "shrinkage_applied": False}
            continue
        asset_aligned = _align_portfolio_and_factors(returns_panel[proxy].dropna().rename("portfolio"), factor_panel, factors)
        n_obs = int(len(asset_aligned))
        if n_obs >= _MIN_INSTITUTIONAL_OBS or n_obs <= len(factors) + 1:
            diagnostics[proxy] = {"region": region, "observations": n_obs, "shrinkage_applied": False}
            continue
        raw_beta, _, _ = _fit_factor_model(asset_aligned["portfolio"], asset_aligned, factors)
        prior = _REGION_PRIORS[region]
        weight = n_obs / (n_obs + _SHRINKAGE_TAU)
        shrunk = {
            factor: weight * float(raw_beta.get(factor, 0.0)) + (1.0 - weight) * float(prior.get(factor, 0.0))
            for factor in factors
        }
        diagnostics[proxy] = {
            "region": region,
            "observations": n_obs,
            "shrinkage_applied": True,
            "tau": _SHRINKAGE_TAU,
            "raw_loadings": {f: round(float(raw_beta.get(f, 0.0)), 4) for f in factors},
            "shrunk_loadings": {f: round(float(shrunk.get(f, 0.0)), 4) for f in factors},
        }
        portfolio_weight = float(proxy_weights.get(proxy, raw_weight))
        for factor in factors:
            delta[factor] += portfolio_weight * (shrunk[factor] - float(raw_beta.get(factor, 0.0)))
    delta = {k: v for k, v in delta.items() if abs(v) > 1e-12}
    return diagnostics, delta


def _region_for_ticker(raw_ticker: str, proxy: str) -> Optional[str]:
    raw = str(raw_ticker or "").upper()
    prox = _canonical_ticker(proxy)
    if prox in _EGYPT_TICKERS or "EGYPT" in raw or raw.endswith(".CA"):
        return "Egypt"
    if prox in _GCC_TICKERS or any(token in raw for token in ("GCC", "KSA", "SAUDI", "UAE", "QATAR", "TADAWUL")):
        return "GCC"
    return None


def _contribution_return(beta: Mapping[str, float], window: pd.DataFrame, factors: Sequence[str]) -> Dict[str, float]:
    return {factor: float(beta.get(factor, 0.0)) * float(window[factor].mean()) * 12.0 for factor in factors}


def _contribution_vol(beta: Mapping[str, float], window: pd.DataFrame, factors: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for factor in factors:
        std = float(window[factor].std(ddof=1)) if len(window) > 1 else 0.0
        out[factor] = abs(float(beta.get(factor, 0.0))) * std * math.sqrt(12.0)
    return out


def _contribution_drawdown(beta: Mapping[str, float], window: pd.DataFrame, factors: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for factor in factors:
        series = float(beta.get(factor, 0.0)) * pd.to_numeric(window[factor], errors="coerce").fillna(0.0)
        wealth = (1.0 + series).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1.0
        out[factor] = float(dd.min()) if not dd.empty else 0.0
    return out


def _factor_warnings(beta: Mapping[str, float], t_stats: Mapping[str, float], stability: float) -> List[str]:
    warnings: List[str] = []
    for factor, value in beta.items():
        b = float(value)
        t = float(t_stats.get(factor, 0.0))
        if abs(b) > 1.8:
            warnings.append(f"Hidden leverage proxy: {factor} loading {b:.2f} exceeds 1.8")
        if abs(t) < 1.0 and abs(b) > 0.3:
            warnings.append(f"Unstable factor exposure: {factor} beta={b:.2f}, t={t:.2f}")
    if abs(float(beta.get("SMB", 0.0))) + abs(float(beta.get("HML", 0.0))) > 2.0:
        warnings.append("Material style concentration detected")
    if stability < 0.3:
        warnings.append("Style drift detected - factor loadings shifted materially within lookback window")
    if sum(1 for value in beta.values() if abs(float(value)) > 0.5) > 3:
        warnings.append("Multi-factor crowding - diversification of factor exposure is limited")
    return warnings


def _reliability_tier(window: int, r_squared: float, stability: float) -> str:
    if window >= _ROLLING_WINDOW and r_squared >= 0.7 and stability >= 0.6:
        return "Institutional"
    if window >= _MIN_INSTITUTIONAL_OBS and r_squared >= 0.5:
        return "Institutional-Lite"
    return "Indicative"


def _fmt_signed_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100.0:+.2f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_abs_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        return f"{abs(float(value)) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "-"


__all__ = [
    "ENGINE_VERSION",
    "SUPPORTED_MODELS",
    "compute_factor_decomposition",
    "render_factor_decomposition_md",
]
