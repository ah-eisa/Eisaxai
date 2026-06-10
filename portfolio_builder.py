"""
portfolio_builder.py — EisaX Portfolio Build Engine
====================================================
Parses Arabic/English "build me a portfolio" requests and routes them
to global_allocator (real QP optimizer) instead of Gemini hallucinations.

Usage (from market_route_handler.py):
    from portfolio_builder import detect_and_build
    result = detect_and_build(message)
    if result:
        return result   # ready-to-send reply string
"""
import re
import logging

logger = logging.getLogger("eisax.portfolio_builder")

# ──────────────────────────────────────────────────────────────────────────────
# Intent Detection
# ──────────────────────────────────────────────────────────────────────────────

_BUILD_SIGNALS_AR = [
    "ابنى", "ابني", "إبني", "ابن لى", "ابن لي",
    "اعمل محفظه", "اعمل محفظة", "إعمل محفظة",
    "عايز محفظه", "عايز محفظة", "عاوز محفظة", "عاوز محفظه",
    "ابنيلى محفظة", "ابنيلي محفظة",
    "بناء محفظة", "تصميم محفظة", "صمملي محفظة",
    "انشئ محفظة", "أنشئ محفظة",
]
_BUILD_SIGNALS_EN = [
    "build me a portfolio", "build a portfolio", "create a portfolio",
    "construct a portfolio", "design a portfolio", "build portfolio",
    "make me a portfolio", "set up a portfolio",
]
_BUILD_VERBS_EN = [
    "build", "create", "construct", "design", "make", "generate",
    "allocate", "rebalance", "re-balance", "set up", "setup",
]
_BUILD_CONTEXT_EN = [
    "portfolio", "allocation", "equities", "stocks", "risk tolerance",
    "max drawdown", "max down", "drawdown", "aggressive",
    "conservative", "balanced", "moderate", "saudi", "us equities",
    "american equities", "gcc",
]
_BUILD_CONTEXT_AR = [
    "محفظة", "محفظه", "خصص", "خصّص", "موازنة", "أعد موازنة", "اعد موازنة",
    "عدواني", "عدوانيه", "محافظ", "متوازن", "مخاطرة", "مخاطر",
]


def _is_build_request(message: str) -> bool:
    ml = message.lower()
    if any(sig in ml for sig in _BUILD_SIGNALS_AR):
        return True
    if any(sig in ml for sig in _BUILD_SIGNALS_EN):
        return True

    has_build_verb = any(sig in ml for sig in _BUILD_VERBS_EN)
    has_build_context = any(sig in ml for sig in _BUILD_CONTEXT_EN) or any(sig in message for sig in _BUILD_CONTEXT_AR)
    return has_build_verb and has_build_context


# ──────────────────────────────────────────────────────────────────────────────
# Parameter Extraction
# ──────────────────────────────────────────────────────────────────────────────

def _parse_params(message: str) -> dict:
    ml = message.lower()

    # ── Capital ────────────────────────────────────────────────────────────────
    capital = 100_000
    m = re.search(r"(\d+)\s*(?:مليار|billion|b\b)", ml)
    if m:
        capital = int(m.group(1)) * 1_000_000_000
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:مليون|million|m\b)", ml)
        if m:
            capital = int(float(m.group(1)) * 1_000_000)
        else:
            m = re.search(r"(\d[\d,]+)", message)
            if m:
                try:
                    capital = int(m.group(1).replace(",", ""))
                except ValueError:
                    pass
    capital = max(10_000, min(capital, 100_000_000_000))

    # ── Horizon ────────────────────────────────────────────────────────────────
    horizon = 5
    m = re.search(r"(\d+)\s*(?:سنه|سنة|سنين|سنوات|year|years|yr)", ml)
    if m:
        horizon = int(m.group(1))

    # ── Risk Profile ───────────────────────────────────────────────────────────
    profile = "balanced"
    if any(w in ml for w in ["عدواني", "عدوانيه", "aggressive", "speculative", "مضاربه", "مضاربة"]):
        profile = "aggressive"
    elif any(w in ml for w in ["نمو", "growth", "dynamic"]):
        profile = "growth"
    elif any(w in ml for w in ["محافظ", "conservative", "stable", "آمن", "امن", "منخفض المخاطر"]):
        profile = "conservative"
    # Horizon-based adjustment
    if horizon <= 3 and profile == "balanced":
        profile = "conservative"
    elif horizon >= 7 and profile == "balanced":
        profile = "growth"

    # ── Regions to Include ─────────────────────────────────────────────────────
    include = []

    if (any(w in ml for w in ["امريكي", "أمريكي", "امريكية", "أمريكية", "us stock", "american", "nasdaq", "s&p", "اسهم امريكيه", "اسهم امريكية"])
            or re.search(r'\b(?:us|usa|u\.s\.|u\.s\.a\.|united\s+states)\b', ml)):
        include.append("US")

    if any(w in ml for w in ["سعودي", "سعوديه", "سعودية", "saudi", "ksa", "تداول", "ارامكو", "تادول"]):
        if "GCC" not in include:
            include.append("GCC")
    if any(w in ml for w in ["اماراتي", "إماراتي", "امارات", "إمارات", "uae", "dubai", "دبي", "ابوظبي", "أبوظبي", "إماراتيه"]):
        if "GCC" not in include:
            include.append("GCC")

    if any(w in ml for w in ["مصري", "مصرية", "egypt", "egx"]):
        include.append("Egypt")

    if any(w in ml for w in ["ذهب", "معادن", "gold", "metal", "فضه", "فضة", "silver", "معدن"]):
        include.append("Gold")

    if any(w in ml for w in ["سندات", "bond", "sukuk", "صكوك", "fixed income", "دخل ثابت"]):
        include.append("Bonds")

    if any(w in ml for w in ["بيتكوين", "كريبتو", "bitcoin", "crypto", "ethereum"]):
        include.append("Crypto")

    if any(w in ml for w in [
        "commodity", "commodities", "سلع", "خامات",
        "نفط", "بترول", "oil", "crude",
        "نحاس", "copper", "raw material", "مواد خام",
    ]):
        if "Commodities" not in include:
            include.append("Commodities")
        if "Gold" not in include:
            include.append("Gold")

    if not include:
        include = ["US", "GCC", "Gold"]   # sensible default

    # ── Max Drawdown ───────────────────────────────────────────────────────────
    max_dd = 1.0   # default = unconstrained
    m = re.search(r"(?:max(?:imum)?\s*(?:draw\s*down|dd|loss|خسارة|هبوط))\s*[:\=]?\s*(\d+)\s*%?", ml)
    if m:
        max_dd = int(m.group(1)) / 100.0
    else:
        m = re.search(r"(?:down|drawdown|أقصى|حد أقصى)\s+(\d+)\s*%?", ml)
        if m:
            max_dd = int(m.group(1)) / 100.0

    # ── Regions to Exclude ─────────────────────────────────────────────────────
    exclude = []
    if "Crypto" not in include:
        exclude.append("Crypto")

    # ── Custom region caps: "up to X% each" / "X% max" near metals/crypto/gold ─
    # Matches patterns like:
    #   "metals and crypto but up to 5% each"
    #   "crypto max 3%"
    #   "gold up to 10%"
    custom_caps: dict[str, float] = {}
    _CAP_TARGETS = {
        "Crypto":      [r"crypto", r"bitcoin", r"كريبتو", r"بيتكوين"],
        "Gold":        [r"metal", r"metals", r"gold", r"silver", r"ذهب", r"معادن", r"فضه", r"فضة"],
        "Commodities": [r"commodit(?:y|ies)", r"oil", r"copper", r"سلع", r"نفط", r"نحاس"],
        "Bonds":       [r"bonds?", r"سندات", r"صكوك", r"sukuk"],
    }
    _cap_regex = re.compile(r"(?:up\s+to\s+|max(?:imum)?\s+|حد\s+|حدّ\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:each|only|max|كحد أقصى|لكل واحدة)?", re.IGNORECASE)
    for _region, _patterns in _CAP_TARGETS.items():
        if not any(re.search(p, ml, re.IGNORECASE) for p in _patterns):
            continue
        # Look for "X%" within 40 chars of any pattern match
        for _p in _patterns:
            for _m in re.finditer(_p, ml, re.IGNORECASE):
                _window_start = max(0, _m.start() - 5)
                _window_end   = min(len(ml), _m.end() + 40)
                _window = ml[_window_start:_window_end]
                _pct_match = _cap_regex.search(_window)
                if _pct_match:
                    _pct = float(_pct_match.group(1)) / 100.0
                    if 0 < _pct < 1:
                        custom_caps[_region] = min(custom_caps.get(_region, 1.0), _pct)
                        break
            if _region in custom_caps:
                break

    return {
        "capital":       capital,
        "profile":       profile,
        "horizon":       horizon,
        "include":       include,
        "exclude":       exclude,
        "max_drawdown":  max_dd,
        "custom_caps":   custom_caps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CIO Sub-Agent
# ──────────────────────────────────────────────────────────────────────────────

def _get_cio_insight(params: dict, result: dict, language: str = "en") -> str:
    import os
    import requests
    
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not ds_key:
        return ""
        
    metrics = result.get("metrics", {})
    regions = " + ".join(params["include"]) if params["include"] else "Global"
    
    _regime_label = (result.get("regime") or {}).get("classification", "—")
    prompt = f"""You are an institutional CIO writing a synthesis paragraph to accompany a deterministic portfolio construction.

CLIENT MANDATE:
- Capital: ${params['capital']:,.0f}
- Horizon: {params['horizon']} years
- Risk Profile: {params['profile'].title()}
- Target Markets: {regions}
- Portfolio Regime: {_regime_label}

PORTFOLIO METRICS:
- Expected Annual Return: ~{metrics.get('expected_return_pct')}%
- Expected Volatility: ~{metrics.get('expected_vol_pct')}%
- Sharpe Ratio: ~{metrics.get('sharpe')}

ALLOCATION (asset → weight %):
{result.get('asset_weights', {})}

Write a concise Institutional CIO synthesis (max 130 words) in {"Arabic" if language == "ar" else "English"}.
TONE: BlackRock / MSCI Barra / Morningstar Direct. Calm, precise, factor-aware.
DO NOT use retail wording. AVOID: "Why it works", "Main risk", "Good timing", "Strong setup", "return enhancer", "AI mega-cap momentum", "Top risk", "Excellent / Weak / Poor".
USE instead: "Investment Rationale", "Primary Risk Vector", "Entry conditions", "Risk-adjusted profile", "Concentration sensitivity", "Factor exposure", "Scenario dependency".

Structure (use these exact headings):
1. **Investment Rationale** — 2–3 sentences on how the construction maps to the client mandate and {params['horizon']}-year horizon (factor exposure, regime fit, diversification logic).
2. **Primary Risk Vector** — 1–2 sentences identifying the single most material risk and its scenario dependency.

Output in Markdown. Do not hallucinate tickers or invent metrics.
"""

    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-v4-flash",
                "messages": [
                    {"role": "system", "content": f"You are an institutional CIO answering in {'Arabic' if language == 'ar' else 'English'}. Be direct, numbers-first."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.3
            },
            timeout=15
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"].strip()
            # properly format blockquotes
            lines = content.split('\\n')
            formatted_content = '\\n'.join([f"> {ln}" if ln.strip() else ">" for ln in lines])
            if language == "ar":
                _heading = "F. طبقة التعليق بالذكاء الاصطناعي — نظرة مدير الاستثمار"
                _subnote = "*تعليق مولّد بالذكاء الاصطناعي. الأقسام A–E أعلاه قائمة على حسابات قابلة للتكرار.*"
            else:
                _heading = "F. AI Commentary Layer — CIO Synthesis"
                _subnote = "*AI-generated synthesis. Sections A–E above are deterministic and reproducible from the optimizer state.*"
            return f"\n## {_heading}\n\n{_subnote}\n\n{formatted_content}\n\n"
    except Exception as e:
        logger.warning(f"CIO Insight failed: {e}")
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Allocator Runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_allocator(params: dict, language: str = "en") -> str:
    from global_allocator import allocate
    from datetime import datetime

    capital     = params["capital"]
    profile     = params["profile"]
    horizon     = params["horizon"]
    include     = params["include"] or None
    exclude     = params["exclude"] or None
    max_drawdown= params.get("max_drawdown", 1.0)
    custom_caps = params.get("custom_caps") or None

    result = allocate(
        profile        = profile,
        region_include = include,
        region_exclude = exclude,
        custom_caps    = custom_caps,
        port_value_usd = capital,
        max_drawdown   = max_drawdown,
        language       = language,
    )

    if "error" in result:
        if language == "ar":
            return (
                f"⚠️ **لم أتمكن من بناء المحفظة**\n\n"
                f"**السبب:** {result['error']}\n\n"
                f"**اقتراح:** جرّب إضافة أسواق أكثر مثل الذهب أو السندات، أو غيّر مستوى المخاطرة."
            )
        return (
            f"⚠️ **Could not build the portfolio**\n\n"
            f"**Reason:** {result['error']}\n\n"
            f"**Suggestion:** Try adding more markets (e.g. Gold or Bonds), or adjust your risk profile / drawdown limit."
        )

    metrics       = result.get("metrics", {})
    profile_label = result.get("profile_label", profile.title())
    report_md     = result.get("report_md", "")
    ret_pct       = metrics.get("expected_return_pct", 0)
    vol_pct       = metrics.get("expected_vol_pct", 0)
    sharpe        = metrics.get("sharpe", 0)
    beta          = metrics.get("beta_world", 0)

    constraint_diag      = result.get("constraint_diagnostics", []) or []
    rebalance_sugs       = result.get("rebalance_suggestions", []) or []
    audit                = result.get("audit", {}) or {}
    regime               = result.get("regime", {}) or {}
    confidence           = result.get("confidence", {}) or {}
    adaptive_disclosures = result.get("adaptive_disclaimers", []) or []
    asset_roles          = result.get("asset_roles", {}) or {}
    implementation       = result.get("implementation", {}) or {}
    benchmark            = result.get("benchmark", {}) or {}
    attribution          = result.get("attribution", {}) or {}

    # Format capital
    if capital >= 1_000_000_000:
        cap_str = f"${capital/1_000_000_000:.1f}B"
    elif capital >= 1_000_000:
        cap_str = f"${capital/1_000_000:.1f}M"
    else:
        cap_str = f"${capital:,.0f}"

    projected = capital * ((1 + ret_pct / 100) ** horizon)
    gain      = projected - capital

    now_str  = datetime.now().strftime("%B %d, %Y")
    regions  = " + ".join(include) if include else "Global"

    # Text severity tags (institutional terminology, no retail wording)
    ret_tag    = "[STRONG]"     if ret_pct > 10 else ("[MODERATE]" if ret_pct > 6 else "[LOW]")
    vol_tag    = "[LOW]"        if vol_pct < 12 else ("[MODERATE]" if vol_pct < 20 else "[HIGH]")
    sharpe_tag = "[STRONG]"     if sharpe > 1.2 else ("[GOOD]" if sharpe > 0.8 else ("[ACCEPTABLE]" if sharpe > 0.5 else "[BELOW MANDATE]"))
    beta_tag   = "[LOW]"        if beta < 0.7 else ("[MODERATE]" if beta < 1.1 else "[HIGH]")

    # Phase D: reduced fake precision — round to 1 dp and prefix with "~"
    def _approx(value, dp=1, suffix=""):
        return f"~{value:.{dp}f}{suffix}"

    # Regime + confidence blocks for Section A
    _regime_class    = regime.get("classification", "—")
    _regime_impl     = regime.get("implication", "")
    _regime_behavior = regime.get("regime_behavior", "")
    _conf_pct     = confidence.get("score_pct", 0)
    _conf_breadth = confidence.get("evidence_breadth", "—")
    _conf_cov     = confidence.get("coverage_quality", "—")
    _conf_tier    = confidence.get("reliability_tier", "—")
    # Confidence tier badge
    if _conf_tier == "Institutional":
        _tier_tag = "[STRONG]"
    elif _conf_tier == "Institutional-Lite":
        _tier_tag = "[MODERATE]"
    else:
        _tier_tag = "[LOW]"

    # Phase E — Implementation feasibility + deployability badges
    _impl_complexity = implementation.get("rebalancing_complexity", "—")
    _impl_liquidity  = implementation.get("liquidity_practicality", "—")
    _impl_friction   = implementation.get("execution_friction", "—")
    _impl_turnover   = implementation.get("est_turnover_pct", 0)
    _impl_slippage   = implementation.get("est_slippage_bp", 0)
    _impl_deploy     = implementation.get("deployability_score", 0)
    _impl_tier       = implementation.get("deployability_tier", "—")
    _impl_note       = implementation.get("institutional_note", "")
    if _impl_tier == "High":
        _deploy_tag = "[STRONG]"
    elif _impl_tier == "Moderate":
        _deploy_tag = "[MODERATE]"
    else:
        _deploy_tag = "[LOW]"
    # Friction badges
    def _impl_tag(level):
        return {"Low": "[LOW]", "Moderate": "[MODERATE]", "High": "[HIGH]",
                "Limited": "[LOW]"}.get(level, f"[{level}]")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION A — Executive Summary
    # ──────────────────────────────────────────────────────────────────────
    if language == "ar":
        section_a = f"""# EisaX Global Portfolio — {profile_label}
**التاريخ:** {now_str}  |  **رأس المال:** {cap_str}  |  **المدة:** {horizon} سنوات  |  **الأسواق:** {regions}

---

## A. الملخص التنفيذي

| المؤشر | القيمة | التقييم |
|--------|--------|---------|
| العائد المتوقع (سنوي) | **{_approx(ret_pct, 1, '%')}** | {ret_tag} |
| التقلب المتوقع | **{_approx(vol_pct, 1, '%')}** | {vol_tag} |
| Sharpe Ratio | **{_approx(sharpe, 2)}** | {sharpe_tag} |
| Beta (vs MSCI World) | **{_approx(beta, 2)}** | {beta_tag} |
| القيمة المتوقعة بعد {horizon} سنوات | **${projected:,.0f}** | ربح متوقع **${gain:,.0f}** |

**تصنيف نظام المحفظة:** **{_regime_class}**
> {_regime_impl}
> **سلوك المحفظة مقابل المؤشر:** {_regime_behavior}

**Confidence Calibration** · Score: **{_conf_pct:.0f}%** · Evidence Breadth: **{_conf_breadth}** · Coverage Quality: **{_conf_cov}** · Reliability Tier: **{_conf_tier}** {_tier_tag}

**Implementation Feasibility** · Deployability: **{_impl_tier}** {_deploy_tag} ({_impl_deploy}/100) · Rebalancing Complexity: **{_impl_complexity}** {_impl_tag(_impl_complexity)} · Liquidity: **{_impl_liquidity}** {_impl_tag(_impl_liquidity)} · Execution Friction: **{_impl_friction}** {_impl_tag(_impl_friction)} · Est. Turnover ~{_impl_turnover:.0f}%/yr · Est. Slippage ~{_impl_slippage:.0f} bp

**Benchmark Context** · Reference: **{benchmark.get('label', '—')}** · Bench Return ~{benchmark.get('expected_ret_pct', 0):.1f}% · Tracking Deviation: **{benchmark.get('tracking_class', '—')}** {_impl_tag(benchmark.get('tracking_class', '—'))} ({benchmark.get('tracking_error_pct', 0):.1f}% TE) · Active Share: **{benchmark.get('active_class', '—')}** {_impl_tag(benchmark.get('active_class', '—'))} ({benchmark.get('active_share_pct', 0):.0f}%) · Style Drift: **{benchmark.get('style_drift', '—')}**

> *الأرقام تقريبية ومبنية على افتراضات تاريخية طويلة المدى. لا تُعدّ ضماناً للأداء المستقبلي.*

---

"""
    else:
        section_a = f"""# EisaX Global Portfolio — {profile_label}
**Date:** {now_str}  |  **Capital:** {cap_str}  |  **Horizon:** {horizon} years  |  **Markets:** {regions}

---

## A. Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Expected Return (annual) | **{_approx(ret_pct, 1, '%')}** | {ret_tag} |
| Expected Volatility | **{_approx(vol_pct, 1, '%')}** | {vol_tag} |
| Sharpe Ratio | **{_approx(sharpe, 2)}** | {sharpe_tag} |
| Beta (vs MSCI World) | **{_approx(beta, 2)}** | {beta_tag} |
| Projected Value in {horizon} years | **${projected:,.0f}** | Expected gain **${gain:,.0f}** |

**Portfolio Regime:** **{_regime_class}**
> {_regime_impl}
> **Regime Behavior vs Benchmark:** {_regime_behavior}

**Confidence Calibration** · Score: **{_conf_pct:.0f}%** · Evidence Breadth: **{_conf_breadth}** · Coverage Quality: **{_conf_cov}** · Reliability Tier: **{_conf_tier}** {_tier_tag}

**Implementation Feasibility** · Deployability: **{_impl_tier}** {_deploy_tag} ({_impl_deploy}/100) · Rebalancing Complexity: **{_impl_complexity}** {_impl_tag(_impl_complexity)} · Liquidity: **{_impl_liquidity}** {_impl_tag(_impl_liquidity)} · Execution Friction: **{_impl_friction}** {_impl_tag(_impl_friction)} · Est. Turnover ~{_impl_turnover:.0f}%/yr · Est. Slippage ~{_impl_slippage:.0f} bp

**Benchmark Context** · Reference: **{benchmark.get('label', '—')}** · Bench Return ~{benchmark.get('expected_ret_pct', 0):.1f}% · Tracking Deviation: **{benchmark.get('tracking_class', '—')}** {_impl_tag(benchmark.get('tracking_class', '—'))} ({benchmark.get('tracking_error_pct', 0):.1f}% TE) · Active Share: **{benchmark.get('active_class', '—')}** {_impl_tag(benchmark.get('active_class', '—'))} ({benchmark.get('active_share_pct', 0):.0f}%) · Style Drift: **{benchmark.get('style_drift', '—')}**

> *Values are approximate, derived from long-run historical assumptions. Not a guarantee of future performance.*

---

"""

    # ──────────────────────────────────────────────────────────────────────
    # SECTION B — Mandate Feasibility Analysis
    # ──────────────────────────────────────────────────────────────────────
    _b_title = "B. تحليل جدوى التفويض" if language == "ar" else "B. Mandate Feasibility Analysis"
    _b_intro = (
        "تحقق من القيود المُفعَّلة قبل التحسين. كل قيد يُعرض مع القيمة الفعلية والحالة."
        if language == "ar" else
        "Constraints enforced during optimization, with active value and status. "
        "Feasible solution requires all constraints in PASS / NEAR CAP / AUTO-RELAXED."
    )
    _b_header_row = ("| القيد | الحد | الفعلي | الحالة |" if language == "ar"
                     else "| Constraint | Limit | Actual | Status |")
    section_b_lines = [
        f"## {_b_title}",
        "",
        f"> {_b_intro}",
        "",
        _b_header_row,
        "|------------|-------|--------|--------|",
    ]
    _STATUS_TAG = {
        "PASS":            "[PASS]",
        "NEAR CAP":        "[NEAR CAP]",
        "AT CAP":          "[AT CAP]",
        "AT FLOOR":        "[AT FLOOR]",
        "BREACH":          "[BREACH]",
        "LOW DIVERSIFICATION": "[LOW DIVERSIFICATION]",
    }
    for c in constraint_diag:
        _name   = c.get("name", "")
        _lim    = c.get("limit_pct")
        _act    = c.get("actual_pct")
        _stat   = c.get("status", "")
        _stat_tag = _STATUS_TAG.get(_stat, f"[{_stat}]" if _stat else "[—]")
        # format limit / actual numerically when possible
        if isinstance(_lim, (int, float)):
            _lim_str = f"{_lim:.2f}" if abs(_lim) < 5 else f"{_lim:.1f}%"
        else:
            _lim_str = str(_lim)
        if isinstance(_act, (int, float)):
            _act_str = f"{_act:.2f}" if abs(_act) < 5 else f"{_act:.1f}%"
        else:
            _act_str = str(_act)
        section_b_lines.append(f"| {_name} | {_lim_str} | {_act_str} | {_stat_tag} |")
    section_b_lines.append("")
    section_b = "\n".join(section_b_lines)

    # ──────────────────────────────────────────────────────────────────────
    # SECTION C + D from allocator's report_md (already structured)
    # ──────────────────────────────────────────────────────────────────────
    section_cd = report_md

    # ──────────────────────────────────────────────────────────────────────
    # SECTION E — Rebalancing Plan (quantified actionability)
    # ──────────────────────────────────────────────────────────────────────
    _e_title = "E. خطة إعادة التوازن" if language == "ar" else "E. Rebalancing Plan"
    if rebalance_sugs:
        _e_intro = (
            "إجراءات مُحدَّدة لتقليل التركز، مع الأثر الكمّي على بيتا، التقلب، ودوران المحفظة."
            if language == "ar" else
            "Targeted reduction candidates with quantified impact on portfolio beta, volatility, and turnover."
        )
        _e_header_row = ("| الإجراء | بيتا قبل→بعد | تقلب قبل→بعد | تقليل التركز | الدوران | الصعوبة |"
                         if language == "ar" else
                         "| Proposed Action | Beta Before→After | Vol Before→After | Concentration Δ | Turnover | Difficulty |")
        section_e_lines = [
            f"## {_e_title}",
            "",
            f"> {_e_intro}",
            "",
            _e_header_row,
            "|-----------------|-------------------|------------------|-----------------|----------|------------|",
        ]
        for s in rebalance_sugs:
            _act_str = f"Reduce {s['asset_name']} ({s['proxy']}) {s['weight_before_pct']:.1f}% → {s['weight_after_pct']:.1f}%"
            _beta_str = f"{s['beta_before']:.3f} → {s['beta_after']:.3f} ({s['beta_delta']:+.3f})"
            _vol_str  = f"{s['vol_before_pct']:.2f}% → {s['vol_after_pct']:.2f}% ({s['vol_delta_pp']:+.2f}pp)"
            _conc_str = f"−{s['concentration_delta_pp']:.1f}pp"
            _tov_str  = f"{s['turnover_pp']:.1f}pp"
            _diff_str = f"[{s['implementation_difficulty']}]"
            section_e_lines.append(f"| {_act_str} | {_beta_str} | {_vol_str} | {_conc_str} | {_tov_str} | {_diff_str} |")
        section_e_lines.append("")
        section_e_lines.append(
            "> *Impact estimates assume pro-rata redistribution of the trimmed weight across remaining holdings.*"
            if language != "ar" else
            "> *تقدير الأثر مبني على إعادة توزيع الوزن المُخفَّض على باقي الأصول بالتناسب.*"
        )
        section_e_lines.append("")
        section_e = "\n".join(section_e_lines)
    else:
        _e_note = (
            "لا توجد مراكز مركزة تفوق 15% — لا حاجة لإعادة توازن فورية."
            if language == "ar" else
            "No concentrated positions above 15% — no immediate rebalancing action required."
        )
        section_e = f"## {_e_title}\n\n> {_e_note}\n\n"

    # ──────────────────────────────────────────────────────────────────────
    # SECTION F — AI Commentary Layer
    # ──────────────────────────────────────────────────────────────────────
    section_f = _get_cio_insight(params, result, language=language)

    # ──────────────────────────────────────────────────────────────────────
    # SECTION G — Audit Appendix
    # ──────────────────────────────────────────────────────────────────────
    _g_title = "G. ملحق المراجعة" if language == "ar" else "G. Audit Appendix"
    _cv = audit.get("constraint_values", {}) or {}
    _custom_caps_str = ", ".join(f"{k} ≤ {v}%" for k, v in _cv.get("custom_caps", {}).items()) or "—"
    if language == "ar":
        section_g = f"""## {_g_title}

| الحقل | القيمة |
|-------|--------|
| معرّف اللقطة (Snapshot ID) | `{audit.get('snapshot_id', '—')}` |
| هاش الكون الاستثماري | `{audit.get('universe_hash', '—')}` |
| الـ Solver | {audit.get('solver_primary', '—')} |
| حالة الـ Solver | {audit.get('solver_status', '—')} |
| عدد الأصول (الكون) | {audit.get('n_assets_universe', 0)} |
| عدد الأصول (المختارة) | {audit.get('n_assets_selected', 0)} |
| Max Beta | {_cv.get('max_beta', '—')} |
| Max Volatility | {_cv.get('max_vol_pct', '—')}% |
| Min Bonds + Cash | {_cv.get('min_bonds_cash_pct', '—')}% |
| Max Drawdown (مطلوب) | {_cv.get('max_drawdown_pct') or 'غير محدد'}% |
| Risk Aversion | {_cv.get('risk_aversion', '—')} |
| Risk-Free Rate | {_cv.get('rf_rate_pct', '—')}% |
| القيود المخصصة | {_custom_caps_str} |

> *قابل للتكرار: نفس المدخلات → نفس Snapshot ID → نفس النتيجة. لا تعديلات صامتة.*

### قيود النموذج — الحدود الهيكلية لمحرك التحليل

{chr(10).join('- ' + _mc for _mc in audit.get('model_constraints', []))}

> *ملاحظة شفافية: القيود أعلاه متأصلة في منهجية المحاكاة التاريخية لبناء المحافظ. عُرضت بشكل صريح لدعم المراجعة المؤسسية والحوكمة.*
"""
    else:
        section_g = f"""## {_g_title}

| Field | Value |
|-------|-------|
| Snapshot ID | `{audit.get('snapshot_id', '—')}` |
| Universe Hash | `{audit.get('universe_hash', '—')}` |
| Solver | {audit.get('solver_primary', '—')} |
| Solver Status | {audit.get('solver_status', '—')} |
| Assets (Universe) | {audit.get('n_assets_universe', 0)} |
| Assets (Selected) | {audit.get('n_assets_selected', 0)} |
| Max Beta | {_cv.get('max_beta', '—')} |
| Max Volatility | {_cv.get('max_vol_pct', '—')}% |
| Min Bonds + Cash | {_cv.get('min_bonds_cash_pct', '—')}% |
| Max Drawdown (Requested) | {_cv.get('max_drawdown_pct') or 'Unconstrained'}% |
| Risk Aversion | {_cv.get('risk_aversion', '—')} |
| Risk-Free Rate | {_cv.get('rf_rate_pct', '—')}% |
| Custom Caps | {_custom_caps_str} |

> *Reproducible: same inputs → same Snapshot ID → identical output. Zero silent corrections.*

### Model Constraints — Structural Limitations of the Engine

{chr(10).join('- ' + _mc for _mc in audit.get('model_constraints', []))}

> *Transparency note: the constraints above are inherent to historical-simulation portfolio engineering. Surfaced explicitly to support institutional review and governance.*
"""

    # ──────────────────────────────────────────────────────────────────────
    # Implementation steps (compact — moved under Section E hint)
    # ──────────────────────────────────────────────────────────────────────
    if language == "ar":
        impl_block = f"""
### خطوات التنفيذ المقترحة

1. افتح حساب وساطة مناسب للأسواق: {regions}
2. وزّع {cap_str} حسب نسب الأصول في القسم D
3. أعد التوازن كل 6-12 شهر
4. تجنّب التصفية في تراجعات السوق قصيرة المدى — أفق {horizon} سنوات يستوعب دورات السوق

"""
    else:
        impl_block = f"""
### Implementation Steps

1. Open a brokerage account for target markets: {regions}
2. Allocate {cap_str} per the asset weights in Section D
3. Rebalance every 6–12 months
4. Avoid liquidating into short-term drawdowns — the {horizon}-year horizon is designed to absorb interim market cycles

"""

    # Combine all sections in order: A · B · C+D · E · F · G
    _final_md = (
        section_a
        + section_b
        + "\n---\n\n"
        + section_cd
        + "\n---\n\n"
        + section_e
        + impl_block
        + "\n---\n\n"
        + section_f
        + "\n---\n\n"
        + section_g
    )

    # ── Phase H — inject computed subsections into the full A-G report ──
    # Engines were run inside allocate(); payloads attached to `result`.
    # We perform markdown injection here so that:
    #   - H1 (benchmark relative)  goes under C
    #   - H4 (factor decomposition) goes under C
    #   - H2 (execution efficiency) goes under E
    #   - H3 (forward scenarios)    becomes ## H., placed before G
    #   - H5 (committee brief)      becomes ## I., placed before G
    #   - audit appendix gets the reproducibility block appended at the end
    try:
        from phase_h.orchestrator import inject_phase_h_sections
        _final_md = inject_phase_h_sections(result, _final_md, language=language)
    except Exception as _ph_exc:
        # Never let Phase H injection break the Phase G report
        import logging as _lg
        _lg.getLogger("portfolio_builder").warning(
            "phase_h injection skipped: %r", _ph_exc
        )

    return _final_md


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_and_build(message: str, language: str = "en") -> str | None:
    """
    Main entry point.
    Returns a formatted markdown reply string if message is a portfolio build request,
    or None if it's not a build request.
    """
    if not _is_build_request(message):
        return None
    try:
        params = _parse_params(message)
        logger.info(
            "[PortfolioBuilder] Build request detected: capital=%s profile=%s horizon=%d regions=%s",
            params["capital"], params["profile"], params["horizon"], params["include"]
        )
        return _run_allocator(params, language=language)
    except Exception as e:
        logger.error("[PortfolioBuilder] Failed: %s", e, exc_info=True)
        return None
