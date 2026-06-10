"""
Analytics Intent Router — EisaX.
Detects natural-language analytics intents in chat messages and routes to
the appropriate analytics module using the user's stored portfolio.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

log = logging.getLogger("eisax.analytics_intent_router")

# ── Intent patterns (Arabic + English) ────────────────────────────────────────
# Each entry: (intent_id, list of regex patterns)
INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("shariah", [
        r"\b(shariah|sharia|halal|haram|اسلامي|إسلامي|شرعي|حلال|حرام|الفحص الشرعي|تدقيق شرعي)\b",
    ]),
    ("macro_sim", [
        r"\b(macro\s*sim|macroeconomic|اقتصاد كلي|محاكاة اقتصاد|سيناريو اقتصادي)\b",
    ]),
    ("forward_scenario", [
        r"\b(forward\s*scenario|forward looking|توقع مستقبلي|سيناريو مستقبلي|توقعات الأشهر|projection)\b",
    ]),
    ("monte_carlo", [
        r"\b(monte\s*carlo|var|cvar|value at risk|قيمة المخاطرة|محاكاة احتمالية|مونت\s*كارلو)\b",
    ]),
    ("regimes", [
        r"\b(market regime|regime|bull and bear|أنظمة السوق|نظام السوق|صعودي وهبوطي|stagflation|ركود تضخمي)\b",
    ]),
    ("optimize", [
        r"\b(optimize portfolio|optimization|efficient frontier|markowitz|sharpe|تحسين المحفظة|الحدود الكفؤة|أوزان مثالية)\b",
    ]),
    ("dividend_income", [
        r"\b(dividend|yield on cost|payout|توزيعات|أرباح موزعة|دخل توزيعات|عائد التوزيعات)\b",
    ]),
    ("budget_plan", [
        r"\b(budget plan|budget allocation|كم سهم اشتري|خطة ميزانية|ميزانية الاستثمار|توزيع ميزانية)\b",
    ]),
]


def detect_intent(message: str) -> Optional[str]:
    """Return matching intent_id or None."""
    if not message:
        return None
    msg_lower = message.lower()
    for intent_id, patterns in INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, msg_lower, re.IGNORECASE):
                return intent_id
    return None


def _load_user_portfolio(user_id: str) -> tuple[pd.DataFrame, float, str]:
    """
    Load user's stored portfolio and return (positions_df, total_value, error_msg).
    Falls back to empty if not found.
    """
    try:
        from core.portfolio_tracker import PortfolioTracker
        tracker = PortfolioTracker()
        data = tracker.get_portfolio(str(user_id))
        positions = data.get("positions", [])
        if not positions:
            return pd.DataFrame(), 0.0, "no_positions"

        rows = []
        total_val = 0.0
        for pos in positions:
            ticker = pos.get("ticker", "")
            shares = float(pos.get("shares", 0) or 0)
            cost   = float(pos.get("purchase_price", 0) or 0)

            # Get current price via market_data
            try:
                from core.market_data import get_full_stock_profile
                profile = get_full_stock_profile(ticker)
                price = float(profile.get("price", cost) or cost)
                sector = profile.get("sector", "Unknown") or "Unknown"
                name = profile.get("name", ticker) or ticker
            except Exception:
                price = cost
                sector = "Unknown"
                name = ticker

            value = price * shares
            total_val += value
            rows.append({
                "ticker":     ticker,
                "name":       name,
                "sector":     sector,
                "qty":        shares,
                "price":      price,
                "cost_basis": cost,
                "value":      value,
            })
        return pd.DataFrame(rows), total_val, ""
    except Exception as e:
        log.warning(f"[intent_router] load portfolio failed: {e}")
        return pd.DataFrame(), 0.0, str(e)


def _no_portfolio_reply(lang: str = "ar") -> str:
    if lang == "ar":
        return ("⚠️ لم أجد محفظة محفوظة لحسابك. أضف صفقاتك أولاً من صفحة المحفظة، "
                "ثم اطلب التحليل مرة أخرى.")
    return ("⚠️ I couldn't find a saved portfolio for your account. "
            "Please add your positions from the portfolio page first, then ask again.")


def _format_money(v) -> str:
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


# ── Per-intent handlers ────────────────────────────────────────────────────────

def handle_shariah(df: pd.DataFrame, lang: str) -> str:
    from core.shariah_screener import screen_portfolio
    result = screen_portfolio(df)

    title = "🕌 الفحص الشرعي للمحفظة" if lang == "ar" else "🕌 Portfolio Shariah Screening"
    lines = [f"## {title}\n",
             f"**{result['summary']}**\n",
             f"- {'نسبة الامتثال' if lang=='ar' else 'Compliance Rate'}: **{result['compliance_rate_pct']:.1f}%**",
             f"- ✅ {'حلال' if lang=='ar' else 'Halal'}: {result['halal_count']}",
             f"- ❌ {'حرام' if lang=='ar' else 'Haram'}: {result['haram_count']}",
             f"- ❓ {'غير محدد' if lang=='ar' else 'Unknown'}: {result['unknown_count']}",
             f"- 💰 {'تقدير التطهير' if lang=='ar' else 'Purification'}: {_format_money(result['purification_estimate'])}",
             ""]
    if not result["results"].empty:
        lines.append("| Ticker | Verdict | Value | Sector | Issues |")
        lines.append("|---|---|---|---|---|")
        for row in result["results"].to_dict(orient="records"):
            lines.append(f"| {row['Ticker']} | {row['Verdict']} | {_format_money(row['Value'])} | {row['Sector']} | {row['Issues']} |")
    return "\n".join(lines)


def handle_macro_sim(df: pd.DataFrame, total: float, lang: str) -> str:
    from core.macro_simulator import MacroScenario, simulate_portfolio
    # Use default baseline + a moderately optimistic scenario
    scen = MacroScenario.from_defaults()
    result = simulate_portfolio(df, total, scen)

    title = "🌍 محاكاة الاقتصاد الكلي (الوضع الحالي)" if lang == "ar" else "🌍 Macro Simulation (Current Conditions)"
    lines = [f"## {title}\n",
             f"- {'التأثير الكلي' if lang=='ar' else 'Total Impact'}: **{result['total_impact_pct']:+.2f}%**",
             f"- {'تغيير القيمة' if lang=='ar' else 'Value Change'}: {_format_money(result['total_impact_value'])}",
             f"- {'القيمة الجديدة' if lang=='ar' else 'New Value'}: {_format_money(result['new_portfolio_value'])}",
             ""]
    lines.append(f"**{'تأثير الاقتصاد على القطاعات' if lang=='ar' else 'Sector Impacts'}:**")
    significant = {k: v for k, v in result["sector_impacts"].items() if abs(v) > 0.05}
    if not significant:
        lines.append(f"_{'لا تأثير قطاعي ملحوظ' if lang=='ar' else 'No significant sector impact'}_")
    else:
        for sec, imp in sorted(significant.items(), key=lambda x: -abs(x[1]))[:8]:
            emoji = "📈" if imp > 0 else "📉"
            lines.append(f"- {emoji} **{sec}**: {imp:+.2f}%")
    lines.append("")
    lines.append(_bi_note(
        "💡 لتغيير الافتراضات الاقتصادية، استخدم لوحة التحكم في صفحة المحفظة.",
        "💡 To customize macro assumptions, use the Portfolio tab controls.", lang))
    return "\n".join(lines)


def handle_forward_scenario(df: pd.DataFrame, total: float, lang: str) -> str:
    from core.macro_simulator import MacroScenario
    from core.scenario_builder import build_forward_scenario
    scen = MacroScenario.from_defaults()
    result = build_forward_scenario(df, total, scen)

    title = "🔭 توقعات المحفظة" if lang == "ar" else "🔭 Forward Projection"
    lines = [f"## {title}\n",
             f"_{result['scenario_label']}_\n",
             f"| {'الأفق' if lang=='ar' else 'Horizon'} | {'القيمة المتوقعة' if lang=='ar' else 'Projected Value'} | {'التغيير' if lang=='ar' else 'Change'} |",
             "|---|---|---|"]
    for h in [3, 6, 12]:
        hd = result["horizons"].get(h, {})
        emoji = "📈" if hd.get("pct_change", 0) >= 0 else "📉"
        lines.append(f"| {h} {'شهور' if lang=='ar' else 'months'} | {_format_money(hd.get('projected_value', total))} | {emoji} {hd.get('pct_change', 0):+.1f}% |")
    return "\n".join(lines)


def handle_monte_carlo(df: pd.DataFrame, total: float, lang: str) -> str:
    from core.monte_carlo import run_portfolio_monte_carlo
    result = run_portfolio_monte_carlo(df, total, n_simulations=3000, horizon_days=252)

    title = "🎲 مونت كارلو / قيمة المخاطرة" if lang == "ar" else "🎲 Monte Carlo / VaR"
    lines = [f"## {title}\n",
             f"_{'محاكاة 3000 مسار لمدة سنة' if lang=='ar' else '3,000 simulations · 1-year horizon'}_\n",
             f"- **VaR 95%**: {result['var'].get(0.95, 0):+.2f}%",
             f"- **VaR 99%**: {result['var'].get(0.99, 0):+.2f}%",
             f"- **CVaR 95%**: {result['cvar'].get(0.95, 0):+.2f}%",
             f"- {'احتمال خسارة' if lang=='ar' else 'P(loss)'} >10%: **{result['prob_loss_gt_threshold']:.1f}%**",
             "",
             f"**{'نتائج محتملة' if lang=='ar' else 'Possible Outcomes'}:**",
             f"- 🟢 {'أفضل (P90)' if lang=='ar' else 'Best (P90)'}: {_format_money(result['best_outcome'])}",
             f"- 🟡 {'متوسط (P50)' if lang=='ar' else 'Median (P50)'}: {_format_money(result['median_outcome'])}",
             f"- 🔴 {'أسوأ (P10)' if lang=='ar' else 'Worst (P10)'}: {_format_money(result['worst_outcome'])}"]
    return "\n".join(lines)


def handle_regimes(df: pd.DataFrame, total: float, lang: str) -> str:
    from core.market_regimes import compare_regimes
    result = compare_regimes(df, total, horizon_months=12)

    title = "🌐 أداء المحفظة في أنظمة السوق" if lang == "ar" else "🌐 Portfolio Under Market Regimes"
    lines = [f"## {title}\n",
             f"| {'النظام' if lang=='ar' else 'Regime'} | {'العائد' if lang=='ar' else 'Return'} | {'القيمة المتوقعة' if lang=='ar' else 'Proj Value'} |",
             "|---|---|---|"]
    for rname in ["bull", "sideways", "bear", "stagflation"]:
        rd = result["regimes"].get(rname, {})
        label = rd.get("label_ar" if lang == "ar" else "label_en", rname)
        ret = rd.get("expected_return_pct", 0)
        val = rd.get("projected_value", total)
        lines.append(f"| {label} | {ret:+.1f}% | {_format_money(val)} |")
    best  = result["regimes"][result["best_regime"]]["label_en"]
    worst = result["regimes"][result["worst_regime"]]["label_en"]
    spread = result["regime_spread_pct"]
    lines.append(f"\n**{'الفارق' if lang=='ar' else 'Spread'}**: {spread:.1f}% ({'بين' if lang=='ar' else 'between'} {best} ↔ {worst})")
    return "\n".join(lines)


def handle_optimize(df: pd.DataFrame, lang: str) -> str:
    from core.portfolio_optimizer import optimize_portfolio
    result = optimize_portfolio(df, objective="max_sharpe")
    if result.get("error"):
        return f"⚠️ {result['error']}"

    title = "📊 تحسين المحفظة (أقصى نسبة شارب)" if lang == "ar" else "📊 Portfolio Optimization (Max Sharpe)"
    cs = result["current_stats"]
    os = result["optimal_stats"]
    imp = result["improvement"]
    lines = [f"## {title}\n",
             f"| | {'الحالي' if lang=='ar' else 'Current'} | {'المثالي' if lang=='ar' else 'Optimal'} |",
             "|---|---|---|",
             f"| {'العائد %' if lang=='ar' else 'Return %'} | {cs['return']:+.2f}% | {os['return']:+.2f}% |",
             f"| {'التذبذب %' if lang=='ar' else 'Volatility %'} | {cs['volatility']:.2f}% | {os['volatility']:.2f}% |",
             f"| Sharpe | {cs['sharpe']:.3f} | {os['sharpe']:.3f} |",
             "",
             f"**{'التحسن' if lang=='ar' else 'Improvement'}**: Sharpe Δ {imp['sharpe_lift']:+.3f}, "
             f"{'عائد' if lang=='ar' else 'return'} {imp['return_lift']:+.2f}%"]
    if result["rebalance_actions"] is not None and not result["rebalance_actions"].empty:
        lines.append(f"\n**{'إعادة التوازن' if lang=='ar' else 'Rebalance'}:**")
        lines.append(f"| Ticker | {'الحالي' if lang=='ar' else 'Current'} | {'المثالي' if lang=='ar' else 'Optimal'} | {'الإجراء' if lang=='ar' else 'Action'} |")
        lines.append("|---|---|---|---|")
        for row in result["rebalance_actions"].to_dict(orient="records"):
            lines.append(f"| {row['Ticker']} | {row['Current %']}% | {row['Optimal %']}% | {row['Action']} |")
    return "\n".join(lines)


def handle_dividend_income(df: pd.DataFrame, lang: str) -> str:
    from core.dividend_engine import project_portfolio_income
    result = project_portfolio_income(df)

    title = "💸 دخل الأرباح الموزعة" if lang == "ar" else "💸 Dividend Income"
    lines = [f"## {title}\n",
             f"- {'الدخل السنوي' if lang=='ar' else 'Annual Income'}: **{_format_money(result['total_annual_income'])}**",
             f"- {'المتوسط الشهري' if lang=='ar' else 'Monthly Avg'}: {_format_money(result['monthly_average_income'])}",
             f"- {'عائد المحفظة' if lang=='ar' else 'Portfolio Yield'}: **{result['portfolio_yield_pct']:.2f}%**"]
    if result.get("yield_on_cost_pct") is not None:
        lines.append(f"- {'العائد على التكلفة' if lang=='ar' else 'Yield on Cost'}: **{result['yield_on_cost_pct']:.2f}%**")
    if result.get("weighted_growth_rate") is not None:
        lines.append(f"- {'نمو التوزيعات (5 سنوات)' if lang=='ar' else '5y Growth Rate'}: {result['weighted_growth_rate']:+.2f}%")
    lines.append(f"- {'استدامة التوزيعات' if lang=='ar' else 'Sustainability'}: **{result['sustainability_score']}/100**")
    return "\n".join(lines)


def _bi_note(ar: str, en: str, lang: str) -> str:
    return ar if lang == "ar" else en


def _detect_lang(message: str) -> str:
    """Quick AR/EN heuristic."""
    arabic_chars = sum(1 for c in message if "؀" <= c <= "ۿ")
    return "ar" if arabic_chars >= 2 else "en"


# ── Main entry point ──────────────────────────────────────────────────────────
def route_message(message: str, user_id: str) -> Optional[str]:
    """
    Detect analytics intent and return a markdown reply, or None if no match.

    Args:
        message: The user's chat message.
        user_id: User ID for portfolio lookup.

    Returns:
        Markdown reply string, or None if no analytics intent matched.
    """
    intent = detect_intent(message)
    if not intent:
        return None

    lang = _detect_lang(message)

    df, total, err = _load_user_portfolio(user_id)
    if df.empty:
        return _no_portfolio_reply(lang)

    try:
        if intent == "shariah":
            return handle_shariah(df, lang)
        if intent == "macro_sim":
            return handle_macro_sim(df, total, lang)
        if intent == "forward_scenario":
            return handle_forward_scenario(df, total, lang)
        if intent == "monte_carlo":
            return handle_monte_carlo(df, total, lang)
        if intent == "regimes":
            return handle_regimes(df, total, lang)
        if intent == "optimize":
            return handle_optimize(df, lang)
        if intent == "dividend_income":
            return handle_dividend_income(df, lang)
        if intent == "budget_plan":
            return _bi_note(
                "💰 لإنشاء خطة ميزانية، استخدم لوحة 'مخطط الميزانية' في صفحة المحفظة "
                "حيث تحدد المبلغ والأوزان المستهدفة.",
                "💰 To build a budget plan, use the Budget Planner in your portfolio page "
                "where you can input the amount and target weights.",
                lang
            )
    except Exception as e:
        log.exception(f"[intent_router] handler failed for {intent}")
        return _bi_note(
            f"⚠️ حدث خطأ في تحليل {intent}: {e}",
            f"⚠️ Error in {intent} analysis: {e}",
            lang
        )
    return None
