# core/services/report_engine.py
import json
from typing import Dict, Any, List

def build_smart_daily_report(market_data: Dict, news_headlines: List[str], portfolio_context: str = "") -> Dict:
    """
    يبني تقريراً ذكياً يعتمد على ربط الأحداث وليس على قوالب جافة.
    """
    
    # 1. استخراج أهم الإشارات (Signals) من البيانات الخام
    signals = extract_key_signals(market_data)
    
    # 2. ربط الإشارات ببعضها لتكوين "رؤية" (Narrative)
    narrative = build_narrative(signals, news_headlines)
    
    # 3. ترجمة الرؤية إلى قرارات استثمارية قابلة للتنفيذ
    decisions = translate_to_decisions(narrative, portfolio_context)
    
    return {
        "signals": signals,
        "narrative": narrative,
        "decisions": decisions,
        "full_report": format_as_markdown(narrative, decisions)
    }

def extract_key_signals(data: Dict) -> Dict:
    """يحول الأرقام إلى إشارات نوعية."""
    signals = {}
    
    # مثال: إشارة النفط
    oil_change = data.get("oil", {}).get("d1_pct", 0)
    if oil_change < -5:
        signals["oil_crash"] = {"severity": "high", "impact": "deflationary", "winners": ["transport", "airlines"], "losers": ["energy"]}
    elif oil_change > 5:
        signals["oil_spike"] = {"severity": "high", "impact": "inflationary", "winners": ["energy"], "losers": ["manufacturing"]}
    
    # مثال: إشارة الذهب
    gold_change = data.get("gold", {}).get("d1_pct", 0)
    if gold_change > 1:
        signals["flight_to_safety"] = {"severity": "medium", "message": "المستثمرون يتحوطون ضد مخاطر جيوسياسية أو تضخمية"}
    
    # أضف المزيد من الإشارات (VIX, DXY, Bonds, etc.)
    return signals

def build_narrative(signals: Dict, headlines: List) -> str:
    """يبني سردية (قصة) واحدة من الإشارات المتعددة."""
    narrative_parts = []
    
    if signals.get("oil_crash"):
        narrative_parts.append(f"هبوط حاد في النفط ({signals['oil_crash']['severity']}) يضغط على قطاع الطاقة لكنه يخفض تكاليف الإنتاج والنقل، مما يفيد الشركات كثيفة الاستهلاك للطاقة.")
    
    if signals.get("flight_to_safety"):
        narrative_parts.append(f"ارتفاع الذهب يشير إلى {signals['flight_to_safety']['message']}. هذا يعني أن السوق لا يزال خائفاً رغم أي مكاسب في المؤشرات.")
    
    if not narrative_parts:
        narrative_parts.append("الأسواق في حالة انتظار. المؤشرات الرئيسية مستقرة ولا توجد إشارة اتجاهية واضحة.")
    
    # ربط الأخبار (مثال)
    if any("Hormuz" in h for h in headlines):
        narrative_parts.append("تصريحات مضاربة حول مضيق هرمز تخلق تقلبات مؤقتة، لكن التأثير على الأصول الحقيقية محدود طالما لم تُغلق المضيق فعلياً.")
    
    return " ".join(narrative_parts)

def translate_to_decisions(narrative: str, portfolio_context: str) -> List[str]:
    """يترجم السردية إلى أوامر تنفيذية."""
    decisions = []
    
    if "هبوط حاد في النفط" in narrative:
        decisions.append("✅ الإجراء: زيادة الوزن النسبي في أسهم النقل والخدمات اللوجستية (مثال: قم بشراء Amazon أو DP World).")
        decisions.append("❌ الإجراء: خفض التعرض لشركات الطاقة (مثال: قم ببيع أو تقليل ARAMCO).")
    
    if "ارتفاع الذهب" in narrative:
        decisions.append("🛡️ التحوط: احتفظ بنسبة 5-10% من المحفظة في الذهب أو صناديق المؤشرات المرتبطة به كتأمين ضد الهبوط.")
    
    if "حالة انتظار" in narrative:
        decisions.append("⏳ التوصية: لا تتخذ أي إجراء جريء. حافظ على السيولة النقدية (Cash) بنسبة 15% على الأقل.")
        decisions.append("👀 المراقبة: راقب مستوى VIX 20. إذا كسره، كن مستعداً لتقليل المخاطر.")
    
    if not decisions:
        decisions.append("📊 التوصية: أعد توازن المحفظة نحو الأهداف الاستراتيجية (60% أسهم، 30% سندات، 10% سيولة).")
    
    return decisions

def format_as_markdown(narrative: str, decisions: List[str]) -> str:
    """يحول المخرجات إلى Markdown جاهز للعرض."""
    md = "## 📈 قراءة السوق\n\n"
    md += narrative + "\n\n"
    md += "## 🎯 قرارات استثمارية فورية\n\n"
    for d in decisions:
        md += f"- {d}\n"
    return md