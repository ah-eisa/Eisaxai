"""
EisaX PDF Export Template
تحويل الـ markdown لـ HTML احترافي قابل للطباعة كـ PDF
"""
import re
from datetime import datetime


EISAX_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a2e;
    background: #fff;
}

.page {
    max-width: 800px;
    margin: 0 auto;
    padding: 40px;
}

/* ── Header ── */
.report-header {
    background: linear-gradient(135deg, #1e3a8a, #2563eb, #1d4ed8);
    color: white;
    padding: 30px 35px;
    border-radius: 12px;
    margin-bottom: 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.report-header .brand {
    font-size: 22pt;
    font-weight: 700;
    letter-spacing: 2px;
    color: #00d4ff;
}

.report-header .meta {
    text-align: right;
    font-size: 9pt;
    color: rgba(255,255,255,0.75);
    line-height: 1.8;
}

.report-header .ticker-badge {
    font-size: 28pt;
    font-weight: 700;
    color: white;
}

.live-price {
    background: #1a1a2e;
    color: #00ff88;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 13pt;
    font-weight: 600;
    margin-top: 10px;
    display: inline-block;
}

/* ── Sections ── */
h1 { font-size: 18pt; color: #0f0c29; margin: 25px 0 10px; border-bottom: 3px solid #00d4ff; padding-bottom: 6px; }
h2 { font-size: 14pt; color: #302b63; margin: 20px 0 8px; padding-left: 10px; border-left: 4px solid #00d4ff; }
h3 { font-size: 12pt; color: #1a1a2e; margin: 15px 0 6px; font-weight: 600; }

p { margin-bottom: 10px; }
ul, ol { margin: 8px 0 8px 20px; }
li { margin-bottom: 4px; }

strong { font-weight: 600; color: #0f0c29; }

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 10pt;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

th {
    background: #302b63;
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 9.5pt;
    letter-spacing: 0.5px;
}

td {
    padding: 9px 14px;
    border-bottom: 1px solid #eef0f4;
    vertical-align: middle;
}

tr:nth-child(even) td { background: #f8f9ff; }
tr:last-child td { border-bottom: none; }

/* ── Score Card ── */
.scorecard {
    background: linear-gradient(135deg, #0f0c29, #302b63);
    color: white;
    border-radius: 12px;
    padding: 25px 30px;
    margin: 20px 0;
}

.scorecard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.scorecard-title {
    font-size: 16pt;
    font-weight: 700;
    color: #00d4ff;
}

.score-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: rgba(255,255,255,0.1);
    border: 3px solid #00d4ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20pt;
    font-weight: 700;
    color: white;
}

.factor-bar {
    margin: 8px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.factor-name {
    width: 160px;
    font-size: 9.5pt;
    color: rgba(255,255,255,0.85);
    flex-shrink: 0;
}

.factor-track {
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.15);
    border-radius: 5px;
    overflow: hidden;
}

.factor-fill {
    height: 100%;
    border-radius: 5px;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
}

.factor-fill.low { background: linear-gradient(90deg, #ff4444, #ff6644); }
.factor-fill.mid { background: linear-gradient(90deg, #ffaa00, #ffcc44); }

.factor-pct {
    width: 40px;
    text-align: right;
    font-size: 9.5pt;
    font-weight: 600;
    color: white;
}

/* ── Verdict Badge ── */
.verdict {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 11pt;
    letter-spacing: 1px;
}
.verdict-buy { background: #00ff88; color: #0f0c29; }
.verdict-hold { background: #ffaa00; color: #0f0c29; }
.verdict-sell { background: #ff4444; color: white; }

/* ── Fact Check ── */
.factcheck {
    background: #f0f9ff;
    border: 1px solid #00d4ff;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 15px 0;
}

.factcheck-title {
    font-weight: 700;
    color: #302b63;
    margin-bottom: 10px;
    font-size: 11pt;
}

/* ── Positioning Guide ── */
.positioning {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 15px 0;
}

.pos-card {
    border-radius: 8px;
    padding: 15px;
    text-align: center;
}

.pos-card.entry { background: #e8fff4; border: 2px solid #00cc66; }
.pos-card.target { background: #fff8e8; border: 2px solid #ffaa00; }
.pos-card.stop { background: #fff0f0; border: 2px solid #ff4444; }

.pos-label { font-size: 8.5pt; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.pos-price { font-size: 14pt; font-weight: 700; }

.pos-card.entry .pos-label { color: #00aa55; }
.pos-card.entry .pos-price { color: #00aa55; }
.pos-card.target .pos-label { color: #cc8800; }
.pos-card.target .pos-price { color: #cc8800; }
.pos-card.stop .pos-label { color: #cc2222; }
.pos-card.stop .pos-price { color: #cc2222; }

/* ── Status indicators ── */
.status-ok { color: #00aa55; font-weight: 700; }
.status-add { color: #0088cc; font-weight: 700; }

/* ── Risk items ── */
.risk-high { color: #cc2222; font-weight: 600; }
.risk-med  { color: #cc8800; font-weight: 600; }
.risk-low  { color: #00aa55; font-weight: 600; }

/* ── Footer ── */
.footer {
    margin-top: 40px;
    padding-top: 15px;
    border-top: 2px solid #eef0f4;
    font-size: 8pt;
    color: #888;
    text-align: center;
}

/* ── Page break ── */
@media print {
    .no-break { page-break-inside: avoid; }
    .page-break { page-break-after: always; }
}

hr { border: none; border-top: 2px solid #eef0f4; margin: 20px 0; }

blockquote {
    border-left: 4px solid #00d4ff;
    padding: 10px 15px;
    background: #f0f9ff;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
    font-style: italic;
    color: #444;
}
"""


def _parse_factor_bars(html: str) -> str:
    """حوّل الـ █░ bars لـ visual CSS bars."""
    def replace_bar_row(m):
        factor = m.group(1).strip()
        pct_str = m.group(2).strip().rstrip('%')
        try:
            pct = int(pct_str)
        except Exception as _e:
            pct = 50
        color_class = "low" if pct < 50 else ("mid" if pct < 75 else "")
        return f"""<div class="factor-bar">
            <div class="factor-name">{factor}</div>
            <div class="factor-track"><div class="factor-fill {color_class}" style="width:{pct}%"></div></div>
            <div class="factor-pct">{pct}%</div>
        </div>"""

    # Pattern: | Factor Name | XX% | emoji `bar` |
    html = re.sub(
        r'\|\s*([^|]+?)\s*\|\s*(\d+)%\s*\|\s*[🟢🟡🔴]?\s*`[█░]+`\s*\|',
        replace_bar_row,
        html
    )
    return html


def _parse_positioning(html: str) -> str:
    """حوّل الـ positioning table لـ cards."""
    def replace_pos(m):
        rows = m.group(0)
        entry = re.search(r'Entry.*?\$([\d,\.]+)', rows)
        target = re.search(r'Target.*?\$([\d,\.]+).*?(\+[\d\.]+%)', rows)
        stop = re.search(r'Stop.*?\$([\d,\.]+)', rows)

        entry_price = entry.group(1) if entry else "N/A"
        target_price = f"${target.group(1)} {target.group(2)}" if target else "N/A"
        stop_price = stop.group(1) if stop else "N/A"

        return f"""<div class="positioning">
            <div class="pos-card entry">
                <div class="pos-label">🟢 Entry Zone</div>
                <div class="pos-price">${entry_price}</div>
            </div>
            <div class="pos-card target">
                <div class="pos-label">🎯 Price Target</div>
                <div class="pos-price">{target_price}</div>
            </div>
            <div class="pos-card stop">
                <div class="pos-label">🔴 Stop Loss</div>
                <div class="pos-price">${stop_price}</div>
            </div>
        </div>"""

    html = re.sub(
        r'📊.*?Positioning Guide.*?(?=\n#{1,3}|\n---|\Z)',
        replace_pos,
        html,
        flags=re.S
    )
    return html


def _extract_header_info(markdown_text: str) -> dict:
    """استخرج المعلومات الأساسية من التقرير."""
    info = {
        'ticker': 'N/A',
        'price': 'N/A',
        'sector': 'N/A',
        'score': 'N/A',
        'verdict': 'BUY',
        'eisax_score': '',
        'conviction': '',
    }

    t = re.search(r'Intelligence Report:\s*([A-Z]{2,6})', markdown_text)
    if t: info['ticker'] = t.group(1)

    p = re.search(r'Live Price.*?\$([\d,\.]+)', markdown_text)
    if p: info['price'] = '$' + p.group(1)

    s = re.search(r'Sector:\s*([^\|<\n]+)', markdown_text)
    if s: info['sector'] = s.group(1).strip()

    q = re.search(r'Quality Score.*?(\d+)/100', markdown_text)
    if q: info['score'] = q.group(1)

    es = re.search(r'EisaX Score.*?(\d+)/100', markdown_text)
    if es: info['eisax_score'] = es.group(1)

    v = re.search(r'VERDICT:\s*([\w ]+)', markdown_text, re.I)
    if v: info['verdict'] = v.group(1).strip()

    c = re.search(r'Conviction.*?:\s*\*\*([\w\-]+)\*\*', markdown_text)
    if c: info['conviction'] = c.group(1)

    return info


def build_eisax_html(markdown_text: str, title: str = "EisaX Report") -> str:
    """
    حوّل الـ markdown لـ HTML احترافي.
    """
    import markdown as md
    from markdown.extensions.tables import TableExtension

    info = _extract_header_info(markdown_text)
    today = datetime.now().strftime("%B %d, %Y")

    verdict_class = "verdict-buy"
    if "SELL" in info['verdict'].upper() or "REDUCE" in info['verdict'].upper():
        verdict_class = "verdict-sell"
    elif "HOLD" in info['verdict'].upper():
        verdict_class = "verdict-hold"

    # تحويل الـ markdown لـ HTML
    body_html = md.markdown(
        markdown_text,
        extensions=['tables', 'fenced_code', 'nl2br']
    )

    # تحسين الـ HTML
    body_html = _parse_factor_bars(body_html)

    # تحسين الـ status emojis في الجداول
    body_html = body_html.replace('>✅<', '><span class="status-ok">✅ Verified</span><')
    body_html = body_html.replace('>➕<', '><span class="status-add">➕ Added</span><')

    # تحسين الـ risk severity
    body_html = re.sub(r'\(Severity:\s*High\)', '<span class="risk-high">(Severity: High)</span>', body_html)
    body_html = re.sub(r'\(Severity:\s*Medium[^)]*\)', '<span class="risk-med">\\g<0></span>', body_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{EISAX_CSS}</style>
</head>
<body>
<div class="page">

<!-- HEADER -->
<div class="report-header">
    <div>
        <div class="brand">EisaX</div>
        <div style="color:rgba(255,255,255,0.6);font-size:9pt;margin-top:4px">AI Investment Intelligence | Abu Dhabi</div>
        {f'<div class="ticker-badge">{info["ticker"]}</div>' if info["ticker"] != "N/A" else ""}
        {f'<div class="live-price">{info["price"]}</div>' if info["price"] != "N/A" else ""}
    </div>
    <div class="meta">
        <div style="font-size:11pt;font-weight:600;color:white">{today}</div>
        <div>Sector: {info["sector"]}</div>
        {f'<div>Quality Score: {info["score"]}/100</div>' if info["score"] != "N/A" else ""}
        {f'<div style="margin-top:8px"><span class="verdict {verdict_class}">{info["verdict"]}</span></div>' if info["verdict"] else ""}
        {f'<div style="color:rgba(255,255,255,0.6);font-size:9pt">Conviction: {info["conviction"]}</div>' if info["conviction"] else ""}
    </div>
</div>

<!-- BODY -->
<div class="no-break">
{body_html}
</div>

<!-- FOOTER -->
<div class="footer">
    <p><strong>EisaX AI Investment Intelligence</strong> | Generated {today} | Abu Dhabi, UAE</p>
    <p>This report is for informational purposes only and does not constitute financial advice. Past performance is not indicative of future results.</p>
</div>

</div>
</body>
</html>"""

    return html
