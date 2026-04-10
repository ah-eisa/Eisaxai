import logging
import sys, re, os
import uuid as _uuid_pdf
from datetime import datetime
logger = logging.getLogger(__name__)
sys.path.insert(0, '/home/ubuntu/investwise')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

NAVY       = colors.HexColor('#0D1B2A')
GOLD       = colors.HexColor('#C9A84C')
LIGHT_GOLD = colors.HexColor('#F5E6C8')
GRAY       = colors.HexColor('#6B7280')
LIGHT_GRAY = colors.HexColor('#F9FAFB')
WHITE      = colors.white
RED        = colors.HexColor('#DC2626')
GREEN      = colors.HexColor('#16A34A')
BORDER     = colors.HexColor('#E5E7EB')

# ── Arabic font registration ───────────────────────────────────────────────────
_ARABIC_FONT      = "Helvetica"        # fallback if TTF missing
_ARABIC_FONT_BOLD = "Helvetica-Bold"
_ARABIC_FONT_PATH = "/home/ubuntu/investwise/static/fonts/Arabic.ttf"
try:
    pdfmetrics.registerFont(TTFont("EisaXArabic", _ARABIC_FONT_PATH))
    _ARABIC_FONT      = "EisaXArabic"
    _ARABIC_FONT_BOLD = "EisaXArabic"   # same TTF — ReportLab will use it for bold too
    logger.debug("Arabic font registered: EisaXArabic")
except Exception as _fe:
    logger.warning("Arabic font not loaded (%s) — falling back to Helvetica", _fe)


# ── Arabic helpers ─────────────────────────────────────────────────────────────

def _has_arabic(text: str) -> bool:
    """Return True if text contains Arabic characters."""
    return bool(re.search(
        r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]',
        text))


def _reshape_arabic(text: str) -> str:
    """
    Apply arabic_reshaper + bidi so Arabic displays correctly in PDFs.
    1. reshape  — connect letters into proper joined forms / ligatures
    2. get_display — reorder to RTL visual order via Unicode BiDi algorithm
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def clean_text(text: str, arabic_mode: bool = False) -> str:
    """
    Clean text for PDF rendering.
    - arabic_mode=False : strip non-ASCII (safe for English / Helvetica)
    - arabic_mode=True  : preserve Arabic Unicode range + reshape for RTL
    """
    if arabic_mode and _has_arabic(text):
        # Strip only emojis / surrogates, keep Arabic + punctuation + digits
        text = re.sub(
            r'[^\u0000-\u007F'
            r'\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'
            r'\uFB50-\uFDFF\uFE70-\uFEFF'
            r'\s\d\.\,\:\;\-\+\%\$\|\/\(\)\*\#\@\!\?]',
            '', text)
        return _reshape_arabic(text.strip())
    else:
        text = text.encode('ascii', 'ignore').decode('ascii')
    replacements = {'--': '-', '->': '->', '|': '|'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


def _ct(text: str, arabic: bool) -> str:
    """Shorthand: clean_text with correct arabic_mode flag."""
    return clean_text(text, arabic_mode=arabic)


# ── Section header block ───────────────────────────────────────────────────────

def section_header(text: str, story, arabic: bool = False):
    """Dark-navy full-width section header with gold underline."""
    fn    = _ARABIC_FONT_BOLD if arabic else 'Helvetica-Bold'
    align = 'RIGHT' if arabic else 'LEFT'
    data  = [[text.upper() if not arabic else text]]
    t = Table(data, colWidths=[176*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), NAVY),
        ('TEXTCOLOR',     (0,0), (-1,-1), WHITE),
        ('FONTNAME',      (0,0), (-1,-1), fn),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('ALIGN',         (0,0), (-1,-1), align),
        ('LINEBELOW',     (0,0), (-1,-1), 2, GOLD),
    ]))
    story.append(Spacer(1, 3*mm))
    story.append(t)
    story.append(Spacer(1, 2*mm))


# ── Main PDF generator ─────────────────────────────────────────────────────────

def generate_cio_pdf(content, output_path, ticker="", title="EisaX Report", lang="en"):
    report_id = 'RPT-' + datetime.now().strftime('%Y%m%d') + '-' + _uuid_pdf.uuid4().hex[:6].upper()
    _ar      = (lang == "ar")
    _fn      = _ARABIC_FONT      if _ar else 'Helvetica'
    _fn_bold = _ARABIC_FONT_BOLD if _ar else 'Helvetica-Bold'
    _align   = TA_RIGHT          if _ar else TA_LEFT
    date_str = datetime.now().strftime('%B %d, %Y')

    # ── Page template (header / footer painted on canvas) ─────────────────────
    def on_page(canvas, doc):
        canvas.saveState()
        w, h = A4
        HEADER_BG     = colors.HexColor('#1E3A8A')
        HEADER_ACCENT = colors.HexColor('#3B82F6')
        canvas.setFillColor(HEADER_BG)
        canvas.rect(0, h-25*mm, w, 25*mm, fill=1, stroke=0)
        canvas.setFillColor(HEADER_ACCENT)
        canvas.rect(0, h-25*mm, w, 1.5*mm, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica-Bold', 14)
        canvas.drawString(15*mm, h-16*mm, 'EisaX')
        canvas.setFillColor(WHITE)
        canvas.setFont('Helvetica', 9)
        canvas.drawString(15*mm, h-22*mm, 'AI-Powered Investment Intelligence')
        if ticker:
            canvas.setFillColor(LIGHT_GOLD)
            canvas.setFont('Helvetica-Bold', 10)
            canvas.drawRightString(w-15*mm, h-16*mm, f'INVESTMENT REPORT: {ticker}')
            canvas.setFillColor(WHITE)
            canvas.setFont('Helvetica', 8)
            canvas.drawRightString(w-15*mm, h-22*mm, date_str)
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(1.5)
        canvas.line(0, h-25*mm, w, h-25*mm)
        # Footer
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, w, 12*mm, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.setFont('Helvetica', 7)
        if _ar:
            canvas.drawRightString(w-15*mm, 4*mm,
                'سري | EisaX Intelligence | للمستثمرين المحترفين فقط')
        else:
            canvas.drawString(15*mm, 4*mm,
                'CONFIDENTIAL | EisaX Intelligence | For Professional Investors Only')
        canvas.setFillColor(WHITE)
        canvas.setFont('Helvetica', 7)
        canvas.drawRightString(w-15*mm, 4*mm, f'Page {doc.page}')
        canvas.setFillColor(colors.HexColor('#D1D5DB'))
        canvas.setFont('Helvetica-Bold', 7)
        canvas.drawString(15*mm, 8.5*mm, f'Report ID: {report_id}')
        canvas.setFillColor(GRAY)
        canvas.setFont('Helvetica', 6)
        if _ar:
            canvas.drawCentredString(w/2, 8.5*mm,
                'هذا التقرير مولّد بالذكاء الاصطناعي ولأغراض إعلامية فقط. ليس نصيحة استثمارية.')
        else:
            canvas.drawCentredString(w/2, 8.5*mm,
                'This report is generated by AI and is for informational purposes only. Not investment advice.')
        canvas.restoreState()

    # ── Document ───────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        topMargin=32*mm, bottomMargin=18*mm,
        leftMargin=15*mm, rightMargin=15*mm,
        title=title, author='EisaX Intelligence')

    # ── Paragraph styles (Arabic-aware) ───────────────────────────────────────
    body_style = ParagraphStyle('body',
        fontSize=9, fontName=_fn,
        textColor=colors.HexColor('#374151'),
        spaceAfter=2*mm, leading=16 if _ar else 14,
        alignment=_align)
    bullet_style = ParagraphStyle('bullet',
        fontSize=9, fontName=_fn,
        textColor=colors.HexColor('#374151'),
        spaceAfter=1.5*mm, leading=15 if _ar else 13,
        leftIndent=0 if _ar else 8*mm,
        rightIndent=8*mm if _ar else 0,
        alignment=_align)
    h2_style = ParagraphStyle('h2',
        fontSize=10, fontName=_fn_bold,
        textColor=NAVY, spaceAfter=2*mm, spaceBefore=4*mm,
        alignment=_align)
    small_style = ParagraphStyle('small',
        fontSize=7, fontName=_fn,
        textColor=GRAY, spaceAfter=1*mm,
        alignment=_align)

    story = []
    brand_header = Table(
        [['EisaX AI', f'Report ID: {report_id}']],
        colWidths=[88*mm, 88*mm],
    )
    brand_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (0, 0), WHITE),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#9CA3AF')),
        ('FONTNAME', (0, 0), (0, 0), _fn_bold),
        ('FONTNAME', (1, 0), (1, 0), _fn),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(brand_header)
    story.append(Spacer(1, 4*mm))
    story.append(Spacer(1, 8*mm))

    # ── Extract key metrics ────────────────────────────────────────────────────
    price    = re.search(r'Live Price[^$]*\$([\d.]+)', content)
    sector   = re.search(r'Sector[:\s|*]+([^|*\n]+)', content)
    score    = re.search(r'Quality Score[:\s|*]+([\d/]+)', content)
    verdict_m = re.search(r'\*\*(BUY|SELL|HOLD)[^*]*\*\*', content)

    price   = price.group(1)                         if price    else 'N/A'
    sector  = _ct(sector.group(1).strip(), _ar)      if sector   else 'N/A'
    score   = score.group(1)                         if score    else 'N/A'
    verdict = verdict_m.group(1)                     if verdict_m else 'N/A'

    # ── Badge ──────────────────────────────────────────────────────────────────
    _badge_txt = _reshape_arabic('تقرير الاستثمار الذكي — EisaX') if _ar else 'INVESTMENT INTELLIGENCE REPORT'
    badge = Table([[_badge_txt]], colWidths=[80*mm])
    badge.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), GOLD),
        ('TEXTCOLOR',     (0,0), (-1,-1), WHITE),
        ('FONTNAME',      (0,0), (-1,-1), _fn_bold),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(badge)
    story.append(Spacer(1, 4*mm))

    # ── Title ──────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        f'<font size="26" color="#0D1B2A"><b>{ticker}</b></font>',
        ParagraphStyle('t', fontSize=26, fontName=_fn_bold,
                       textColor=NAVY, spaceAfter=2*mm, alignment=_align)))
    story.append(Paragraph(
        f'<font color="#C9A84C">{sector}</font>',
        ParagraphStyle('s', fontSize=13, fontName=_fn,
                       textColor=GOLD, spaceAfter=3*mm, alignment=_align)))
    story.append(HRFlowable(width='100%', thickness=2, color=GOLD, spaceAfter=4*mm))

    # ── Metrics banner ─────────────────────────────────────────────────────────
    v_color     = '#16A34A' if verdict == 'BUY' else ('#DC2626' if verdict == 'SELL' else '#C9A84C')
    _lbl_price  = _reshape_arabic('السعر الحالي') if _ar else 'LIVE PRICE'
    _lbl_sector = _reshape_arabic('القطاع')       if _ar else 'SECTOR'
    _lbl_score  = _reshape_arabic('مؤشر الجودة')  if _ar else 'QUALITY SCORE'
    _lbl_verdict= _reshape_arabic('التوصية')      if _ar else 'VERDICT'
    metrics = Table(
        [[_lbl_price, _lbl_sector, _lbl_score, _lbl_verdict],
         [f'${price}', sector[:18], score, verdict]],
        colWidths=[42*mm, 55*mm, 42*mm, 37*mm])
    metrics.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  NAVY),
        ('TEXTCOLOR',     (0,0), (-1,0),  GOLD),
        ('FONTNAME',      (0,0), (-1,0),  _fn_bold),
        ('FONTSIZE',      (0,0), (-1,0),  7),
        ('BACKGROUND',    (0,1), (-1,-1), LIGHT_GRAY),
        ('TEXTCOLOR',     (0,1), (-1,1),  NAVY),
        ('FONTNAME',      (0,1), (-1,1),  _fn_bold),
        ('FONTSIZE',      (0,1), (-1,1),  13),
        ('BACKGROUND',    (3,1), (3,1),   colors.HexColor(v_color)),
        ('TEXTCOLOR',     (3,1), (3,1),   WHITE),
        ('FONTNAME',      (3,1), (3,1),   _fn_bold),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID',          (0,0), (-1,-1), 0.5, BORDER),
        ('LINEBELOW',     (0,0), (-1,0),  1.5, GOLD),
    ]))
    story.append(metrics)
    story.append(Spacer(1, 5*mm))

    # ── Parse content line by line ─────────────────────────────────────────────
    lines     = content.split('\n')
    skip_code = False
    chart_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # ── Code / ASCII-chart blocks ──────────────────────────────────────────
        if line.startswith('```'):
            if not skip_code:
                skip_code   = True
                chart_lines = []
            else:
                skip_code = False
                if chart_lines:
                    ct = Table(
                        [[Paragraph(
                            '<font name="Courier" size="7">%s</font>'
                            % _ct(cl, False)
                              .replace('&','&amp;')
                              .replace('<','&lt;')
                              .replace('>','&gt;'),
                            body_style)]
                         for cl in chart_lines if cl.strip()],
                        colWidths=[176*mm])
                    ct.setStyle(TableStyle([
                        ('BACKGROUND',   (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
                        ('TOPPADDING',   (0,0), (-1,-1), 1),
                        ('BOTTOMPADDING',(0,0), (-1,-1), 1),
                        ('LEFTPADDING',  (0,0), (-1,-1), 8),
                        ('RIGHTPADDING', (0,0), (-1,-1), 4),
                        ('LINEABOVE',    (0,0), (-1,0),  0.5, colors.HexColor('#cbd5e1')),
                        ('LINEBELOW',    (0,-1),(-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                        ('LINEBEFORE',   (0,0), (0,-1),  3,   colors.HexColor('#3B82F6')),
                    ]))
                    story.append(ct)
                    story.append(Spacer(1, 2*mm))
                    chart_lines = []
            i += 1
            continue
        if skip_code:
            chart_lines.append(line)
            i += 1
            continue

        # ── Skip known boilerplate lines ───────────────────────────────────────
        if any(x in line for x in [
            '# EisaX Intelligence Report', 'MEMORANDUM', 'Live Price.*Sector',
            'To: Investment', 'From: EisaX', 'Re: Investment Analysis',
            'eisax-chart', 'data-ticker']):
            i += 1
            continue
        if re.match(r'^\*\*Live Price', line):
            i += 1
            continue

        # ── H3 section header ──────────────────────────────────────────────────
        if line.startswith('###'):
            text = _ct(line.lstrip('#').strip(), _ar)
            section_header(text, story, arabic=_ar)
            i += 1
            continue

        # ── H2 ─────────────────────────────────────────────────────────────────
        if line.startswith('##'):
            text = _ct(line.lstrip('#').strip(), _ar)
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(f'<b>{text}</b>', h2_style))
            story.append(HRFlowable(width='100%', thickness=0.5, color=GOLD, spaceAfter=2*mm))
            i += 1
            continue

        # ── H1 (skip) ──────────────────────────────────────────────────────────
        if line.startswith('#'):
            i += 1
            continue

        # ── Markdown table ─────────────────────────────────────────────────────
        if line.startswith('|') and '|' in line[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in table_lines:
                if re.match(r'^\|[\s\-|:]+\|$', tl):
                    continue
                cells = [_ct(c.strip(), _ar) for c in tl.split('|')[1:-1]]
                if any(cells):
                    rows.append(cells)
            if rows:
                num_cols  = max(len(r) for r in rows)
                for r in rows:
                    while len(r) < num_cols:
                        r.append('')
                cw        = 176*mm / num_cols
                tbl_align = 'RIGHT' if _ar else 'LEFT'
                para_rows = []
                for ri, row in enumerate(rows):
                    para_row = []
                    for cell in row:
                        if ri == 0:
                            p = Paragraph(f'<b>{cell}</b>',
                                ParagraphStyle('th', fontSize=8,
                                    fontName=_fn_bold, textColor=WHITE,
                                    alignment=TA_RIGHT if _ar else TA_LEFT))
                        else:
                            fc = colors.HexColor('#374151')
                            if cell in ('BUY','STRONG BUY','OK'): fc = GREEN
                            elif cell in ('SELL','STRONG SELL'):   fc = RED
                            p = Paragraph(cell,
                                ParagraphStyle('td', fontSize=8.5,
                                    fontName=_fn, textColor=fc,
                                    alignment=TA_RIGHT if _ar else TA_LEFT))
                        para_row.append(p)
                    para_rows.append(para_row)
                t = Table(para_rows, colWidths=[cw]*num_cols, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND',    (0,0), (-1,0),  NAVY),
                    ('ALIGN',         (0,0), (-1,-1), tbl_align),
                    ('TOPPADDING',    (0,0), (-1,-1), 5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ('LEFTPADDING',   (0,0), (-1,-1), 6),
                    ('RIGHTPADDING',  (0,0), (-1,-1), 6),
                    ('GRID',          (0,0), (-1,-1), 0.3, BORDER),
                    ('LINEBELOW',     (0,0), (-1,0),  1, GOLD),
                    ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
                ]))
                story.append(t)
                story.append(Spacer(1, 3*mm))
            continue

        # ── News item (markdown link) ──────────────────────────────────────────
        if line.startswith('- [') and 'http' in line:
            m = re.search(r'\[([^\]]+)\]', line)
            if m:
                story.append(Paragraph(f'• {_ct(m.group(1), _ar)}', bullet_style))
            i += 1
            continue

        # ── Bullet ─────────────────────────────────────────────────────────────
        if line.startswith('- ') or line.startswith('* '):
            text = _ct(line[2:], _ar)
            text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
            bullet_char = '•'
            story.append(Paragraph(f'{bullet_char} {text}', bullet_style))
            i += 1
            continue

        # ── Numbered list ──────────────────────────────────────────────────────
        if re.match(r'^\d+\.', line):
            text = _ct(re.sub(r'^\d+\.\s*', '', line), _ar)
            text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
            story.append(Paragraph(f'• {text}', bullet_style))
            i += 1
            continue

        # ── Verdict box ────────────────────────────────────────────────────────
        if re.search(r'\*\*(BUY|SELL|HOLD)', line):
            text = _ct(line, _ar)
            text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
            vc   = GREEN if 'BUY' in line else (RED if 'SELL' in line else GOLD)
            vt   = Table([[Paragraph(text,
                ParagraphStyle('v', fontSize=11, fontName=_fn_bold,
                               textColor=WHITE, alignment=TA_CENTER))]],
                colWidths=[176*mm])
            vt.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,-1), vc),
                ('TOPPADDING',    (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(vt)
            story.append(Spacer(1, 3*mm))
            i += 1
            continue

        # ── Horizontal separator ───────────────────────────────────────────────
        if re.match(r'^[-=*]{3,}$', line):
            i += 1
            continue

        # ── Regular text ───────────────────────────────────────────────────────
        text = _ct(line, _ar)
        if text and len(text) > 3:
            text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*([^*]+)\*',     r'<i>\1</i>', text)
            story.append(Paragraph(text, body_style))
        i += 1

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width='100%', thickness=1, color=GOLD))
    story.append(Spacer(1, 2*mm))
    if _ar:
        disc = _reshape_arabic(
            'إخلاء المسؤولية: هذا التقرير مولّد بواسطة EisaX AI ولأغراض إعلامية فقط. '
            'لا يمثل نصيحة استثمارية. استشر مستشارًا ماليًا مؤهلًا دائمًا. '
            'مدعوم بـ EisaX Intelligence | المهندس أحمد عيسى')
    else:
        disc = ('DISCLAIMER: This report is generated by EisaX AI and is for informational purposes only. '
                'It does not constitute investment advice. Always consult a qualified financial advisor. '
                'Powered by EisaX Intelligence | Eng. Ahmed Eisa')
    story.append(Paragraph(disc, small_style))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    logger.debug(f"OK: {output_path}")
    return output_path, report_id


# ── Smoke-test (runs only when executed directly) ──────────────────────────────
if __name__ == '__main__':
    os.makedirs('/home/ubuntu/investwise/static/exports', exist_ok=True)
    test_en = """# EisaX Intelligence Report: NVDA
**Live Price:** $189.82 (+1.02%) | **Sector:** Technology | **Quality Score:** 100/100
### 1. Executive Summary
NVIDIA presents a compelling BUY case with 114% revenue growth and world-class margins.
### 2. Fundamental Analysis
**Growth:** Revenue TTM $130.5B growing 114% YoY. Gross Margin 70.1%.
### 6. EisaX Verdict
**BUY | Conviction: High**
"""
    generate_cio_pdf(test_en,
        '/home/ubuntu/investwise/static/exports/test_cio_en.pdf', 'NVDA', lang='en')

    test_ar = """# تقرير EisaX: أرامكو
**السعر الحالي:** $8.25 | **القطاع:** الطاقة | **مؤشر الجودة:** 88/100
### 1. الملخص التنفيذي
تقدم شركة أرامكو السعودية فرصة استثمارية قوية بنمو إيرادات 18% وهامش ربح استثنائي.
### 2. التحليل الأساسي
**النمو:** الإيرادات السنوية 600 مليار ريال بنمو 18% سنوياً. هامش الربح الإجمالي 52%.
### 6. توصية EisaX
**شراء | الاقتناع: مرتفع**
"""
    generate_cio_pdf(test_ar,
        '/home/ubuntu/investwise/static/exports/test_cio_ar.pdf', '2222.SR',
        title='تقرير EisaX — أرامكو', lang='ar')
    print("✅ Both PDFs generated.")
