"""
core/services/export_service.py
─────────────────────────────────
PDF / DOCX export pipeline extracted from process_message.

Public API
──────────
    handle_export(message, history, session_id) -> dict
        Generates a PDF or DOCX from the last assistant message in *history*
        and returns a response dict with a download_url.

Notes
─────
- HTML is sanitized before PDF conversion to prevent XSS in generated files.
- Charts are rendered with matplotlib (Agg backend, no display) and embedded
  as base64 data-URIs so the exported file is fully self-contained.
- DOCX generation delegates to core.cio_docx.generate_cio_docx.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime as _dt
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Export output directory (relative to project root)
from core.config import EXPORTS_DIR as _cfg_exports
_EXPORT_DIR = _cfg_exports

# Characters/tags that are safe to keep in the HTML body
_SAFE_HTML_TAGS = {
    "p", "br", "b", "strong", "i", "em", "u", "s",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "table", "thead", "tbody", "tr", "th", "td",
    "blockquote", "code", "pre",
    "div", "span", "hr",
    "a", "img",
}


def _sanitize_html(html: str) -> str:
    """
    Remove potentially dangerous HTML attributes (onclick, javascript:, etc.)
    while keeping structural/styling tags intact.

    Uses a lightweight regex approach so we avoid an extra library dependency.
    bleach is preferred when available.
    """
    try:
        import bleach
        allowed_attrs = {
            "*":   ["class", "style", "id"],
            "a":   ["href", "title"],
            "img": ["src", "alt", "style", "width", "height"],
            "td":  ["colspan", "rowspan", "style"],
            "th":  ["colspan", "rowspan", "style"],
            "div": ["style", "class", "data-ticker"],
        }
        return bleach.clean(html, tags=_SAFE_HTML_TAGS, attributes=allowed_attrs, strip=False)
    except ImportError:
        logger.debug("[ExportService] bleach not installed; using regex HTML sanitizer fallback")

    # Fallback: strip event handlers and javascript: hrefs
    html = re.sub(r'\s+on\w+\s*=\s*"[^"]*"', "", html, flags=re.I)
    html = re.sub(r'\s+on\w+\s*=\s*\'[^\']*\'', "", html, flags=re.I)
    html = re.sub(r'href\s*=\s*"javascript:[^"]*"', 'href="#"', html, flags=re.I)
    html = re.sub(r'href\s*=\s*\'javascript:[^\']*\'', "href='#'", html, flags=re.I)
    return html


def _generate_chart(ticker: str) -> str | None:
    """
    Download 3-month daily price data for *ticker* and render a base64 PNG.
    Returns the base64 string or None on failure.
    """
    try:
        import yfinance as _yf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.dates as _mdates
        from io import BytesIO
        import base64

        df = _yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty:
            return None

        fig, ax = _plt.subplots(figsize=(10, 3))
        ax.plot(df.index, df["Close"], color="#c9a84c", linewidth=2)
        ax.fill_between(df.index, df["Close"], alpha=0.1, color="#c9a84c")
        ax.set_facecolor("#0a0a1a")
        fig.patch.set_facecolor("#0a0a1a")
        ax.tick_params(colors="white")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%b %d"))
        ax.set_title(f"{ticker} — 90 Day Price", color="white", fontsize=11)

        buf = BytesIO()
        _plt.savefig(buf, format="png", bbox_inches="tight",
                     facecolor="#0a0a1a", dpi=120)
        _plt.close(fig)
        logger.debug("[PDF] Chart generated for %s", ticker)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        logger.warning("[PDF] Chart failed for %s: %s", ticker, exc)
        return None


def _build_html(content: str, today: str) -> str:
    """Convert markdown *content* to a styled HTML document ready for PDF."""
    import markdown as _md

    # Resolve chart divs → inline images
    tickers_in_report = re.findall(r'data-ticker="([A-Z0-9.\-]{1,10})"', content)
    for tic in tickers_in_report:
        b64 = _generate_chart(tic)
        if b64:
            img_tag = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;margin:10px 0">'
            content = content.replace(
                f'<div class="eisax-chart" data-ticker="{tic}"></div>', img_tag
            )
            content = content.replace(f'data-ticker="{tic}"></div>', img_tag)

    body = _md.markdown(content, extensions=["tables", "nl2br", "fenced_code"])

    # Replace ASCII progress-bar rows with visual HTML bars
    def _fix_bar_row(m: re.Match) -> str:
        name = m.group(1).strip()
        try:
            pct = int(m.group(2))
        except Exception:
            logger.debug("[export] _fix_bar_row: cannot parse pct from %r — keeping raw match", m.group(0), exc_info=True)
            return m.group(0)
        color = "#00cc66" if pct >= 75 else ("#ffaa00" if pct >= 50 else "#ff4444")
        w = int(pct * 1.5)
        return (
            f"<tr>"
            f'<td style="padding:7px 12px;font-size:9.5pt;width:160px">{name}</td>'
            f'<td style="padding:7px 12px">'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="width:150px;height:10px;background:#e0e4f0;border-radius:5px;overflow:hidden;flex-shrink:0">'
            f'<div style="width:{w}px;height:100%;background:{color};border-radius:5px"></div></div>'
            f'<b style="color:{color};font-size:9.5pt;white-space:nowrap">{pct}%</b>'
            f"</div></td></tr>"
        )

    body = re.sub(
        r"<tr>\s*<td>([^<]{3,40})</td>\s*<td>(\d+)%</td>\s*<td>[^<]*<code>[^<]*</code></td>\s*</tr>",
        _fix_bar_row, body,
    )

    # Replace ASCII score bars with a visual overall-score block
    body = re.sub(
        r"<code>[█░\s]+</code>\s*(\d+)/100",
        lambda m: (
            f'<div style="margin:12px 0;padding:12px 16px;background:#f0f4ff;border-radius:8px;border:1px solid #c0caf0">'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<span style="font-weight:700;color:#302b63;white-space:nowrap">Overall Score</span>'
            f'<div style="flex:1;height:14px;background:#e0e4f0;border-radius:7px;overflow:hidden">'
            f'<div style="width:{int(m.group(1))}%;height:100%;background:linear-gradient(90deg,#302b63,#00d4ff);border-radius:7px"></div>'
            f"</div>"
            f'<b style="font-size:15pt;color:#302b63;white-space:nowrap">{m.group(1)}/100</b>'
            f"</div></div>"
        ),
        body,
    )

    css = (
        "body{font-family:Arial,sans-serif;font-size:10.5pt;line-height:1.75;color:#1a1a2e;max-width:780px;margin:0 auto;padding:30px 40px}"
        "h1{font-size:16pt;color:#0f0c29;border-bottom:3px solid #00d4ff;padding-bottom:6px;margin:22px 0 10px}"
        "h2{font-size:13pt;color:#302b63;border-left:4px solid #00d4ff;padding-left:10px;margin:18px 0 8px}"
        "h3{font-size:11.5pt;font-weight:700;margin:14px 0 6px}"
        "p{margin-bottom:10px} ul,ol{margin:6px 0 10px 22px} li{margin-bottom:5px}"
        "strong{font-weight:700} hr{border:none;border-top:2px solid #eef0f4;margin:16px 0}"
        "table{width:100%;border-collapse:collapse;margin:14px 0;font-size:10pt;border-radius:8px;overflow:hidden}"
        "th{background:#302b63;color:white;padding:10px 13px;text-align:left;font-size:9.5pt}"
        "td{padding:8px 13px;border-bottom:1px solid #eef0f4;vertical-align:middle}"
        "tr:nth-child(even) td{background:#f8f9ff}"
        "blockquote{border-left:4px solid #00d4ff;background:#f0f8ff;padding:10px 15px;margin:10px 0;border-radius:0 8px 8px 0;font-style:italic}"
        "a{color:#302b63;text-decoration:none}"
        ".eisax-chart{display:none}"
        "code{background:#f4f4f4;padding:2px 5px;border-radius:3px;font-size:9pt}"
        "@page{margin:12mm 14mm;size:A4}"
    )

    header = f"""
        <div style="background:#0a0a1a;padding:0;margin-bottom:0;">
          <div style="background:linear-gradient(90deg,#0a0a1a 0%,#1a1040 40%,#0d0d2b 100%);padding:28px 40px 20px;border-bottom:3px solid #c9a84c">
            <div style="display:flex;justify-content:space-between;align-items:flex-end">
              <div>
                <div style="font-family:Arial Black,Arial,sans-serif;font-size:36pt;font-weight:900;letter-spacing:6px;color:white;line-height:1">EISAX</div>
                <div style="font-size:9pt;color:#c9a84c;letter-spacing:2px;margin-top:5px;text-transform:uppercase">AI Investment Intelligence &nbsp;|&nbsp; Abu Dhabi</div>
              </div>
              <div style="text-align:right">
                <div style="font-size:9pt;color:rgba(255,255,255,0.5);letter-spacing:1px">{today}</div>
                <div style="font-size:8pt;color:rgba(255,255,255,0.35);margin-top:3px">CONFIDENTIAL | FOR INSTITUTIONAL USE</div>
              </div>
            </div>
          </div>
        </div>
    """

    footer = (
        f'<div style="margin-top:35px;padding:12px 40px;border-top:2px solid #eef0f4;'
        f'font-size:8pt;color:#888;text-align:center">'
        f"EisaX AI Investment Intelligence | Abu Dhabi, UAE | Generated {today}<br>"
        "This report is for informational purposes only and does not constitute financial advice."
        "</div>"
    )

    # Sanitize the converted body before embedding in the final HTML
    safe_body = _sanitize_html(body)

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<style>{css}</style></head><body>"
        + header
        + '<div style="padding:0 40px 40px;">'
        + safe_body
        + "</div>"
        + footer
        + "</body></html>"
    )


async def handle_export(
    message:    str,
    history:    list[dict],
    session_id: str,
) -> dict:
    """
    Generate a PDF or DOCX export from *history* and return a response dict.

    Parameters
    ──────────
    message    : the original user message (used to detect pdf vs docx)
    history    : full chat history list[{role, content}]
    session_id : propagated into the response dict

    Returns
    ───────
    dict with keys: reply, download_url, format, session_id
    (or reply + session_id on error)
    """
    if len([m for m in history if m.get("role") == "assistant"]) < 1:
        return {
            "reply": "لا يوجد محتوى للتصدير. قم بالتحليل أولاً.",
            "session_id": session_id,
        }

    fmt = "docx" if ("word" in message.lower() or "docx" in message.lower()) else "pdf"

    try:
        reports = [
            m["content"] for m in history
            if m.get("role") == "assistant" and len(m.get("content", "")) > 10
        ]
        src = reports[-1] if reports else ""

        today    = _dt.now().strftime("%B %d, %Y")
        filename = f"EisaX_{time.strftime('%Y%m%d_%H%M%S')}.{fmt}"
        filepath = str(_EXPORT_DIR / filename)
        _EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        if fmt == "pdf":
            from core.playwright_pdf import html_to_pdf
            html_content = _build_html(src, today)
            html_to_pdf(html_content, filepath)
            os.chmod(filepath, 0o644)
        else:
            from core.cio_docx import generate_cio_docx
            ticker_m = re.search(r"[A-Z]{2,5}", src[:500])
            ticker   = ticker_m.group(0) if ticker_m else ""
            generate_cio_docx(src, filepath, ticker=ticker)
            os.chmod(filepath, 0o644)

        return {
            "reply":        f"✅ {'PDF' if fmt == 'pdf' else 'Word'} report is ready!",
            "download_url": f"/v1/download/{filename}",
            "format":       fmt,
            "session_id":   session_id,
        }

    except Exception as exc:
        logger.error("[ExportService] export failed: %s", exc)
        return {"reply": f"❌ خطأ في التصدير: {exc}", "session_id": session_id}
