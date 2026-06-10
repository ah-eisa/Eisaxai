"""
EisaX Playwright PDF Generator
Converts HTML to PDF using headless Chromium (Playwright).
Drop-in replacement for weasyprint.HTML(...).write_pdf()
"""
import asyncio
import concurrent.futures
import logging
import os

logger = logging.getLogger(__name__)


async def _async_html_to_pdf(html: str, output_path: str) -> None:
    """Async core: render HTML → PDF with headless Chromium."""
    from playwright.async_api import async_playwright
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        page = await browser.new_page()
        await page.set_content(html, wait_until="load")
        # Wait for network idle so Google Fonts (Cairo, Tajawal, Inter) fully load
        try:
            await page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass  # timeout OK — fonts may already be cached
        # Extra render tick for font swap + Arabic shaping
        await page.wait_for_timeout(800)
        await page.pdf(
            path=output_path,
            format="A4",
            margin={"top": "12mm", "bottom": "12mm", "left": "14mm", "right": "14mm"},
            print_background=True,
        )
        await browser.close()
    logger.info("[PlaywrightPDF] Saved → %s", output_path)


def html_to_pdf(html: str, output_path: str) -> None:
    """
    Render HTML string to PDF file.
    Works from both sync and async contexts (FastAPI/uvicorn).
    Runs Playwright in a fresh thread with its own event loop to avoid
    conflicts with uvicorn's running loop.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _async_html_to_pdf(html, output_path))
        future.result(timeout=90)


def inject_print_css(html: str, extra_css: str = "") -> str:
    """
    Inject print-friendly CSS (and any extra_css) into an HTML document.
    If the document already has a <head>, the style is appended inside it.
    Otherwise a minimal <head> is prepended.
    NOTE: Does NOT override font-family when the HTML already specifies Arabic fonts
    (Cairo/Tajawal) — overriding with Arial breaks Arabic character shaping.
    """
    # Detect Arabic HTML — if it already declares Cairo/Tajawal, don't override font
    _has_arabic_font = "cairo" in html.lower() or "tajawal" in html.lower()
    _font_rule = "" if _has_arabic_font else "body{font-family:Arial,sans-serif;font-size:10pt}"

    base_css = (
        _font_rule
        + ".eisax-chart{display:block;background:#f1f5f9;padding:12px;border-radius:8px;"
          "font-family:Courier,monospace;font-size:8pt;white-space:pre-wrap}"
        "canvas{max-width:100%!important;border-radius:8px}"
        "a{color:#302b63;text-decoration:none}"
        "pre,code{font-size:8.5pt}"
    )
    style_tag = f"<style>{base_css}{extra_css}</style>"
    if "</head>" in html:
        return html.replace("</head>", f"{style_tag}</head>", 1)
    return f"<head><meta charset='utf-8'>{style_tag}</head>{html}"
