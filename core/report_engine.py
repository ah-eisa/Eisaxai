"""EisaX Report Engine — generates PDF reports from chat history."""
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportEngine:
    def __init__(self, output_dir="static/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def create_report(self, data: dict) -> str:
        """
        Generate a PDF report from chat data dict.
        data keys: date, author, content (list of {role, content} dicts)
        Returns the file path on success or an error string.
        """
        try:
            from core.playwright_pdf import html_to_pdf

            ts = data.get("date", datetime.now().strftime("%Y-%m-%d"))
            author = data.get("author", "EisaX User")

            rows_html = ""
            for msg in data.get("content", []):
                role = str(msg.get("role", "User")).upper()
                content = (
                    str(msg.get("content", ""))
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                rows_html += (
                    f'<div style="margin-bottom:12px">'
                    f'<b style="font-size:10pt">{role}:</b><br>'
                    f'<span style="font-size:10pt;white-space:pre-wrap">{content}</span>'
                    f'</div>'
                    f'<hr style="border:none;border-top:1px solid #eee;margin:8px 0">'
                )

            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body{{font-family:Arial,sans-serif;font-size:10.5pt;color:#1a1a2e;padding:30px 40px}}
  h1{{font-size:16pt;text-align:center;color:#1e1b4b}}
  .sub{{text-align:center;font-size:9pt;color:#888;margin-bottom:24px}}
  .prepared{{font-size:11pt;font-weight:700;margin-bottom:16px}}
</style>
</head><body>
<h1>EisaX Intelligence Report</h1>
<div class="sub">Date: {ts}</div>
<div class="prepared">Prepared for: {author}</div>
{rows_html}
</body></html>"""

            file_name = f"EisaX_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path = os.path.join(self.output_dir, file_name)
            html_to_pdf(html, file_path)
            logger.info("[ReportEngine] Report saved: %s", file_path)
            return file_path
        except Exception as e:
            logger.error("[ReportEngine] Generation failed: %s", e)
            return f"Generation failed: {str(e)}"
