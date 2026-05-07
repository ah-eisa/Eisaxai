# Auto-extracted mixin — do not edit directly; source of truth is git history.
from __future__ import annotations
from typing import Any, Dict, Optional
import logging
import os
import state
import config
from datetime import datetime
logger = logging.getLogger(__name__)


class ExportMixin:
    def _handle_report(self, sid: str, mem: Dict[str, Any], msg: str) -> Dict[str, Any]:
        """Legacy report handler (generates fresh). Redirects to export if explicit."""
        if "export" in msg.lower() or "pdf" in msg.lower():
            return self._handle_export(sid, mem, msg)
            
        # ... logic to generate report body ...
        # Reuse existing logic but ensuring we populate last_artifact too
        return self._handle_export(sid, mem, msg, force_refresh=True)

    def _handle_export(self, sid: str, mem: Dict[str, Any], msg: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Handles explicit export requests.
        Prioritizes state.last_artifact.
        Falls back to generating a report from memory tokens.
        """
        from core.report_engine import ReportEngine

        # ── Hard Rejection Gate ────────────────────────────────────────────────
        _perf_export = mem.get("performance") or {}
        _exp_ret_ex  = float(_perf_export.get("expected_return", 0) or 0)
        _sharpe_ex   = float(_perf_export.get("sharpe", 0) or 0)
        if _perf_export and (_exp_ret_ex < 0.045 or _sharpe_ex < 0):
            _w_ex  = mem.get("weights") or mem.get("weights_raw") or {}
            _top3e = sorted(_w_ex.items(), key=lambda x: -x[1])[:3] if _w_ex else []
            _fix_lines = []
            if _exp_ret_ex < 0.045:
                _fix_lines.append("أضف أسهم US أو Gold أو Bonds لرفع العائد المتوقع فوق معدل الخطر الصفري (4.5%)")
            if _sharpe_ex < 0:
                _fix_lines.append("العائد المتوقع أقل من معدل الخطر الصفري — المحفظة لا تُعوّض المستثمر عن المخاطرة")
            _fixes_md = "\n".join(f"- {f}" for f in _fix_lines) if _fix_lines else "- راجع مكونات المحفظة"
            _rejection = (
                "# ❌ Portfolio Rejected — Strategy Invalid\n\n"
                "**لا يمكن تنفيذ هذه المحفظة — المعايير الأساسية غير مستوفاة.**\n\n"
                "| المؤشر | القيمة | المطلوب |\n"
                "|--------|--------|---------|\n"
                f"| العائد المتوقع | **{_exp_ret_ex*100:.2f}%** | > 4.5% |\n"
                f"| Sharpe Ratio | **{_sharpe_ex:.2f}** | > 0 |\n\n"
                "## السبب\n\n"
                "المحفظة المقترحة **تخسر قيمتها** أو لا تُعوّض عن مخاطرها. "
                "تنفيذها سيضر بالمستثمر بدلاً من مساعدته.\n\n"
                "## الإصلاحات المقترحة\n\n"
                f"{_fixes_md}\n\n"
                "## جرّب بدلاً من ذلك\n\n"
                "> ابني محفظة **balanced** باستخدام **US + GCC + Gold** لضمان عائد إيجابي ومتوازن.\n"
            )
            return {"type": "chat.reply", "reply": _rejection}
        # ── End Rejection Gate ─────────────────────────────────────────────────

        # ── Normalize stale tickers in session memory ─────────────────────────
        # Old sessions may have "UAE", "SAUDI" etc. in mem — fix them in-place
        # before any report path runs.
        try:
            if mem.get("tickers"):
                mem["tickers"] = pm._normalize_tickers(mem["tickers"])
            if mem.get("weights"):
                _old_w = mem["weights"]
                _new_w = {}
                for _t, _w in _old_w.items():
                    _mapped = pm._TICKER_MAP.get(_t.upper(), _t)
                    if _mapped:
                        _new_w[_mapped] = _new_w.get(_mapped, 0) + _w
                mem["weights"] = _new_w
        except Exception:
            pass

        # ── Placeholder gate (also covers stale cached artifacts) ─────────────
        _w_for_gate = mem.get("weights") or mem.get("weights_raw") or {}
        _ph = pm.has_placeholder_tickers(_w_for_gate)
        if _ph:
            _block_msg = (
                "# ⛔ Report Blocked — Unverified Assets\n\n"
                f"**Placeholder tickers detected:** `{'`, `'.join(_ph)}`\n\n"
                "EisaX cannot produce a client-facing report with unidentified securities.\n\n"
                "**Fix:** Ask EisaX to rebuild the portfolio — "
                "the optimizer will select verified assets from the live market library.\n\n"
                "> **Rule:** Every asset must be verified by ticker, name, and market.\n"
            )
            return {"type": "chat.reply", "reply": _block_msg}
        # ─────────────────────────────────────────────────────────────────────

        content_to_export = ""
        title = f"EISAX Report {datetime.now().strftime('%Y-%m-%d')}"

        # 1. Check valid artifact — but invalidate cache if it contains placeholders
        _artifact_cached = state.get_artifact(sid)
        _cached_content = _artifact_cached.get("content", "") if _artifact_cached else ""
        _cache_has_placeholder = any(
            f"`{p}`" in _cached_content or f" {p} " in _cached_content or f"/{p}" in _cached_content
            for p in pm._ALL_FAKE_TICKERS
        ) if _cached_content else False

        if not force_refresh and _artifact_cached and _artifact_cached.get("exportable") and not _cache_has_placeholder:
            content_to_export = _artifact_cached["content"]
            # Try to derive title from content header?
            first_line = content_to_export.strip().split('\n')[0]
            if first_line.startswith("# "):
                title = first_line[2:].strip()
        else:
            # 2. Fallback: Generate from Memory (Tickers)
            tickers = mem.get("tickers", [])
            if not tickers and not state.get_artifact(sid):
                return {"type": "chat.reply", "reply": "I don't have a generated report or portfolio to export yet. Try 'Optimize my portfolio' first."}
            
            # Generate fresh report
            base_report_md = f"# Investment Report\n\nPortfolio: {', '.join(tickers)}\n\n"
            base_report_md += pm.build_portfolio_report_body(mem)
            content_to_export = pm.generate_executive_report_llm(
                model=config.DEFAULT_MODEL, temperature=0.2, mem=mem, base_report_md=base_report_md
            )
            
            # SAVE ARTIFACT (so prompt loop stops)
            state.set_artifact(sid, {
                "type": "report",
                "content": content_to_export,
                "source": "self_generated",
                "exportable": True,
                "timestamp": datetime.now()
            })

        # 3. Generate PDF
        try:
            engine = ReportEngine()
            pdf_path = engine.generate_pdf(title, content_to_export)
            filename = os.path.basename(pdf_path)
            
            logger.info(f"[Finance] EXPORT SUCCESS: {filename} | Content length: {len(content_to_export)}")
            return {
                "type": "report.export",
                # DIRECT LINK, NO QUESTIONS
                "reply": f"Here is your PDF document: **{title}**\n\n[Download PDF](/static/reports/{filename})", 
                "data": {
                    "format": "pdf", 
                    "printable": True, 
                    "url": f"/static/reports/{filename}",
                    "download_url": f"/static/reports/{filename}",
                    "filename": filename,
                    "title": title
                }
            }
        except Exception as e:
             return {"type": "error", "reply": f"Export failed: {e}"}

    def _save_to_brain(self, target, reply_text, real_price, analyst_target, fund, news_sent):
        """Save analysis verdict and stock knowledge to the Brain DB."""
        try:
            from learning_engine import get_engine
            _ru = reply_text.upper()
            _bv = "SELL" if ("SELL" in _ru or "REDUCE" in _ru) else "HOLD" if "HOLD" in _ru else "BUY"
            if real_price and real_price > 0:
                _bc = get_engine()._get_conn()
                _bc.execute(
                    "INSERT INTO predictions (ticker, prediction_date, verdict, price_at_prediction, target_price, horizon_days) VALUES (?, date('now'), ?, ?, ?, 30)",
                    (target, _bv, real_price, analyst_target or None)
                )
                _bc.execute(
                    "INSERT INTO stock_knowledge (ticker, company_name, sector, summary, last_price, last_verdict, last_sentiment, analysis_count, first_seen, last_updated, tags) VALUES (?, ?, ?, ?, ?, ?, ?, 1, date('now'), datetime('now'), '[]') ON CONFLICT(ticker) DO UPDATE SET last_price=excluded.last_price, last_verdict=excluded.last_verdict, last_updated=excluded.last_updated, analysis_count=analysis_count+1",
                    (target, fund.get('company_name', target), fund.get('sector', 'Unknown'), f"{_bv} @ ${real_price:.2f}", real_price, _bv, news_sent or 'Neutral')
                )
                _bc.commit()
                _bc.close()
                logger.info(f"[Brain] Saved: {target} {_bv} @ ${real_price:.2f}")
        except Exception as _be:
            logger.warning(f"[Brain] Warning: {_be}")


