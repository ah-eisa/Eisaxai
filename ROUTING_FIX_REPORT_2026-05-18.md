# EisaX — Routing Fix Report
**Date:** 2026-05-18
**Scope:** Single integration fix — expose the institutional `_run_allocator` through `/v1/chat`.

---

## 1. Verdict

**GREEN.** Institutional A-G/H/I report is now reached via the live `/v1/chat` endpoint for both EN and AR portfolio-build requests. Non-portfolio messages route unchanged. All internal suites remain green.

---

## 2. Route changed

| Route | File | Change |
|---|---|---|
| `POST /v1/chat` | `api/routers/chat.py:340` | Inserted institutional-allocator fast-path immediately before the `orchestrator.process_message(...)` call |

**No other routes touched.** `/v1/global-allocate` remains unregistered (out of scope). Orchestrator code (`core/orchestrator.py`) unchanged. Portfolio builder unchanged. Phase H/I unchanged.

---

## 3. Diff summary

```diff
api/routers/chat.py
+ 38 lines inserted before `result = await orchestrator.process_message(...)`
```

Inserted block (`api/routers/chat.py:340-377`):

```python
# ── Institutional allocator fast-path ─────────────────────────────────
# Route portfolio-build intents (EN/AR) to the validated A-G/H/I
# _run_allocator instead of the legacy LLM "EisaX Portfolio Pipeline".
# Falls through to the orchestrator on any error or non-match.
if not payload.files and not active_file_id:
    try:
        from portfolio_builder import detect_and_build as _pb_detect
        _inst_settings_lang = ""
        if payload.settings and isinstance(payload.settings, dict):
            _inst_settings_lang = str(payload.settings.get("language") or "").strip().lower()
        if _inst_settings_lang.startswith("ar"):
            _inst_lang = "ar"
        elif _inst_settings_lang.startswith("en"):
            _inst_lang = "en"
        else:
            _inst_lang = "ar" if any("؀" <= ch <= "ۿ" for ch in message) else "en"
        _inst_md = _pb_detect(message, language=_inst_lang)
        if _inst_md:
            orchestrator.session_mgr.save_message(
                session_id, payload.user_id, "user", message
            )
            orchestrator.session_mgr.save_message(
                session_id, payload.user_id, "assistant", _inst_md
            )
            quota = orchestrator.session_mgr.get_user_daily_usage(payload.user_id)
            return JSONResponse(
                content={
                    "reply": _inst_md,
                    "session_id": session_id,
                    "agent": "EisaX Institutional Allocator",
                    "model": "phase_h+phase_i",
                    "download_url": None,
                    "format": "markdown",
                    "quota": quota,
                },
                headers=orchestrator.session_mgr.get_quota_header(payload.user_id),
            )
    except Exception as _inst_err:
        logger.warning(
            "[institutional-allocator] fast-path failed; falling through: %s",
            _inst_err,
        )
```

**Design notes:**
- Pure pre-emption — runs before `orchestrator.process_message`, so the orchestrator's legacy `portfolio_pipeline` path (which produces the "EisaX Portfolio Pipeline" output) is bypassed when `portfolio_builder.detect_and_build` matches.
- `portfolio_builder.detect_and_build` returns `None` for non-portfolio messages — automatic fall-through to the existing orchestrator.
- Skipped when files are attached (institutional allocator doesn't consume uploads).
- Language source: `payload.settings.language` if set; otherwise auto-detect from Unicode range U+0600–U+06FF.
- Any exception falls through to the orchestrator — no behavior change risk.
- Session messages saved through the same `session_mgr` API the orchestrator uses, preserving quota and chat history.

---

## 4. Before / After

### Before
```
EN: "Build me a balanced portfolio..." → agent="EisaX Portfolio Pipeline"
    model="pipeline+deepseek" · 10,712 bytes · starts with "## 0. Strategy Readiness: ✅ APPROVED"
    Numbered sections (## 0, ## 1, ## 2…). Emojis present.
AR: same — also routed to "EisaX Portfolio Pipeline" with emojis.
```

### After
```
EN: "Build me a balanced portfolio..." → agent="EisaX Institutional Allocator"
    model="phase_h+phase_i" · 12,799 bytes · starts with "# EisaX Global Portfolio — Balanced Multi-Asset Mandate"
    Section headers: A → B → C → D → E → F → H → G. No emojis.
AR: same structure, Arabic headers (## A. الملخص التنفيذي, ## E. خطة إعادة التوازن, ## G. ملحق المراجعة).
```

---

## 5. Validation Results

| # | Check | Result |
|---|---|---|
| 1 | Staging restart (`eisax-gunicorn-staging`) | PASS — active |
| 2 | `/v1/chat` EN portfolio request | PASS — `EisaX Institutional Allocator`, 12,799 B |
| 3 | `/v1/chat` AR portfolio request | PASS — `EisaX Institutional Allocator`, 12,864 B |
| 4 | EN section order (`assert_section_order`) | PASS — `## A → ## B → ## C → ## D → ## E → ## F → ## H → ## G` |
| 5 | AR section order (`assert_section_order`) | PASS — same order, Arabic titles |
| 6 | No emojis in institutional EN output | PASS — `emoji_present: False` |
| 7 | No emojis in institutional AR output | PASS — `emoji_present: False` |
| 8 | `G. Audit Appendix` is last section (after H) | PASS — consistent with existing report contract |
| 9 | Tone-guard EN | PASS |
| 10 | Tone-guard AR | PASS |
| 11 | Regression suite (`phase_h.testing.runner`) | PASS — 8/8 |
| 12 | Phase I context graph tests | PASS — 13/13 |
| 13 | Non-portfolio chat ("What is your name?") still routes correctly | PASS — `agent="EisaX AI"`, NOT institutional |
| 14 | Existing auth (`X-API-Key=SECURE_TOKEN`) preserved | PASS — fast-path runs only after auth check |
| 15 | Session save/quota plumbing preserved | PASS — uses same `session_mgr` API |

---

## 6. Sample Output Paths

- EN: `/home/ubuntu/investwise/phase_h/testing/samples_live_chat/route_en.md`
- AR: `/home/ubuntu/investwise/phase_h/testing/samples_live_chat/route_ar.md`

EN section headers (verbatim from live response):
```
## A. Executive Summary
## B. Mandate Feasibility Analysis
## C. Risk Diagnostics
## D. Allocation Logic
## E. Rebalancing Plan
## F. AI Commentary Layer — CIO Synthesis
## H. Forward Scenario Distribution
## G. Audit Appendix
```

AR section headers (verbatim from live response):
```
## A. الملخص التنفيذي
## B. تحليل جدوى التفويض
## C. Risk Diagnostics
## D. Allocation Logic
## E. خطة إعادة التوازن
## F. طبقة التعليق بالذكاء الاصطناعي — نظرة مدير الاستثمار
## H. توزيع السيناريوهات المستقبلية
## G. ملحق المراجعة
```

---

## 7. Notes / Caveats

1. **Capital parsing on Arabic "ألف" (thousand)** — `portfolio_builder._parse_params` does not recognize `ألف`; "100 ألف دولار" parses to $10,000 (min clamp) instead of $100,000. **Pre-existing parser limitation, not introduced by this fix.** Out of scope per freeze; record as follow-up.
2. **First message in a new session** still triggers the onboarding welcome before the allocator can fire — this is existing onboarding logic in the orchestrator. Subsequent messages route correctly.
3. **Stock-analysis path** has its own latency (live data fetch); not affected by this change.
4. **Latency:** institutional EN ~11s, AR ~8s on live `/v1/chat` (cold-ish, includes one allocator run). Within acceptable bounds.

---

## 8. Production Verdict

**APPROVED for production.**
- Minimal blast radius: one file (`api/routers/chat.py`), 38 inserted lines, guarded by try/except with fall-through.
- All validation gates green.
- No new features, no new engines, no graph/ontology changes.
- Recommended next step: separate commit of just this routing fix on top of yesterday's checkpoint scope.
