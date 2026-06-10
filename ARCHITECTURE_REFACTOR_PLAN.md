# EisaX — Architecture Refactor Plan
**As of: 2026-05-04 | Author: Head of System Architecture**

---

## Executive Summary

Total project code: **~99,584 lines** across ~150 Python files.  
Six files account for **~26,000 lines (26%)** of the entire codebase.  
Root causes: monolith handlers, data-as-code, copy-paste duplication, zero separation of concerns.

---

## Critical Problems Found

### Problem Files (Sorted by Severity)

| Priority | File | Lines | Issues |
|----------|------|-------|--------|
| 🔴 P1 | `core/agents/finance.py` | 7,398 | One class, 31 methods, `_handle_analytics` alone = 4,334 lines |
| 🔴 P1 | `api_bridge_v2.py` | 6,425 | 197 fns, 0 class structure, 3 duplicated symbols, 30+ routes mixed |
| 🟠 P2 | `core/services/market_updates.py` | 4,314 | 5 mixed domains, no classes, report+data+social all together |
| 🟠 P2 | `core/local_tickers.py` | 3,635 | Data dictionary masquerading as Python (only 4 real functions) |
| 🟡 P3 | `arab_dashboard_fixed.py` | 2,977 | UI + data logic + AI agent calls in root module |
| 🟡 P3 | `api/routes/chat.py` | 1,893 | Copy-paste file: `MessagePayload` ×3, `_check_admin` ×2 |

---

## Phase 1 — `core/agents/finance.py` (7,398 → ~600 lines each)

### Problem
`FinancialAgent` contains **31 methods** across **8 unrelated domains**.  
`_handle_analytics()` = lines 2784–7118 = **4,334 lines** (59% of file).  
`_build_scorecard_md()` = lines 1806–2335 = **530 lines**.

### Target Structure
```
core/agents/
├── finance.py              ← thin dispatcher (~300 lines, imports from below)
├── base.py                 ← existing BaseAgent
├── finance_helpers.py      ← existing helpers
├── handlers/
│   ├── __init__.py
│   ├── analytics.py        ← _handle_analytics() (L2784–7118, ~4334 lines)
│   ├── scorecard.py        ← _build_scorecard_md() + _build_factcheck_block() (L1806–2525)
│   ├── report_data.py      ← _precompute_report_data() (L2549–2783)
│   ├── cio.py              ← _handle_cio_analysis() (L398–897)
│   ├── fixed_income.py     ← _handle_fixed_income() + _handle_egypt_bonds() (L928–1366)
│   ├── portfolio.py        ← _handle_portfolio_show/add/remove/account (L7299–7398)
│   ├── trade.py            ← _handle_trade() + _handle_greeks() + _handle_forecast() (L7119–7298)
│   └── export.py           ← _handle_export() + _handle_report() (L1367–1508)
```

### Migration Steps
1. Create `core/agents/handlers/` package
2. Extract each handler method into its own module as a standalone function/class
3. Pass `self` context as explicit arguments (avoids deep inheritance coupling)
4. Replace method bodies in `FinancialAgent` with one-line delegations:
   ```python
   def _handle_analytics(self, sid, mem, msg, settings):
       from core.agents.handlers.analytics import handle_analytics
       return handle_analytics(self, sid, mem, msg, settings)
   ```
5. After delegation works: remove handler methods from `FinancialAgent`

**Target size per handler file: 400–800 lines. `finance.py`: ~300 lines.**

---

## Phase 2 — `api_bridge_v2.py` (6,425 → ~300 lines each)

### Problem
- 197 functions, 30+ routes, all in one file
- **Duplicated symbols:**
  - `_require_jwt` defined at L64 AND L5510
  - `_coerce_chat_payload` defined at L174 AND L3864
  - `MessagePayload` defined multiple times
- Route domains mixed: staging, admin, guest mgmt, chat, portfolio, export, TTS

### Target Structure
```
api/
├── app.py                          ← FastAPI app creation + include_router calls only (~50 lines)
├── middleware/
│   ├── __init__.py
│   └── auth.py                     ← _resolve_auth, _require_jwt (single definition)
├── models/
│   ├── __init__.py
│   └── payloads.py                 ← MessagePayload, TTSRequest, HtmlExportPayload, etc. (single defs)
├── routers/
│   ├── __init__.py
│   ├── staging.py                  ← /staging-api/* endpoints (L1146–2154)
│   ├── admin.py                    ← /v1/admin/*, guest user mgmt (L1537–1688)
│   ├── chat.py                     ← /v1/chat, /chat (L3973–4145)
│   ├── portfolio.py                ← /v1/portfolio/* (L3587–3808)
│   ├── reports.py                  ← /v1/report, /v1/pilot-report (L3876–3972)
│   └── export.py                   ← upload, TTS, translate, HTML export
├── helpers/
│   ├── staging_helpers.py          ← all _staging_* functions (L395–1145)
│   ├── guest_helpers.py            ← _guest_trial_*, _read_htpasswd_* (L1201–1480)
│   └── report_helpers.py           ← _sanitize_*, _apply_report_meta_* (L856–1018)
└── api_bridge_v2.py               ← DEPRECATED: thin shim that imports from api/app.py
```

### Migration Steps
1. Create `api/models/payloads.py` — move all `BaseModel` classes, **eliminating duplicates**
2. Create `api/middleware/auth.py` — single `_require_jwt`, `_resolve_auth`, `_bearer`
3. Extract route groups into `api/routers/` using `APIRouter`
4. Create `api/app.py` that assembles all routers
5. Point existing `api_bridge_v2.py` to `from api.app import app` (backward compat)

**Target: 0 duplicate symbols. Each router file: 300–500 lines.**

---

## Phase 3 — `core/services/market_updates.py` (4,314 → ~500 lines each)

### Problem
Five distinct domains in one flat module — no classes, no separation:
- Market data collection (`_collect_market_data`, `_get_fear_greed`)
- Regime detection (`_determine_regime`, `build_eisax_stance`)
- Report text generation (`_generate_full_report_text`, `_generate_cio_daily_report_text`)
- LinkedIn/social content (`_generate_linkedin_text_ai`, formatting hooks)
- Text formatters / utilities (`_clean_text`, `_format_report_number`, etc.)

### Target Structure
```
core/services/
├── market_updates.py           ← orchestrator only: calls other modules (~150 lines)
├── market_data_collector.py    ← _collect_market_data, _get_fear_greed, _load_pipeline* (~600 lines)
├── market_regime.py            ← _determine_regime, build_eisax_stance, build_invalidation_logic, _build_asset_allocation_view (~400 lines)
├── market_report.py            ← _generate_full_report_text, _generate_cio_daily_report_text, _enrich_full_report (~700 lines)
├── market_social.py            ← LinkedIn hooks, _daily_linkedin_hook, _weekly_linkedin_hook, _fit_word_window (~400 lines)
└── market_formatters.py        ← _clean_text*, _format_report_number, _snapshot_brief, _trigger_hierarchy_lines (~300 lines)
```

---

## Phase 4 — `core/local_tickers.py` (3,635 → ~50 lines)

### Problem
3,600 lines of Python dictionary literals for ticker data. Only **4 functions** at the end.  
Data is hardcoded in source — no hot-reload, no tooling, bloats IDE and grep.

### Target Structure
```
data/
└── tickers/
    ├── saudi.json          ← SAUDI_TICKERS dict → JSON
    ├── uae.json            ← UAE_TICKERS dict → JSON
    └── egypt.json          ← EGX_TICKERS dict → JSON

core/local_tickers.py       ← loader only (~50 lines):
                               import json, functools
                               @functools.cache
                               def _load(market): return json.load(...)
                               SAUDI_TICKERS = _load("saudi")
                               # ... same API, zero behavior change
```

### Migration Steps
1. Write a one-time script: `scripts/export_tickers_to_json.py`
2. Run it → generates `data/tickers/*.json`
3. Replace `core/local_tickers.py` body with JSON loader
4. Verify `get_all_tickers_flat()`, `get_market_sectors()`, etc. still pass tests

**Benefit: IDE performance, grep usability, allows non-dev updates to ticker data.**

---

## Phase 5 — `api/routes/chat.py` (1,893 → ~400 lines)

### Problem
Copy-paste concatenation. Exact duplicates in same file:
- `MessagePayload` — defined 3×
- `TTSRequest` — defined 2×
- `HtmlExportPayload` — defined 2×
- `_check_admin` — defined 2×
- `TranslatePayload` — defined 2×

### Fix
1. All `BaseModel` classes → `api/models/payloads.py` (shared with `api_bridge_v2.py` cleanup)
2. `_check_admin` → `api/middleware/auth.py`
3. Remove all duplicates; import from shared modules
4. Split remaining routes: chat-specific logic stays, file/export routes → `api/routes/export.py`

---

## Phase 6 — `arab_dashboard_fixed.py` (2,977 lines, lower priority)

### Problem
Root-level file mixing: Dash/UI layout, AI agent calls, data formatting, CSS generation (713 lines of CSS in Python!).

### Target
```
dashboard/
├── __init__.py
├── app.py                  ← Dash app init + layout assembly
├── ai_bridge.py            ← _agent_chat, _get_eisax_agent, _should_use_agent_for_ai
├── data_layer.py           ← _get_pipeline, build_ai_market_context
├── ui_components.py        ← Dash component builders
└── styles.py               ← _build_css (713 lines → CSS file)
```

---

## Secondary Files to Address (P3, after Phases 1–4)

| File | Lines | Action |
|------|-------|--------|
| `core/portfolio_manager.py` | 1,551 | Split: position tracking vs. reporting vs. risk |
| `core/services/market_route_handler.py` | 1,473 | Extract to router + service layer |
| `core/orchestrator.py` | 1,608 | Merge `_classify_intent` into existing `core/intent_classifier.py` |
| `core/fixed_income.py` | 1,711 | Split: Egypt bonds vs. Gulf bonds vs. yield calcs |
| `core/scorecard.py` | 1,181 | Extract sector-specific scorers |

---

## Execution Sequence

```
Phase 1: finance.py handlers      → Week 1  (highest risk, highest ROI)
Phase 2: api_bridge_v2.py routers → Week 2  (deduplicate, route isolation)
Phase 3: market_updates.py split  → Week 3  (functional decomposition)
Phase 4: local_tickers → JSON     → Week 3  (data/code separation, 1 day)
Phase 5: chat.py dedup            → Week 4  (quick cleanup)
Phase 6: dashboard refactor       → Week 5  (lower priority, UI changes)
```

---

## Rules for All Phases

1. **One extraction per PR** — no mixing phase 1 and phase 2 changes
2. **No behavior changes** — extract only, no refactoring logic at same time
3. **Thin delegation first** — add `from handlers.x import fn; return fn(...)` before deleting old code
4. **Keep backward-compat imports** — `from core.agents.finance import FinancialAgent` must keep working
5. **Test after each extraction** — run the staging API on port 8001 and hit each endpoint

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Largest file | 7,398 lines | ~800 lines |
| Files > 1,500 lines | 10 files | 0 files |
| Duplicate symbol count | 7+ | 0 |
| Data lines in Python | 3,500+ | 0 |
| `api_bridge_v2.py` | 6,425 lines | ~100 lines (shim) |
