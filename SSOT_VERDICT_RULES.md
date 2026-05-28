# SSOT Verdict & Worker — Reference Rules

Last updated: 2026-05-28
Owner: EisaX core. Touched by staging.py / decision_engine.py / fact_sheet.py / report_reconciler.py.

This file documents three things that explain why a given response looks the way it does:

1. **Verdict score-band rules** (F3/F4) — score → verdict mapping in `decision_engine.py` (TG3 + Rule 8A)
2. **Confidence label rules** (F5) — `_build_report_meta` extraction logic in `pilot_report_parsers.py`
3. **Multi-worker post_fork test plan** (F6) — what to verify before scaling `workers=1 → N`

---

## 1. Verdict Score Bands (decision_engine.py)

The composite EisaX score (`_scorecard_score`, 0–100) is reconciled with the upstream verdict by three terminal guardrails. All rules require `_scorecard_score > 0`.

| Rule | Trigger | Action | Source |
|---|---|---|---|
| **TG3-1** | `score < 55` **AND** `downside_high` **AND** verdict ∈ {BUY, HOLD} | verdict ← **REDUCE** | line 281 |
| **TG3-2** | `60 ≤ score ≤ 74` **AND** `no_clear_edge` **AND** verdict = BUY | verdict ← **HOLD** | line 293 |
| **Rule 8A** | `score ≥ 75` **AND** `upside_pct ≥ 20%` **AND** verdict = HOLD | verdict ← **BUY** (Tactical) | line 313 |

Where:

```
downside_high  =  risk_score > 60  OR  beta > 1.8  OR  bearish_count ≥ 3
no_clear_edge  =  upside_pct < 15%  AND  risk_score > 40
```

### Worked examples from 2026-05-28 regression

| Ticker | Score | Inputs | Upstream verdict | Rule fired | Final verdict |
|---|---|---|---|---|---|
| ADNOCGAS.AE | 75 | upside < 20% | BUY | (none — neither TG3-2 since not BUY-in-range nor 8A since upside<20) | **Hold** (passed through) |
| EMAAR.AE | 58 | downside_high not set | REDUCE | (none — score=58 not <55) | **Reduce** (came from scorecard upstream) |
| 2222.SR (Aramco) | 75 | upside ≥ 20% | HOLD → 8A → BUY | Rule 8A | **Buy** |
| QNBK.QA | 69 | no_clear_edge | BUY → TG3-2 → HOLD? Yet log shows Buy | (8A protect path or upside sufficient) | **Buy** |

> ⚠ **Behavioral note**: Score=58 producing Reduce did not come from TG3-1 (that needs <55). It came from the scorecard layer (`scorecard.py`) before TG3. Document the scorecard table separately if/when that becomes a question.

### When verdict changes between two runs on the same day
Expected — `_scorecard_score` is recomputed live per request from technicals + fundamentals. The score, risk_score, beta, and upside_pct can all drift between runs depending on TV cache freshness, options-IV feed, and news sentiment. The pipeline guarantees **internal consistency** within a single response (FactSheet ↔ body ↔ verdict) via the reconciler, not stability across runs.

---

## 2. Confidence Label (`_build_report_meta`)

### Sources tried in order

For `confidence_score` (0–100):

1. `_parse_percent_after_label(text, "Verdict Confidence")` — legacy strict format `Verdict Confidence: 60%`
2. `_parse_percent_loose(text, r"Confidence\s+Calibration[^|\n]{0,40}?Score")` — handles `**Confidence Calibration** · Score: **60%**`
3. `_parse_percent_loose(text, r"Confidence\s+Score")` — handles markdown table cell `| Confidence Score | 60% |`
4. Fallback: `conviction_score` argument (passed in by caller)

For `confidence_label` (Low/Medium/High):

1. `_parse_level_label(text, "Confidence")` — strict `Confidence: Medium`
2. `_parse_level_label_loose(text, r"Confidence\s+Calibration")` — wider scan after the calibration title
3. Fallback: `_report_label_from_score(confidence_score)` where ≤59=Low, 60–74=Medium, ≥75=High

### Why all 8 tickers returned `Low` before the fix
The body templates evolved to use `Confidence Calibration · Score: 60%` and `| Confidence Score | 60% |`. Neither matched the legacy strict regex `Verdict Confidence: X%`, so `confidence_score` fell through to the stale `conviction_score=51` argument → `_report_label_from_score(51) = Low`.

After F5 patch: extraction matches both new formats, labels follow score correctly.

---

## 3. Multi-worker post_fork Test Plan (F6)

Current staging config: `workers = 1`. The `post_fork` hook in `gunicorn_staging.conf.py` resets three module-level locks before uvicorn boots:

- `core.analysis_cache._lock` (RLock)
- `core.news_engine_client._cache_lock` (Lock)
- `core.ticker_index._INDEX_LOCK` (Lock)

If we ever scale to `workers ≥ 2`, validate:

| Check | How |
|---|---|
| **C1** — `post_fork` fires on each child | grep `Booting worker with pid` → `Application startup complete` pairs in error log; each pair must be present per worker |
| **C2** — No worker stuck in futex (RSS=24M) after 60s | `ps -o pid,rss,stat -p <child_pids>` — RSS should be ≥80M when ready |
| **C3** — Shared mutable state safe | analysis_cache writes from worker A must not corrupt worker B's reads. Hit the same ticker N=20 times across both workers via parallel curl, then diff the cached parquet rows |
| **C4** — News engine scheduler only fires once | APScheduler is started at import time → if 2 workers boot, two schedulers run. Either gate with `if worker.age == 0` in `post_fork` or move scheduler to a sidecar service |
| **C5** — File descriptors for log files don't collide | Each worker should hold its own fd to `gunicorn_staging_test_*.log`. After 1h, run `lsof -p <child_pids>` and confirm no duplicate inode references |

### Open risk if scaled today
- APScheduler in `engine.py` (line 1 in journald: `EisaX News Collection`) starts at import time. With workers=2, news collection would fire 2× every 15min. Trivial to dedupe via `if os.getpid() == server.pid + 1` in `post_fork`, but untested.

---

## Quick reference — file → role

| File | Role |
|---|---|
| `core/services/fact_sheet.py` | SSOT FactSheet (authoritative price, sma, verdict, sector, currency) |
| `core/services/report_reconciler.py` | Single-pass post-LLM corrections (price/sma/sector/news swaps) |
| `core/services/pilot_report_parsers.py` | Body → report_meta extraction (verdict, scores, confidence labels) |
| `core/services/decision_engine.py` | TG3 + Rule 8A guardrails for upstream verdict |
| `api/routers/staging.py` | Response payload assembly; consumes FactSheet for verdict passthrough |
| `gunicorn_staging.conf.py` | post_fork hook resets module-level locks → kills futex_do_wait deadlock |
