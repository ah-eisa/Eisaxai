# Production Promotion Checklist — SSOT / workers=2 / scheduler-gate

Written: 2026-05-28
Baseline commit: `1f83c87`
Baseline tag: `ssot-staging-baseline-2026-05-28`
Status: **document-only, do not execute** until staging 24h soak completes (2026-05-29 ≈22:45) with no alerts.

---

## A. Files Affected

### A.1 Already on disk under `/home/ubuntu/investwise/` (no copy needed)
Production reads the same `core/services/*.py` and `api_bridge_v2.py` as staging via Python imports. A production restart is sufficient to pick these up.

| File | Change | Risk |
|---|---|---|
| `core/services/fact_sheet.py` | NEW — SSOT builder, crypto/commodity detection, yfinance fallback | Low — pure new module, no in-place mutations |
| `core/services/report_reconciler.py` | NEW — post-LLM corrections | Low — same |
| `core/services/pilot_report_parsers.py` | MODIFIED — `_parse_percent_loose` + `_parse_level_label_loose` helpers + tolerant confidence extraction in `_build_report_meta` | Low — additive; legacy regex still tried first |
| `api_bridge_v2.py` | MODIFIED — line 143 gated on `EISAX_SCHEDULER_OWNER` env var (default `"1"` so single-worker keeps starting scheduler) | Medium — affects scheduler startup. Default behaves identically to pre-change. |
| `api/routers/staging.py` | MODIFIED — `_ssot_fs` hoisting + verdict passthrough | None for production — production does NOT use `/staging-api/*` routes. Only relevant if Phase 3 ports SSOT into `/v1/report`. |
| `SSOT_VERDICT_RULES.md` | NEW — docs | None |

### A.2 New for production
| File | Purpose |
|---|---|
| `gunicorn_production.conf.py` (TO BE CREATED) | Mirror of `gunicorn_staging.conf.py` with prod settings: bind 127.0.0.1:8000, workers=2, timeout=120, error/access logs `gunicorn_test_*.log`, same `post_fork` lock-reset + `when_ready` + O_EXCL scheduler-owner lock at `/tmp/eisax-production-scheduler.lock` |
| `/etc/systemd/system/eisax-gunicorn.service` (TO BE EDITED) | Replace inline `gunicorn` flags with `--config /home/ubuntu/investwise/gunicorn_production.conf.py` |

---

## B. Services Needing Restart / Reload

| Service | Action | Reason |
|---|---|---|
| `eisax-gunicorn.service` | **restart** (not reload) | Picks up new shared-module code + new config file. HUP reload won't load a new config file. |
| `eisax-monitor.service` | none | Only reads metrics; doesn't import the changed modules at runtime |
| `eisax-news.service` | none | Runs independently — own scheduler, not the in-process APScheduler |
| `eisax-pipeline.service` | none | Same |
| `eisax-learning.service` | none | Same |
| `eisax-inbound.service` | none | Same |
| `eisax-telegram.service` | none | Same |
| `eisax.service` (legacy, port 8512) | optional restart | If it imports the same core modules and you want consistency, restart. Otherwise leave. |

**Expected downtime:** ≤ 5 seconds for `eisax-gunicorn` restart (workers boot in 2s on staging; even doubled, ≤5s). No other service interrupted.

---

## C. Pre-checks (run BEFORE promotion)

```bash
# C-1: Staging soak is clean
tail -5 /home/ubuntu/investwise/logs/staging_soak.jsonl
test -s /home/ubuntu/investwise/logs/staging_soak_alerts.log && echo "FAIL: alerts exist" || echo "PASS: no alerts"

# C-2: Local git matches origin
git -C /home/ubuntu/investwise fetch origin
test "$(git -C /home/ubuntu/investwise rev-parse HEAD)" = "$(git -C /home/ubuntu/investwise rev-parse origin/main)" && echo "PASS" || echo "FAIL"

# C-3: Production currently healthy
curl -s -H "X-API-Key: $TOKEN" http://127.0.0.1:8000/health | jq .status   # expect "online"

# C-4: Capture baseline responses for 3 tickers (compare later)
mkdir -p /tmp/prod-baseline
for T in ADNOCGAS.AE AAPL EMAAR.AE; do
  curl -s -H "X-API-Key: $TOKEN" -X POST http://127.0.0.1:8000/v1/report \
    -d "{\"query\":\"$T\"}" -H "Content-Type: application/json" \
    > /tmp/prod-baseline/$T.json
done

# C-5: Snapshot service unit file + log positions
sudo cp /etc/systemd/system/eisax-gunicorn.service \
        /etc/systemd/system/eisax-gunicorn.service.bak.$(date +%Y%m%d_%H%M)
stat -c %s /home/ubuntu/investwise/logs/gunicorn_test_error.log > /tmp/prod-pre-promote-errlog-offset.txt

# C-6: Confirm no in-flight Redis traffic spikes / external SLO at risk
redis-cli ping                            # PONG
sudo journalctl -u eisax-gunicorn -n 20   # no recent errors

# C-7: Confirm gunicorn_production.conf.py written and sanity-loadable
/home/ubuntu/investwise/venv/bin/python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('cfg','/home/ubuntu/investwise/gunicorn_production.conf.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.workers == 2
assert mod.bind == '127.0.0.1:8000'
assert hasattr(mod, 'post_fork') and hasattr(mod, 'when_ready')
print('PASS: prod config valid')
"
```

All seven pre-checks must PASS before proceeding.

---

## D. Promotion Steps

### Phase 1 — Restart only (low risk, picks up F5 + reconciler in shared core/)
*Do this first; observe before Phase 2.*

```bash
# 1. Daemon-reload (in case unit file edited)
sudo systemctl daemon-reload                                            # no-op if unchanged

# 2. Restart
sudo systemctl restart eisax-gunicorn.service

# 3. Wait for "Application startup complete" × workers
until [ $(sudo journalctl -u eisax-gunicorn --since "30 seconds ago" 2>/dev/null \
          | grep -c "Application startup complete") -ge 2 ]; do sleep 1; done
echo "OK: both workers booted"
```

**Estimated time:** 5 seconds restart + 2 seconds boot = ≤7 s downtime.

### Phase 2 — Production gunicorn config (the futex fix + scheduler gate)
*Only run after Phase 1 has soaked for at least a few hours and is stable.*

```bash
# 1. Edit /etc/systemd/system/eisax-gunicorn.service:
#    Replace ExecStart= line with:
ExecStart=/home/ubuntu/investwise/venv/bin/gunicorn api_bridge_v2:app \
  --config /home/ubuntu/investwise/gunicorn_production.conf.py

# 2. Reload + restart
sudo systemctl daemon-reload
sudo systemctl restart eisax-gunicorn.service

# 3. Verify scheduler single-fire (CRITICAL — different from staging because workers=2)
SVC_SINCE=$(systemctl show eisax-gunicorn -p ActiveEnterTimestamp --value)
test "$(sudo journalctl -u eisax-gunicorn --since "$SVC_SINCE" | grep -c 'Scheduler started')" = "1" \
  && echo "PASS: scheduler single-fire" \
  || echo "FAIL: scheduler fired N times"
test -s /tmp/eisax-production-scheduler.lock \
  && echo "PASS: lock file present" \
  || echo "FAIL: lock missing"
```

**Estimated time:** 5 s daemon-reload + 7 s restart = ≤12 s downtime.

### Phase 3 — SSOT wiring in `/v1/report` and `/v1/pilot-report` (deferred)
*Not part of this promotion. Requires identifying the production-side response shaper and mirroring the `_ssot_fs = build_fact_sheet(...) / reconcile_report(...)` block from `api/routers/staging.py`. Plan separately.*

---

## E. Post-checks (run AFTER each phase)

```bash
# P-1: Service health
sudo systemctl status eisax-gunicorn --no-pager | head -8
test "$(systemctl is-active eisax-gunicorn)" = "active" && echo "PASS" || echo "FAIL"

# P-2: Health endpoint
curl -s -H "X-API-Key: $TOKEN" http://127.0.0.1:8000/health | jq .status     # online

# P-3: Both workers up (Phase 2 onwards)
test "$(pgrep -P $(systemctl show eisax-gunicorn -p MainPID --value) | wc -l)" -ge 2 \
  && echo "PASS" || echo "FAIL"

# P-4: Re-run the same 3 baseline tickers, diff vs baseline
for T in ADNOCGAS.AE AAPL EMAAR.AE; do
  curl -s -H "X-API-Key: $TOKEN" -X POST http://127.0.0.1:8000/v1/report \
    -d "{\"query\":\"$T\"}" -H "Content-Type: application/json" \
    > /tmp/prod-post-promote/$T.json
  jq -S '{verdict: .verdict, risk: .risk_level, confidence: .confidence}' \
    /tmp/prod-baseline/$T.json > /tmp/baseline-$T.norm.json
  jq -S '{verdict: .verdict, risk: .risk_level, confidence: .confidence}' \
    /tmp/prod-post-promote/$T.json > /tmp/post-$T.norm.json
  diff -u /tmp/baseline-$T.norm.json /tmp/post-$T.norm.json && \
    echo "$T: no field diff" || echo "$T: review diff"
done

# P-5: No new errors / no futex
sudo journalctl -u eisax-gunicorn --since "5 minutes ago" 2>&1 \
  | grep -E "Traceback|futex|deadlock" | head -10

# P-6: Scheduler ran exactly once (only valid after Phase 2)
sudo journalctl -u eisax-gunicorn --since "$SVC_SINCE" 2>&1 \
  | grep -E "Scheduler started|scheduler skipped" | wc -l    # expect 2 (1 start + 1 skipped)

# P-7: News collection still firing on schedule
sudo journalctl -u eisax-gunicorn --since "30 minutes ago" 2>&1 \
  | grep -c "EisaX News Collection"     # expect at least 1 (depends on cron position)

# P-8: Memory stable
sudo systemctl status eisax-gunicorn --no-pager | grep -E "Memory:|Peak"
```

Acceptable verdict drift is permitted (LLM/DecisionState variance — documented in `SSOT_VERDICT_RULES.md` §1). **Confidence label must not regress** to `Low` for all tickers (that signals the F5 fix didn't load).

---

## F. Rollback Commands

| Failure mode | Command | Time |
|---|---|---|
| **Phase 1 fails (code regression)** | `git -C /home/ubuntu/investwise checkout 2cb42aa -- core/services/fact_sheet.py core/services/report_reconciler.py core/services/pilot_report_parsers.py api_bridge_v2.py && sudo systemctl restart eisax-gunicorn` | ≤30 s |
| **Phase 1 fails (and you want clean revert via tag)** | `git -C /home/ubuntu/investwise reset --hard 2cb42aa && sudo systemctl restart eisax-gunicorn` (NOTE: this also reverts unstaged work — use cautiously) | ≤30 s |
| **Phase 2 fails (worker won't boot / futex hangs)** | `sudo cp /etc/systemd/system/eisax-gunicorn.service.bak.<timestamp> /etc/systemd/system/eisax-gunicorn.service && sudo systemctl daemon-reload && sudo systemctl restart eisax-gunicorn` | ≤30 s |
| **Phase 2 fails (scheduler double-fires)** | Edit `gunicorn_production.conf.py`: set `workers=1` temporarily. `sudo systemctl restart eisax-gunicorn`. Investigate gate. | ≤2 min |
| **Cannot connect after restart at all** | `sudo kill -9 $(pgrep -f gunicorn_production.conf.py); sudo cp /etc/systemd/system/eisax-gunicorn.service.bak.<timestamp> ...; sudo systemctl restart` | ≤2 min |
| **Repo broken on host** | Use `/home/ubuntu/eisax-backups/ssot-pre-w2-test-20260528_223323.tar.gz` to restore SSOT files | ≤5 min |

**Tag-based revert ceiling:** `git reset --hard 2cb42aa` puts the working tree at the pre-SSOT baseline. Useful as an absolute last resort.

---

## G. Expected Downtime

| Action | Downtime |
|---|---|
| Phase 1 restart | **≤7 s** (gunicorn graceful_timeout = `60` on prod; worker boot ≈2 s) |
| Phase 2 restart with new config | **≤12 s** (daemon-reload + restart + boot) |
| Cumulative if Phase 1 → Phase 2 done back-to-back | ≤20 s |
| Rollback (any) | **≤30 s** to ≤2 min depending on path |

All within a single 60-second customer-facing window if everything goes smoothly. No multi-minute outage expected.

---

## H. Remaining Risks (since production-specific factors)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **P-R1** | Production `timeout=120s` is tight for cold-cache LLM calls (staging timed out at 90s in some calls, would have been at 100s on prod). | Medium | Consider bumping prod timeout to `180s` in the new config file. Decision needed before Phase 2. |
| **P-R2** | Production runs on `ENVIRONMENT=test` not `production` — code may have dev-mode guards still active. | Low | Audit `core/config.py` for `ENVIRONMENT == "test"` branches before Phase 2. |
| **P-R3** | Workers=2 + 2 concurrent LLM calls × multiple users → real 429 pressure on OpenAI/Anthropic. Staging only tested 4 calls. | Medium | Monitor news-filter 429s in post-checks. F-W3 carry-forward. |
| **P-R4** | Parquet cache writes from 2 workers — small-sample safe on staging; production has higher QPS. | Medium | Watch `analysis_cache` warnings in journal for 24 h post-promotion. |
| **P-R5** | Phase 3 (SSOT in `/v1/report`) not done — production responses won't benefit from FactSheet reconciler until that phase ships. Only `/staging-api/*` does. | Acknowledged | Phase 1+2 give us the futex fix and scheduler dedup. SSOT benefits await Phase 3. |
| **P-R6** | `gunicorn_production.conf.py` not yet written. | Low (work item) | Mirror staging config 1:1 with prod port + log paths. ≤30 min work. |
| **P-R7** | If both `eisax.service` (legacy, port 8512) AND `eisax-gunicorn` share import-time module state and you only restart one, state drift is possible. | Low | Restart legacy too at Phase 1 — adds 5 s extra downtime. |

---

## I. Promotion Sequence Recap

```
Day 0 (TODAY): write checklist (this doc)
Day 1 (after 24h soak passes): Phase 1 restart (Window: low-traffic hour)
Day 1 + 2-4h observation
Day 1 (later or Day 2): Phase 2 config swap (Window: low-traffic hour)
Day 2-7: soak Phase 2
Day 8+: Plan Phase 3 (SSOT in /v1/report) as separate engineering task
```

**Estimated total customer-impact:** ≤20 seconds across both phases.
