# EisaX Security / Ops Runbook

**Purpose:** keep the current hardened state stable and catch regressions early.  
**Applies to:** `ai.eisax.com`, `agent.eisax.com`, mail/admin surfaces, and the production server at `141.145.153.23`.

## 1. Daily Checks

Run these from any admin machine:

```powershell
curl.exe -s https://ai.eisax.com/v1/version
curl.exe -s https://agent.eisax.com/health
curl.exe -s -o NUL -D - https://ai.eisax.com/v1/download/test.pdf
curl.exe -s -o NUL -D - https://agent.eisax.com/backups/test.bak
curl.exe -s -o NUL -D - https://ai.eisax.com/admin/stats
```

Expected:
- `/v1/version` => `{"status":"ok"}`
- `agent.eisax.com/health` => `{"status":"ok"}`
- unauthenticated `/v1/download/*` => `401`
- `/backups/*` or `*.bak` on `agent.eisax.com` => `404`
- `/admin/stats` without cookie => `403`

Then confirm services on the server:

```bash
sudo systemctl is-active nginx
sudo systemctl is-active eisax-gunicorn.service
sudo systemctl is-active eisax-gunicorn-staging.service
```

All three must return `active`.

## 2. Weekly Checks

SSH to the server and verify logs, rotation, and disk growth:

```bash
cd /home/ubuntu/investwise
ls -lh backend.log backend.log.* gunicorn_error.log pipeline.log
sudo cat /etc/logrotate.d/eisax
df -h /
free -h
```

Expected:
- `backend.log.1/2/3` exist over time
- `/etc/logrotate.d/eisax` exists and includes:
  - `daily`
  - `rotate 7`
  - `compress`
  - `copytruncate`
  - `maxsize 20M`
- disk and memory remain comfortably below pressure thresholds

Check that no backup artifacts or secret backups have reappeared:

```bash
cd /home/ubuntu/investwise
find . -maxdepth 1 \\( -name "*.bak*" -o -name "*.diff" -o -name "current_diff.txt" \\)
find . -maxdepth 1 -name ".env.bak*"
```

Expected: no results.

## 3. After Every Security-Sensitive Change

If any of these change:
- auth/session logic
- `/admin/*`
- `/v1/download/*`
- upload / file parsing
- nginx rules
- staging analyze payload shape

run this smoke suite:

```powershell
curl.exe -s https://ai.eisax.com/v1/version
curl.exe -s https://agent.eisax.com/health
curl.exe -s -o NUL -D - https://ai.eisax.com/v1/download/test.pdf
curl.exe -s -o NUL -D - https://ai.eisax.com/admin/stats
curl.exe -s -o NUL -D - https://agent.eisax.com/backups/test.bak
$body='query=Analyze%20BTC&report_language=en'
curl.exe -s -X POST https://agent.eisax.com/api/analyze -H "Content-Type: application/x-www-form-urlencoded" --data $body
```

Confirm:
- staging analyze still returns `summary/verdict/.../teaser`
- staging analyze does **not** return `full_report`, `html_report`, or `report_json`
- no public admin access
- no public download access
- no backup artifact serving

## 4. Admin Session Checks

Use these when validating the admin login flow:

1. Wrong password:
   - `POST /admin/login` => `403`
2. Correct `ADMIN_PASSPHRASE`:
   - `POST /admin/login` => `200` with cookie:
     - `HttpOnly`
     - `Secure`
     - `SameSite=Strict`
     - `Max-Age=14400`
3. `GET /admin/stats` with cookie => `200`
4. `POST /admin/logout` => `200`
5. `GET /admin/stats` after logout => `403`

Never reintroduce:
- `X-Admin-Key` from browser JavaScript
- token in query string
- admin token in `localStorage` or `sessionStorage`
- `/admin/auth`

## 5. File Parsing Checks

When touching `core/file_processor.py`, verify:

- blocked:
  - `.exe`
  - `.sh`
  - fake `.pdf`
  - fake `.png`
  - malformed Office ZIP
- allowed:
  - valid `.txt`
  - valid `.csv`
  - valid `.pdf`

Must remain true:
- extension allowlist enforced
- magic-bytes check enforced
- `python-magic` MIME check used when installed
- ZIP uncompressed guard remains `50 MB`
- subprocess timeout remains `20s`
- image dimension cap remains `8000px`
- unknown-type fallback stays disabled

## 6. Token / Secret Hygiene

Monthly or after any incident:

1. Rotate:
   - `SECURE_TOKEN`
   - `ADMIN_PASSPHRASE`
2. Restart:

```bash
sudo systemctl restart eisax-gunicorn.service
sudo systemctl restart eisax-gunicorn-staging.service
```

3. Re-run the smoke suite
4. Confirm old tokens return `403`

Do not:
- keep secret backups in the repo root
- copy secrets into notes, diffs, or static files
- expose build metadata again via `/v1/version`

## 7. Incident Triage Order

If something looks wrong, check in this order:

1. Public edge regression
   - `/v1/download/*` returning anything other than `401`
   - `*.bak` or `/backups/*` not returning `404`
   - `/admin/stats` public access
2. Auth regression
   - browser sees admin token
   - admin pages stop using cookie flow
3. Staging payload regression
   - `full_report` / `report_json` reappear publicly
4. Parser regression
   - unsupported files accepted
   - worker hangs

## 8. Key Paths

- App root: `/home/ubuntu/investwise`
- Production service: `eisax-gunicorn.service`
- Staging service: `eisax-gunicorn-staging.service`
- Nginx site: `/etc/nginx/sites-available/agent.eisax.com`
- Logrotate config: `/etc/logrotate.d/eisax`
- Final closure reference:
  - `/home/ubuntu/investwise/EISAX_SECURITY_FINAL_CLOSURE_2026-04-19.md`

## 9. Non-Negotiables

These should never regress:

- no `SECURE_TOKEN` in login responses
- no public report downloads
- no public backup artifacts
- no browser-side admin token handling
- no caller-controlled `user_id` overriding JWT ownership
- no return of `full_report` / `html_report` / `report_json` from public staging

