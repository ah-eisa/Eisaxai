# EisaX Security — Final Closure Reference

**Closure Date:** 2026-04-19

This reference captures the verified security closure state after the April 19 remediation cycle.

## 1. Admin Session — HttpOnly Cookie

| Item | Status |
|---|---|
| `/admin/login` accepts `ADMIN_PASSPHRASE` only | ✅ |
| `/admin/auth` (legacy) removed | ✅ `404` |
| `SECURE_TOKEN` rejected as admin password | ✅ `403` |
| Cookie flags: `HttpOnly; Secure; SameSite=Strict; Max-Age=14400` | ✅ |
| `admin_logs.html` + `admin_analytics.html`: no token in JS storage path for admin session | ✅ |

## 2. User Routes — JWT Ownership

| Item | Status |
|---|---|
| `/api/history/*`, `/v1/download/*`, and portfolio routes require Bearer JWT | ✅ |
| `user_id` is derived from JWT payload, not request body | ✅ |
| Download links use opaque UUID tokens with TTL and `user_id` binding | ✅ |
| Timestamp-guessable export filenames are no longer exposed as direct download identifiers | ✅ |

## 3. File Parsing — Validation + Process Isolation

| Item | Status |
|---|---|
| Extension allowlist rejects `.exe`, `.sh`, and unknown types | ✅ |
| Magic-bytes validation blocks mismatched file content | ✅ |
| `python-magic` MIME detection adds a second validation layer when available | ✅ |
| Size limits enforced before parsing (PDF 10MB, Excel 5MB, Image 8MB) | ✅ |
| ZIP uncompressed-size guard 50MB for `xlsx` / `docx` / `pptx` | ✅ |
| Parsing runs in `ProcessPoolExecutor` with a 20s timeout | ✅ |
| Image dimension cap 8000px | ✅ |
| Unknown-type fallback removed; explicit reject only | ✅ |

## 4. SECURE_TOKEN — Isolated & Rotated

| Item | Status |
|---|---|
| Token rotated | ✅ |
| Token no longer returned in login responses | ✅ |
| Browser no longer receives the token for admin flows | ✅ |
| Token retained only for internal services / CLI fallback paths | ✅ |
| `.env.bak.*` secret backup files removed | ✅ |

## 5. Infrastructure Guards

| Item | Status |
|---|---|
| Session IDs use `uuid4` instead of timestamps | ✅ |
| Nginx blocks `.bak`, `/backups/`, and dotfiles | ✅ |
| Request body size limit: 4MB max | ✅ |
| Rate limits applied to admin endpoints (5–30/min) | ✅ |
| `backend.log` rotation via Python `RotatingFileHandler` (10MB × 3 backups) | ✅ |
| `gunicorn_error.log`, `pipeline.log`, and companion logs covered by `/etc/logrotate.d/eisax` | ✅ |

## Hardening Residuals

These are not treated as open public breaches, but remain follow-up hardening work:

- `SECURE_TOKEN` fallback still exists in `_check_admin` for internal callers and CLI/service compatibility.
- File parsing uses process isolation and validation hardening, but not cgroup/container-grade sandboxing.
- Secrets still live in `.env` instead of a managed secret store.
