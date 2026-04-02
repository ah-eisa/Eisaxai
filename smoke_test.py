#!/usr/bin/env python3
"""
EisaX Smoke Tests
Usage:  python3 smoke_test.py [--url http://localhost:8000] [--full]
        --full  includes slow stock-analysis tests (adds ~60s)
"""
import sys, os, time, json, subprocess, argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--url",  default="http://localhost:8000")
parser.add_argument("--full", action="store_true", help="include slow stock tests")
args = parser.parse_args()

BASE  = args.url.rstrip("/")
TOKEN = os.getenv("SECURE_TOKEN", "")
ADMIN = os.getenv("ADMIN_TOKEN",  "")

PASS = FAIL = WARN = 0

# ── Colours ───────────────────────────────────────────────────────────────────
G, R, Y, B, RST = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"

def ok(name, detail=""):
    global PASS; PASS += 1
    print(f"  {G}✅ PASS{RST}  {name}" + (f"  — {detail}" if detail else ""))

def fail(name, detail=""):
    global FAIL; FAIL += 1
    print(f"  {R}❌ FAIL{RST}  {name}" + (f"  — {detail}" if detail else ""))

def warn(name, detail=""):
    global WARN; WARN += 1
    print(f"  {Y}⚠️  WARN{RST}  {name}" + (f"  — {detail}" if detail else ""))

def section(title):
    print(f"\n{B}── {title} {'─'*(52-len(title))}{RST}")

# ── HTTP helper (uses curl — avoids Python urllib quirks) ─────────────────────
def req(method, path, body=None, token=TOKEN, timeout=20, extra_headers=None):
    """Returns (status_code:int, body:dict|str)"""
    cmd = [
        "curl", "-s", "-o", "/tmp/_smoke_body", "-w", "%{http_code}",
        "-X", method,
        "-H", f"access-token: {token}",
        "-H", "Content-Type: application/json",
        "--max-time", str(timeout),
        "--connect-timeout", "5",
    ]
    if extra_headers:
        for h in extra_headers:
            cmd += ["-H", h]
    if body:
        cmd += ["-d", json.dumps(body)]
    cmd.append(BASE + path)

    try:
        t0  = time.monotonic()
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout+5)
        ms  = (time.monotonic() - t0) * 1000
        status = int(out.strip())
        try:
            with open("/tmp/_smoke_body") as f:
                raw = f.read()
            data = json.loads(raw)
        except Exception:
            data = {}
        return status, data, ms
    except subprocess.TimeoutExpired:
        return 0, {}, timeout * 1000
    except Exception as e:
        return 0, {"error": str(e)}, 0


# ═══════════════════════════════════════════════════════════════
section("1. Server Reachability")

status, _, ms = req("GET", "/", timeout=10)
if status in (200, 301, 302): ok("GET /",  f"{ms:.0f}ms → HTTP {status} (redirect to eisax.com)")
else:                          fail("GET /", f"HTTP {status}")

# ═══════════════════════════════════════════════════════════════
section("2. Auth Guard")

status, _, _ = req("GET", "/v1/health", token="wrong-token")
if status == 403:   ok("Wrong token → 403")
else:               fail("Wrong token → 403", f"got {status}")

status, _, _ = req("GET", "/v1/health", token="")
if status == 403:   ok("Empty token → 403")
else:               warn("Empty token", f"got {status}")

# ═══════════════════════════════════════════════════════════════
section("3. Health Check")

status, body, ms = req("GET", "/v1/health", timeout=90)
if status in (200, 207):
    overall = body.get("status", "?")
    ok("GET /v1/health", f"{ms:.0f}ms  overall={overall}")
    for svc, info in body.get("services", {}).items():
        s   = info.get("status", "?")
        lat = f"{info.get('latency_ms',0):.0f}ms"
        detail = (info.get("detail") or "")
        dtl = detail[:55]
        if   s == "ok":       ok(f"  {svc}", lat)
        elif "429" in detail: warn(f"  {svc}", f"{lat} {dtl}")
        elif s == "degraded": warn(f"  {svc}", f"{lat} {dtl}")
        else:                 fail(f"  {svc}", f"{lat} {dtl}")
else:
    fail("GET /v1/health", f"HTTP {status}")

# ═══════════════════════════════════════════════════════════════
section("4. Chat — Greetings (fast)")

status, body, ms = req("POST", "/v1/chat",
    body={"message":"مرحبا","user_id":"smoke","session_id":"smoke-g1"}, timeout=30)
reply = body.get("reply","")
if status == 200 and reply:  ok("Arabic greeting", f"{ms:.0f}ms — {reply[:45]}")
else:                         fail("Arabic greeting", f"HTTP {status}")

status, body, ms = req("POST", "/v1/chat",
    body={"message":"Hello","user_id":"smoke","session_id":"smoke-g2"}, timeout=30)
if status == 200 and body.get("reply"):  ok("English greeting", f"{ms:.0f}ms")
else:                                     fail("English greeting", f"HTTP {status}")

# ═══════════════════════════════════════════════════════════════
section("5. Portfolio Gate")

status, body, ms = req("POST", "/v1/chat",
    body={"message":"ابني لي محفظة باجمالي 100 الف دولار","user_id":"smoke","session_id":"smoke-p1"}, timeout=60)
reply = body.get("reply","")
if status == 200 and any(w in reply for w in ["محتاج","المبلغ","المخاطرة","معلومات","تفاصيل","بالتأكيد"]):
    ok("Vague portfolio → asks questions", f"{ms:.0f}ms")
elif status == 200:
    warn("Portfolio gate did not trigger", reply[:80])
elif status == 0:
    warn("Portfolio gate timeout/connection issue", "curl returned 0")
else:
    fail("Portfolio gate", f"HTTP {status}")

# ═══════════════════════════════════════════════════════════════
section("6. Error Handling")

status, body, ms = req("POST", "/v1/chat",
    body={"message":"","user_id":"smoke","session_id":"smoke-e1"}, timeout=20)
if status in (200, 400, 422):  ok("Empty message handled", f"HTTP {status}")
elif status == 0:               warn("Empty message → timeout/connection issue", f"curl returned 0")
else:                           warn("Empty message", f"HTTP {status}")

long_msg = "ا" * 4000
status, body, ms = req("POST", "/v1/chat",
    body={"message": long_msg,"user_id":"smoke","session_id":"smoke-e2"}, timeout=30)
if status in (200, 400, 413, 422):  ok("Long message handled", f"HTTP {status}")
else:                                warn("Long message", f"HTTP {status}")

# ═══════════════════════════════════════════════════════════════
section("7. Ticker Resolver (unit)")

try:
    sys.path.insert(0, "/home/ubuntu/investwise")
    from core.tools.ticker_resolver import resolve_ticker
    cases = [("gold","GC=F"),("الذهب","GC=F"),("bitcoin","BTC-USD"),
             ("ارامكو","2222.SR"),("sp500","^GSPC"),("AAPL","AAPL")]
    bad = [(q,r,resolve_ticker(q)) for q,r in cases if resolve_ticker(q)!=r]
    if not bad:  ok("resolve_ticker", f"{len(cases)} queries correct")
    else:
        for q,exp,got in bad:
            fail(f"resolve_ticker({q!r})", f"got {got!r} expected {exp!r}")
except Exception as e:
    fail("Ticker resolver", str(e))

# ═══════════════════════════════════════════════════════════════
section("8. Admin Endpoints")

status, _, ms = req("GET", "/admin/logs", timeout=10)
if status == 200:  ok("GET /admin/logs", f"{ms:.0f}ms")
else:              warn("GET /admin/logs", f"HTTP {status}")

# Rate limit: 429 returns Arabic message
# (skip actual flood test to avoid self-DoS)

# ═══════════════════════════════════════════════════════════════
if args.full:
    section("9. Stock Analysis (slow — ~60s)")

    status, body, ms = req("POST", "/v1/chat",
        body={"message":"what is the price of AAPL?","user_id":"smoke","session_id":"smoke-s1"},
        timeout=150)
    reply = body.get("reply","")
    if status == 200 and reply and len(reply) > 20:
        ok("AAPL price query", f"{ms:.0f}ms")
    else:
        fail("AAPL price query", f"HTTP {status} reply={reply[:60]}")

    status, body, ms = req("POST", "/v1/chat",
        body={"message":"سعر الذهب","user_id":"smoke","session_id":"smoke-s2"},
        timeout=150)
    reply = body.get("reply","")
    if status == 200 and reply and len(reply) > 10:
        ok("Gold price Arabic", f"{ms:.0f}ms")
    else:
        warn("Gold price Arabic", f"HTTP {status} reply={reply[:60]}")

# ═══════════════════════════════════════════════════════════════
total = PASS + FAIL + WARN
print(f"\n{'═'*57}")
print(f"  {B}{PASS} passed  {FAIL} failed  {WARN} warnings  ({total} total){RST}")
if args.full:
    print("  (full mode — stock analysis included)")
print(f"{'═'*57}\n")

sys.exit(1 if FAIL > 0 else 0)
