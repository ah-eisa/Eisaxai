#!/home/ubuntu/investwise/venv/bin/python3
"""
EisaX smoke check — runs automatically after every Python file edit (PostToolUse hook).

Two-stage guard:
  1. py_compile  → catches SyntaxError / IndentationError
  2. import test → catches ImportError / NameError / AttributeError in module-level code

Streamlit entry points are skipped at stage 2 (they'd launch a server).
"""
import sys
import os
import re
import py_compile
import subprocess

# ── Config ────────────────────────────────────────────────────────────────────
VENV_PYTHON  = "/home/ubuntu/investwise/venv/bin/python3"
PROJECT_ROOT = "/home/ubuntu/investwise"

# Streamlit pages / entry-points: skip import test, only syntax-check
SKIP_IMPORT = {
    "arab_dashboard_fixed.py",
    "arab_dashboard.py",
    "app.py",
}

# Relative-path → dotted module name for every importable core file
IMPORTABLE = {
    "portfolio.py":                 "portfolio",
    "core/auth.py":                 "core.auth",
    "core/config.py":               "core.config",
    "core/user_db.py":              "core.user_db",
    "core/streamlit_auth.py":       "core.streamlit_auth",
    "core/portfolio_tracker.py":    "core.portfolio_tracker",
    "core/portfolio_db.py":         "core.portfolio_db",
    "core/market_data.py":          "core.market_data",
    "core/editorial.py":            "core.editorial",
    "core/polish_cache.py":         "core.polish_cache",
    "scripts/alert_monitor.py":     "scripts.alert_monitor",
    "scripts/weekly_digest.py":     "scripts.weekly_digest",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def rel(filepath: str) -> str:
    return os.path.relpath(filepath, PROJECT_ROOT)

def resolve_module(filepath: str):
    """Return dotted module or None (skip)."""
    r = rel(filepath)
    basename = os.path.basename(filepath)
    if basename in SKIP_IMPORT:
        return None          # skip — Streamlit entry point
    return IMPORTABLE.get(r) # None = unknown file, still run syntax check only

def scan_silent_excepts(filepath: str):
    """Advisory (non-blocking): flag `except ...: pass` — swallowed errors.

    The 429-storm bug (summarizer hammering a dead quota silently) was exactly
    this pattern. Nudge the author to log when the except wraps an external call.
    """
    try:
        with open(filepath, encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return
    hits = [
        i + 1                                 # 1-based line of the `except`
        for i in range(len(lines) - 1)
        if re.match(r"^\s*except\b.*:\s*$", lines[i])
        and re.match(r"^\s*pass\s*$", lines[i + 1])
    ]
    if hits:
        print(f"⚠ Silent excepts  : {os.path.basename(filepath)} — `except: pass` at "
              f"line(s) {hits} — log the error if it wraps an external/LLM/DB call")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: smoke_check.py <file.py>")
        sys.exit(0)

    filepath = sys.argv[1]

    if not filepath.endswith(".py"):
        sys.exit(0)                    # not a Python file — nothing to do

    name = os.path.basename(filepath)

    # ── Stage 1: syntax ───────────────────────────────────────────────────────
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✓ Syntax OK       : {name}")
    except py_compile.PyCompileError as exc:
        print(f"✗ SYNTAX ERROR    : {exc}")
        sys.exit(1)

    # ── Stage 3 (advisory, non-blocking): silent except: pass ─────────────────
    scan_silent_excepts(filepath)

    # ── Stage 2: import smoke test ────────────────────────────────────────────
    module = resolve_module(filepath)

    if module is None:
        # Streamlit entry point: skip gracefully
        basename = os.path.basename(filepath)
        if basename in SKIP_IMPORT:
            print(f"⚡ Import skipped  : {name}  (Streamlit entry-point)")
        sys.exit(0)

    if module == "":
        # Unknown file outside the importable map — just syntax check is enough
        sys.exit(0)

    snippet = (
        f"import sys, os; "
        f"sys.path.insert(0, {repr(PROJECT_ROOT)}); "
        f"from dotenv import load_dotenv; load_dotenv({repr(str(PROJECT_ROOT + '/.env'))}); "
        f"import {module}; "
        f"print('✓ Import OK       : {module}')"
    )
    result = subprocess.run(
        [VENV_PYTHON, "-c", snippet],
        capture_output=True, text=True, timeout=20,
        cwd=PROJECT_ROOT
    )

    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"✗ IMPORT ERROR     : {module}")
        # Print only the last few lines of stderr — skip the traceback noise
        lines = result.stderr.strip().splitlines()
        for line in lines[-6:]:
            print("  ", line)
        sys.exit(1)


if __name__ == "__main__":
    main()
