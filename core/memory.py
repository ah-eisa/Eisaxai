import logging
import json
import os
import re
from pathlib import Path
from typing import Any
logger = logging.getLogger(__name__)

# ============================================================
# PERSISTENT SESSION MEMORY (disk-based, safe)
# ============================================================
MEMORY_DIR = Path(os.getenv("SESSION_MEMORY_DIR", "session_memory"))
MEMORY_DIR.mkdir(exist_ok=True)

_SID_SAFE_RE = re.compile(r"[^a-zA-Z0-9_\-]+")

def safe_sid(sid: str) -> str:
    sid = (sid or "default").strip()
    sid = _SID_SAFE_RE.sub("_", sid)[:64]
    return sid or "default"

def get_sid_from_settings(settings: dict | None) -> str:
    if not settings:
        return "default"
    return safe_sid(str(settings.get("session_id") or settings.get("sid") or "default"))

def get_memory_path(sid: str) -> Path:
    return MEMORY_DIR / f"{safe_sid(sid)}.json"

def get_memory(sid: str) -> dict[str, Any]:
    p = get_memory_path(sid)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    # atomic on Windows/Unix
    try:
        tmp.replace(path)
    except OSError:
        # Fallback for Windows if file exists and is locked logic issues (rare but possible)
        if path.exists():
            path.unlink()
        tmp.rename(path)

def set_memory(sid: str, data: dict[str, Any]) -> None:
    p = get_memory_path(sid)
    try:
        atomic_write(p, json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        # don't crash the app on memory write errors
        logger.error(f"[memory] write failed for {p.name}: {e}")
def clear_memory(sid: str) -> None:
    p = get_memory_path(sid)
    try:
        if p.exists():
            p.unlink()
    except Exception as e:
        logger.error(f"[memory] clear failed for {p.name}: {e}")