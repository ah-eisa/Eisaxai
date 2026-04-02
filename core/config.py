"""
core/config.py
──────────────
Single source of truth for all filesystem paths in the EisaX project.

Usage
─────
    from core.config import BASE_DIR, APP_DB, CORE_DB, EXPORTS_DIR

All paths are resolved relative to this file's location (core/config.py),
so they work correctly regardless of where the process is launched from.

Path hierarchy
──────────────
  BASE_DIR/                      ← /home/ubuntu/investwise/
    investwise.db                ← APP_DB   (sessions, portfolios, users)
    price_cache.db               ← PRICE_CACHE_DB
    eisax_playbook.md            ← PLAYBOOK_PATH
    .env                         ← ENV_FILE
    core/
      investwise.db              ← CORE_DB  (brain, fundamentals, market data)
      local_tickers.py           ← LOCAL_TICKERS_FILE
    static/
      exports/                   ← EXPORTS_DIR
    data/
      historical/                ← HISTORICAL_DATA_DIR
      portfolios/                ← PORTFOLIOS_DIR
    logs/
      eisax_modifications.log    ← MODIFICATIONS_LOG
      data_fetcher.log           ← DATA_FETCHER_LOG
    backups/                     ← BACKUPS_DIR
    file_cache/                  ← FILE_CACHE_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Anchors ───────────────────────────────────────────────────────────────────
# core/config.py lives inside BASE_DIR/core/ → parent is BASE_DIR
BASE_DIR: Path = Path(__file__).parent.parent.resolve()
CORE_DIR: Path = Path(__file__).parent.resolve()

# ── Databases ─────────────────────────────────────────────────────────────────
# Root-level DB: sessions, portfolios, users, vectors
APP_DB:         Path = BASE_DIR / "investwise.db"

# Core-level DB: brain/learning, fundamentals, DFM/EGX/Saudi market data
CORE_DB:        Path = CORE_DIR / "investwise.db"

# Price cache (fast TTL cache for live prices)
PRICE_CACHE_DB: Path = BASE_DIR / "price_cache.db"

# ── Config / content files ────────────────────────────────────────────────────
PLAYBOOK_PATH:  Path = Path(os.getenv("PLAYBOOK_PATH", str(BASE_DIR / "eisax_playbook.md")))
ENV_FILE:       Path = BASE_DIR / ".env"

# ── Output directories ────────────────────────────────────────────────────────
EXPORTS_DIR:    Path = BASE_DIR / "static" / "exports"
FILE_CACHE_DIR: Path = BASE_DIR / "file_cache"
BACKUPS_DIR:    Path = Path(os.getenv("BACKUP_DIR",  str(BASE_DIR / "backups")))
LOGS_DIR:       Path = BASE_DIR / "logs"
DATA_DIR:       Path = BASE_DIR / "data"
HISTORICAL_DATA_DIR: Path = DATA_DIR / "historical"
PORTFOLIOS_DIR: Path = DATA_DIR / "portfolios"

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR:     Path = BASE_DIR / "static"

# ── Log files ─────────────────────────────────────────────────────────────────
MODIFICATIONS_LOG: Path = Path(os.getenv("LOG_PATH", str(LOGS_DIR / "eisax_modifications.log")))
DATA_FETCHER_LOG:  Path = LOGS_DIR / "data_fetcher.log"
BACKEND_LOG:       Path = BASE_DIR / "backend.log"

# ── Data files ────────────────────────────────────────────────────────────────
LOCAL_TICKERS_FILE: Path = CORE_DIR / "local_tickers.py"
DFM_CSV:            Path = CORE_DIR / "DFM.csv"
ADX_CSV:            Path = CORE_DIR / "ADX.csv"
