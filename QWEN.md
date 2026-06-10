# InvestWise / EisaX — Project Context

## Project Overview

**EisaX** (also known as InvestWise) is an AI-powered financial analysis and portfolio management platform built with FastAPI. It provides intelligent chat-based financial advice, stock screening, portfolio optimization, and market analysis capabilities — with a focus on Arab/MENA markets (UAE, Saudi, Egypt, Qatar, Kuwait, Oman, Bahrain).

The system uses a **Multi-Agent Orchestrator** architecture that routes user queries to specialized agents (stock analysis, portfolio management, bonds, crypto, macro analysis) and integrates multiple LLM providers (DeepSeek, Gemini, OpenAI) with fallback chains for reliability.

## Architecture

```
Client (UI/API)
       │
       ▼
api_bridge_v2.py (FastAPI entry point — 3900+ lines)
       │
       ▼
MultiAgentOrchestrator (core/orchestrator.py)
       │
       ├──▶ SessionManager — manages user sessions
       ├──▶ Memory System (core/memory_manager.py) — user context, facts, interests
       ├──▶ Router — classifies intent (STOCK_ANALYSIS, PORTFOLIO, BOND, CRYPTO, MACRO, GENERAL)
       │
       └──▶ Specialized Agents (core/agents/)
              ├── finance.py (2580+ lines — core financial analysis)
              └── ...
```

### Key Components

| Component | Path | Purpose |
|-----------|------|---------|
| **API Bridge** | `api_bridge_v2.py` | FastAPI app, routes, auth, rate limiting |
| **Orchestrator** | `core/orchestrator.py` | Multi-agent routing and coordination |
| **Finance Agent** | `core/agents/finance.py` | Stock analysis, screening, financial metrics |
| **Memory System** | `core/memory_manager.py` | User context persistence, fact extraction |
| **Export Engine** | `core/export_engine.py` | Generate reports (PDF, DOCX, Excel) |
| **Data Fetcher** | `core/data_fetcher.py` | Market data from yfinance, Yahoo Query, etc. |
| **Portfolio** | `core/portfolio.py`, `core/portfolio_manager.py` | Portfolio CRUD and optimization |
| **Auth** | `core/auth.py` | JWT authentication, API keys (`eixa_` prefix) |

## Technologies

- **Python 3.11**
- **FastAPI** + **Uvicorn** / **Gunicorn**
- **SQLite** (investwise.db) — primary data store
- **yfinance**, **yahooquery** — market data
- **pyportfolioopt**, **cvxpy** — portfolio optimization
- **pandas**, **numpy**, **scipy**, **scikit-learn** — data & analytics
- **matplotlib**, **seaborn** — visualization
- **Google Gemini** (primary + backup API keys) — LLM
- **DeepSeek** — LLM (deepseek-chat, deepseek-reasoner)
- **OpenAI** — LLM
- **Playwright** — PDF generation
- **chromadb**, **sentence-transformers** — vector embeddings
- **bcrypt**, **PyJWT** — authentication

## Building and Running

### Local Development

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn api_bridge_v2:app --reload --host 0.0.0.0 --port 8000

# Or use the start script (uses systemd or screen)
./start.sh
```

### Docker

```bash
docker-compose up -d
```

This starts two services:
- `eisax-api` — FastAPI backend on port 8000
- `eisax-learning` — Learning engine (runs `run_learning_engine.py`)

### Production (systemd)

```bash
sudo systemctl status eisax.service
sudo systemctl restart eisax.service
sudo journalctl -u eisax.service -f
```

### API Access

- **Base URL:** `http://localhost:8000`
- **Authentication:** `X-API-Key: eixa_...` or `Authorization: Bearer eixa_...`
- **Health check:** `GET /health` (requires `X-API-Key` header)

## Key Directories

| Directory | Contents |
|-----------|----------|
| `core/` | Core logic: orchestrator, agents, memory, auth, data fetchers, export |
| `api/` | FastAPI route handlers (chat, portfolio) |
| `static/` | Frontend assets (HTML, CSS, JS) |
| `data/` | Market data files (Excel reports, cached data) |
| `logs/` | Application logs |
| `backups/` | Database backups |
| `file_cache/` | Cached API responses and reports |
| `uploads/` | User-uploaded files |
| `session_memory/` | JSON session files |
| `scripts/` | Utility scripts |
| `tests/` | Test files |

## Configuration

Environment variables (via `.env` file):

- `DEEPSEEK_API_KEY` — DeepSeek API key
- `GEMINI_API_KEY` / `GEMINI_API_KEY_BACKUP` — Gemini API keys
- `OPENAI_API_KEY` — OpenAI API key
- `SECURE_TOKEN` — Legacy admin token
- `MODEL_NAME` — DeepSeek model name (default: `deepseek-chat`)

See `config.py` for all configurable defaults (portfolio weights, timeouts, etc.).

## LLM Fallback Chain

The system has built-in reliability:
1. **Primary:** Gemini 2.5 Flash (fast, economical)
2. **Backup:** Gemini 2.5 Pro (more capable)
3. **Fallback:** DeepSeek (chat + reasoner models)
4. **Cache:** Cached responses for repeated queries

## Notable Files

| File | Description |
|------|-------------|
| `DEVELOPMENT_PLAN.md` | Comprehensive development plan (Arabic, approved for execution) |
| `DATABASE_DEVELOPMENT_PLAN.md` | Database development plan |
| `FINANCE_AGENT_DEVELOPMENT_PLAN.md` | Finance agent development plan |
| `CLAUDE.md` | Claude session instructions (role, delegation model, review checklist) |
| `eisax_playbook.md` | Operational playbook |
| `docker-compose.yml` | Docker compose with API + learning services |
| `Dockerfile` | Docker image (Python 3.11-slim) |
| `start.sh` | Start script (systemd → screen fallback) |
| `gunicorn.ctl` | Gunicorn process control |

## Key Design Decisions

1. **Orchestrator pattern** — single entry point routes to specialized agents
2. **LLM fallback chains** — multiple providers ensure reliability
3. **Memory-first approach** — user context persisted across sessions
4. **API key system** — `eixa_` prefixed keys with tier-based access
5. **SQLite** — lightweight, file-based database for portability
6. **Export engine** — generates PDF/DOCX/Excel reports via Playwright
7. **Rate limiting** — SlowApi middleware for request throttling

## Testing

```bash
# Smoke test
python -m tests.smoke_chat

# Pytest
pytest
```

## Development Conventions

- **Claude = architect/production manager** — decides what and how to build
- **Codex = executor** — runs well-scoped tasks, outputs reviewed before acceptance
- **Token conservation** — command-first responses, no unnecessary explanations
- **Codex Review Checklist:** syntax check → no TODOs/placeholders → name matching → caller updates → smoke check passes
