# Clawdbot

FastAPI + local agent project for intelligent chat and portfolio management.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload

# Open UI in browser
http://127.0.0.1:8000
```

> The root `/` serves `chat.html` directly. Static assets are at `/static/`.

---

## Architecture (Orchestrator Layer)

The project uses an **Orchestrator layer** to separate concerns and add testable boundaries **without changing the /chat API contract**.

### High-Level Flow

```
Client (UI/API)
       │
       ▼
/chat endpoint  (api/chat_routes.py)
       │
       ▼
AgentOrchestrator  (core/orchestrator.py)
       │
       ├──▶ MemoryManager   (core/memory_manager.py)
       │         └──▶ memory_store.py (JSON files)
       │
       ├──▶ PromptBuilder   (core/prompt_builder.py)
       │
       └──▶ agent.handle_message  (agent.py)
```

### New Files

| File                     | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `core/orchestrator.py`   | Coordinates message flow (context → prompt → agent) |
| `core/memory_manager.py` | Wraps `memory_store.py` (get/save/clear session)    |
| `core/prompt_builder.py` | Builds prompts from system + history + user input   |
| `tests/smoke_chat.py`    | Smoke test verifying `/chat` returns expected keys  |

### Orchestrator Responsibilities

`AgentOrchestrator` coordinates:

1. **Load session context** - via `MemoryManager.get_context(session_id)`
2. **Build prompt payload** - via `PromptBuilder.build_prompt_payload(...)`
3. **Call the agent** - invokes `agent.handle_message(...)`
4. **Persist the turn** - via `MemoryManager.save_turn(...)`
5. **Return response** - same schema as before

---

## API Contract (Unchanged)

### POST /chat

**Request:**
```json
{
  "message": "Your message here",
  "settings": { "session_id": "abc", "memory": true },
  "history": []
}
```

**Response:**
```json
{
  "type": "chat.reply",
  "reply": "Assistant response...",
  "data": null
}
```

> Response keys `type`, `reply`, `data` remain stable for frontend compatibility.

---

## Testing

### Smoke Test

```bash
# Terminal 1
uvicorn app:app --reload

# Terminal 2
python -m tests.smoke_chat
```

### Unit Testing Boundaries

Each component can be tested in isolation:

```python
# Test MemoryManager
from core.memory_manager import MemoryManager
mm = MemoryManager()
mm.save_turn("test_session", "Hello", "Hi there!")
ctx = mm.get_context("test_session")
assert "history" in ctx

# Test PromptBuilder
from core.prompt_builder import PromptBuilder
pb = PromptBuilder({"assistant": "You are helpful."})
prompt = pb.get_system_prompt("assistant")
assert "helpful" in prompt

# Test Orchestrator (with mock agent)
from core.orchestrator import AgentOrchestrator
def mock_agent(**kwargs):
    return {"type": "test", "reply": "mock", "data": None}
orch = AgentOrchestrator(agent_callable=mock_agent)
result = orch.handle_message("sess", "test", meta={"settings": {"memory": False}})
assert result["reply"] == "mock"
```

---

## Project Structure

```
clawdbot/
├── app.py                 # FastAPI app (mounts routes & static, serves / → chat.html)
├── agent.py               # Main agent logic (handle_message, detect_intent, etc.)
├── state.py               # Shared state (settings, SYSTEM_PROMPTS)
├── memory_store.py        # Low-level JSON file storage (mem_get, mem_set)
├── api/
│   ├── chat_routes.py     # /chat endpoint (uses AgentOrchestrator)
│   └── portfolio_routes.py
├── core/
│   ├── orchestrator.py    # AgentOrchestrator class
│   ├── memory_manager.py  # MemoryManager class
│   ├── prompt_builder.py  # PromptBuilder class
│   ├── data.py            # Price data utilities
│   ├── metrics.py         # Portfolio metrics
│   ├── policy.py          # Investment policies
│   └── portfolio.py       # Portfolio optimization
├── static/
│   └── chat.html          # Main UI
├── session_memory/        # JSON session files (auto-created)
└── tests/
    └── smoke_chat.py      # Endpoint smoke test
```

---

## Key Design Decisions

1. **Thin wiring only** - `chat_routes.py` minimally changed to instantiate orchestrator
2. **No storage format changes** - `MemoryManager` wraps `memory_store.py` as-is
3. **Same response schema** - `{type, reply, data}` unchanged
4. **agent_callable** - Currently set to `agent.handle_message`, easily swappable
