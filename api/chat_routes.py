from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict

from state import agent_settings, SYSTEM_PROMPTS
from core.router import Router
from core.orchestrator import _orchestrator
from agent import handle_message
from core.file_generator import maybe_generate_file

# =========================
# Setup Institutional Router
# =========================
_router = Router(
    agent_callable=handle_message,
    orchestrator=_orchestrator
)

router = APIRouter(tags=["chat"])

class ChatRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    prompt: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    history: Optional[list[dict]] = None 

    def resolved(self) -> str:
        for v in (self.message, self.text, self.prompt):
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

@router.post("/chat", operation_id="chat_post_v3")
def chat(req: ChatRequest):
    """
    Main chat endpoint - Updates to use Institutional Router.
    Validates session, classifies intent, checks policy, then executes.
    """
    msg = req.resolved()
    if not msg:
        raise HTTPException(status_code=422, detail="Missing message/text/prompt in request body.")

    try:
        # 1. Prepare Context
        s = dict(agent_settings)
        if isinstance(req.settings, dict):
            s.update(req.settings)

        import state
        
        # Mandatory Session ID
        session_id = s.get("session_id")
        if not session_id or session_id == "default":
            raise HTTPException(status_code=400, detail="Missing or invalid session_id.")

        # Request ID for tracing
        request_id = str(uuid.uuid4())

        # Collect files
        active_files = list(state.uploaded_files)
        active_file_id = req.settings.get("active_file_id") if req.settings else None

        # Build Meta for Router
        meta = {
            "request_id": request_id,
            "settings": s,
            "files": active_files,
            "active_file_id": active_file_id or state.active_file_id,
            "history": req.history,
        }

        # --- OPTIMIZED ROUTING: Skip MiniCPM classification overhead ---
        # The Orchestrator/GeneralAgent already handles intent classification
        # internally via IntentClassifier. No need to classify twice.
        # MiniCPM is only used for vision tasks (image analysis).

        # Check for image in active file (vision task)
        image_path = None
        use_vision = False
        if active_file_id:
            f = next((f for f in state.uploaded_files if f["id"] == active_file_id), None)
            if f and f.get("filename", "").lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                use_vision = True

        reply = ""
        result_type = "chat.reply"
        data = None

        if use_vision and image_path:
            # VISION PATH: MiniCPM (Local) — its actual strength
            from core.minicpm_client import minicpm_client
            reply = minicpm_client.analyze_image(msg, image_path)
        else:
            # MAIN PATH: Orchestrator handles everything (chat + finance)
            result = _router.handle_request(
                session_id=session_id,
                text=msg,
                meta=meta
            )
            
            # Handle File Generation (if applicable)
            if result.get("type", "") == "file.ready":
                 result = maybe_generate_file(result)
            
            reply = result.get("reply", "")
            result_type = result.get("type", "chat.reply")
            data = result.get("data", None)

        # --- OPTIMIZED ROUTING END ---

        # FINAL HARD GUARD — never return long/template replies
        if isinstance(reply, str):
            reply = _router._normalize_reply(reply)

        return {
            "type": result_type,
            "reply": reply,
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
