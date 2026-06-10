"""
Prompt Builder - Constructs prompts/messages for the agent.

This module handles the construction of prompt payloads, combining:
- System prompt (based on mode)
- Session context/history
- File attachments
- Current user input
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import requests

class PromptBuilder:
    """
    Builds prompt/message structures for the agent.
    
    Combines system prompts, session context, file attachments, 
    and user input into the expected format for the agent.
    """
    
    def scrape_urls(self, text: str, max_chars: int = 5000) -> str:
        """Find URLs in text, fetch their content, and return a context string."""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
        urls = re.findall(url_pattern, text)
        if not urls:
            return ""
            
        contexts = []
        for url in urls[:2]: # Limit to 2 URLs for performance
            try:
                # Simple fetch
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
                if resp.status_code == 200:
                    # Very basic text extraction (remove HTML tags)
                    page_text = re.sub(r'<[^>]+>', ' ', resp.text)
                    page_text = ' '.join(page_text.split()) # normalize whitespace
                    snippet = page_text[:max_chars]
                    contexts.append(f"[WEBSITE CONTENT: {url}]\n{snippet}\n[END WEBSITE]")
            except Exception as e:
                contexts.append(f"[WEBSITE ERROR: {url}] Failed to fetch: {str(e)}")
                
        return "\n\n".join(contexts)
    
    def __init__(self, system_prompts: Dict[str, str]):
        """
        Initialize the prompt builder.
        
        Args:
            system_prompts: Dictionary mapping mode names to system prompt strings.
        """
        self.system_prompts = system_prompts
    
    def get_system_prompt(self, mode: str) -> str:
        """
        Get the system prompt for a given mode.
        
        Args:
            mode: The mode name (e.g., 'assistant', 'code_review').
            
        Returns:
            The system prompt string for that mode.
        """
        return self.system_prompts.get(mode, self.system_prompts.get("assistant", ""))
    
    def build_file_context(
        self,
        files: List[Dict[str, Any]],
        active_file_id: Optional[str] = None,
        max_files: int = 1,
        max_chars_per_file: int = 20000
    ) -> str:
        """
        Build a file context string from uploaded files.
        
        Args:
            files: List of file dicts with 'id', 'filename', 'text' keys.
            active_file_id: If set, prefer this file; otherwise use last N files.
            max_files: Maximum number of files to include.
            max_chars_per_file: Maximum characters per file.
            
        Returns:
            Formatted string with file contents for injection into prompt.
        """
        if not files:
            return ""
        
        # Filter to files with text content
        text_files = [f for f in files if f.get("text")]
        if not text_files:
            return ""
        
        # Prefer active file if set
        if active_file_id:
            chosen = [f for f in text_files if f["id"] == active_file_id]
            if chosen:
                text_files = chosen
        
        # Limit to max_files (take last N)
        files_to_use = text_files[-max_files:] if max_files > 0 else []
        
        parts = []
        for f in files_to_use:
            text = f.get("text", "")
            snippet = text[:max_chars_per_file]
            is_truncated = len(text) > max_chars_per_file
            note = " (TRUNCATED)" if is_truncated else ""
            
            if text:
                parts.append(
                    f"[ATTACHED FILE: {f['filename']}{note}]\n{snippet}\n[/END FILE]"
                )
            else:
                parts.append(
                    f"[ATTACHED FILE: {f['filename']}]\n"
                    "Binary/non-text file attached. Ask user what to do with it.\n"
                    "[/END FILE]"
                )
        
        return "\n".join(parts)
    
    def build_user_message(
        self,
        user_text: str,
        file_context: Optional[str] = None
    ) -> str:
        """
        Build the final user message with optional file context.
        
        Args:
            user_text: The user's input text.
            file_context: Optional file context string.
            
        Returns:
            The combined user message ready for the agent.
        """
        if not file_context:
            return user_text
        
        return (
            "You have the following attached file(s) for context:\n\n"
            f"{file_context}\n\n"
            "User request:\n"
            f"{user_text}"
        )
    
    def build_messages(
        self,
        user_text: str,
        mode: str = "assistant",
        history: Optional[List[Dict[str, str]]] = None,
        file_context: Optional[str] = None,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Build the full messages array for chat completion APIs.
        
        Args:
            user_text: The current user input.
            mode: The agent mode for system prompt selection.
            history: Optional list of previous messages.
            file_context: Optional file context to prepend.
            include_system: Whether to include system message.
            
        Returns:
            List of message dicts ready for chat completion API.
        """
        messages = []
        
        # Add system message
        if include_system:
            system_prompt = self.get_system_prompt(mode)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add current user message
        final_user_msg = self.build_user_message(user_text, file_context)
        messages.append({"role": "user", "content": final_user_msg})
        
        return messages
    
    def build_prompt_payload(
        self,
        user_text: str,
        session_context: Dict[str, Any],
        settings: Dict[str, Any],
        files: Optional[List[Dict[str, Any]]] = None,
        active_file_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build the complete prompt payload for handle_message.
        
        This is the main entry point that combines everything needed
        for the agent to process a request.
        
        Args:
            user_text: The user's input text.
            session_context: The session memory/context.
            settings: Agent settings (mode, model, temperature, etc.)
            files: Optional list of uploaded files.
            active_file_id: Optional active file ID.
            
        Returns:
            Dictionary with 'message', 'file_context', 'history', etc.
        """
        mode = settings.get("mode", "assistant")
        max_files = settings.get("max_context_files", 1)
        max_chars = settings.get("max_file_chars", 20000)
        
        # Build file context if files provided
        file_context = ""
        if files:
            file_context = self.build_file_context(
                files=files,
                active_file_id=active_file_id,
                max_files=max_files,
                max_chars_per_file=max_chars
            )
        
        # Scrape URLs from user text
        web_context = self.scrape_urls(user_text)
        
        # Build the combined message
        combined_message = self.build_user_message(user_text, file_context)
        if web_context:
            combined_message = f"You have the following website content for context:\n\n{web_context}\n\n{combined_message}"
        
        # Get history from session context
        history = session_context.get("history", [])
        
        # Prepare system prompt (appending style card instructions if present)
        base_system_prompt = self.get_system_prompt(mode)
        append_prompt = settings.get("system_prompt_append", "")
        final_system_prompt = base_system_prompt + append_prompt
        
        return {
            "message": combined_message,
            "raw_user_text": user_text,
            "file_context": file_context,
            "web_context": web_context,
            "history": history,
            "system_prompt": final_system_prompt,
            "mode": mode
        }
