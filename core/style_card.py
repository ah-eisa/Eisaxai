from __future__ import annotations

class StyleCard:
    """
    Defines response style guidelines — natural and adaptive.
    """
    MAX_LINES = 30
    MAX_BULLETS = 8
    
    # The system prompt injection for style guidance (not enforcement)
    SYSTEM_INSTRUCTION = (
        "\n\nSTYLE GUIDANCE:\n"
        "- Be concise but natural. Use paragraphs for explanations, bullets only when listing items.\n"
        "- Match response length to question complexity. Short question → short answer.\n"
        "- Tone: Warm, intelligent, genuine. Like a knowledgeable friend.\n"
        "- No corporate fluff. No template phrases. Be real.\n"
        "- If the topic is serious (finance, health), be professional but still human.\n"
    )
    
    @staticmethod
    def format_fallback(text: str) -> str:
        """
        Light formatting cleanup — only truncates extremely long outputs.
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) <= 30:
            return "\n".join(lines)
            
        # Only truncate truly excessive output
        return "\n".join(lines[:30])
