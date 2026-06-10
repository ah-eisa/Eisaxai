from __future__ import annotations
import json
import config
from core.llm import get_client

# Simple local logger or import from utils if available
def _log(event: str, **fields):
    print(f"[{event}] {fields}")

def generate_slide_blueprint(content: str, title: str = "Presentation") -> list[dict]:
    """
    Use LLM to convert content into a structured slide blueprint.
    
    Returns: list of {"title": str, "bullets": list[str]}
    Each slide has:
    - title: max 8 words
    - bullets: 3-5 short points (max 15 words each)
    """
    try:
        client = get_client()
        response = client.create_completion(
            model=config.DEFAULT_MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a presentation designer. Convert the given content into a slide blueprint.\n\n"
                        "RULES:\n"
                        "- Output ONLY valid JSON array\n"
                        "- Each slide: {\"title\": \"...\", \"subtitle\": \"...\", \"bullets\": [\"...\", ...], \"insight\": \"...\"}\n"
                        "- Title: max 6 words, sharp and impactful\n"
                        "- Subtitle: max 12 words, provides context to the title\n"
                        "- Bullets: exactly 3-5 per slide\n"
                        "- Insight: Optional, one short strategic takeaway (max 12 words)\n"
                        "- Each bullet: max 15 words, start with action verb or key concept\n"
                        "- No paragraphs, no long sentences\n"
                        "- One main idea per slide\n"
                        "- 4-7 slides total\n\n"
                        "Example output:\n"
                        '[{"title": "Portfolio Optimization", "subtitle": "Risk-adjusted performance analysis", "bullets": ["Allocation to alpha strategies", "Hedging against volatility", "Liquidity management"], "insight": "Institutional focus remains on capital preservation."}]'
                    ),
                },
                {"role": "user", "content": f"Create slide blueprint for:\n\n{content[:3000]}"},
            ],
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up JSON (remove markdown code blocks if present)
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        result = result.strip()
        
        # Parse JSON
        slides = json.loads(result)
        
        # Validate structure
        if isinstance(slides, list) and len(slides) > 0:
            validated_slides = []
            for slide in slides[:8]:  # Max 8 slides
                if isinstance(slide, dict) and "title" in slide:
                    validated_slides.append({
                        "title": str(slide.get("title", ""))[:60],
                        "subtitle": str(slide.get("subtitle", ""))[:100] if slide.get("subtitle") else None,
                        "bullets": [str(b)[:100] for b in slide.get("bullets", [])[:5]],
                        "insight": str(slide.get("insight", ""))[:120] if slide.get("insight") else None
                    })
            if validated_slides:
                _log("slide_blueprint_success", slide_count=len(validated_slides))
                return validated_slides
                
    except Exception as e:
        _log("slide_blueprint_error", error=str(e))
    
    # Fallback: return heuristic based blueprint
    # For now, just a simple list as fallback
    return [{"title": title, "bullets": ["Content processing failed", "Please try again"]}]


def generate_slide_blueprint_for_topic(topic: str) -> list[dict]:
    """
    Use LLM to generate a complete slide blueprint for a NEW topic.
    
    This creates slides FROM SCRATCH, not converting existing content.
    
    Returns: list of {"title": str, "bullets": list[str]}
    """
    try:
        client = get_client()
        response = client.create_completion(
            model=config.DEFAULT_MODEL,
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert presentation designer. Create a professional slide deck on the given topic.\n\n"
                        "OUTPUT RULES (STRICT):\n"
                        "- Output ONLY valid JSON array, nothing else\n"
                        "- Each slide: {\"title\": \"...\", \"subtitle\": \"...\", \"bullets\": [\"...\", ...], \"insight\": \"...\"}\n"
                        "- Create 5-7 slides total\n"
                        "- Insight: One short tactical or strategic analytical note (max 12 words)\n"
                        "- Each slide title: max 6 words, sharp and impactful\n"
                        "- Each slide subtitle: max 12 words, context for the title\n"
                        "- Each slide: exactly 3-5 bullets (except framework slides)\n"
                        "- Each bullet: max 15 words, start with action verb or key concept\n\n"
                        "SLIDE STRUCTURE:\n"
                        "1. Cover Title slide - the main topic title\n"
                        "2. Strategy Framework slide - MUST be titled 'Why [Topic] Matters Now'. 3 bullets max + 1 insight.\n"
                        "3. Industrial Stack slide - MUST be titled 'The [Topic] Stack'. Columns: 1. Legal Layer, 2. Tech Layer, 3. Market Layer. Insight: Strategic framework driven.\n"
                        "4-5. Content slides - key tactical points or market sectors\n"
                        "6. Strategic Takeaways - final conclusions or next steps\n\n"
                        "DO NOT include generic slides like 'Welcome'. Focus on institutional-grade content."
                    ),
                },
                {"role": "user", "content": f"Create a professional slide deck on: {topic}"},
            ],
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up JSON (remove markdown code blocks if present)
        if "```" in result:
            # Extract content between code blocks
            parts = result.split("```")
            for part in parts:
                if part.strip().startswith("json"):
                    result = part.strip()[4:].strip()
                    break
                elif part.strip().startswith("["):
                    result = part.strip()
                    break
        
        # Parse JSON
        slides = json.loads(result)
        
        # Validate and clean structure
        if isinstance(slides, list) and len(slides) > 0:
            validated_slides = []
            for slide in slides[:8]:  # Max 8 slides
                if isinstance(slide, dict) and "title" in slide:
                    title = str(slide.get("title", ""))[:60]
                    subtitle = str(slide.get("subtitle", ""))[:100] if slide.get("subtitle") else None
                    bullets = [str(b)[:100] for b in slide.get("bullets", [])[:5]]
                    insight = str(slide.get("insight", ""))[:120] if slide.get("insight") else None
                    if title and bullets:
                        validated_slides.append({
                            "title": title,
                            "subtitle": subtitle,
                            "bullets": bullets,
                            "insight": insight
                        })
            if validated_slides:
                _log("topic_slide_blueprint_success", topic=topic, slide_count=len(validated_slides))
                return validated_slides
                
    except Exception as e:
        _log("topic_slide_blueprint_error", topic=topic, error=str(e))
    
    # Fallback: return a minimal structure
    return [
        {"title": topic.title(), "bullets": [
            f"Overview of {topic}",
            "Key concepts and principles",
            "Important considerations"
        ]},
        {"title": "Key Points", "bullets": [
            "Main idea or concept",
            "Supporting details",
            "Practical applications"
        ]},
        {"title": "Summary", "bullets": [
            "Key takeaways",
            "Next steps to consider"
        ]}
    ]
