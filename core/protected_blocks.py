"""Protect report spans from broad editorial regex transforms."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re


logger = logging.getLogger("eisax.protected_blocks")


@dataclass
class ProtectedSpan:
    sentinel: str
    original: str


_SENTINEL_RE = re.compile(r"\A⟦PROTECTED_[A-Z]+_\d+⟧\Z")

_PATTERNS = [
    (
        "DISCLAIMER",
        re.compile(
            r"(?:^|\n)(?:>\s*)?\*{0,2}(?:⚠️\s*)?\*{0,2}Disclaimer:\*{0,2}[\s\S]+?(?:\n\n|\Z)",
            re.MULTILINE,
        ),
    ),
    (
        "AUDIT",
        re.compile(
            r"(?:^|\n)#{2,3}\s+Audit\s+Trail[\s\S]+?(?=\n#{1,3}\s+|\Z)",
            re.MULTILINE,
        ),
    ),
    (
        "FACT",
        re.compile(
            r"(?:^|\n)\|.*?FACT[- ]?CHECK[\s\S]+?(?=\n\n|\Z)",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "FOOTER",
        re.compile(
            r"(?:^|\n)>?[\s🛡️]*(?:This report|For informational purposes)[\s\S]+?(?=\n\n|\Z)",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
]

_URL_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def _next_sentinel(category: str, counts: dict[str, int]) -> str:
    idx = counts.get(category, 0)
    counts[category] = idx + 1
    return f"⟦PROTECTED_{category}_{idx}⟧"


def protect(text: str) -> tuple[str, list[ProtectedSpan]]:
    """
    Scan text for protected regions and replace each with a unique sentinel.

    The replacement is intentionally one-way until restore() is called with the
    returned spans. Existing sentinel-only text is left unchanged, which keeps
    repeated protect() calls from adding new protection layers.
    """
    if not text:
        return text, []

    spans: list[ProtectedSpan] = []
    counts: dict[str, int] = {}
    out = text

    for category, pattern in _PATTERNS:

        def _swap(match: re.Match[str], _cat: str = category) -> str:
            original = match.group(0)
            if _SENTINEL_RE.fullmatch(original):
                return original
            sentinel = _next_sentinel(_cat, counts)
            spans.append(ProtectedSpan(sentinel=sentinel, original=original))
            return sentinel

        out = pattern.sub(_swap, out)

    def _swap_url(match: re.Match[str]) -> str:
        display_text = match.group(1)
        url = match.group(2)
        if _SENTINEL_RE.fullmatch(url):
            return match.group(0)
        sentinel = _next_sentinel("URL", counts)
        spans.append(ProtectedSpan(sentinel=sentinel, original=url))
        return f"[{display_text}]({sentinel})"

    out = _URL_PATTERN.sub(_swap_url, out)
    logger.debug(
        "[ProtectedBlocks] protected=%d categories=%s",
        len(spans),
        {span.sentinel.split("_")[1] for span in spans},
    )
    return out, spans


def restore(text: str, spans: list[ProtectedSpan]) -> str:
    """Replace each sentinel back with its original protected text."""
    if not spans:
        return text

    out = text
    for span in reversed(spans):
        out = out.replace(span.sentinel, span.original)
    return out


def _smoke_disclaimer() -> None:
    text = (
        "Intro paragraph.\n\n"
        "> ⚠️ **Disclaimer:** This is not an offer to buy or sell any security.\n\n"
        "Action: Buy candidate."
    )
    protected, spans = protect(text)
    transformed = protected.replace("Buy", "accumulation")
    restored = restore(transformed, spans)
    assert "buy or sell any security" in restored
    assert "Action: accumulation candidate." in restored
    print("scenario=disclaimer")
    print("before:", text)
    print("after:", restored)


def _smoke_audit_url() -> None:
    text = (
        "See [link](https://example.com?a=Buy).\n\n"
        "## Audit Trail\n"
        "- Source URL: [audit](https://example.com/audit?rating=Buy)\n"
        "\n## Next Section\n"
        "Buy note."
    )
    protected, spans = protect(text)
    transformed = protected.replace("Buy", "accumulation")
    restored = restore(transformed, spans)
    assert "[link](https://example.com?a=Buy)" in restored
    assert "[audit](https://example.com/audit?rating=Buy)" in restored
    assert "accumulation note." in restored
    print("scenario=audit_url")
    print("before:", text)
    print("after:", restored)


def _smoke_empty() -> None:
    protected, spans = protect("")
    assert protected == ""
    assert spans == []
    print("scenario=empty")
    print("before:", "")
    print("after:", protected, spans)


if __name__ == "__main__":
    _smoke_disclaimer()
    _smoke_audit_url()
    _smoke_empty()
