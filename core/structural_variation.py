from __future__ import annotations

import hashlib
import logging
import random
import re


logger = logging.getLogger("eisax.structural_variation")


def _seed(ticker: str, salt: int = 0) -> int:
    h = hashlib.md5(ticker.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + salt) % 10**9


def _render(template: str, *parts: str) -> str:
    rendered = template
    for index, part in enumerate(parts, start=1):
        rendered = rendered.replace("{" + str(index) + "}", part.strip())
    return rendered


def _contains_section_header(value: str) -> bool:
    return "### " in value


def _clean_clause(value: str) -> str:
    return value.strip().rstrip(" ,;:")


def _sentence_case(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return cleaned
    return cleaned[:1].upper() + cleaned[1:]


def _rebuild_line_block(lines: list[str], trailing_newline: bool | None = None) -> list[str]:
    if not lines:
        return []
    stripped = [line.rstrip("\n") for line in lines]
    if trailing_newline is None:
        trailing_newline = lines[-1].endswith("\n")
    rebuilt = [line + "\n" for line in stripped[:-1]]
    rebuilt.append(stripped[-1] + ("\n" if trailing_newline else ""))
    return rebuilt


def _select_active_transforms(ticker: str, total: int) -> set[int]:
    count = 3 + (_seed(ticker, salt=100) % 3)
    indices = list(range(total))
    random.Random(_seed(ticker, salt=101)).shuffle(indices)
    return set(indices[:count])


def _apply_transform_but_clause(text: str, ticker: str, salt: int) -> tuple[str, int]:
    variants = [
        "{1} but {2}.",
        "Despite {2}, {1}.",
        "{1} — although {2}.",
        "While {2}, {1}.",
    ]
    variant = variants[_seed(ticker, salt=salt) % len(variants)]
    patterns = [
        (re.compile(r"(^|\n|[.!?]\s+)([A-Z][^\n\.]{15,120})\s+but\s+([^\n\.]{15,120})\."), "pxy"),
        (re.compile(r"(^|\n|[.!?]\s+)Despite\s+(.{15,120}),\s+([A-Z][^\n\.]{15,120})\."), "pyx"),
        (re.compile(r"(^|\n|[.!?]\s+)([A-Z][^\n\.]{15,120})\s+— although\s+([^\n\.]{15,120})\."), "pxy"),
        (re.compile(r"(^|\n|[.!?]\s+)While\s+(.{15,120}),\s+([A-Z][^\n\.]{15,120})\."), "pyx"),
    ]

    applied = 0
    for pattern, order in patterns:
        def _sub(match: re.Match[str]) -> str:
            nonlocal applied
            segment = match.group(0)
            if _contains_section_header(segment):
                return segment
            prefix = match.group(1)
            if order == "pxy":
                first, second = match.group(2), match.group(3)
            else:
                second, first = match.group(2), match.group(3)
            if variant == "{1} but {2}.":
                first_out = first.strip()
                second_out = second.strip()
            else:
                first_out = _clean_clause(first)
                second_out = _clean_clause(second)
            rewritten = prefix + _render(variant, first_out, second_out)
            if rewritten != segment:
                applied += 1
            return rewritten

        text = pattern.sub(_sub, text)
    return text, applied


def _apply_transform_verdict_because(text: str, ticker: str, salt: int) -> tuple[str, int]:
    variants = [
        "{1} because {2}.",
        "Given {2}, {1}.",
        "{2} supports the {1} stance.",
    ]
    variant = variants[_seed(ticker, salt=salt) % len(variants)]
    patterns = [
        (re.compile(r"(^|\n|[.!?]\s+)(Buy|Hold|Reduce|Sell)\s+because\s+([^\n\.]{15,150})\."), "pvr"),
        (re.compile(r"(^|\n|[.!?]\s+)Given\s+(.{15,150}),\s+(Buy|Hold|Reduce|Sell)\."), "prv"),
        (re.compile(r"(^|\n|[.!?]\s+)(.{15,150})\s+supports\s+the\s+(Buy|Hold|Reduce|Sell)\s+stance\."), "prv"),
    ]

    applied = 0
    for pattern, order in patterns:
        def _sub(match: re.Match[str]) -> str:
            nonlocal applied
            segment = match.group(0)
            if _contains_section_header(segment):
                return segment
            prefix = match.group(1)
            if order == "pvr":
                verdict, reason = match.group(2), match.group(3)
            else:
                reason, verdict = match.group(2), match.group(3)
            reason = _clean_clause(reason)
            if variant.startswith("{2} supports"):
                reason = _sentence_case(reason)
            rewritten = prefix + _render(variant, verdict, reason)
            if rewritten != segment:
                applied += 1
            return rewritten

        text = pattern.sub(_sub, text)
    return text, applied


def _apply_transform_risk_hierarchy(text: str, ticker: str, salt: int) -> tuple[str, int]:
    variants = [
        "{1} {2}",
        "{2} {1}",
    ]
    variant = variants[_seed(ticker, salt=salt) % len(variants)]
    patterns = [
        (re.compile(r"(Strong\s+[^\.]+\.)\s+(Risk[^\.]+\.)"), "sr"),
        (re.compile(r"(Risk[^\.]+\.)\s+(Strong\s+[^\.]+\.)"), "rs"),
    ]

    applied = 0
    for pattern, order in patterns:
        def _sub(match: re.Match[str]) -> str:
            nonlocal applied
            segment = match.group(0)
            if _contains_section_header(segment):
                return segment
            if order == "sr":
                thesis, risk = match.group(1), match.group(2)
            else:
                risk, thesis = match.group(1), match.group(2)
            rewritten = _render(variant, thesis, risk)
            if rewritten != segment:
                applied += 1
            return rewritten

        text = pattern.sub(_sub, text)
    return text, applied


def _apply_transform_bullet_order(text: str, ticker: str, salt: int) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    applied = 0
    rng_seed = _seed(ticker, salt=salt)

    while index < len(lines):
        if not lines[index].startswith("- **"):
            output.append(lines[index])
            index += 1
            continue

        end = index
        while end < len(lines) and lines[end].startswith("- **"):
            end += 1

        block = lines[index:end]
        if 3 <= len(block) <= 5 and not any(_contains_section_header(line) for line in block):
            baseline = sorted(block, key=lambda item: re.sub(r"\s+", " ", item.strip()).casefold())
            shuffled = baseline[:]
            random.Random(rng_seed).shuffle(shuffled)
            shuffled = _rebuild_line_block(shuffled)
            if shuffled != block:
                output.extend(shuffled)
                applied += 1
            else:
                output.extend(block)
        else:
            output.extend(block)
        index = end

    return "".join(output), applied


def _apply_transform_trigger_order(text: str, ticker: str, salt: int) -> tuple[str, int]:
    variants = [
        ("Upgrade Trigger:", "Downgrade Trigger:", "Invalidation:"),
        ("Invalidation:", "Upgrade Trigger:", "Downgrade Trigger:"),
        ("Downgrade Trigger:", "Upgrade Trigger:", "Invalidation:"),
    ]
    order = variants[_seed(ticker, salt=salt) % len(variants)]
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    applied = 0

    while index < len(lines):
        block = lines[index:index + 3]
        if len(block) < 3:
            output.extend(block)
            break

        mapping: dict[str, str] = {}
        valid = True
        for line in block:
            stripped = line.rstrip("\n")
            if _contains_section_header(stripped):
                valid = False
                break
            matched_label = None
            for label in ("Upgrade Trigger:", "Downgrade Trigger:", "Invalidation:"):
                if stripped.startswith(label):
                    matched_label = label
                    break
            if matched_label is None or matched_label in mapping:
                valid = False
                break
            mapping[matched_label] = stripped

        if valid and len(mapping) == 3:
            reordered = [mapping[label] for label in order]
            rebuilt = _rebuild_line_block(reordered, trailing_newline=block[-1].endswith("\n"))
            if rebuilt != block:
                output.extend(rebuilt)
                applied += 1
            else:
                output.extend(block)
            index += 3
            continue

        output.append(lines[index])
        index += 1

    return "".join(output), applied


def apply_structural_variation(
    text: str,
    ticker: str,
    decision_data: dict | None = None,
) -> str:
    """
    Walk report text and apply 3-5 structural transforms deterministically
    based on hash(ticker). Returns transformed text.

    Same ticker → same transforms every time (idempotent, stable).
    Different tickers → potentially different transforms.

    EISAX_DISABLE_VARIATION=1 short-circuits all transforms so the report
    flows in a single fixed sequence across tickers and runs.
    """
    del decision_data

    import os as _os_sv
    if _os_sv.getenv("EISAX_DISABLE_VARIATION", "").strip().lower() in {"1","true","yes","on"}:
        logger.info("[StructVar] %s: skipped (EISAX_DISABLE_VARIATION=1)", ticker)
        return text

    if not text or len(text) < 200 or not ticker:
        logger.info("[StructVar] %s: transforms_applied=%d", ticker, 0)
        return text

    ticker_norm = ticker.upper().strip()
    transforms = [
        _apply_transform_but_clause,
        _apply_transform_verdict_because,
        _apply_transform_risk_hierarchy,
        _apply_transform_bullet_order,
        _apply_transform_trigger_order,
    ]
    active = _select_active_transforms(ticker_norm, len(transforms))

    transforms_applied = 0
    varied = text
    for index, transform in enumerate(transforms):
        if index not in active:
            continue
        varied, applied = transform(varied, ticker_norm, index)
        transforms_applied += applied

    logger.info("[StructVar] %s: transforms_applied=%d", ticker_norm, transforms_applied)
    return varied


if __name__ == "__main__":
    sample = (
        "### Executive Summary\n"
        "The demand outlook is improving across the export book, but margin normalization still limits near-term operating leverage. "
        "Buy because free cash flow remains visible, contract renewal risk is manageable, and balance-sheet flexibility supports capital returns. "
        "Strong backlog coverage improves revenue visibility. Risk remains that input-cost volatility delays the margin recovery case.\n\n"
        "- **Demand**: Volume growth remains durable across key end markets.\n"
        "- **Margins**: Cost discipline is offsetting some inflation pressure.\n"
        "- **Balance Sheet**: Net leverage stays conservative versus peers.\n"
        "- **Catalysts**: Contract wins could tighten consensus estimates.\n\n"
        "Upgrade Trigger: A sustained acceleration in bookings with cleaner gross-margin conversion.\n"
        "Downgrade Trigger: A multi-quarter deterioration in order quality or pricing power.\n"
        "Invalidation: Evidence that utilization gains fail to translate into cash generation.\n\n"
        "The medium-term setup remains constructive because demand durability offsets the current capex cycle drag, while management still faces a measured execution burden. "
        "Portfolio role remains defensive, and the report keeps the canonical verdict taxonomy intact."
    )

    for ticker_value in ["ADNOCGAS.AE", "2222.SR", "AAPL", "NVDA", "TSLA"]:
        print(f"\n=== {ticker_value} ===")
        print(apply_structural_variation(sample, ticker_value)[:400])
