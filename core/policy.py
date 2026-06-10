from __future__ import annotations

def apply_policy(
    weights: dict[str, float],
    *,
    universe: list[str] | None = None,
    min_assets: int = 4,
    max_w: float = 0.20,
    min_w: float = 0.0,
    seed_w: float = 0.02
) -> dict[str, float]:
    """
    Institutional policy (smart):
    - Drop < min_w, cap at max_w
    - Renormalize
    - If holdings < min_assets, add assets from universe with small seed weights
      then renormalize (keeps optimizer shape)
    """
    if not weights:
        weights = {}

    w = {str(k).upper().strip(): float(v) for k, v in weights.items()}
    w = {k: v for k, v in w.items() if v > 0 and v >= min_w}

    # cap
    w = {k: min(v, max_w) for k, v in w.items()}

    # renormalize
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    # add missing holdings if possible
    if universe:
        uni = [str(u).upper().strip() for u in universe if u and u.strip()]
        missing = max(0, min_assets - len(w))
        if missing > 0:
            candidates = [t for t in uni if t not in w]
            # add up to missing candidates with small seed weight
            for t in candidates[:missing]:
                w[t] = seed_w

            # re-cap just in case + renormalize again
            w = {k: min(v, max_w) for k, v in w.items()}
            total = sum(w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}

    # final cleanup
    w = {k: float(v) for k, v in w.items() if v > 0}
    return w
