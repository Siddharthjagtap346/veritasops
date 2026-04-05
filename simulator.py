# simulator.py

def apply_spread_dynamics(claim: dict):
    """
    Update spread based on moderation status.

    Improvements:
    - warning labels reduce spread
    - spread limits can nearly flatten virality
    - very low growth can still happen, but not always strongly positive
    """
    multiplier = 1.0

    if claim.get("spread_limited", False):
        multiplier *= 0.35
    elif claim.get("warning_label_active", False):
        multiplier *= 0.70

    growth = claim["spread"]["growth_rate"] * multiplier
    growth = max(0.98, growth)

    claim["spread"]["views"] = int(claim["spread"]["views"] * growth)
    claim["spread"]["shares"] = int(claim["spread"]["shares"] * growth)


def reduce_uncertainty(claim: dict, amount: float = 0.2):
    claim["uncertainty"] = max(0.0, claim["uncertainty"] - amount)


def add_extra_evidence(claim: dict):
    claim["evidence"].append({
        "text": "Additional verification source added",
        "credibility": 0.75
    })
    reduce_uncertainty(claim, 0.15)