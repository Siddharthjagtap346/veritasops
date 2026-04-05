# rewards.py

def compute_step_reward(
    action_type: str,
    claim: dict | None,
    valid: bool = True,
    repeated: bool = False,
    premature_finalize: bool = False,
):
    """
    Compute per-step reward.

    Design goals:
    - discourage invalid / repeated / wasteful actions
    - reward sensible moderation behavior
    - avoid making hidden truth the ONLY source of signal
    """

    if premature_finalize:
        return -0.4

    if not valid:
        return -0.5

    if repeated:
        return -0.3

    if claim is None:
        return -0.2

    reward = 0.0

    gt = claim.get("ground_truth")
    risk = claim.get("risk_level", 0.0)
    unc = claim.get("uncertainty", 0.0)

    # Verification / investigation actions
    if action_type == "verify_claim":
        if unc > 0.45 and risk > 0.55:
            reward += 0.22
        elif unc > 0.35:
            reward += 0.10
        else:
            reward -= 0.08

    elif action_type == "request_more_evidence":
        if unc > 0.55:
            reward += 0.15
        elif unc > 0.40:
            reward += 0.08
        else:
            reward -= 0.10

    # Resolution actions
    elif action_type == "mark_false":
        reward += 0.20 if risk > 0.60 else 0.05
        if gt == "false":
            reward += 0.35
        else:
            reward -= 0.45

    elif action_type == "mark_supported":
        reward += 0.12 if unc < 0.40 else -0.05
        if gt == "true":
            reward += 0.35
        else:
            reward -= 0.45

    elif action_type == "mark_uncertain":
        reward += 0.12 if unc > 0.55 else 0.0
        if gt == "uncertain":
            reward += 0.25
        elif gt == "true" and unc > 0.65:
            reward += 0.05
        else:
            reward -= 0.15

    # Interventions
    elif action_type == "add_warning_label":
        if risk > 0.65:
            reward += 0.18
        else:
            reward += 0.04

        if gt == "false":
            reward += 0.18
        elif gt == "true":
            reward -= 0.18

    elif action_type == "limit_spread":
        if risk > 0.75:
            reward += 0.22
        else:
            reward += 0.03

        if gt == "false":
            reward += 0.22
        elif gt == "true":
            reward -= 0.40

    elif action_type == "broadcast_alert":
        if risk > 0.85:
            reward += 0.20
        else:
            reward -= 0.05

        if gt == "false":
            reward += 0.25
        elif gt == "true":
            reward -= 0.50

    elif action_type == "set_priority":
        reward += 0.05

    elif action_type == "finalize_strategy":
        reward += 0.0

    return round(reward, 4)