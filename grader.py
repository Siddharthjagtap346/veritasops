#grader.py

def grade_episode(state: dict) -> float:
    claims = state["claims"]
    total = len(claims)

    correct = 0
    harm_control = 0.0
    false_censorship_penalty = 0.0

    # Resource efficiency
    used_ver = (
        state["initial_resources"]["verification_budget"]
        - state["resources"]["verification_budget"]
    )
    used_int = (
        state["initial_resources"]["intervention_budget"]
        - state["resources"]["intervention_budget"]
    )

    max_ver = state["initial_resources"]["verification_budget"]
    max_int = state["initial_resources"]["intervention_budget"]

    budget_efficiency = 1.0
    if max_ver + max_int > 0:
        budget_efficiency = 1.0 - (
            ((used_ver + used_int) / max(1, (max_ver + max_int))) * 0.18
        )

    # Timing quality: finishing too early with unresolved claims should hurt
    unresolved = sum(1 for c in claims if c["status"] == "unresolved")
    time_step = state.get("time_step", 0)
    max_steps = state.get("max_steps", 1)

    if unresolved == 0:
        timing_quality = 1.0
    else:
        step_ratio = time_step / max(1, max_steps)
        if step_ratio < 0.5:
            timing_quality = 0.45
        elif step_ratio < 0.75:
            timing_quality = 0.70
        else:
            timing_quality = 0.85

    false_claims = [c for c in claims if c["ground_truth"] == "false"]

    for claim in claims:
        gt = claim["ground_truth"]
        status = claim["status"]

        # Resolution accuracy
        if gt == "false" and status == "false":
            correct += 1
        elif gt == "true" and status == "supported":
            correct += 1
        elif gt == "uncertain" and status == "uncertain":
            correct += 1

        # Harm control weighted by actual spread containment
        if gt == "false":
            controlled = 0.0

            if claim.get("warning_label_active"):
                controlled += 0.4
            if claim.get("spread_limited"):
                controlled += 0.6

            # Bonus if spread was kept lower
            views = claim["spread"]["views"]
            shares = claim["spread"]["shares"]

            if views < 20000:
                controlled += 0.15
            if shares < 3000:
                controlled += 0.10

            harm_control += min(1.0, controlled)

        # False censorship penalty
        if gt == "true":
            if claim.get("spread_limited"):
                false_censorship_penalty += 1.0
            elif claim.get("warning_label_active"):
                false_censorship_penalty += 0.5

    resolution_accuracy = correct / max(1, total)
    harm_control = harm_control / max(1, len(false_claims))
    false_censorship_avoidance = max(
        0.0, 1.0 - false_censorship_penalty / max(1, total)
    )

    score = (
        0.34 * resolution_accuracy
        + 0.24 * harm_control
        + 0.18 * false_censorship_avoidance
        + 0.14 * budget_efficiency
        + 0.10 * timing_quality
    )

    return round(max(0.0, min(1.0, score)), 4)