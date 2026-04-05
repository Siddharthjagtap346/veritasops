from statistics import mean

from inference import should_finalize
from server.veritasops_environment import VeritasopsEnvironment
from models import VeritasopsAction
from grader import grade_episode


def rule_based_action(obs) -> VeritasopsAction:
    claims = sorted(
        obs.active_claims,
        key=lambda c: (
            c.status == "unresolved",
            c.risk_level,
            c.spread.views,
            c.spread.growth_rate,
            c.priority,
        ),
        reverse=True,
    )

    for c in claims:
        if c.status != "unresolved":
            continue

        # 1) VERY dangerous unresolved claims: verify first if possible
        if (
            c.risk_level >= 0.8
            and c.uncertainty >= 0.45
            and obs.resources.verification_budget > 0
        ):
            return VeritasopsAction(action_type="verify_claim", claim_id=c.claim_id)

        # 2) High-risk unresolved claim: warning label
        if (
            c.risk_level >= 0.75
            and obs.resources.intervention_budget > 0
            and not c.warning_label_active
        ):
            return VeritasopsAction(action_type="add_warning_label", claim_id=c.claim_id)

        # 3) High-risk spreading claim: limit spread
        if (
            c.risk_level >= 0.75
            and obs.resources.intervention_budget > 0
            and not c.spread_limited
        ):
            return VeritasopsAction(action_type="limit_spread", claim_id=c.claim_id)

        # 4) If evidence/uncertainty now points false, resolve it
        if c.uncertainty < 0.35 and c.risk_level >= 0.6:
            return VeritasopsAction(action_type="mark_false", claim_id=c.claim_id)

        # 5) If low-risk and fairly certain, support it
        if c.uncertainty < 0.25 and c.risk_level < 0.45:
            return VeritasopsAction(action_type="mark_supported", claim_id=c.claim_id)

        # 6) Very uncertain: gather more evidence
        if c.uncertainty >= 0.7 and obs.resources.verification_budget > 0:
            return VeritasopsAction(action_type="request_more_evidence", claim_id=c.claim_id)

        # 7) If still unresolved and uncertain enough, close it as uncertain
        if c.uncertainty >= 0.55:
            return VeritasopsAction(action_type="mark_uncertain", claim_id=c.claim_id)

        # 8) Fallback classification for medium-high risk unresolved claims
        if c.risk_level >= 0.6:
            return VeritasopsAction(action_type="mark_false", claim_id=c.claim_id)

        # 9) Otherwise resolve conservatively
        return VeritasopsAction(action_type="mark_uncertain", claim_id=c.claim_id)

    if should_finalize(obs):
        return VeritasopsAction(action_type="finalize_strategy")

    return VeritasopsAction(action_type="finalize_strategy")


def run_task(task_id: str):
    env = VeritasopsEnvironment(task_id=task_id)
    obs = env.reset()
    done = False
    steps = 0

    while not done:
        action = rule_based_action(obs)
        obs = env.step(action)
        done = obs.metadata.get("done", False)
        steps += 1

    final_score = obs.final_score if obs.final_score is not None else grade_episode(env._data)
    return {
        "task_id": task_id,
        "steps": steps,
        "score": round(final_score, 4),
    }


def run_benchmark():
    task_ids = ["task_easy", "task_medium", "task_hard"]
    results = [run_task(task_id) for task_id in task_ids]
    avg = round(mean(r["score"] for r in results), 4)

    print("\n=== VERITASOPS BENCHMARK RESULTS ===")
    for r in results:
        print(f"{r['task_id']}: score={r['score']} steps={r['steps']}")
    print(f"average_score: {avg}")

    return {
        "results": results,
        "average_score": avg,
    }


if __name__ == "__main__":
    run_benchmark()