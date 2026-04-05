# inference.py
import json
import os
from typing import Optional

from openai import OpenAI

from server.veritasops_environment import VeritasopsEnvironment
from models import VeritasopsAction
from grader import grade_episode
# =========================
# REQUIRED ENV VARIABLES
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "veritasops"
TASKS = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS = 8
TEMPERATURE = 0.0


# =========================
# RULE-BASED FALLBACK POLICY
# =========================
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

    unresolved = [c for c in claims if c.status == "unresolved"]

    # If nothing unresolved, finalize
    if not unresolved:
        return VeritasopsAction(action_type="finalize_strategy")

    for c in unresolved:
        # 1) Very dangerous + uncertain → verify first
        if (
            c.risk_level >= 0.8
            and c.uncertainty >= 0.45
            and obs.resources.verification_budget > 0
        ):
            return VeritasopsAction(action_type="verify_claim", claim_id=c.claim_id)

        # 2) Dangerous unresolved claim → add warning label
        if (
            c.risk_level >= 0.75
            and obs.resources.intervention_budget > 0
            and not c.warning_label_active
        ):
            return VeritasopsAction(action_type="add_warning_label", claim_id=c.claim_id)

        # 3) Dangerous spreading claim → limit spread
        if (
            c.risk_level >= 0.75
            and obs.resources.intervention_budget > 0
            and not c.spread_limited
        ):
            return VeritasopsAction(action_type="limit_spread", claim_id=c.claim_id)

        # 4) Strongly false-looking → resolve as false
        if c.uncertainty < 0.35 and c.risk_level >= 0.6:
            return VeritasopsAction(action_type="mark_false", claim_id=c.claim_id)

        # 5) Likely true and low-risk → support
        if c.uncertainty < 0.25 and c.risk_level < 0.45:
            return VeritasopsAction(action_type="mark_supported", claim_id=c.claim_id)

        # 6) Very uncertain → request more evidence
        if c.uncertainty >= 0.7 and obs.resources.verification_budget > 0:
            return VeritasopsAction(action_type="request_more_evidence", claim_id=c.claim_id)

        # 7) Still uncertain → resolve as uncertain
        if c.uncertainty >= 0.55:
            return VeritasopsAction(action_type="mark_uncertain", claim_id=c.claim_id)

        # 8) Medium/high-risk unresolved → false
        if c.risk_level >= 0.6:
            return VeritasopsAction(action_type="mark_false", claim_id=c.claim_id)

        # 9) Conservative fallback
        return VeritasopsAction(action_type="mark_uncertain", claim_id=c.claim_id)

    # Only finalize if truly nothing useful remains
    if should_finalize(obs):
        return VeritasopsAction(action_type="finalize_strategy")

    # Emergency fallback: try to resolve anything left
    for c in unresolved:
        if c.uncertainty >= 0.5:
            return VeritasopsAction(action_type="mark_uncertain", claim_id=c.claim_id)
        return VeritasopsAction(action_type="mark_false", claim_id=c.claim_id)

    return VeritasopsAction(action_type="finalize_strategy")


# =========================
# LLM HELPERS
# =========================
def format_observation_for_model(obs) -> str:
    claims_payload = []
    for c in obs.active_claims:
        claims_payload.append(
            {
                "claim_id": c.claim_id,
                "text": c.text,
                "uncertainty": c.uncertainty,
                "risk_level": c.risk_level,
                "views": c.spread.views,
                "shares": c.spread.shares,
                "growth_rate": c.spread.growth_rate,
                "status": c.status,
                "warning_label_active": c.warning_label_active,
                "spread_limited": c.spread_limited,
                "priority": c.priority,
                "evidence": [
                    {"text": e.text, "credibility": e.credibility}
                    for e in c.evidence
                ],
            }
        )

    payload = {
        "time_step": obs.time_step,
        "max_steps": obs.max_steps,
        "remaining_steps": obs.remaining_steps,
        "resources": {
            "verification_budget": obs.resources.verification_budget,
            "intervention_budget": obs.resources.intervention_budget,
        },
        "incoming_reports": obs.incoming_reports,
        "last_action_result": obs.last_action_result,
        "active_claims": claims_payload,
    }

    return json.dumps(payload, ensure_ascii=False)


def get_llm_action(obs) -> Optional[VeritasopsAction]:
    if not HF_TOKEN:
        return None

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )

        system_prompt = """You are an AI misinformation crisis coordinator.

Choose exactly ONE best next moderation action.

Your goal:
1. Reduce harmful misinformation spread
2. Resolve high-risk claims before low-risk ones
3. Avoid unnecessary censorship
4. Use verification and intervention budgets efficiently
5. Do NOT finalize early if unresolved risky claims remain

Preferred strategy:
- For dangerous uncertain claims: verify first
- For harmful spreading claims: warn and/or limit spread
- For likely false claims: mark_false
- For highly uncertain unresolved claims: request_more_evidence or mark_uncertain
- For low-risk well-supported claims: mark_supported
- Finalize only when no meaningful action remains

Return ONLY valid JSON in this exact schema:
{
  "action_type": "verify_claim | request_more_evidence | mark_supported | mark_false | mark_uncertain | add_warning_label | limit_spread | broadcast_alert | set_priority | finalize_strategy",
  "claim_id": "C1 or null",
  "priority_level": "low | medium | high | null"
}

Do not explain.
Do not include markdown.
Return JSON only.
"""

        user_prompt = format_observation_for_model(obs)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Clean accidental code fences if model misbehaves
        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)

        return VeritasopsAction(
            action_type=data["action_type"],
            claim_id=data.get("claim_id"),
            priority_level=data.get("priority_level"),
        )

    except Exception:
        return None

def is_reasonable_action(obs, action: VeritasopsAction) -> bool:
    if action.action_type == "finalize_strategy":
        return should_finalize(obs)

    if action.claim_id is None:
        return action.action_type in ["broadcast_alert", "finalize_strategy"]

    claim = next((c for c in obs.active_claims if c.claim_id == action.claim_id), None)
    if claim is None:
        return False

    if claim.status != "unresolved":
        return False

    if action.action_type == "verify_claim" and obs.resources.verification_budget <= 0:
        return False

    if action.action_type in ["add_warning_label", "limit_spread"] and obs.resources.intervention_budget <= 0:
        return False

    if action.action_type == "add_warning_label" and claim.warning_label_active:
        return False

    if action.action_type == "limit_spread" and claim.spread_limited:
        return False

    return True

# =========================
# LOGGING HELPERS (MANDATORY FORMAT)
# =========================
def format_action(action: VeritasopsAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), ensure_ascii=False, separators=(",", ":"))


def format_reward(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "0.00"


def safe_error(obs) -> str:
    msg = getattr(obs, "last_action_result", None)
    if not msg:
        return "null"

    lowered = msg.lower()
    if any(
        bad in lowered
        for bad in [
            "invalid",
            "missing",
            "no verification budget",
            "no intervention budget",
            "error",
            "failed",
        ]
    ):
        return msg

    return "null"


# =========================
# TASK RUNNER
# =========================
def run_task(task_id: str) -> float:
    env = VeritasopsEnvironment(task_id=task_id)
    obs = env.reset()

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    rewards = []
    step_num = 0
    success = False

    try:
        done = obs.metadata.get("done", False)

        while not done and step_num < MAX_STEPS:
            step_num += 1

            action = get_llm_action(obs)

            if action is None or not is_reasonable_action(obs, action):
                action = rule_based_action(obs)

            obs = env.step(action)

            reward = obs.metadata.get("reward", 0.0)
            done = obs.metadata.get("done", False)
            error_msg = safe_error(obs)

            rewards.append(float(reward))

            print(
                f"[STEP] step={step_num} "
                f"action={format_action(action)} "
                f"reward={format_reward(reward)} "
                f"done={'true' if done else 'false'} "
                f"error={error_msg}"
            )

        score = grade_episode(env._data)
        success = score >= 0.7

        return score

    except Exception as e:
        print(
            f"[STEP] step={step_num + 1} "
            f"action=null "
            f"reward=0.00 "
            f"done=true "
            f"error={str(e)}"
        )
        return 0.0

    finally:
        rewards_str = ",".join(format_reward(r) for r in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_num} "
            f"score={score:.4f} "
            f"rewards={rewards_str}"
        )
# =========================
# FINALIZATION CHECK            
# =========================

def should_finalize(obs) -> bool:
    unresolved = [c for c in obs.active_claims if c.status == "unresolved"]

    # If anything is still unresolved, try not to finalize
    if not unresolved:
        return True

    verification_left = obs.resources.verification_budget > 0
    intervention_left = obs.resources.intervention_budget > 0

    # If we still have any resources, keep working
    if verification_left or intervention_left:
        return False

    # No resources left = forced finalize
    return True

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    scores = {}

    for task in TASKS:
        score = run_task(task)
        scores[task] = score

    avg = round(sum(scores.values()) / len(scores), 4)

    print("\n=== FINAL SCORES ===")
    for task, score in scores.items():
        print(f"{task}: {score:.4f}")
    print(f"average: {avg:.4f}")