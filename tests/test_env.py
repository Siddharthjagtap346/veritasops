# test_env.py
from veritasops.server.veritasops_environment import VeritasopsEnvironment
from veritasops.models import VeritasopsAction


def test_reset_and_step():
    env = VeritasopsEnvironment("task_easy")
    obs = env.reset()
    assert obs.time_step == 0

    result = env.step(VeritasopsAction(action_type="verify_claim", claim_id="C1"))
    assert result.time_step == 1


def test_invalid_claim_id_penalty():
    env = VeritasopsEnvironment("task_easy")
    env.reset()
    obs = env.step(VeritasopsAction(action_type="verify_claim", claim_id="BAD"))
    assert obs.metadata["reward"] < 0


def test_finalize_works():
    env = VeritasopsEnvironment("task_easy")
    env.reset()
    obs = env.step(VeritasopsAction(action_type="finalize_strategy"))
    assert obs.metadata["done"] is True