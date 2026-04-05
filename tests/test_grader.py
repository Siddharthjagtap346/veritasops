# test_grader.py
from veritasops.grader import grade_episode
from veritasops.server.veritasops_environment import VeritasopsEnvironment


def test_grader_range():
    env = VeritasopsEnvironment("task_easy")
    env.reset()
    score = grade_episode(env._data)
    assert 0.0 <= score <= 1.0


def test_debug_state_exists():
    env = VeritasopsEnvironment("task_easy")
    env.reset()
    state = env.debug_state
    assert "claims" in state
    assert "resources" in state