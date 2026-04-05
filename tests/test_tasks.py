# test_tasks.py
from veritasops.tasks import list_tasks, load_task


def test_all_tasks_load():
    for task_id in list_tasks():
        task = load_task(task_id)
        assert "claims" in task
        assert "resources" in task