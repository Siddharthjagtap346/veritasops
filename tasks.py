# tasks.py
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_task(task_id: str):
    mapping = {
        "task_easy": "task_easy.json",
        "task_medium": "task_medium.json",
        "task_hard": "task_hard.json",
    }

    if task_id not in mapping:
        raise ValueError(f"Unknown task_id: {task_id}")

    path = DATA_DIR / mapping[task_id]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_tasks():
    return ["task_easy", "task_medium", "task_hard"]