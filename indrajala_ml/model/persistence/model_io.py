import json
from typing import Any


def save_json(path: str, state: dict[str, Any]) -> None:
    """Writes state as JSON: a format-2 file (format2.py)."""
    with open(path, "w") as f:
        json.dump(state, f)


def load_json(path: str) -> dict[str, Any]:
    """
    Reads a JSON file, with no envelope assumptions: a format-2 file or a legacy envelope.
    """
    with open(path) as f:
        return json.load(f)
