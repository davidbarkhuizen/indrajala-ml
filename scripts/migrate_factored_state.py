"""
Rewrites files written before the factored state (the factored-state workplan, D9: old files are
migrated, not read through shims) into its layout, in place, printing each file's old and new
sha256.

A saved window-recurrent network (format 2): its "recurrence", {"state_size": s, "codes": ...,
"cell": ..., "pooling": ...}, becomes {"factors": [{"state_size": s, "codes": ...}], "cross":
false, "cell": ..., "pooling": ...}, the one factor it had. Its weights and optimizer state are
unchanged: a factor's parts are in the order the network's parts were (pooling, cell, codes,
initial state), so the file loads as the same network, by bits. A file already migrated is left
alone.

    .venv/bin/python scripts/migrate_factored_state.py network FILE...

Stage 4 migrates the saved-model fixtures with it; stage 5 adds the studies' results and
certificates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def migrate_network(state: dict[str, Any]) -> dict[str, Any]:
    """A format-2 network's state with its recurrence in the factored layout; other shapes unchanged."""
    recurrence = state.get("recurrence")
    if recurrence is None or "factors" in recurrence:
        return state
    factored = {
        "factors": [{"state_size": recurrence["state_size"], "codes": recurrence["codes"]}],
        "cross": False,
        "cell": recurrence["cell"],
        "pooling": recurrence["pooling"],
    }
    return {key: factored if key == "recurrence" else value for key, value in state.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate_network_file(path: Path) -> tuple[str, str]:
    """path rewritten by migrate_network, as save() writes (model_io.py); its old and new sha256."""
    old = _sha256(path)
    state = json.loads(path.read_text())
    migrated = migrate_network(state)
    if migrated is not state:
        path.write_text(json.dumps(migrated))
    return old, _sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    network = commands.add_parser("network", help="saved window-recurrent networks (format 2)")
    network.add_argument("files", nargs="+", type=Path)
    arguments = parser.parse_args()
    for path in arguments.files:
        old, new = migrate_network_file(path)
        print(f"{path}  {old}  {new}{'  (unchanged)' if old == new else ''}")


if __name__ == "__main__":
    main()
