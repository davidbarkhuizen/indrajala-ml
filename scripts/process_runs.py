"""
The one-process-per-measurement machinery the timing scripts share: a worker is this Python
running a script's worker mode, which prints its result as JSON on its last stdout line, and
interleaved_runs spreads the repeats of every cell across processes in a rotating order.

    from process_runs import interleaved_runs, run_json_worker
"""

import json
import subprocess
import sys
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, TypeVar

CellT = TypeVar("CellT", bound=Hashable)


def run_json_worker(command: Sequence[str], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run command in its own process and return the JSON its last stdout line prints."""
    output = subprocess.run(command, env=env, check=True, capture_output=True, text=True).stdout
    return json.loads(output.strip().splitlines()[-1])


def interleaved_runs(
    cells: Sequence[CellT], repeats: int, run_cell: Callable[[CellT], dict[str, Any]]
) -> dict[CellT, list[dict[str, Any]]]:
    """
    {cell: [run_cell(cell) for each repeat]}, each repeat running every cell once. The order
    rotates by one each repeat and reverses on odd repeats, so no cell always follows the same one.
    """
    cells = list(cells)
    runs: dict[CellT, list[dict[str, Any]]] = {cell: [] for cell in cells}
    for repeat in range(repeats):
        order = cells[repeat % len(cells) :] + cells[: repeat % len(cells)]
        if repeat % 2:
            order.reverse()
        for cell in order:
            runs[cell].append(run_cell(cell))
        print(f"repeat {repeat + 1}/{repeats} done", file=sys.stderr, flush=True)
    return runs
