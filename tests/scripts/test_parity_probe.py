import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import parity_probe  # scripts/ isn't a package


def _record(path: Path, outputs: list[float]) -> Path:
    inputs = [0.0, math.inf, -0.0]
    arrays = {"op/inputs": parity_probe._bits(inputs), "op/outputs": parity_probe._bits(outputs)}
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]
    return path


def test_compare_counts_a_nan_of_the_other_sign_as_a_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    negative_nan = struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000000))[0]
    first = _record(tmp_path / "a.npz", [1.0, math.nan, 0.0])
    second = _record(tmp_path / "b.npz", [1.0, negative_nan, 0.0])
    assert parity_probe.compare(str(first), str(first))
    assert not parity_probe.compare(str(first), str(second))
    lines = capsys.readouterr().out.splitlines()
    assert lines[-3:] == [
        "op: 2 of 3 identical",
        "  at inf (0x7ff0000000000000): nan (0x7ff8000000000000) against nan (0xfff8000000000000)",
        "NOT bit-identical",
    ]


def test_inputs_are_the_same_on_every_run() -> None:
    assert parity_probe._uniform("exp near", 3, -20.0, 20.0) == parity_probe._uniform("exp near", 3, -20.0, 20.0)
    assert len(parity_probe.probes()["adam"][0]) == len(parity_probe.ADAM_STEPS)
