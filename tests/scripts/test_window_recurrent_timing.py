"""
scripts/window_recurrent_timing.py: the case names parse to the cells they name, the old names
keep their meaning, and each kind's step runs and its breakdown accounts for the step. Nothing is timed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import window_recurrent_timing as timing  # scripts/ isn't a package


def test_the_old_names_are_the_transformer_s_training_step() -> None:
    assert timing.parse_case("k1-s4-m16-B32-W12") == (
        "learn",
        "transformer",
        {"k": 1, "s": 4, "m": 16, "B": 32, "W": 12},
        None,
    )


def test_every_default_case_parses() -> None:
    parsed = [timing.parse_case(case) for case in timing.DEFAULT_CASES]
    assert [kind for kind, _, _, _ in parsed] == ["learn"] * 4 + ["forward"] * 2 + ["long-test"]
    assert parsed[-1] == ("long-test", "family", {"p": 7, "f": 32, "n": 1024, "c": 1023}, "U1")
    assert parsed[3] == ("learn", "family", {"p": 7, "f": 32, "B": 64, "W": 16}, None)


@pytest.mark.parametrize(
    "case",
    [
        "learn-linear-k1-s16-m16-B4-W3",
        "learn-family-p5-f16-B4-W3",
        "forward-linear-k1-s16-m16-n8-r4",
        "forward-family-p5-f16-n8-r4",
        "long-test-family-p5-f16-U1-n8-c5",
        "study-family-p5-f16-e2",
    ],
)
def test_each_kind_runs_and_its_breakdown_accounts_for_the_step(case: str) -> None:
    step, inference = timing.case_step(case)
    step()
    assert inference == (not case.startswith("learn"))
    lines = timing.breakdown(case)
    shares = [float(line.rsplit(", ", 1)[1].rstrip("%")) for line in lines[1:]]
    assert all(share >= 0 for share in shares) and sum(shares) == pytest.approx(100, abs=0.5)
