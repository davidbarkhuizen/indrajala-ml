"""machine_reading: a network's machine read against its target (the code-consolidation workplan's stage 2)."""

import numpy as np
import pytest

from indrajala_ml.studies import machine_reading as mr
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_tasks as wt

# a mod-3 counter: token 0 the identity, token 1 adds one; the output the count after the token
COUNTER = wt.Mealy(np.array([[0, 1], [1, 2], [2, 0]]), np.array([[0, 1], [1, 2], [2, 0]]))


class Table:
    """A snapped machine given by tables, one factor: next[c][t] the code after t, out[c][t] its label."""

    def __init__(self, next: list[list[int]], out: list[list[int]]) -> None:
        self.next, self.out = next, out

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return [len(self.next)]

    def start(self) -> tuple[wc.State, list[wc.FloatArray]]:
        return (0,), [np.zeros(len(self.next))]

    def step(self, state: wc.State, windows: wc.IntArray) -> wc.Steps:
        tokens = [int(w[0]) for w in windows]
        c = state[0]
        return wc.Steps(
            np.array([[self.out[c][t]] for t in tokens], dtype=np.int64),
            np.ones((len(tokens), 1)),
            np.array([[self.next[c][t]] for t in tokens], dtype=np.int64),
            [np.zeros((len(tokens), len(self.next)))],
        )


def test_a_copy_of_the_target_is_exact_at_every_length_and_breaks_nothing():
    walk = mr.product(Table([[0, 1], [1, 2], [2, 0]], [[0, 1], [1, 2], [2, 0]]), COUNTER)
    assert len(walk.codes) == 3 and len(walk.pairs) == 3
    rows = mr.by_length(walk, [1, 5, 20], mr.word_length(COUNTER))
    assert all(r["last"] == r["exact"] == r["named"] == 1.0 for r in rows)
    result = mr.breaks(walk, mr.word_length(COUNTER))
    assert result["first_codes"] == 3 and result["right_pairs"] == 3
    assert result["first"] == {"broken": 0, "transitions": 6} and result["extra"] == {"broken": 0, "transitions": 0}


def test_an_extra_code_that_breaks_is_counted_apart_from_the_first_codes():
    # code 3 stands for count 0 after 2 + 1 (count 0's first code is 0), right for one token, but its
    # token 1 goes to code 2 (count 2) where count 1 is due
    walk = mr.product(Table([[0, 1], [1, 2], [2, 3], [3, 2]], [[0, 1], [1, 2], [2, 0], [0, 1]]), COUNTER)
    assert len(walk.codes) == 4
    d = mr.word_length(COUNTER)
    extra = walk.pairs.index((3, 0))
    assert mr.right_codes(walk, 1)[extra]  # its outputs right one token ahead
    assert not mr.right_codes(walk, d)[extra]  # but not d tokens ahead
    result = mr.breaks(walk, 1)
    assert result["first_right"] == {"broken": 0, "transitions": 6}  # the core, 0 1 2, exact
    assert result["extra_right"] == {"broken": 1, "transitions": 2}  # code 3's token 1
    rows = mr.by_length(walk, [1, 4, 5, 8], d)
    exact = [r["exact"] for r in rows]
    # the first wrong answer is the fifth token of 1 1 1 1 t: exact to length 4, then falling
    assert exact[:2] == [1.0, 1.0] and exact[-1] < exact[2] < 1.0


def test_a_factor_s_partition_is_named_only_over_every_state_and_in_two_blocks_or_more():
    coordinates = wt.Coordinates(("A",), np.array([[0], [1], [0], [1]]), 2)
    assert wc.name_partition([[0, 2], [1, 3]], coordinates) == "A"
    assert wc.name_partition([[0, 1, 2, 3]], coordinates) is None  # one block: a constant form fits
    assert wc.name_partition([[0], [1]], coordinates) is None  # only some states reached


@pytest.mark.parametrize("name", ["word-problem-s4", "random-4-0", "modular-division-5"])
def test_d_is_the_target_s_distinguishing_length_plus_one(name: str):
    machine = wt.task_named(name).machine
    distinguishing = machine.distinguishing_length()
    assert distinguishing is not None and mr.word_length(machine) == distinguishing + 1
