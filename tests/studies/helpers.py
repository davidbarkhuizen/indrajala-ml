"""
What the study tests share (the DRY third pass workplan, D10): the carry machine as a test double,
and the smallest study settings a real network's run takes.
"""

import numpy as np

from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt

ADDITION = wt.ADDITION

# a real network's run in seconds: one epoch of 64 examples a step, two steps, the tests' smallest
# reads; a test changes what it needs with dataclasses.replace
TINY = wr.Settings(
    epoch_examples=64,
    max_epochs=1,
    patience=1,
    screen=20,
    steps=2,
    test_lengths=(3,),
    sampled=50,
    relation_sampled=10,
    per_stratum=5,
    codes_cases=20,
    seeds=1,
)


class CarryTable:
    """The carry machine at k 1 as a WindowMachine, with flip (state, window) handing on the next carry instead."""

    def __init__(self, flip: tuple[int, int] | None = None) -> None:
        self.flip = flip

    @property
    def columns(self) -> int:
        return 1

    @property
    def codes(self) -> list[int]:
        return [3]

    def start(self) -> tuple[tuple[int, ...], list[np.ndarray]]:
        return (0,), [np.array([0.0, 1.0, 1.0])]

    def step(self, state: tuple[int, ...], windows: np.ndarray) -> wc.Steps:
        (carry,) = state
        labels, after = ADDITION.machine.windows(windows)
        digits, nexts = labels[carry], after[carry].copy()
        if self.flip is not None and self.flip[0] == carry:
            nexts[windows[:, 0] == self.flip[1]] = (nexts[windows[:, 0] == self.flip[1]] + 1) % 3
        distances = np.ones((len(windows), 3))
        distances[np.arange(len(windows)), nexts] = 0.0
        return wc.Steps(digits, np.full(digits.shape, 5.0), nexts[:, np.newaxis], [distances])
