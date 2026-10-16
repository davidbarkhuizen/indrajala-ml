"""The code margins (indrajala_ml/studies/window_replay.py), on a stand-in snapped network at k 1."""

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest

from indrajala_ml.studies import window_replay as wrp
from indrajala_ml.studies import window_tasks as wt

ADDITION = wt.ADDITION
OFFSET = 0.1  # every state's distance from its code, per coordinate


class _Backend:
    @staticmethod
    def matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
        return np.array(rows, dtype=float)


class SnappedCarryNetwork:
    """
    A stand-in that runs the carry machine: its state the carry's one-hot vector plus OFFSET in
    every coordinate, beside a codebook of the one-hot vectors and one code no state is near.
    """

    backend = _Backend()
    columns = 1

    def __init__(self, snapped: bool = True) -> None:
        states = len(ADDITION.machine.transition)
        E = np.vstack([np.eye(states), np.full(states, 5.0)]) if snapped else None
        self.states = states
        self.recurrence = SimpleNamespace(factors=[SimpleNamespace(state_size=states)])
        self.recurrence_layer = SimpleNamespace(
            factors=[SimpleNamespace(codes=None if E is None else SimpleNamespace(E=E))]
        )

    def window_states(self, X: np.ndarray) -> tuple[None, list[np.ndarray], list[np.ndarray]]:
        machine = ADDITION.machine
        rows, width = X.shape
        carry = np.zeros(rows, dtype=int)
        states = [np.eye(self.states)[carry] + OFFSET]
        for t in range(width):
            carry = machine.transition[carry, X[:, t].astype(int)]
            states.append(np.eye(self.states)[carry] + OFFSET)
        return None, states, states


def test_the_margins_read_the_codes_in_use_and_not_the_unused_one():
    margins = wrp.code_margins(SnappedCarryNetwork(), ADDITION, [4, 8], cases=40)
    assert margins is not None
    (factor,) = margins["factors"]
    used = factor["codes_used"]
    assert len(used) >= 2 and max(used) < SnappedCarryNetwork().states  # the far code is never used
    assert factor["closest_used"] == pytest.approx(np.sqrt(2))
    states = SnappedCarryNetwork().states
    assert factor["farthest_state"] == pytest.approx(OFFSET * np.sqrt(states))
    # the next nearest code: one coordinate at 1 + OFFSET, one at 1 - OFFSET, the rest at OFFSET
    q2 = (1 + OFFSET) ** 2 + (1 - OFFSET) ** 2 + (states - 2) * OFFSET**2
    assert margins["least_snap_ratio"] == pytest.approx(np.sqrt(q2 / (states * OFFSET**2)))
    assert "closest pair 1.414" in wrp.margins_text(margins)


def test_an_unsnapped_network_has_no_margins():
    assert wrp.code_margins(SnappedCarryNetwork(snapped=False), ADDITION, [4], cases=10) is None
    assert wrp.margins_text(None) == "no snapped factor"
