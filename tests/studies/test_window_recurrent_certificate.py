"""
The window-recurrent adder's certificate (indrajala_ml/studies/window_recurrent_certificate.py)
against stand-in transducers that add, drop a carry, carry only inside a window or snap or choose a
digit on a near tie, and against real networks, whose machine is checked against their own
forward pass.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from indrajala_ml.data import addition_data as ad
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import Codes, WindowRecurrence
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr

ALL_TWOS = 26  # the column token of digits (2, 2, 2): column sum 6
NEAR = 1.01  # a distance ratio inside SNAP_RATIO


class CarryMachine:
    """
    A stand-in transducer: code q holds the carry q % 3, copies codes a carry (q = carry + 3 *
    copy, the copy the window's last token's parity), each window's digits the sum's and its next
    code its carry out's, every other code twice as far as the one it snaps to. Its faults:
    "drops" hands a carry out of 2 on as 1; "inside" hands on no carry (it carries only inside a
    window); "snap-tie" puts the code of the next carry, "copy-tie" the carry's other copy, at
    NEAR after an all-twos window; "digit-tie" gives an all-twos window's digits a lead of 1e-9.
    """

    def __init__(self, k: int, copies: int = 1, fault: str | None = None) -> None:
        self.columns = k
        self.copies = copies
        self.fault = fault
        self.codes = 3 * copies

    def start(self) -> tuple[int, np.ndarray]:
        return 0, self._distances(np.zeros(1, dtype=np.int64))[0]

    def _distances(self, codes: np.ndarray) -> np.ndarray:
        distances = np.full((len(codes), self.codes), 2.0)
        distances[np.arange(len(codes)), codes] = 1.0
        return distances

    def step(self, code: int, windows: np.ndarray) -> wc.Steps:
        want_digits, want_carry = wc.expected(windows)
        digits, carry = want_digits[code % 3], want_carry[code % 3].copy()
        if self.fault == "drops":
            carry[carry == 2] = 1
        if self.fault == "inside":
            carry[:] = 0
        codes = carry + 3 * (windows[:, -1] % self.copies)
        distances = self._distances(codes)
        twos = (windows == ALL_TWOS).all(axis=1)
        if self.fault == "snap-tie":
            distances[twos, (carry[twos] + 1) % 3 + 3 * (windows[twos, -1] % self.copies)] = NEAR
        if self.fault == "copy-tie":
            distances[twos, carry[twos] + 3 * ((windows[twos, -1] + 1) % self.copies)] = NEAR
        leads = np.full(digits.shape, 5.0)
        if self.fault == "digit-tie":
            leads[twos] = 1e-9
        return wc.Steps(digits, leads, codes, distances)


class MachineNetwork:
    """A WindowMachine as the network WindowAdder reads: each example's windows run in turn from the start."""

    def __init__(self, machine: CarryMachine) -> None:
        self.machine = machine

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> list[tuple[float, ...]]:
        return [tokens for tokens, _labels in rows]

    def classify_rows(self, prepared: Sequence[tuple[float, ...]]) -> list[tuple[int, ...]]:
        k = self.machine.columns
        predictions: list[tuple[int, ...]] = []
        for tokens in prepared:
            code, digits = self.machine.start()[0], list[int]()
            for w in range(0, len(tokens), k):
                steps = self.machine.step(code, np.array([tokens[w : w + k]], dtype=np.int64))
                digits += [int(d) for d in steps.digits[0]]
                code = int(steps.codes[0])
            predictions.append(tuple(digits))
        return predictions


def fails_as_said(counterexample: wc.Counterexample, machine: CarryMachine) -> bool:
    """The counterexample's windows run through the machine give what it says they give."""
    code, digits = machine.start()[0], list[int]()
    for window, before in zip(counterexample.windows, counterexample.codes, strict=True):
        assert code == before
        steps = machine.step(code, np.array([window], dtype=np.int64))
        code, digits = int(steps.codes[0]), [int(d) for d in steps.digits[0]]
    if counterexample.kind == "digits":
        return digits == counterexample.got != counterexample.expected
    return counterexample.got != counterexample.expected


# the stand-ins


@pytest.mark.parametrize("k", [1, 2, 4])
def test_an_adding_machine_is_certified_and_minimises_to_the_carry_automaton(k: int):
    certificate = wc.certify(CarryMachine(k), 3)
    assert certificate.certified and certificate.counterexample is None
    assert certificate.reachable == [0, 1, 2] and certificate.phi == {0: 0, 1: 1, 2: 2}
    assert certificate.minimised == [[0], [1], [2]]
    assert certificate.snap.smallest == 2.0 and certificate.snap.short == 0
    assert certificate.evaluations == 3 * 27**k


def test_copies_of_a_carry_are_certified_and_merged_by_the_minimised_machine():
    certificate = wc.certify(CarryMachine(2, copies=2), 6)
    assert certificate.certified and sorted(certificate.reachable) == list(range(6))
    assert certificate.phi == {q: q % 3 for q in range(6)}
    assert certificate.minimised == [[0, 3], [1, 4], [2, 5]]


@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_machine_that_drops_a_carry_of_two_fails_on_the_first_all_twos_window(k: int):
    certificate = wc.certify(CarryMachine(k, fault="drops"), 3)
    assert not certificate.certified and certificate.wrong_digits == 0 and certificate.wrong_carries > 0
    assert certificate.wrong_ties == 0
    assert certificate.reachable == [0, 1]
    counterexample = certificate.counterexample
    assert counterexample is not None and counterexample.kind == "carry"
    # the smallest: the first window in order whose carry out is 2, from the start
    assert counterexample.windows == [[0] * (k - 1) + [ALL_TWOS]] and counterexample.codes == [0]
    assert counterexample.expected == [2] and counterexample.got == [1]
    assert fails_as_said(counterexample, CarryMachine(k, fault="drops"))
    triple, n = counterexample.addition()
    assert triple == (2 * 3 ** (k - 1),) * 3 and n == 2 * k


@pytest.mark.parametrize("k", [1, 2, 4])
def test_a_machine_that_carries_only_inside_a_window_fails_on_the_first_carry_out(k: int):
    certificate = wc.certify(CarryMachine(k, fault="inside"), 3)
    assert not certificate.certified and certificate.reachable == [0]
    counterexample = certificate.counterexample
    assert counterexample is not None and counterexample.kind == "carry"
    # token 5 is the digits (0, 1, 2), the first column sum of 3 in token order
    assert counterexample.windows == [[0] * (k - 1) + [5]]
    assert counterexample.expected == [1] and counterexample.got == [0]
    # as an addition at alignment 0, through the zero window, the network gets its sum wrong
    triple, n = counterexample.addition()
    tokens, labels = wr.example(triple, n - k, k, 0)
    network = MachineNetwork(CarryMachine(k, fault="inside"))
    assert network.classify_rows([tokens]) != [labels]


def test_a_snap_near_tie_branches_and_fails_when_the_other_code_carries_otherwise():
    certificate = wc.certify(CarryMachine(2, fault="snap-tie"), 3)
    assert not certificate.certified and certificate.snap.short > 0
    assert certificate.snap.smallest == pytest.approx(NEAR)
    counterexample = certificate.counterexample
    # the network's own snap is right; the near tie's other code isn't
    assert certificate.wrong_ties == certificate.wrong_carries > 0
    assert counterexample is not None and counterexample.kind == "carry"
    assert counterexample.windows == [[ALL_TWOS, ALL_TWOS]] and counterexample.expected == [2]
    assert counterexample.got == [0]


def test_a_snap_near_tie_between_copies_of_a_carry_is_certified_and_counted():
    certificate = wc.certify(CarryMachine(2, copies=2, fault="copy-tie"), 6)
    assert certificate.certified and certificate.snap.short > 0
    assert certificate.snap.smallest == pytest.approx(NEAR) and certificate.snap.window is not None


def test_a_digits_near_tie_fails_the_certificate_though_every_digit_is_right():
    certificate = wc.certify(CarryMachine(1, fault="digit-tie"), 3)
    assert certificate.correct and not certificate.certified
    assert certificate.lead.short == 3 and certificate.lead.window == [ALL_TWOS]
    assert certificate.to_json()["certified"] is False


def test_a_certified_machine_passes_the_long_tests():
    k = 4
    assert wc.certify(CarryMachine(k, copies=2), 6).certified
    model = wr.WindowAdder(MachineNetwork(CarryMachine(k, copies=2)), k)
    catalogue = ad.catalogue(sampled=20, per_stratum=2, doubling=True)
    long = [p for p in catalogue if p.id in ("B1", "M3")]
    results = ad.evaluate(model.adder(1024), 1024, long, shrink=False)
    assert all(r.holds for r in results)
    assert {r.id: r.reach() for r in results}["M3"] == 1024


def test_a_counterexample_is_json_with_its_addition():
    value = wc.certify(CarryMachine(1, fault="drops"), 3).to_json()
    assert value["certified"] is False and value["phi"] == {"0": 0, "1": 1}
    assert value["counterexample"]["addition"] == {"triple": [2, 2, 2], "n": 2}
    assert "fails" in wc.summary({**value, "seconds": 0.0})


# real networks


@pytest.mark.parametrize("network_class", [WindowRecurrentArrayNetwork, WindowRecurrentRustArrayNetwork])
def test_a_networks_machine_is_its_forward_pass_window_by_window(network_class: Any):
    k = 2
    network: Any = network_class.randomized((k,), wr.window_specs(), WindowRecurrence(4, Codes(4)), Adam(), seed=3)
    machine = wc.NetworkMachine(network)
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 27, size=(40, 3 * k))
    expected = network.classify_rows(network.prepare_dataset([(tuple(map(float, r)), (0,) * len(r)) for r in rows]))
    codes = network.state_codes(network.backend.matrix(rows.astype(float).tolist()))
    for row, want, want_codes in zip(rows, expected, codes, strict=True):
        code, distances = machine.start()
        assert code == want_codes[0] and int(np.argmin(distances)) == code
        digits: list[int] = []
        for w, window in enumerate(row.reshape(-1, k)):
            steps = machine.step(code, window[np.newaxis, :])
            digits += [int(d) for d in steps.digits[0]]
            code = int(steps.codes[0])
            assert code == want_codes[w + 1]
        assert tuple(digits) == tuple(want)


def test_an_untrained_networks_counterexample_is_its_own_output():
    k = 2
    network = wr.build(k, 4, 4, seed=1)
    certificate = wc.certify(wc.NetworkMachine(network), 4)
    assert not certificate.certified and set(certificate.reachable) <= set(range(4))
    counterexample = certificate.counterexample
    assert counterexample is not None
    tokens = [float(t) for window in counterexample.windows for t in window]
    (prediction,) = network.classify_rows(network.prepare_dataset([(tuple(tokens), (0,) * len(tokens))]))
    if counterexample.kind == "digits":
        assert list(prediction[-k:]) == counterexample.got != counterexample.expected
