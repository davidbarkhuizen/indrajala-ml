"""
The window-recurrent adder's certificate (docs/window-recurrent-claims-workplan.md, E1, stage 2;
scripts/window_recurrent_study.py's certify).

A window's work is a function of the state it receives and its k column tokens, and a snapped state
is one of the network's m codes, exactly (the codes' forward pass gives the code's own vector), so
the network is a finite-state transducer: its states the codes, its input symbols the 27^k windows,
its outputs k digits. Starting from the snapped initial state (W8), breadth first over every window,
certify collects the reachable codes and, for each reachable code q and window w, the digits
D(q, w) and the next code G(q, w). The network is certified when a map phi from its reachable codes
to carries {0, 1, 2}, with phi(start) = 0, has for every reachable q and every w:

- D(q, w) the k sum digits of w's columns with carry in phi(q), and
- phi(G(q, w)) the carry out of w's columns,

and every digit leads by its margin (D3): its log-probability (its logit) at least LOGIT_LEAD ahead
of the next digit's. phi is fixed as the search goes, a code's carry by the first window that
reaches it, then checked by every other. An addition is a start state, a word of windows (the zero
columns of the alignment, W6-i, and the final window, W9, are windows like any other) and the
transducer's outputs, so a certified network adds correctly at every length and alignment.
Conversely, a code reached with two carries fails on the zero window, whose digit is the carry, for
one of them: the certificate is exact, up to its margins.

The snap's margin (D3): a snap holds when every other code is at least SNAP_RATIO times as far as
the code it snaps to. Rounding elsewhere (another batch, another BLAS order) could snap a near tie
to another code, so a near tie branches: every code within the ratio is a next code too, explored
and checked as G's is. A certificate with near ties is still a proof, for either snap; they are
counted (ties) apart from wrong outputs. A digit's near tie can't branch (one of its digits is
wrong): it fails the certificate. (D3 as written failed every near tie; branching proves the same
for both snaps instead, settled in stage 2.)

The search runs every window from every reachable code, failing or not: the counts of wrong digits
and carries, the margins and the minimised machine (the coarsest partition of the reachable codes
that keeps D and G, its size 3, the carry automaton, for a certified network) describe a near miss
too. The first failure in breadth-first order, the fewest windows, is the counterexample, as the
windows from the start and as an addition at alignment 0 (through one zero window more).

A WindowMachine is what certify reads: WindowRecurrentShape's networks through NetworkMachine, or a
stand-in (the tests').
"""

from __future__ import annotations

import hashlib
import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

from indrajala_ml.data.addition_data import BASE, Triple
from indrajala_ml.model.persistence.load_network import load_network

SNAP_RATIO = 1.05  # D3: every other code's distance over the snapped code's, at least
LOGIT_LEAD = 1e-6  # D3: a digit's logit over the next's, at least
CARRIES = 3  # three base-3 operands: a column's sum and carry in are at most 8, its carry out at most 2
BATCH = 8_192  # windows a forward pass

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class Steps(NamedTuple):
    """
    A batch of windows from one code: each window's k digits and their logit leads, the code it
    snaps to and its state's distance to every code.
    """

    digits: IntArray  # (N, k)
    leads: FloatArray  # (N, k)
    codes: IntArray  # (N,)
    distances: FloatArray  # (N, m)


class WindowMachine(Protocol):
    """A network as a transducer over its codes: its window size, its start and its windows from a code."""

    @property
    def columns(self) -> int: ...

    def start(self) -> tuple[int, FloatArray]:
        """The initial state's code and its distance to every code (m,)."""
        ...

    def step(self, code: int, windows: IntArray) -> Steps:
        """windows (N, k) of column tokens, each run from the code's state."""
        ...


def distances_to(H: FloatArray, E: FloatArray) -> FloatArray:
    """Each row of H's Euclidean distance to each code (N, m)."""
    return np.sqrt(((H[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2).sum(axis=2))


def snap_margins(codes: IntArray, distances: FloatArray, tolerance: float) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """
    Each snap's ratio, the nearest other code's distance over the snapped code's (inf when it is
    alone, 1 when both are 0), and the other codes within tolerance of it (N, m), its near ties.
    """
    rows = np.arange(len(codes))
    chosen = distances[rows, codes]
    others = distances.copy()
    others[rows, codes] = np.inf
    nearest_other = others.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(chosen > 0, nearest_other / chosen, np.where(nearest_other > 0, np.inf, 1.0))
        near = others < tolerance * chosen[:, np.newaxis]
    near |= (others == 0) & (chosen == 0)[:, np.newaxis]
    return ratios, near


def logit_leads(probabilities: FloatArray, classes: int) -> tuple[IntArray, FloatArray]:
    """Each column's most probable digit, and its log-probability's lead over the next's."""
    P = probabilities.reshape(probabilities.shape[0], -1, classes)
    top = np.sort(P, axis=2)[:, :, -2:]
    with np.errstate(divide="ignore"):
        leads = np.log(top[:, :, 1]) - np.log(top[:, :, 0])
    return np.argmax(P, axis=2), np.where(top[:, :, 0] == top[:, :, 1], 0.0, leads)


class NetworkMachine:
    """A window-recurrent array network (WindowRecurrentShape, either backend) as a WindowMachine."""

    def __init__(self, network: Any) -> None:
        self.network = network
        self.E = np.array(network.codes.E.tolist(), dtype=np.float64)

    @property
    def columns(self) -> int:
        return int(self.network.columns)

    def start(self) -> tuple[int, FloatArray]:
        H, indices = self.network.first_step()
        return int(indices[0]), distances_to(np.array(H.tolist(), dtype=np.float64), self.E)[0]

    def step(self, code: int, windows: IntArray) -> Steps:
        backend = self.network.backend
        X = backend.matrix(windows.astype(np.float64).tolist())
        C = backend.matrix(np.repeat(self.E[code][np.newaxis, :], len(windows), axis=0).tolist())
        output, H, indices = self.network.window_step(X, C)
        digits, leads = logit_leads(np.array(output.tolist(), dtype=np.float64), self.network.class_count)
        distances = distances_to(np.array(H.tolist(), dtype=np.float64), self.E)
        return Steps(digits, leads, np.array(indices, dtype=np.int64), distances)


def all_windows(k: int) -> IntArray:
    """Every window of k column tokens, (27^k, k), in lexicographic order of its columns."""
    tokens = BASE**3
    return np.array(list(itertools.product(range(tokens), repeat=k)), dtype=np.int64).reshape(-1, k)


def expected(windows: IntArray) -> tuple[IntArray, IntArray]:
    """Per carry in (axis 0): each window's k sum digits (3, N, k) and its carry out (3, N)."""
    sums = windows // (BASE * BASE) + windows // BASE % BASE + windows % BASE
    digits = np.zeros((CARRIES, *windows.shape), dtype=np.int64)
    carry_out = np.zeros((CARRIES, windows.shape[0]), dtype=np.int64)
    for carry in range(CARRIES):
        c = np.full(windows.shape[0], carry, dtype=np.int64)
        for j in range(windows.shape[1]):
            total = sums[:, j] + c
            digits[carry, :, j] = total % BASE
            c = total // BASE
        carry_out[carry] = c
    return digits, carry_out


def as_addition(windows: Sequence[Sequence[int]]) -> tuple[Triple, int]:
    """A word of windows as the addition it is at alignment 0: its triple and its column count."""
    columns = [t for window in windows for t in window]
    operands = [0, 0, 0]
    for t, token in enumerate(columns):
        for i, digit in enumerate((token // (BASE * BASE), token // BASE % BASE, token % BASE)):
            operands[i] += digit * BASE**t
    return (operands[0], operands[1], operands[2]), len(columns)


@dataclass(frozen=True)
class Counterexample:
    """
    The first failure in breadth-first order: kind "digits" (D(q, w) wrong for phi(q)) or "carry"
    (a next code of (q, w) whose carry isn't w's carry out); the windows from the start through w,
    the codes before each, the carry in to w, and the digits expected and given, or the carries.
    """

    kind: str
    windows: list[list[int]]
    codes: list[int]
    carry_in: int
    expected: list[int]
    got: list[int]

    def addition(self) -> tuple[Triple, int]:
        """The counterexample as an addition at alignment 0, through one zero window more (W9)."""
        return as_addition([*self.windows, [0] * len(self.windows[0])])


@dataclass(frozen=True)
class Margin:
    """The smallest margin of a kind, where it is (the code and window), and how many fall short of it."""

    smallest: float
    code: int | None
    window: list[int] | None
    short: int


@dataclass(frozen=True)
class Certificate:
    """certify's result for one network."""

    columns: int
    codes: int
    start: int
    reachable: list[int]  # in breadth-first order, near ties' codes among them
    phi: dict[int, int]  # each reachable code's carry, as first fixed
    wrong_digits: int  # (code, window) pairs whose digits are wrong for the code's carry
    wrong_carries: int  # (code, window) pairs with a next code whose carry is wrong
    wrong_ties: int  # of those, the pairs whose snapped next code is right: wrong only on a near tie
    counterexample: Counterexample | None
    snap: Margin  # short: the near ties, branched
    lead: Margin  # short: the digits' near ties, each a failure
    minimised: list[list[int]]  # the minimised machine's states (over G), each the reachable codes it merges
    evaluations: int
    seconds: float

    @property
    def correct(self) -> bool:
        return self.wrong_digits == 0 and self.wrong_carries == 0

    @property
    def certified(self) -> bool:
        return self.correct and self.lead.short == 0

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["phi"] = {str(code): carry for code, carry in self.phi.items()}
        value.update(correct=self.correct, certified=self.certified)
        if self.counterexample is not None:
            triple, n = self.counterexample.addition()
            value["counterexample"]["addition"] = {"triple": list(triple), "n": n}
        return value


class _Smallest:
    """The running minimum of a margin, its place, and the count under the tolerance."""

    def __init__(self, tolerance: float) -> None:
        self.tolerance = tolerance
        self.value = np.inf
        self.code: int | None = None
        self.window: list[int] | None = None
        self.short = 0

    def add(self, values: FloatArray, code: int, windows: IntArray | None) -> None:
        self.short += int((values < self.tolerance).sum())
        i = int(np.argmin(values))
        if values[i] < self.value:
            self.value = float(values[i])
            self.code = code
            self.window = None if windows is None else [int(t) for t in windows[i]]

    def margin(self) -> Margin:
        return Margin(self.value, self.code, self.window, self.short)


def minimise(
    reachable: Sequence[int], D: Mapping[int, npt.NDArray[np.int8]], G: Mapping[int, npt.NDArray[np.int16]]
) -> list[list[int]]:
    """
    The coarsest partition of the reachable codes in which equivalent codes give the same digits
    and move to equivalent codes on every window (Moore's refinement).
    """
    block = {q: D[q].tobytes() for q in reachable}
    while True:
        ids = {key: i for i, key in enumerate(dict.fromkeys(block.values()))}
        number = np.zeros(max(reachable) + 1, dtype=np.int64)
        for q in reachable:
            number[q] = ids[block[q]]
        refined = {q: block[q] + number[G[q]].tobytes() for q in reachable}
        if len(set(refined.values())) == len(ids):
            break
        block = refined
    classes: dict[bytes, list[int]] = {}
    for q in sorted(reachable):
        classes.setdefault(block[q], []).append(q)
    return sorted(classes.values())


def certify(
    machine: WindowMachine, codes: int, snap_ratio: float = SNAP_RATIO, logit_lead: float = LOGIT_LEAD
) -> Certificate:
    """The certificate of a machine of codes codes (the module's docstring)."""
    begun = time.perf_counter()
    k = machine.columns
    windows = all_windows(k)
    want_digits, want_carry = expected(windows)
    snap, lead = _Smallest(snap_ratio), _Smallest(logit_lead)
    start, start_distances = machine.start()
    ratios, near = snap_margins(np.array([start]), start_distances[np.newaxis, :], snap_ratio)
    snap.add(ratios, start, None)
    starts = [start, *(int(r) for r in np.flatnonzero(near[0]))]
    phi = dict.fromkeys(starts, 0)
    parent: dict[int, tuple[int, int] | None] = dict.fromkeys(starts)
    queue = list(starts)
    D: dict[int, npt.NDArray[np.int8]] = {}
    G: dict[int, npt.NDArray[np.int16]] = {}
    wrong_digits = wrong_carries = wrong_ties = 0
    first: tuple[str, int, int, int] | None = None  # kind, code, window, the next code that failed
    for q in queue:  # breadth first: the queue grows as codes are reached
        parts = [machine.step(q, windows[i : i + BATCH]) for i in range(0, len(windows), BATCH)]
        # narrow: at k 4, 531,441 windows a code
        D[q] = np.concatenate([p.digits for p in parts]).astype(np.int8)
        G[q] = np.concatenate([p.codes for p in parts]).astype(np.int16)
        ties: list[npt.NDArray[np.bool_]] = []
        for i, p in enumerate(parts):
            batch = windows[i * BATCH : (i + 1) * BATCH]
            ratios, near = snap_margins(p.codes, p.distances, snap_ratio)
            snap.add(ratios, q, batch)
            lead.add(p.leads.min(axis=1), q, batch)
            ties.append(near)
        # every next code, (window, code), G's then the near ties', in window order
        tie_windows, tie_codes = np.nonzero(np.concatenate(ties))
        nexts = np.concatenate([np.arange(len(windows)), tie_windows]), np.concatenate([G[q], tie_codes])
        carry = phi[q]
        order = np.lexsort((nexts[1], nexts[0]))
        _, firsts = np.unique(nexts[1][order], return_index=True)  # each next code's first window
        for j in order[np.sort(firsts)]:
            w, r = int(nexts[0][j]), int(nexts[1][j])
            if r not in phi:
                phi[r], parent[r] = int(want_carry[carry][w]), (q, w)
                queue.append(r)
        carried = np.full(codes, -1, dtype=np.int64)
        carried[list(phi)] = list(phi.values())
        bad_digits = np.flatnonzero((D[q] != want_digits[carry]).any(axis=1))
        wrong_next = carried[nexts[1]] != want_carry[carry][nexts[0]]
        bad_carries = np.unique(nexts[0][wrong_next])
        wrong_digits += len(bad_digits)
        wrong_carries += len(bad_carries)
        wrong_ties += len(bad_carries) - int(wrong_next[: len(windows)].sum())
        if first is None and (len(bad_digits) or len(bad_carries)):
            d = int(bad_digits[0]) if len(bad_digits) else len(windows)
            c = int(bad_carries[0]) if len(bad_carries) else len(windows)
            if d <= c:
                first = ("digits", q, d, -1)
            else:
                failed = nexts[1][wrong_next & (nexts[0] == c)]
                first = ("carry", q, c, int(failed[0]))
    counterexample = None
    if first is not None:
        kind, q, w, r = first
        path: list[tuple[int, int]] = [(q, w)]
        while (step := parent[path[-1][0]]) is not None:
            path.append(step)
        path.reverse()
        carry = phi[q]
        counterexample = Counterexample(
            kind,
            [[int(t) for t in windows[i]] for _, i in path],
            [code for code, _ in path],
            carry,
            [int(x) for x in want_digits[carry][w]] if kind == "digits" else [int(want_carry[carry][w])],
            [int(x) for x in D[q][w]] if kind == "digits" else [phi[r]],
        )
    return Certificate(
        k,
        codes,
        start,
        list(queue),
        dict(phi),
        wrong_digits,
        wrong_carries,
        wrong_ties,
        counterexample,
        snap.margin(),
        lead.margin(),
        minimise(queue, D, G),
        len(queue) * len(windows),
        time.perf_counter() - begun,
    )


def certify_file(path: Path, snap_ratio: float = SNAP_RATIO) -> dict[str, Any]:
    """
    A saved network's certificate as JSON, with the file's path, sha256 and snap ratio (1.0: strict,
    the network's own snaps only, no near tie branched).
    """
    network = load_network(str(path))
    certificate = certify(NetworkMachine(network), network.recurrence.codes.count, snap_ratio)
    return {
        "network": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "snap_ratio": snap_ratio,
        **certificate.to_json(),
    }


def summary(result: dict[str, Any]) -> str:
    """One line of a certificate's JSON: the verdict, the codes, the margins, the counterexample."""
    verdict = "certified" if result["certified"] else "correct, a digit's near tie" if result["correct"] else "fails"
    line = (
        f"{verdict}: {len(result['reachable'])} codes reached, minimised to {len(result['minimised'])}, "
        f"{result['wrong_digits']} wrong digits, {result['wrong_carries']} wrong carries "
        f"({result['wrong_ties']} only on a near tie); "
        f"snap ratio >= {result['snap']['smallest']:.3f} ({result['snap']['short']} near ties), "
        f"logit lead >= {result['lead']['smallest']:.3g}; {result['seconds']:.0f} s"
    )
    counterexample = result["counterexample"]
    if counterexample is not None:
        addition = counterexample["addition"]
        line += (
            f"; e.g. {counterexample['kind']} after {len(counterexample['windows'])} windows "
            f"(carry in {counterexample['carry_in']}: expected {counterexample['expected']}, got "
            f"{counterexample['got']}), the addition {tuple(addition['triple'])} at n {addition['n']}"
        )
    return line
